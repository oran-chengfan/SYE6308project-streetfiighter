"""Single-machine raw-observation recurrent Double DQN and branch replay.

Both arms share actor, learner, exploration and total native-frame credits.
There is no PPO rollout, action-probability target or imitation prerequisite.
"""
import argparse
import copy
import json
import random
import time
from collections import Counter, deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.league_adaptive.runtime import ROOT, Actor, setup, write, save_fight, restore_fight
from experiments.league_adaptive.characters import CHARACTERS, proposal_sequence
from experiments.league_adaptive.environment import FightEnv
from experiments.league_adaptive.evaluate import evaluate
from experiments.ryu_direct_league.policy import save
from experiments.ryu_counterq.policy import random_sequence_q, construct_sequence_q
from experiments.ryu_counterq.replay import Episode,SequenceReplay,sequence_double_q_targets,recurrent_drift_audit,TrajectoryRecorder
from experiments.ryu_counterq.training_state import (
    learner_components,load_checkpoint,save_checkpoint,restore_components,random_state,restore_random_state,progress_metadata)


def initial_pool(stars=(1,), styles=('mixed',),charge_decisions=10):
    pool=[]
    for char in CHARACTERS:
        for star in stars:pool.append(dict(id=f'cpu_{char}_{star}',kind='cpu',character=char,stars=star))
        for style in styles:pool.append(dict(id=f'{style}_{char}',kind='generic',style=style,character=char,period=24,charge_decisions=charge_decisions))
    return pool


class Exploration:
    def __init__(self,seed,probability=.5,charge_decisions=10,complete_half_circle=False,proposal_style='mixed'):
        self.rng=np.random.default_rng(seed);self.pending=deque();self.probability=probability
        self.charge_decisions=int(charge_decisions)
        self.complete_half_circle=bool(complete_half_circle)
        self.proposal_style=proposal_style
    def choose(self,greedy,epsilon,definition,raw,player):
        if self.pending:return self.pending.popleft()
        if self.rng.random()>=epsilon:return greedy
        if self.rng.random()<self.probability:
            seq=proposal_sequence(definition['actions'],raw,player,self.rng,style=self.proposal_style,charge_decisions=self.charge_decisions,
                                  complete_half_circle=self.complete_half_circle)
            self.pending.extend(seq[1:]);return seq[0]
        return int(self.rng.integers(len(definition['actions'])))


def sample_sequence_batch(replay,cfg):
    """One sampling implementation for training and frozen-stream diagnostics."""
    group=None
    if cfg.get('replay_group_weights'):
        available=sorted({replay.episodes[k[0]]['group'] for k in replay.priorities})
        mass=np.array([cfg['replay_group_weights'][g] for g in available],float)
        group=available[int(replay.rng.choice(len(available),p=mass/mass.sum()))]
    return replay.sample(cfg['batch_size'],beta=cfg['priority_beta'],group=group)


def optimize(policy,target,optimizer,replay,cfg,*,data=None,audit=False,current_update=None):
    if replay is None and data is None:raise ValueError('A fixed audit batch or live replay is required')
    data=sample_sequence_batch(replay,cfg) if data is None else data
    burn=data['burn'];learn=cfg['learn_length']
    extra={} if 'input_durations' not in data else dict(input_durations=data['input_durations'])
    policy.set_training_mode(True)
    q,_=policy.observation_sequence_values(data['images'],data['positions'],data['previous'],data['hidden'],burn,**extra)
    with torch.no_grad():
        tq,_=target.observation_sequence_values(data['images'],data['positions'],data['previous'],data['hidden'],burn,**extra)
        rule=cfg.get('backup_rule');trace_mask=None
        if rule=='collector_cut':
            # Conditional collector-policy prefixes, current-online selection
            # at the final Double-Q bootstrap. This retains policy-lag bias;
            # it is not an unbiased current-greedy return. Unknown old labels
            # fall back to strict current-greedy cuts, never guessed programs.
            trace_mask=torch.where(data['collector_known'],data['collector_greedy'],data['actions']==q.detach().argmax(-1))
        desired,backup=sequence_double_q_targets(q.detach(),tq,data['rewards'],data['dones'],burn,learn,
            cfg.get('backup_steps',cfg['nstep']),cfg['gamma'],data['valid'],actions=data['actions'],
            cut_non_greedy=rule in ('watkins','collector_cut'),trace_mask=trace_mask,return_details=True)
    chosen=q[:,burn:burn+learn].gather(2,data['actions'][:,burn:burn+learn,None]).squeeze(2)
    mask=data['valid'][:,burn:burn+learn]
    weights=mask.float()*data['weights'][:,None]
    td=chosen-desired
    weighted_errors=F.smooth_l1_loss(chosen,desired,reduction='none')*weights
    loss=weighted_errors.sum()/weights.sum().clamp_min(1.)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite Double DQN loss')
    optimizer.zero_grad();loss.backward()
    duration_gradient=0.
    if policy.input_history is not None:
        duration_gradient=float(policy.input_duration_projection.weight.grad.norm())
    grad=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],cfg['gradient_clip'])
    if not torch.isfinite(grad):raise RuntimeError('Nonfinite Q gradient')
    optimizer.step()
    if replay is not None:
        replay.update(data['keys'],td,mask);replay.account_retention_samples(data,mask)
    is_branch=torch.tensor([source=='branch' for source in data['sources']],device=mask.device)
    is_prefill=torch.tensor([phase=='prefill' for phase in data['data_phases']],device=mask.device)
    is_history=torch.tensor([part=='history' for part in data.get('retention_sources',['recent']*len(data['sources']))],device=mask.device)
    is_seed=torch.tensor([phase=='historical_seed' for phase in data['data_phases']],device=mask.device)
    result=dict(loss=float(loss.detach()),gradient_norm=float(grad),learn_transitions=int(mask.sum()),
                branch_samples=int(is_branch.sum()),branch_learn_transitions=int(mask[is_branch].sum()),
                prefill_samples=int(is_prefill.sum()),prefill_learn_transitions=int(mask[is_prefill].sum()),
                history_samples=int(is_history.sum()),history_learn_transitions=int(mask[is_history].sum()),
                seed_learn_transitions=int(mask[is_seed].sum()),
                input_duration_gradient_norm=duration_gradient,
                recorded_input_duration_nonzero_TD=int((mask&(data['input_durations'][:,burn:burn+learn].amax(-1)>0)).sum()) if extra else 0,
                burn=data['burn'],sample_group=data['sample_groups'][0],
                backup_steps_mean=float(backup['steps'][mask].float().mean()),
                backup_gt1_fraction=float((backup['steps'][mask]>1).float().mean()),
                backup_greedy_cut_fraction=float(backup['greedy_cut'][mask].float().mean()),
                backup_reward_abs_mean=float(backup['reward_sum'][mask].abs().mean()),
                backup_bootstrap_abs_mean=float(backup['bootstrap'][mask].abs().mean()))
    for label,selected in (('history',is_history),('recent',~is_history)):
        roots=mask&selected[:,None]
        result[f'{label}_backup_steps']=int(backup['steps'][roots].sum())
        result[f'{label}_backup_roots']=int(roots.sum())
    versions=data.get('collector_update_versions',[])
    lags=[] if current_update is None else [current_update-v for v in versions if v>=0]
    result.update(known_collector_versions=len(lags),collector_lag_sum=sum(lags),collector_lag_max=max(lags,default=0))
    if cfg.get('training_focus') and replay is not None:
        focus=set(cfg['training_focus']['opponents'])
        selected=torch.tensor([replay.episodes[n]['opponent_config']['id'] in focus for n,_ in data['keys']],device=mask.device)
        for name,rows in (('focus',selected),('background',~selected)):
            result[name+'_TD_transitions']=int(mask[rows].sum())
            result[name+'_TD_weight_sum']=float(weights[rows].sum())
            result[name+'_loss_contribution']=float(weighted_errors[rows].detach().sum()/weights.sum().clamp_min(1.))
    if audit:
        result['audit']=dict(keys=[list(k) for k in data['keys']],burn=burn,
            actions=data['actions'][:,burn:burn+learn].tolist(),valid=mask.tolist(),
            q_before=chosen.detach().tolist(),targets=desired.tolist(),
            **{key:value.tolist() for key,value in backup.items()})
    return result


def branch_proposals(env,episode,definition,rng,cfg,cf):
    """Training-only alternatives based on observed distance and real keys.

    Continuous held directions are evidence about input history, not a claim
    that a fighter is out of hitstun or guaranteed to execute a special move.
    """
    raw=env.game.raw;port=env.player;length=int(cf['branch_decisions']);context='generic'
    charge={};proposals=[]
    if cf.get('guile_contextual',False):
        if definition['character']!='Guile':raise ValueError('Guile proposals require the actual Guile actor')
        distance=abs(raw[f'p{3-port}_x']-raw[f'p{port}_x'])
        if raw[f'p{3-port}_y']<176 and distance<150:context='airborne_opponent';styles=('guile_flash','guile_hold')
        elif distance>140:context='far_distance';styles=('guile_boom','guile_hold')
        elif distance<75:context='close_range';styles=('guile_flash','guile_hold')
        else:context='mid_range';styles=('guile_boom','guile_flash')
        length=int(cf['horizons'][context])
        back='LEFT' if raw[f'p{3-port}_x']>=raw[f'p{port}_x'] else 'RIGHT'
        keys=[keys for action in episode.actions[-10:] for keys in env.commands[action]]
        charge=dict(continuous_back_60=len(keys)==60 and all(back in k for k in keys),
                    continuous_down_60=len(keys)==60 and all('DOWN' in k for k in keys))
        for style in styles:
            if style=='guile_hold':
                seq=proposal_sequence(definition['actions'],raw,port,rng,style=style,length=length)
            else:
                cycle=proposal_sequence(definition['actions'],raw,port,rng,style=style,charge_decisions=cfg['charge_decisions'])
                ready=charge['continuous_back_60' if style=='guile_boom' else 'continuous_down_60']
                first=cycle[cfg['charge_decisions']:] if ready else cycle
                seq=(first+cycle*((length+len(cycle)-1)//len(cycle)))[:length]
            proposals.append(seq)
        if len(proposals)!=cf['alternatives'] or proposals[0]==proposals[1]:
            raise RuntimeError('Contextual branch proposals must remain physically distinct')
    else:
        for _ in range(cf['alternatives']):
            seq=proposal_sequence(definition['actions'],raw,port,rng,length=length,charge_decisions=cfg.get('charge_decisions',10),complete_half_circle=cfg.get('complete_half_circle',False))
            if any(seq==x for x in proposals):seq[0]=(seq[0]+1)%len(definition['actions'])
            proposals.append(seq)
    return context,charge,proposals


def coverage_schedule(rivals,order='pool',offsets=None):
    """Visit the same conditions once, optionally spacing source groups.

    Proportional deficits spread both ports and learned opponents over the
    whole cycle. Zipping unequal rival lists leaves a long CPU-only tail.
    This scheduler is training metadata and never a policy input.
    """
    pairs=[(i,p) for i,r in enumerate(rivals) for p in ((1,) if r['kind']=='cpu' else (1,2))]
    schedule=[(i,p,0 if rivals[i]['kind']=='cpu' else 12*(n%2)) for n,(i,p) in enumerate(pairs)]
    if offsets is not None:
        schedule=[(i,p,o) for i,p in pairs for o in ((0,) if rivals[i]['kind']=='cpu' else offsets)]
    if order=='pool':return schedule
    if order!='stratified':raise ValueError(order)
    groups={}
    for index,port,offset in schedule:
        kind=rivals[index]['kind']
        group=('cpu' if kind=='cpu' else 'agent' if kind=='agent' else 'rule',port)
        groups.setdefault(group,[]).append((index,port,offset))
    used={key:0 for key in groups};mixed=[];total=len(schedule)
    for step in range(total):
        available=[key for key in groups if used[key]<len(groups[key])]
        group=max(available,key=lambda key:(step+1)*len(groups[key])-used[key]*total)
        mixed.append(groups[group][used[group]]);used[group]+=1
    return mixed


def training_success(info,criterion='match_result'):
    """Curriculum feedback, separate from both native scoring and TD reward."""
    if criterion=='cpu_ko' and info['opponent_kind']=='cpu':
        return float(info['result']=='win' and info['end_reason'] not in ('native_time_up','frame_cap'))
    return 1. if info['result']=='win' else .5 if info['result']=='draw' else 0.


def plan_success_refresh(evaluation,replay,progress,spec):
    """Equal collection opportunities; source labels never enter the actor."""
    groups={g:[] for g in ('pve','p1','p2')};known={}
    for row in evaluation['rows']:
        if not (row['ko_win'] if row['opponent_kind']=='cpu' else row['result']=='win'):continue
        group='pve' if row['opponent_kind']=='cpu' else f"p{row['player']}"
        groups[group].append(row)
        known[row['opponent'],row['player'],row['offset']]=row
    cursor=progress.get('success_refresh_cursor',0);chosen=[]
    for group,rows in groups.items():
        if not rows:continue
        shift=cursor%len(rows);groups[group]=rows[shift:]+rows[:shift]
    pending={g:[] for g in groups}
    for item in progress.get('unverified_success_candidates',[]):
        row=known.get((item['opponent'],item['player'],item['offset']))
        pending[item['group']].append(dict(opponent=item['opponent'],player=item['player'],offset=item['offset'],
            expected=None if row is None else {k:row[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')}))
    for g,rows in groups.items():
        for row in rows:
            if not any((p['opponent'],p['player'],p['offset'])==(row['opponent'],row['player'],row['offset']) for p in pending[g]):
                pending[g].append(dict(opponent=row['opponent'],player=row['player'],offset=row['offset'],
                    expected={k:row[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')}))
    for index in range(max((len(r) for r in pending.values()),default=0)):
        for group in ('pve','p1','p2'):
            if index<len(pending[group]):chosen.append(pending[group][index])
    progress['success_refresh_cursor']=cursor+1
    progress['pending_success_refresh']=chosen[:spec['maximum_new_successes_per_refresh']]


def collect_pending_successes(policy,definition,env,replay,rivals,progress,costs,config,out,base_frames,budget):
    """Independent full greedy games, accounted as new learning interactions."""
    spec=config.get('history_refresh');pending=progress.get('pending_success_refresh',[])
    if not spec or not pending:return
    # Never leave a half-collected policy family at a budget boundary. Pending
    # verification persists in the complete state and runs before further SGD.
    unknown=[p for p in pending if p['expected'] is None]
    if budget-env.native_frames<7200*(len(pending)+len(unknown)):return
    version=f"{progress['lineage_id']}/update{progress['optimizer_updates']}"
    selected=[r for r in rivals if r['id'] in {p['opponent'] for p in pending}]
    env.close();rng=random_state();directory=out/f"success_refresh_{env.native_frames}";failures=[]
    def collect(conditions,destination):
        recorder=TrajectoryRecorder(selected,version);actor=Actor(copy.deepcopy(policy),definition)
        try:
            result=evaluate(actor,selected,destination,cpu=False,offsets=(0,12),ports=(1,2),
                conditions=[{k:p[k] for k in ('opponent','player','offset')} for p in conditions],
                arena_call_sequence=True,transition_observer=recorder,reward_kind=config['learner']['reward'],progress=False)
        finally:actor.close()
        env.native_frames+=result['native_frames'];costs['success_collection']+=result['native_frames']
        costs['success_collection_setup']+=result.get('setup_native_frames',0)
        costs['success_collection_games']+=len(result['rows'])
        calls=sum(len(e['actions']) for e in recorder.episodes)
        costs['success_collection_decisions']+=calls;costs['environment_calls']+=calls
        return result,recorder.episodes
    try:
        if unknown:
            verified,episodes=collect(unknown,directory/'verification')
            rows={(r['opponent'],r['player'],r['offset']):r for r in verified['rows']}
            for p in unknown:
                row=rows[p['opponent'],p['player'],p['offset']]
                if (row['ko_win'] if row['opponent_kind']=='cpu' else row['result']=='win'):
                    p['expected']={k:row[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')}
                else:
                    failures.extend(e for e in episodes if e.get('validated_complete') and (e['opponent_config']['id'],e['player'],e['offset'])==(p['opponent'],p['player'],p['offset']))
            attempted={(p['opponent'],p['player'],p['offset']) for p in unknown}
            progress['unverified_success_candidates']=[p for p in progress.get('unverified_success_candidates',[]) if (p['opponent'],p['player'],p['offset']) not in attempted]
        checked=[p for p in pending if p['expected'] is not None]
        if checked:outcome,episodes=collect(checked,directory/'independent_collection')
        else:outcome=dict(rows=[]);episodes=[]
    finally:restore_random_state(rng)
    expected={(p['opponent'],p['player'],p['offset']):p['expected'] for p in pending}
    for row in outcome['rows']:
        old=expected[row['opponent'],row['player'],row['offset']]
        if any(row[k]!=old[k] for k in old):raise RuntimeError('Independent success reproduction changed its complete native result')
    additions=[]
    for episode in episodes:
        episode.update(data_phase='refreshed_success',retention_role='discovered_success',
            actor_update_version=progress['optimizer_updates'])
        number,new=replay.insert_verified_episode(episode,base_frames+env.native_frames)
        additions.append(dict(episode=number,new_distinct_trace=new,condition=episode['opponent_config'],
            player=episode['player'],offset=episode['offset'],policy_version=version))
    for episode in failures:
        episode.update(data_phase='verified_failure',retention_role='difficult_failure',
            actor_update_version=progress['optimizer_updates'])
        replay.insert_verified_episode(episode,base_frames+env.native_frames,require_success=False)
    attempted={(p['opponent'],p['player'],p['offset']) for p in pending}
    progress['unverified_success_candidates']=[p for p in progress.get('unverified_success_candidates',[]) if (p['opponent'],p['player'],p['offset']) not in attempted]
    progress['pending_success_refresh']=[];progress['native_frames']=base_frames+env.native_frames
    event=replay.refresh_history(progress['native_frames'],spec);event['independent_collections']=additions
    with (out/'history_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event,ensure_ascii=False)+'\n')


def train_run(config,character,seed,arm,budget,out,*,pool=None,checkpoints=None,start=None,dev_pool=None,resume=None,episode_limit=None):
    setup();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    out=Path(out).resolve()
    if out.resolve().is_relative_to((ROOT/'models/counter_ryu_final').resolve()):
        raise ValueError('The retained reference directory is read-only for research')
    if (out/'config.json').exists() or (out/'agent.pt').exists():
        raise FileExistsError('Existing output is preserved. Use a fresh directory; --resume requires complete training_state.pt, while --start is weight initialization.')
    out.mkdir(parents=True,exist_ok=True)
    cfg=config['learner'];started=time.monotonic()
    if arm not in config.get('allowed_arms',('ordinary','counterfactual')):raise ValueError(arm)
    if start and resume:raise ValueError('Choose weight initialization or full resume, not both')
    rivals=copy.deepcopy(pool or config.get('pool') or initial_pool())
    resumed=load_checkpoint(resume,config,rivals,character) if resume else None
    if resumed:
        definition=resumed['definition'];policy=construct_sequence_q(definition)
    elif start:
        source=torch.load(start,map_location='cpu',weights_only=False);definition=source['definition']
        if definition.get('character')!=character:raise ValueError('Continuation character differs')
        policy=construct_sequence_q(definition);policy.load_state_dict(source['state_dict'])
    else:policy,definition=random_sequence_q(character,config['network'])
    target,optimizer=learner_components(policy,cfg['learning_rate'],source.get('optimizer_state') if start else None)
    actor=Actor(copy.deepcopy(policy),definition);actor.policy.optimizer=None
    for parameter in actor.policy.parameters():parameter.requires_grad_(False)
    env=FightEnv(definition,rivals,reward=cfg['reward'])
    replay=SequenceReplay(capacity=cfg['replay_transitions'],seed=seed+17,burn=cfg['burn_in'],
                          learn=cfg['learn_length'],nstep=cfg['nstep'],byte_capacity=cfg['replay_bytes'],alpha=cfg['priority_alpha'])
    replay_source=None
    if cfg.get('resume_replay',False):
        if start is None:raise ValueError('Replay initialization requires a declared parent')
        replay_source=Path(start).parent/'replay.pt'
        cached=torch.load(replay_source,map_location='cpu',weights_only=False)
        if (cached['definition']!=definition or cached['reward_kind']!=cfg['reward']
                or cached['rules']!=config.get('rules',{}) or any(r not in rivals for r in cached['pool'])):
            raise ValueError('Replay action/observation definition or opponent version differs')
        replay.load_state_dict(cached['replay'])
    explore=Exploration(seed+29,cfg['structured_exploration_probability'],cfg.get('charge_decisions',10),cfg.get('complete_half_circle',False),cfg.get('proposal_style','mixed'))
    branch_rng=np.random.default_rng(seed+41)
    schedule=coverage_schedule(rivals,cfg.get('coverage_order','pool'),cfg.get('coverage_offsets'))
    costs=Counter(factual_training=0,counterfactual_training=0,optimizer_updates=0)
    rows=[];branch_rows=[];curve=[];pending=list(checkpoints or [budget])
    run_id=out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out.name
    progress=dict(native_frames=0,optimizer_updates=0,target_next_sync_update=cfg['target_every_updates'],
        target_sync_count=0,target_last_sync_update=0,episode_number=0,coverage_cursor=0,payoff={},
        epsilon_decay_frames=cfg.get('epsilon_decay_native_frames',max(1,int(budget*cfg['epsilon_decay_fraction']))),
        lineage_id=config.get('lineage_tag',run_id),arm=arm,seed=seed,storage_policy=None,
        next_autosave_frame=cfg.get('autosave_native_frames',0))
    history_seed_ids=[]
    if resumed:
        progress=restore_components(resumed,policy=policy,target=target,optimizer=optimizer,replay=replay,
            explore=explore,branch_rng=branch_rng,actor=actor,env=env)
        if progress['arm']!=arm:raise ValueError('Full resume changed the training arm')
        if progress['seed']!=seed:raise ValueError('Full resume cannot silently replace its random seed')
        history_seed_ids=resumed.get('history_seed_ids',[])
        requested=config.get('active_replay_policy',progress.get('storage_policy'))
        if requested!=progress.get('storage_policy'):
            if resumed['purpose']!='constructed_common_parent':raise ValueError('Cannot change retention during an exact continuation')
            replay.configure_history(history_seed_ids,requested['capacity_fraction'],requested['sample_fraction'])
            progress['storage_policy']=requested
    base_frames=progress['native_frames'];base_updates=progress['optimizer_updates']
    payoff=progress['payoff'];episode_number=progress['episode_number'];coverage_cursor=progress['coverage_cursor']
    finish_episode=cfg.get('finish_episode_at_budget',False)
    initial_history_transitions=sum(len(replay.episodes[n]['actions']) for n in replay.history_ids)
    initial_conv=policy.features_extractor.visual[0][0].weight.detach().clone()
    witness_groups=set()
    last_update={};best=-1;best_checkpoint=None
    write(out/'config.json',dict(config=config,character=character,seed=seed,arm=arm,budget=budget,
          source=str(resume) if resume else str(start) if start else 'independent random Q initialization',pool=rivals,
          full_resume=None if resume is None else str(resume),initial_progress=progress_metadata(progress),
          initial_history_transitions=initial_history_transitions,
          replay_source=None if replay_source is None else str(replay_source),
          initial_replay_transitions=replay.transitions,
          continuation_replay=('Complete boundary checkpoint restored: online, target, Adam, replay/priorities, RNG, sampling, update and exploration clocks.' if resumed else
              'Weight/replay initialization only. Target and progress restart as configured; this is not a full training resume.')))
    try:
        while (env.native_frames<budget if finish_episode else env.native_frames+6<=budget):
            collect_pending_successes(policy,definition,env,replay,rivals,progress,costs,config,out,base_frames,budget)
            if env.native_frames>=budget:break
            if episode_limit is not None and len(rows)>=episode_limit:break
            if cfg.get('collect_only_coverage',False) and episode_number>=len(schedule):break
            if cfg.get('sampling_mode')=='fixed_groups':
                from experiments.league_adaptive.environment import sample_training_condition
                index,port,offset=sample_training_condition(rivals,cfg,explore.rng)
            elif coverage_cursor<len(schedule):
                index,port,offset=schedule[coverage_cursor];coverage_cursor+=1
            else:
                # Half uniform coverage, half squared weakness. All roles keep
                # positive probability; no opponent identity enters the actor.
                scores=np.asarray([payoff.get((i,p),.5) for i,p,_ in schedule])
                base=np.ones(len(schedule),float)
                if cfg.get('training_group_weights'):
                    groups=['pve' if rivals[i]['kind']=='cpu' else f'p{p}' for i,p,_ in schedule]
                    sizes=Counter(groups)
                    base=np.asarray([cfg['training_group_weights'][g]/sizes[g] for g in groups])
                if any(r.get('training_weight',1.)!=1. for r in rivals):
                    sampling_weights=np.asarray([rivals[i].get('training_weight',1.) for i,_,_ in schedule],float)
                    if np.any(sampling_weights<=0) or not np.isfinite(sampling_weights).all():
                        raise ValueError('Training rival weights must be positive finite numbers')
                    if cfg.get('training_group_weights'):
                        for group in set(groups):
                            selected=np.asarray([g==group for g in groups])
                            sampling_weights[selected]/=sampling_weights[selected].mean()
                    base*=sampling_weights
                base/=base.sum();weak=base*((1-scores)**2+.02)
                w=.5*base+.5*weak/weak.sum()
                index,port,_=schedule[int(explore.rng.choice(len(schedule),p=w))]
                offset=12*(episode_number%2)
            if rivals[index]['kind']=='cpu' or (not finish_episode and budget-env.native_frames<24):offset=0
            obs,_=env.reset(seed=seed+episode_number,options=dict(opponent=index,player=port,offset=offset))
            costs['factual_training']+=offset
            group='pve' if rivals[index]['kind']=='cpu' else f'p{port}'
            family=f"{progress['lineage_id']}/episode{episode_number}"
            actor.policy.load_state_dict(policy.state_dict());explore.pending.clear()
            if cfg.get('reset_actor_each_training_episode',False):actor.reset()
            actor_update_version=int(progress['optimizer_updates'])
            episode=Episode(obs,group,family);root_counter=0;info={};factual_readings=[dict(env.game.raw)];factual_native=[]
            episode_branch_begin=len(branch_rows)
            while not env.ended and (finish_episode or env.native_frames+6<=budget):
                cf=config['counterfactual'];root_step=env.episode_steps
                longest=max(cf.get('horizons',{}).values(),default=cf['branch_decisions']) if cf.get('guile_contextual',False) else cf['branch_decisions']
                enough=budget-env.native_frames>=6*(cf['alternatives']*longest+1)
                first_root=int(cf.get('first_root_decision',cf['every_factual_decisions']))
                if arm=='counterfactual' and root_step>=first_root and (root_step-first_root)%cf['every_factual_decisions']==0 and enough:
                    saved=save_fight(env,actor);prefix=episode.prefix(cfg['burn_in'])
                    witness_key='neural' if rivals[index]['kind']=='agent' else group
                    keep_witness=witness_key not in witness_groups
                    if keep_witness:
                        began=time.monotonic();witness_dir=out/'root_witnesses';witness_dir.mkdir(exist_ok=True)
                        witness_root=copy.deepcopy(saved);witness_root['env']['controller']=0
                        torch.save(dict(kind='native_training_witness',definition=definition,reward_kind=cfg['reward'],
                            policy_state_dict={k:v.detach().cpu() for k,v in actor.policy.state_dict().items()},
                            root=witness_root,prefix=prefix.pack(actor.policy),pool=[rivals[index]],
                            policy_version=dict(seed=seed,arm=arm,episode=episode_number,learner_updates=actor_update_version)),
                            witness_dir/f'{witness_key}_root.pt')
                        costs['witness_seconds']+=time.monotonic()-began;witness_groups.add(witness_key)
                    context,charge,proposals=branch_proposals(env,episode,definition,branch_rng,cfg,cf)
                    root_raw=dict(env.game.raw)
                    for k,seq in enumerate(proposals):
                        bobs=restore_fight(env,actor,saved);branch=copy.deepcopy(prefix);branch.family=family+f'/root{root_step}/branch{k}'
                        before_frames=env.native_frames;dealt=taken=0;trace=[]
                        forced_prefix=min(int(cf.get('forced_prefix_decisions',len(seq))),len(seq))
                        for step,proposal in enumerate(seq):
                            unforced=int(actor.predict(bobs,True)[0]);branch.before(actor.policy)
                            forced=proposal if step<forced_prefix else unforced
                            actor.policy.record_executed_actions(forced,bobs['position'][None])
                            bobs,reward,t,tr,bi=env.step(forced);costs['environment_calls']+=1
                            branch.append(forced,reward,t,bobs);dealt+=bi['damage_dealt'];taken+=bi['damage_taken']
                            trace.append(dict(action=forced,unforced_greedy=unforced,opponent_action=bi['opponent_action'],
                                forced_input=step<forced_prefix,
                                reward=reward,damage_dealt=bi['damage_dealt'],damage_taken=bi['damage_taken'],
                                coordinates={f'p{p}_{axis}':env.game.raw[f'p{p}_{axis}'] for p in (1,2) for axis in ('x','y')},
                                hp=[env.game.raw[f'p{p}_hp'] for p in (1,2)]))
                            if t or tr:break
                        used=env.native_frames-before_frames;costs['counterfactual_training']+=used
                        packed=branch.pack(actor.policy)
                        packed.update(source='branch',result=bi['result'],player=port,offset=env.offset,character=character,opponent_config=dict(rivals[index]),
                                      actor_update_version=actor_update_version)
                        replay.add(packed);root_counter+=1
                        branch_rows.append(dict(family=branch.family,root=root_step,root_readings=root_raw,
                            context=context,input_history=charge,planned_decisions=len(seq),
                            actions=packed['actions'][branch.learn_start:].tolist(),native_frames=used,
                            forced_prefix_decisions=min(forced_prefix,len(trace)),
                            result=bi['result'],end_reason=bi['end_reason'],damage_dealt=dealt,damage_taken=taken,
                            final_readings=dict(env.game.raw),_final_frame=bobs['image'][-1].copy(),trace=trace,
                            frozen_policy_version=dict(seed=seed,arm=arm,episode=episode_number,learner_updates=costs['optimizer_updates']),
                            continuation=f'episode-frozen Q version at {episode_number}; first{forced_prefix}calls forced, subsequent calls greedy, bootstrap at nonterminal cut',
                            scope='Finite responsive branch, not an independent full match or optimal Q label.'))
                    obs=restore_fight(env,actor,saved)
                greedy=int(actor.predict(obs,True)[0]);episode.before(actor.policy)
                epsilon=cfg['epsilon_start']+(cfg['epsilon_end']-cfg['epsilon_start'])*min(1.,(base_frames+env.native_frames)/max(1,progress['epsilon_decay_frames']))
                pending_program=bool(explore.pending)
                action=explore.choose(greedy,epsilon,definition,env.game.raw,port)
                costs['factual_decisions']+=1
                costs['factual_non_greedy_decisions']+=int(action!=greedy)
                costs['factual_pending_program_decisions']+=int(pending_program)
                actor.policy.record_executed_actions(action,obs['position'][None])
                obs,reward,t,tr,info=env.step(action);costs['environment_calls']+=1
                costs['factual_training']+=info['elapsed_native_frames']
                costs['factual_group_'+group]+=info['elapsed_native_frames']
                episode.append(action,reward,t,obs)
                factual_readings.append(dict(env.game.raw))
                factual_native.append(info['elapsed_native_frames'])
            for record in branch_rows[episode_branch_begin:]:
                root=record['root'];n=len(record['actions']);end=min(len(episode.actions),root+n)
                actual_actions=episode.actions[root:end]
                record['factual_actions']=actual_actions
                record['different_from_factual']=record['actions'][:len(actual_actions)]!=actual_actions
                actual_native=sum(factual_native[root:end])
                same_horizon=end-root==n and actual_native==record['native_frames']
                record['factual_same_horizon_complete']=same_horizon
                record['factual_window_native_frames']=actual_native
                record['factual_final_readings']=factual_readings[end]
                record['factual_episode_result']=info.get('result') if env.ended else 'incomplete'
                record['different_native_outcome']=record['final_readings']!=factual_readings[end] if same_horizon else None
                fields=[f'p{p}_{field}' for p in (1,2) for field in ('x','y','hp')]
                record['different_positions_or_health']=any(record['final_readings'][f]!=factual_readings[end][f] for f in fields) if same_horizon else None
                record['different_legal_pixels']=not np.array_equal(record.pop('_final_frame'),episode.frames[end+3]) if same_horizon else None
                record.pop('_final_frame',None)
            if episode.actions:
                packed=episode.pack(actor.policy)
                packed.update(source='factual',result=info.get('result','incomplete'),player=port,offset=env.offset,character=character,
                              data_phase='prefill' if cfg.get('collect_only_coverage',False) else 'online',
                              opponent_config=dict(rivals[index]),actor_update_version=actor_update_version)
                packed.update(end_reason=info.get('end_reason'),validated_complete=env.ended and info.get('result')!='cutoff',
                    ko_win=info.get('result')=='win' and info.get('end_reason') in ('signed_hp_ko','round_counter'),
                    final_self_hp=info.get('self_hp'),final_opponent_hp=info.get('opponent_hp'))
                if cfg.get('audit_action_sequences',False):packed['audit_actions']=list(episode.actions)
                replay.add(packed)
            episode_number+=1
            if info:
                row=dict(episode=episode_number,opponent=rivals[index]['id'],opponent_config=rivals[index],player=port,offset=env.offset,
                    **{k:info[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')},
                    complete=env.ended,branch_sequences=root_counter,total_training_frames=env.native_frames)
                rows.append(row)
                if env.ended:
                    value=training_success(info,cfg.get('sampling_success','match_result'))
                    payoff[(index,port)]=.8*payoff.get((index,port),.5)+.2*value
                    if config.get('history_refresh') and value==1.:
                        candidate=dict(opponent=rivals[index]['id'],player=port,offset=env.offset,group=group)
                        queue=progress.setdefault('unverified_success_candidates',[])
                        if candidate not in queue:queue.append(candidate)
                        del queue[:-64]
            credit=max(0,(base_frames+env.native_frames-cfg['warmup_native_frames'])//cfg['update_every_native_frames'])
            while replay.priorities and progress['optimizer_updates']<credit:
                began=time.monotonic()
                last_update=optimize(policy,target,optimizer,replay,cfg,current_update=progress['optimizer_updates'])
                costs['optimizer_updates']+=1;progress['optimizer_updates']+=1
                costs['replay_learn_transitions']+=last_update['learn_transitions']
                costs['branch_replay_samples']+=last_update['branch_samples']
                costs['branch_learn_transitions']+=last_update['branch_learn_transitions']
                costs['prefill_replay_samples']+=last_update['prefill_samples']
                costs['prefill_learn_transitions']+=last_update['prefill_learn_transitions']
                costs['history_replay_samples']+=last_update['history_samples']
                costs['history_learn_transitions']+=last_update['history_learn_transitions']
                costs['seed_learn_transitions']+=last_update['seed_learn_transitions']
                costs['input_duration_gradient_norm_sum']+=last_update['input_duration_gradient_norm']
                costs['recorded_input_duration_nonzero_TD']+=last_update['recorded_input_duration_nonzero_TD']
                costs['replay_group_'+last_update['sample_group']]+=last_update['learn_transitions']
                costs['replay_burn_calls']+=last_update['burn']*cfg['batch_size']
                for key in ('history_backup_steps','recent_backup_steps','history_backup_roots','recent_backup_roots',
                            'known_collector_versions','collector_lag_sum'):
                    costs[key]+=last_update[key]
                for key in ('focus_TD_transitions','background_TD_transitions','focus_TD_weight_sum','background_TD_weight_sum',
                            'focus_loss_contribution','background_loss_contribution'):
                    costs[key]+=last_update.get(key,0)
                if progress['optimizer_updates']>=progress['target_next_sync_update']:
                    target.load_state_dict(policy.state_dict());progress['target_sync_count']+=1
                    progress['target_last_sync_update']=progress['optimizer_updates']
                    progress['target_next_sync_update']+=cfg['target_every_updates']
                costs['optimization_seconds']+=time.monotonic()-began
            costs['menu_setup']=env.game.setup_frames
            costs['wall_seconds']=round(time.monotonic()-started,3)
            progress.update(native_frames=base_frames+env.native_frames,episode_number=episode_number,
                coverage_cursor=coverage_cursor,payoff=payoff)
            write(out/'progress.json',dict(arm=arm,character=character,seed=seed,frames=env.native_frames,budget=budget,
                  cumulative_native_frames=progress['native_frames'],cumulative_updates=progress['optimizer_updates'],
                  epsilon=epsilon,target_next_sync_update=progress['target_next_sync_update'],
                  episodes=episode_number,costs=dict(costs),last_update=last_update,replay_bytes=replay.bytes,
                  recent=rows[-12:],status='running'))
            if pending and (env.native_frames>=pending[0] or env.native_frames+6>budget):
                threshold=pending.pop(0);path=out/f'checkpoint_{threshold}.pt'
                save(path,policy,definition,training=dict(algorithm='Double DQN',arm=arm,seed=seed,costs=dict(costs)),optimizer=optimizer)
                item=dict(threshold=threshold,actual_frames=env.native_frames,checkpoint=path.relative_to(ROOT).as_posix(),last_update=last_update)
                if policy.input_history is not None:
                    item['input_history']=dict(mode=policy.input_history['mode'],
                        projection_l2=float(policy.input_duration_projection.weight.detach().norm()))
                if dev_pool:
                    # The native core permits one emulator per process.
                    # Evaluation happens between episodes; no live state is lost.
                    env.close();rng=random_state()
                    try:
                        evaluation=evaluate(Actor(copy.deepcopy(policy),definition),dev_pool,out/f'development_{threshold}',cpu=False,offsets=(0,),arena_call_sequence=True,progress=False)
                        if config.get('history_refresh'):
                            plan_success_refresh(evaluation,replay,progress,config['history_refresh'])
                        if config.get('stale_history_audit'):
                            audit=recurrent_drift_audit(policy,replay,config['stale_history_audit']['sequences'])
                            write(out/f'history_audit_{threshold}.json',audit)
                            item['recurrent_history_audit']=audit
                    finally:restore_random_state(rng)
                    costs['evaluation']+=evaluation['native_frames'];item['development']=evaluation
                    score=evaluation['overall']['wins']+evaluation['pve']['wins']
                    if score>best:best=score;best_checkpoint=path.relative_to(ROOT).as_posix()
                item.update(costs=dict(costs),learning_clock=progress_metadata(progress))
                if config.get('history_refresh'):item['history_usage']=copy.deepcopy(replay.retention_usage)
                curve.append(item);write(out/'checkpoints.json',curve)
                print('CHECKPOINT',character,seed,arm,env.native_frames,last_update,flush=True)
            cadence=cfg.get('autosave_native_frames',0)
            if cfg.get('save_training_state') and cadence and progress['native_frames']>=progress['next_autosave_frame']:
                while progress['next_autosave_frame']<=progress['native_frames']:progress['next_autosave_frame']+=cadence
                save_checkpoint(out/'training_state.pt',policy=policy,target=target,optimizer=optimizer,replay=replay,
                    definition=definition,config=config,pool=rivals,progress=progress,explore=explore,
                    branch_rng=branch_rng,actor=actor,env=env,history_seed_ids=history_seed_ids)
        save(out/'agent.pt',policy,definition,training=dict(algorithm='Double DQN',arm=arm,seed=seed,costs=dict(costs)),optimizer=optimizer)
        if cfg.get('save_replay',False):
            torch.save(dict(definition=definition,pool=rivals,reward_kind=cfg['reward'],rules=config.get('rules',{}),
                replay=replay.state_dict()),out/'replay.pt')
        examples=replay.save_examples(out/'experiences')
        if cfg.get('save_training_state'):
            save_checkpoint(out/'training_state.pt',policy=policy,target=target,optimizer=optimizer,replay=replay,
                definition=definition,config=config,pool=rivals,progress=progress,explore=explore,
                branch_rng=branch_rng,actor=actor,env=env,history_seed_ids=history_seed_ids)
        costs['wall_seconds']=round(time.monotonic()-started,3)
        if costs['factual_training']+costs['counterfactual_training']+costs['success_collection']!=env.native_frames:
            raise RuntimeError('Cumulative native-frame ledger mismatch')
        result=dict(status='completed',character=character,seed=seed,arm=arm,budget=budget,actual_training_frames=env.native_frames,
            costs=dict(costs),curve=curve,rows=rows,best_development_checkpoint=best_checkpoint,
            replay_transitions=replay.transitions,replay_bytes=replay.bytes,
            history_usage=copy.deepcopy(replay.retention_usage),
            saved_continuous_examples=examples,
            coverage_completed=coverage_cursor>=len(schedule),
            full_resume=None if resume is None else str(resume),cumulative_progress=progress_metadata(progress),
            actual_frame_overrun=max(0,env.native_frames-budget),
            visual_filter_update_l2=float((policy.features_extractor.visual[0][0].weight.detach()-initial_conv).norm()),
            branch_summary=dict(sequences=len(branch_rows),
                different_from_factual_sequences=sum(bool(r['different_from_factual']) for r in branch_rows),
                different_position_or_health_sequences=sum(bool(r['different_positions_or_health']) for r in branch_rows),
                different_visible_frame_sequences=sum(bool(r['different_legal_pixels']) for r in branch_rows)),
            algorithm='n-step recurrent dueling Double DQN, online selection and target evaluation',
            limitations=['Stored actor h can become stale; real burn-in reduces but does not eliminate recurrent representational drift.',
                         'Short counterfactual cuts bootstrap rather than fabricate terminal outcomes. Branches are not independent wins.'])
        write(out/'summary.json',result);write(out/'branches.json',branch_rows)
        if (out/'progress.json').exists():
            final_progress=json.loads((out/'progress.json').read_text(encoding='utf-8'))
            final_progress.update(status='completed',costs=dict(costs));write(out/'progress.json',final_progress)
        return result
    finally:env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/counterfactual_dqn.json')
    parser.add_argument('--character',choices=CHARACTERS,default='Ryu')
    parser.add_argument('--seed',type=int,default=52031)
    parser.add_argument('--arm',choices=['ordinary','counterfactual'],default='ordinary')
    parser.add_argument('--frames',type=int,default=1200000)
    parser.add_argument('--out',required=True)
    parser.add_argument('--pool');parser.add_argument('--start');parser.add_argument('--resume')
    parser.add_argument('--episode-limit',type=int)
    parser.add_argument('--checkpoints',nargs='+',type=int)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    pool=json.loads((ROOT/args.pool).read_text(encoding='utf-8')) if args.pool else None
    train_run(config,args.character,args.seed,args.arm,args.frames,ROOT/args.out,pool=pool,start=args.start,
              checkpoints=args.checkpoints or [v for v in config['comparison']['checkpoints'] if v<=args.frames],
              dev_pool=config.get('development_pool'),resume=args.resume,episode_limit=args.episode_limit)


if __name__=='__main__':main()
