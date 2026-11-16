"""Fixed-pool experiment coordination using the shared DDQN and evaluator.

No learning loop or simulator is implemented here. Full replay/checkpoints are
local artifacts; public results keep independent matches and actual costs.
"""
import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

from experiments.league_adaptive.runtime import ROOT,Actor,load_actor,write,same_state,compiled
from experiments.league_adaptive.environment import FightEnv
from experiments.league_adaptive.evaluate import evaluate,paired_regression
from experiments.ryu_counterq.policy import construct_sequence_q
from experiments.ryu_counterq.replay import SequenceReplay,TrajectoryRecorder,select_history,recurrent_drift_audit,reconstruct_input_durations
from experiments.ryu_counterq.train import train_run,Exploration,coverage_schedule,training_success
from experiments.ryu_counterq.training_state import (learner_components,save_checkpoint,load_checkpoint,
    write_checkpoint,transfer_adam_state,contract,progress_metadata)
from experiments.ryu_direct_league.policy import save


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(result):
    cpu=[r for r in result['rows'] if r['opponent_kind']=='cpu']
    return dict(cpu_ko=sum(r['ko_win'] for r in cpu),cpu_games=len(cpu),
        cpu_time_wins=sum(r['result']=='win' and r['time_up'] for r in cpu),
        pvp=result['overall'],ports=result['by_port'])


def resume_check(config,artifact,update):
    spec=config['resume_validation'];out=artifact/'resume_validation'
    proof=out/'comparison.json'
    cfg=copy.deepcopy(config);cfg['learner'].update(target_every_updates=spec['target_every_updates'],
        warmup_native_frames=spec['warmup_native_frames'],epsilon_decay_native_frames=spec['epsilon_decay_native_frames'],
        autosave_native_frames=0,replay_transitions=6000,replay_bytes=100663296,audit_action_sequences=True)
    cfg.pop('stale_history_audit',None);cfg['lineage_tag']='ddqn_stabilization/resume_validation'
    cfg.pop('active_replay_policy',None)
    rivals=[r for r in cfg['pool'] if r['id'] in ('cpu_Guile_1','cpu_Ryu_1','mixed_Ryu')]
    fixture=copy.deepcopy(cfg);fixture['pool']=rivals;fixture['development_pool']=[]
    write(out/'configuration.json',fixture)
    if proof.exists():return read(proof)
    seed=spec['seed'];maximum=spec['maximum_native_frames_per_path']
    initial=out/'initial'
    if not (initial/'summary.json').exists():train_run(cfg,'Guile',seed,'ordinary',0,initial,pool=rivals,episode_limit=0)
    paths={}
    for name,parent,episodes in [('continuous',initial,spec['episodes']),
            ('first',initial,spec['split_after_episodes']),
            ('restored',out/'first',spec['episodes']-spec['split_after_episodes'])]:
        destination=out/name
        if not (destination/'summary.json').exists():
            train_run(cfg,'Guile',seed,'ordinary',maximum,destination,pool=rivals,
                resume=parent/'training_state.pt',episode_limit=episodes)
        paths[name]=read(destination/'summary.json')
    a=torch.load(out/'continuous/training_state.pt',map_location='cpu',weights_only=False)
    b=torch.load(out/'restored/training_state.pt',map_location='cpu',weights_only=False)
    fields=('online','target','optimizer','replay','progress','rng','exploration','branch_rng','environment','actor_state')
    different=[k for k in fields if not same_state(a[k],b[k])]
    keys=('opponent','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')
    old=[tuple(r[k] for k in keys) for r in paths['continuous']['rows']]
    new=[tuple(r[k] for k in keys) for d in ('first','restored') for r in paths[d]['rows']]
    if old!=new:different.append('complete_match_sequence')
    witness={}
    for name in paths:
        point=read(out/name/'progress.json')
        witness[name]={key:point[key] for key in ('frames','cumulative_native_frames',
            'cumulative_updates','epsilon','target_next_sync_update','episodes')}
    result=dict(passed=not different,different_fields=different,complete_games_per_path=len(old),
        schedule_witness=witness,
        native_frames=sum(p['actual_training_frames'] for p in paths.values()),
        optimizer_updates=sum(p['costs']['optimizer_updates'] for p in paths.values()),
        cumulative_progress=a['progress']|{'payoff_conditions':len(a['progress']['payoff'])})
    result['cumulative_progress'].pop('payoff')
    write(proof,result);update(resume_validation=result)
    if different:raise RuntimeError(f'Native boundary resume changed future learning: {different}')
    return result


def prepare(config,artifact,update):
    results={};cfg=config['learner']
    all_rows=[json.loads(line) for line in (ROOT/config['preparation']['old_reference_rows']).read_text(encoding='utf-8').splitlines()]
    for seed in config['preparation']['seeds']:
        out=artifact/'prepared'/str(seed);summary=out/'summary.json'
        if summary.exists():results[str(seed)]=read(summary);continue
        source=ROOT/config['preparation']['source_pattern'].format(seed=seed)
        collection=out/'initial_development';recorder=TrajectoryRecorder(config['development_pool'],str(source.relative_to(ROOT)))
        actor=load_actor(source)
        try:
            ev=evaluate(actor,config['development_pool'],collection,cpu=False,offsets=(0,),arena_call_sequence=True,
                progress=False,transition_observer=recorder,reward_kind=cfg['reward'])
        finally:actor.close()
        old=[r for r in all_rows if r['panel']==config['preparation']['old_reference_panel'].format(seed=seed)]
        key=lambda r:tuple(r[k] for k in ('opponent','player','offset'))
        expected={key(r):r for r in old}
        fields=('result','end_reason','self_hp','opponent_hp','native_frames')
        differences=[dict(condition=key(r),field=f,old=expected[key(r)][f],new=r[f])
            for r in ev['rows'] for f in fields if expected[key(r)][f]!=r[f]]
        if len(old)!=40 or len(recorder.episodes)!=40 or differences or ev['native_frames']>config['preparation']['native_frame_cap_per_seed']:
            raise RuntimeError(f'Parent collection changed the fixed complete panel: {differences}')
        raw=torch.load(source,map_location='cpu',weights_only=False);definition=raw['definition']
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        policy=construct_sequence_q(definition);policy.load_state_dict(raw['state_dict'])
        target,optimizer=learner_components(policy,cfg['learning_rate'],raw['optimizer_state'])
        replay=SequenceReplay(capacity=cfg['replay_transitions'],seed=seed+17,burn=cfg['burn_in'],learn=cfg['learn_length'],
            nstep=cfg['nstep'],byte_capacity=cfg['replay_bytes'],alpha=cfg['priority_alpha'])
        for episode in recorder.episodes:replay.add(episode)
        selected=select_history(replay,config['retention']['history_episode_quotas'])
        hn=sum(len(replay.episodes[n]['actions']) for n in selected)
        hb=sum(replay.episode_bytes(replay.episodes[n]) for n in selected)
        if hn>int(replay.capacity*config['retention']['history_capacity_fraction']) or hb>int(replay.byte_capacity*config['retention']['history_capacity_fraction']):
            raise ValueError('Declared historical trajectories exceed their bounded allocation')
        env=FightEnv(definition,config['pool'],reward=cfg['reward']);actor=Actor(copy.deepcopy(policy),definition)
        explore=Exploration(seed+29,cfg['structured_exploration_probability'],cfg['charge_decisions'],
            cfg.get('complete_half_circle',False),cfg['proposal_style'])
        known=raw['training']['costs'];frames=known['factual_training']+known.get('counterfactual_training',0);updates=known['optimizer_updates']
        pool_index={r['id']:i for i,r in enumerate(config['pool'])};payoff={}
        for row in ev['rows']:
            i=pool_index[row['opponent']]
            if config['pool'][i]!=row['opponent_config']:raise ValueError('Parent and training opponent versions differ')
            payoff[(i,row['player'])]=.8*.5+.2*training_success(row,cfg['sampling_success'])
        progress=dict(native_frames=frames,optimizer_updates=updates,target_next_sync_update=updates+cfg['target_every_updates'],
            target_last_sync_update=updates,target_sync_count=0,episode_number=0,
            coverage_cursor=len(coverage_schedule(config['pool'],cfg['coverage_order'])),payoff=payoff,
            epsilon_decay_frames=cfg['epsilon_decay_native_frames'],lineage_id=f'ddqn_stabilization/seed{seed}',
            seed=seed,arm='ordinary',storage_policy={'name':'common','capacity_fraction':0.,'sample_fraction':0.},
            next_autosave_frame=frames+cfg['autosave_native_frames'])
        try:
            save_checkpoint(out/'training_state.pt',policy=policy,target=target,optimizer=optimizer,replay=replay,
                definition=definition,config=config,pool=config['pool'],progress=progress,explore=explore,
                branch_rng=np.random.default_rng(seed+41),actor=actor,env=env,purpose='constructed_common_parent',history_seed_ids=selected)
            audit=recurrent_drift_audit(policy,replay,config['stale_history_audit']['sequences'])
            if not same_state(policy.state_dict(),raw['state_dict']) or not same_state(optimizer.state_dict(),raw['optimizer_state']):
                raise RuntimeError('Common preparation changed source weights or Adam')
        finally:env.close();actor.close()
        write(out/'history_audit_0.json',audit)
        result=dict(source=source.relative_to(ROOT).as_posix(),full_state=(out/'training_state.pt').relative_to(ROOT).as_posix(),
            initial_development=stats(ev),complete_conditions=len(ev['rows']),native_frames=ev['native_frames'],optimizer_updates=0,
            replay_transitions=replay.transitions,replay_bytes=replay.bytes,historical_transitions=hn,historical_bytes=hb,
            history=[dict(opponent=replay.episodes[n]['opponent_config'],player=replay.episodes[n]['player'],
                offset=replay.episodes[n]['offset'],result=replay.episodes[n]['result'],ko_win=replay.episodes[n]['ko_win']) for n in selected],
            original_result_differences=differences,
            original_weights_unchanged=True,original_Adam_unchanged=True,
            origin='Old online weights and Adam initialize a newly constructed common parent. Old target/replay/RNG/sampler were unavailable; no claim of restoring them.',
            reconstructed_state='Target initialized to online; target cadence starts here. Known old frame/update/epsilon progress carried. Sampler/RNG constructed once and thereafter restored exactly.',
            history_audit=audit)
        write(summary,result);results[str(seed)]=result;update(prepared=results,stage='common_parents_prepared')
    return results


def run_training_jobs(jobs,workers_file):
    """One bounded subprocess owner, shared by the two controlled protocols."""
    processes=[];handles=[]
    try:
        for job in jobs:
            handle=Path(job['log']).open('wb');handles.append(handle)
            child=subprocess.Popen(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            processes.append((job['name'],child));print('STARTED',job['name'],child.pid,flush=True)
        write(workers_file,dict(workers=[dict(name=n,pid=p.pid) for n,p in processes]))
        while any(p.poll() is None for _,p in processes):time.sleep(1)
        if any(p.returncode for _,p in processes):raise RuntimeError(f'Training failed: {[(n,p.returncode) for n,p in processes]}')
    finally:
        for _,process in processes:
            if process.poll() is None:process.terminate();process.wait()
        for handle in handles:handle.close()


def paired_training(config,artifact,update,completed):
    if not resume_check(config,artifact,update)['passed']:raise RuntimeError('Resume validation required')
    parents=prepare(config,artifact,update);runs={}
    for arm in config['comparison']['arms']:
        cfg=copy.deepcopy(config)
        cfg['active_replay_policy']=dict(name=arm,capacity_fraction=0. if arm=='A' else config['retention']['history_capacity_fraction'],
            sample_fraction=0. if arm=='A' else config['retention']['history_sample_fraction'])
        resolved=artifact/f'configuration_{arm}.json';write(resolved,cfg)
        jobs=[]
        for seed in config['comparison']['seeds']:
            out=artifact/'comparison'/str(seed)/arm;budget=config['comparison']['frames_per_arm_seed']
            if completed(out,cfg,'Guile',seed,'ordinary',budget,config['pool']):continue
            command=[sys.executable,'-m','experiments.ryu_counterq.train','--config',str(resolved),
                '--character','Guile','--seed',str(seed),'--arm','ordinary','--frames',str(budget),'--out',str(out),
                '--resume',str(ROOT/parents[str(seed)]['full_state']),'--checkpoints',*map(str,config['comparison']['checkpoints'])]
            jobs.append(dict(name=f'{arm}_{seed}',command=command,log=artifact/f'train_{arm}_{seed}.log'))
        run_training_jobs(jobs,artifact/'workers.json')
        for seed in config['comparison']['seeds']:
            data=read(artifact/'comparison'/str(seed)/arm/'summary.json')
            runs[f'{arm}/{seed}']=dict(native_frames=data['actual_training_frames'],costs=data['costs'],
                curve=[dict(threshold=p['threshold'],actual_frames=p['actual_frames'],development=stats(p['development'])) for p in data['curve']])
        update(comparison=runs,stage='paired_training_'+arm+'_complete')
    return runs


def input_parent(config,source,mode,out):
    """Explicit zero-fusion conversion, retaining every available learning state."""
    source=Path(source);source=source if source.is_absolute() else ROOT/source
    out=Path(out);destination=out/'training_state.pt'
    cfg=copy.deepcopy(config);spec=cfg['conditional_input_history']
    cfg['network']['input_history']=dict(mode=mode,cap_native_frames=spec['cap_native_frames'],normalization=spec['normalization'])
    cfg['active_replay_policy']=dict(name='B',capacity_fraction=cfg['retention']['history_capacity_fraction'],
        sample_fraction=cfg['retention']['history_sample_fraction'])
    if destination.exists():load_checkpoint(destination,cfg,cfg['pool'],'Guile');return cfg,destination
    state=load_checkpoint(source,config,config['pool'],'Guile')
    if state['definition'].get('input_history') is not None:raise ValueError('Expected the declared unaugmented complete parent')
    old=construct_sequence_q(state['definition']);old.load_state_dict(state['online'])
    definition=copy.deepcopy(state['definition']);definition['input_history']=copy.deepcopy(cfg['network']['input_history'])
    policy=construct_sequence_q(definition);target=construct_sequence_q(definition)
    for model,previous in ((policy,state['online']),(target,state['target'])):
        values=model.state_dict()
        if set(values)-set(previous)!={'input_duration_projection.weight'}:raise ValueError('Unexpected architecture change')
        for name,value in previous.items():
            if values[name].shape!=value.shape:raise ValueError('Shared network matrix changed shape')
            values[name]=value.clone()
        model.load_state_dict(values)
    optimizer,optimizer_transfer=transfer_adam_state(old,policy,state['optimizer'],config['learner']['learning_rate'])
    replay=SequenceReplay(capacity=config['learner']['replay_transitions'],seed=0,burn=config['learner']['burn_in'],
        learn=config['learner']['learn_length'],nstep=config['learner']['nstep'],
        byte_capacity=config['learner']['replay_bytes'],alpha=config['learner']['priority_alpha'])
    replay.load_state_dict(state['replay']);commands=compiled(definition['actions']);before_ids=set(replay.episodes)
    for episode in replay.episodes.values():
        episode['input_durations']=reconstruct_input_durations(episode,commands,spec['cap_native_frames'])
    replay.bytes=sum(replay.episode_bytes(e) for e in replay.episodes.values())
    replay.configure_history(state['history_seed_ids'],cfg['retention']['history_capacity_fraction'],cfg['retention']['history_sample_fraction'])
    if set(replay.episodes)!=before_ids:raise ValueError('Added3byte history must not silently remove initial experience')
    state.update(definition=definition,contract=contract(cfg,cfg['pool'],'Guile'),online=policy.state_dict(),
        target=target.state_dict(),optimizer=optimizer.state_dict(),replay=replay.state_dict(),actor_state=None,
        purpose='converted_input_history_parent')
    state['progress']['storage_policy']=copy.deepcopy(cfg['active_replay_policy'])
    write_checkpoint(destination,state)
    save(out/'initial_agent.pt',policy,definition,training=dict(purpose='Zero-fusion initialization identity',source=str(Path(source).relative_to(ROOT))))
    with np.load(ROOT/'models/counter_ryu_final/reload_inputs.npz',allow_pickle=False) as data:
        observations={k:data[k].copy() for k in ('image','position')}
    old.predict(observations,deterministic=True);policy.predict(observations,deterministic=True)
    error=float((old.last_context_q_values-policy.last_context_q_values).abs().max())
    if error or not torch.equal(old.last_context_q_values.argmax(1),policy.last_context_q_values.argmax(1)):
        raise RuntimeError('Zero-fusion conversion changed initial Q inference')
    write(out/'summary.json',dict(source=Path(source).relative_to(ROOT).as_posix(),mode=mode,
        shared_weights_and_target_unchanged=True,new_projection_zero=True,optimizer_transfer=optimizer_transfer,
        initial_fixed80_max_q_error=error,initial_progress=progress_metadata(state['progress']),
        replay_transitions=replay.transitions,replay_bytes=replay.bytes,
        scope='Explicit architecture initialization, not a claim of resuming an old model with missing state. Old target, Adam moments, replay/priorities and all RNG/clocks retained; only the input projection is new.'))
    load_checkpoint(destination,cfg,cfg['pool'],'Guile')
    return cfg,destination


def input_history_training(config,artifact,report,update,completed):
    """One input-only ablation after the complete paired retention experiment."""
    for seed in config['comparison']['seeds']:
        for arm in config['comparison']['arms']:
            result=read(artifact/'comparison'/str(seed)/arm/'summary.json')
            if result['status']!='completed':raise RuntimeError('Finish primary comparison first')
    root=artifact/'input_history';root.mkdir(exist_ok=True);spec=config['conditional_input_history']
    seeds=[spec['first_seed']];done=[];costs={};initializations={};decision=None
    for seed in seeds:
        parent_panel=read(artifact/'prepared'/str(seed)/'initial_development/summary.json')
        terminal=read(artifact/'comparison'/str(seed)/'B/development_2000000/summary.json')
        change=paired_regression(parent_panel['rows'],terminal['rows'])
        old_good={tuple(r[k] for k in ('opponent','opponent_kind','player','offset')) for r in parent_panel['rows']
            if (r['ko_win'] if r['opponent_kind']=='cpu' else r['result']=='win')}
        new_good={tuple(r[k] for k in ('opponent','opponent_kind','player','offset')) for r in terminal['rows']
            if (r['ko_win'] if r['opponent_kind']=='cpu' else r['result']=='win')}
        use_terminal=change['inheritance_gate_passed'] and old_good<=new_good and bool(new_good-old_good)
        source=(artifact/'comparison'/str(seed)/'B' if use_terminal else artifact/'prepared'/str(seed))/'training_state.pt'
        baseline=terminal if use_terminal else parent_panel
        write(root/'prepared'/str(seed)/'initial_development/summary.json',baseline)
        write(root/'prepared'/str(seed)/'baseline_provenance.json',dict(source_checkpoint=source.relative_to(ROOT).as_posix(),
            reused_known_panel=True,new_native_frames=0,zero_projection_keeps_complete_parent_policy=True,
            scope='Same complete parent policy plus identically zero input fusion; not a newly replayed40-game result.'))
        configurations={};jobs=[]
        for mode in ('real','zero'):
            cfg,parent=input_parent(config,source,mode,root/'parents'/str(seed)/mode)
            cfg['comparison'].update(seeds=seeds.copy(),arms=['real','zero'],frames_per_arm_seed=spec['initial_budget_per_arm_seed'],checkpoints=spec['checkpoints'])
            resolved=root/f'configuration_{mode}.json';write(resolved,cfg);configurations[mode]=cfg
            initializations[f'{mode}/{seed}']=read(parent.parent/'summary.json')
            out=root/'comparison'/str(seed)/mode;budget=spec['initial_budget_per_arm_seed']
            if not completed(out,cfg,'Guile',seed,'ordinary',budget,config['pool']):
                command=[sys.executable,'-m','experiments.ryu_counterq.train','--config',str(resolved),
                    '--character','Guile','--seed',str(seed),'--arm','ordinary','--frames',str(budget),
                    '--out',str(out),'--resume',str(parent),'--checkpoints',*map(str,spec['checkpoints'])]
                jobs.append(dict(name=f'input_{mode}_{seed}',command=command,log=root/f'train_{mode}_{seed}.log'))
        proof=resume_check(configurations['real'],root/'native_check',lambda **values:write(root/'native_validation.json',values))
        if not proof['passed']:raise RuntimeError('Input-history full continuation failed')
        update(conditional_input_history=dict(status='training',current_seed=seed,initializations=initializations,
            native_resume_validation=proof,replication_decision=decision))
        run_training_jobs(jobs,root/'workers.json');done.append(seed)
        analysis_config=copy.deepcopy(configurations['real']);analysis_config['comparison']['seeds']=done.copy()
        trends=analyze(analysis_config,root,report,lambda **values:None,
            record_name='input_history_episodes.jsonl',figure_name='input_history_learning_curves.png')
        for mode in ('real','zero'):
            costs[f'{mode}/{seed}']=read(root/'comparison'/str(seed)/mode/'summary.json')['costs']
        last={mode:trends[f'{mode}/{seed}']['points'][-1] for mode in ('real','zero')}
        scores={mode:p['development']['cpu_ko']+p['development']['pvp']['wins'] for mode,p in last.items()}
        if seed==spec['first_seed']:
            repeat=scores['real']>scores['zero'];decision=dict(first_seed=seed,complete_outcome_scores=scores,
                repeat=repeat,reason='Repeat only a positive complete-game objective difference, not feature gradients or input counts.')
            if repeat:seeds.append(spec['replication_seed'])
        update(conditional_input_history=dict(status='completed_seed',completed_seeds=done.copy(),trends=trends,
            initializations=initializations,native_resume_validation=proof,replication_decision=decision,costs=costs,
            parent_selection='Original complete parent unless B terminal keeps every original known win/KO and adds a new complete capability.',
            scope='Input values alone differ: real continuous physical-key durations versuszero. Both share zero-initialized480parameter fusion and every old matrix shape; no new actor permissions, reward, replay, opponent or exploration rule.'))
    update(stage='input_history_comparison_complete')
    return trends


def analyze(config,artifact,report,update,*,record_name='episodes.jsonl',figure_name='learning_curves.png',result_key='learning_trends'):
    """Read already completed panels; never start another simulator or learner."""
    summaries={};public_rows=[]
    columns=('opponent','opponent_kind','opponent_config','opponent_character','player','offset',
        'result','end_reason','self_hp','opponent_hp','native_frames','ko_win','time_up','cpu_stars')
    key=lambda r:tuple(r[k] for k in ('opponent','opponent_kind','player','offset'))
    success=lambda r:r['ko_win'] if r['opponent_kind']=='cpu' else r['result']=='win'
    for seed in config['comparison']['seeds']:
        parent=read(artifact/'prepared'/str(seed)/'initial_development/summary.json')
        old={key(r):r for r in parent['rows']};old_good={k for k,r in old.items() if success(r)}
        for row in parent['rows']:public_rows.append(dict(panel=f'parent/{seed}',**{k:row[k] for k in columns}))
        for arm in config['comparison']['arms']:
            root=artifact/'comparison'/str(seed)/arm;points=[];previous_new=set();ever_new=set();previous_costs={}
            final=read(root/'summary.json') if (root/'summary.json').exists() else None
            for threshold in config['comparison']['checkpoints']:
                panel=root/f'development_{threshold}/summary.json'
                if not panel.exists():continue
                ev=read(panel);current={key(r):r for r in ev['rows']}
                if old.keys()!=current.keys() or any(old[k]['opponent_config']!=current[k]['opponent_config'] for k in old):
                    raise ValueError('Checkpoint conditions or exact opponent versions changed')
                changes=paired_regression(parent['rows'],ev['rows'])
                good={k for k,r in current.items() if success(r)};new=good-old_good
                item=dict(threshold=threshold,development=stats(ev),
                    original_capabilities=len(old_good),original_capabilities_kept=len(good&old_good),
                    parent_wins=changes['reference_wins'],parent_wins_kept=changes['old_wins_kept'],
                    parent_ko=changes['reference_pve_ko'],parent_ko_kept=changes['pve_ko_kept'],
                    old_match_wins_lost=[dict(condition=r['condition'],result=r['candidate']['result'],
                        end_reason=r['candidate']['end_reason']) for r in changes['old_wins_lost']],
                    old_KOs_lost=[dict(condition=r['condition'],result=r['candidate']['result'],
                        end_reason=r['candidate']['end_reason']) for r in changes['pve_ko_lost']],
                    parent_capabilities_lost=[dict(condition=list(k),result=current[k]['result'],
                        end_reason=current[k]['end_reason']) for k in sorted(old_good-good)],
                    newly_solved=[list(k) for k in sorted(new)],newly_solved_count=len(new),
                    first_seen_solved_now=[list(k) for k in sorted(new-ever_new)],
                    newly_solved_details=[dict(condition=list(k),parent_result=old[k]['result'],
                        parent_end_reason=old[k]['end_reason'],candidate_end_reason=current[k]['end_reason'],
                        candidate_self_hp=current[k]['self_hp'],candidate_opponent_hp=current[k]['opponent_hp']) for k in sorted(new)],
                    previous_nonwins_to_wins=[r['condition'] for r in changes['new_wins']],
                    previous_new_kept=len(previous_new&new),previous_new_count=len(previous_new),
                    ever_new_kept=len(ever_new&new),ever_new_count=len(ever_new))
                if final:
                    point=next(p for p in final['curve'] if p['threshold']==threshold)
                    item.update(actual_training_frames=point['actual_frames'],costs=point['costs'],
                        learning_clock=point['learning_clock'])
                    counts=point['costs'];delta=lambda field:counts.get(field,0)-previous_costs.get(field,0)
                    item.update(interval_TD_history_fraction=delta('history_learn_transitions')/max(1,delta('replay_learn_transitions')),
                        interval_initial_experience_TD_fraction=delta('seed_learn_transitions')/max(1,delta('replay_learn_transitions')),
                        interval_optimizer_updates=delta('optimizer_updates'))
                    previous_costs=counts
                audit=root/f'history_audit_{threshold}.json'
                if audit.exists():
                    records=read(audit)['rows']
                    item['history_audit']=dict(sequences=len(records),decisions=sum(r['decisions'] for r in records),
                        stored_h_max_q_error=max(r['stored_h_max_q_error'] for r in records),
                        stored_h_action_differences=sum(r['stored_h_action_differences'] for r in records),
                        zero_h_max_q_error=max(r['zero_h_max_q_error'] for r in records),
                        zero_h_action_differences=sum(r['zero_h_action_differences'] for r in records),
                        local_details=audit.relative_to(ROOT).as_posix())
                points.append(item);previous_new=new;ever_new|=new
                for row in ev['rows']:public_rows.append(dict(panel=f'{arm}/{seed}/{threshold}',**{k:row[k] for k in columns}))
            progress_file=root/'progress.json'
            current_costs=(final or (read(progress_file) if progress_file.exists() else {})).get('costs',{})
            denominator=current_costs.get('replay_learn_transitions',0)
            summaries[f'{arm}/{seed}']=dict(status='completed' if final else 'running' if progress_file.exists() else 'pending',
                initial=stats(parent),points=points,costs=current_costs,
                history_update_fraction=current_costs.get('history_replay_samples',0)/max(1,config['learner']['batch_size']*current_costs.get('optimizer_updates',0)),
                history_TD_fraction=current_costs.get('history_learn_transitions',0)/max(1,denominator),
                initial_experience_TD_fraction=current_costs.get('seed_learn_transitions',0)/max(1,denominator))
    report.mkdir(parents=True,exist_ok=True)
    (report/record_name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in public_rows),encoding='utf-8')
    update(**{result_key:summaries})
    if any(value['points'] for value in summaries.values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n=len(config['comparison']['seeds']);figure,axes=plt.subplots(n,3,figsize=(13,3.5*n),sharex=True,squeeze=False)
        for index,seed in enumerate(config['comparison']['seeds']):
            for arm in config['comparison']['arms']:
                data=summaries[f'{arm}/{seed}'];points=data['points'];x=[0]+[p['threshold']/1e6 for p in points]
                axes[index,0].plot(x,[data['initial']['cpu_ko']]+[p['development']['cpu_ko'] for p in points],marker='o',label=arm)
                for port in (1,2):
                    axes[index,port].plot(x,[data['initial']['ports'][str(port)]['wins']]+
                        [p['development']['ports'][str(port)]['wins'] for p in points],marker='o',label=arm)
            for column,axis in enumerate(axes[index]):
                axis.set_title(f'Seed {seed}: '+('CPU KO /24' if column==0 else f'PvP P{column} wins /8'))
                axis.set_ylim(0,24 if column==0 else 8);axis.grid(alpha=.3);axis.legend();axis.set_xlabel('Additional training native frames (millions)')
        figure.tight_layout();figure.savefig(report/figure_name,dpi=140);plt.close(figure)
    return summaries


def publish(config,artifact,report):
    """Select compact evidence only after actual process cleanup and analysis."""
    import shutil
    from experiments.ryu_counterq.report import retain_weights,model_identity
    state=read(report/'summary.json');archive=artifact/'full_study_state.json'
    if state.get('format')=='compact_ddqn_stabilization':state=read(archive)
    cleanup=state.get('process_cleanup',{})
    if not state.get('analysis_complete') or not cleanup.get('verified') or cleanup.get('active_related_processes')!=[]:
        raise RuntimeError('Finish analysis and verify actual learner/evaluator process cleanup before publishing')
    trends=analyze(config,artifact,report,lambda **values:state.update(values))
    if any(d['status']!='completed' or len(d['points'])!=len(config['comparison']['checkpoints']) for d in trends.values()):
        raise RuntimeError('Every declared seed, arm and milestone must be reported')
    models={};identities=[];budget={};parents={}
    for seed in config['comparison']['seeds']:
        preparation=read(artifact/'prepared'/str(seed)/'summary.json')
        parents[str(seed)]={k:preparation[k] for k in ('source','initial_development','native_frames','optimizer_updates',
            'replay_transitions','replay_bytes','historical_transitions','historical_bytes','original_weights_unchanged',
            'original_Adam_unchanged','origin','reconstructed_state')}
        parents[str(seed)]['history_conditions']=[dict(opponent=r['opponent']['id'],player=r['player'],
            offset=r['offset'],result=r['result'],ko_win=r['ko_win']) for r in preparation['history']]
        parents[str(seed)]['local_full_training_state']=(artifact/'prepared'/str(seed)/'training_state.pt').relative_to(ROOT).as_posix()
        for arm in config['comparison']['arms']:
            name=f'{arm}_{seed}';root=artifact/'comparison'/str(seed)/arm
            result=read(root/'summary.json');source=(root/'agent.pt').relative_to(ROOT).as_posix()
            destination=f'models/ddqn_stabilization/{name}.pt'
            retain_weights(source,destination);identities.append(model_identity(source,destination))
            inherited=torch.load(ROOT/preparation['source'],map_location='cpu',weights_only=False)['training']['costs']
            models[name]=dict(path=destination,own_character='Guile',algorithm='recurrent dueling Double DQN',
                original_parent=preparation['source'],original_training_native_frames=inherited['factual_training']+inherited.get('counterfactual_training',0),
                preparation_native_frames=preparation['native_frames'],new_training_native_frames=result['actual_training_frames'],
                local_full_training_state=(root/'training_state.pt').relative_to(ROOT).as_posix(),
                delivered_file_is_full_resume=False,
                purpose='Research endpoint; not automatically promoted')
            budget[name]=result['costs']|dict(actual_frame_overrun=result['actual_frame_overrun'])
    for name,source in state.get('additional_models',{}).items():
        destination=f'models/ddqn_stabilization/{name}.pt'
        retain_weights(source,destination);identities.append(model_identity(source,destination))
        raw=torch.load(ROOT/source,map_location='cpu',weights_only=False)
        models[name]=dict(path=destination,source=source,own_character=raw['definition'].get('character'),
            algorithm='recurrent dueling Double DQN',input_history=raw['definition'].get('input_history'),
            purpose='Additional explicitly labelled research evidence')
        models[name].update(state.get('additional_model_metadata',{}).get(name,{}))
    videos=[]
    for directory in state.get('selected_video_dirs',[]):
        source=ROOT/directory;destination=report/'videos'/source.name;destination.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/'replay.mp4',destination/'replay.mp4')
        decisions=read(source/'decisions.json')
        (destination/'decisions.json').write_text('[\n'+',\n'.join(
            json.dumps(row,ensure_ascii=False,separators=(',',':')) for row in decisions)+'\n]\n',encoding='utf-8')
        details=read(source/'summary.json');details['agent']=Path(details['agent']).relative_to(ROOT).as_posix()
        details['video']=(destination/'replay.mp4').relative_to(ROOT).as_posix();write(destination/'summary.json',details)
        videos.append(dict(directory=destination.relative_to(ROOT).as_posix(),result=details['result'],
            native_frames=details['native_frames'],scope='Known-case complete replay, not an additional independent test victory'))
    stills=[]
    for item in state.get('selected_video_frames',[]):
        source=ROOT/item['source'];destination=report/'videos'/item['video']/source.name
        if not destination.resolve().is_relative_to((report/'videos').resolve()):raise ValueError('Invalid video evidence destination')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        stills.append(dict(path=destination.relative_to(ROOT).as_posix(),caption=item['caption']))
    write(archive,state)
    # Run costs have their own ledger. Keep one compact view of each learning
    # curve instead of embedding the same cumulative counters at every point.
    for collection in (trends,state.get('conditional_input_history',{}).get('trends',{})):
        for data in collection.values():
            for point in data['points']:
                point.pop('costs',None)
                if 'learning_clock' in point:
                    point['learning_clock']={k:point['learning_clock'][k] for k in ('native_frames','optimizer_updates',
                        'target_next_sync_update','target_last_sync_update','target_sync_count','epsilon_decay_frames')}
    totals=dict(preparation_native_frames=sum(v['native_frames'] for v in parents.values()),
        resume_validation_native_frames=state['resume_validation']['native_frames'],
        main_training_native_frames=sum(v['factual_training'] for v in budget.values()),
        main_optimizer_updates=sum(v['optimizer_updates'] for v in budget.values()),
        main_evaluation_native_frames=sum(v.get('evaluation',0) for v in budget.values()),
        counterfactual_native_frames=0,counter_training_native_frames=0,
        all_new_native_frames=state.get('combined_budget',{}).get('native_frames'),
        all_new_optimizer_updates=state.get('combined_budget',{}).get('optimizer_updates'),
        other_costs=state.get('additional_costs',{}))
    compact=dict(format='compact_ddqn_stabilization',configuration='configs/ddqn_stabilization.json',
        started_at_utc=state['started_at_utc'],stage='completed_research_pending_Mac_review',
        hypothesis='Complete continuation plus bounded sustained historical TD sampling can reduce capability loss while allowing new useful responses.',
        own_character='Guile',formal_reference=config['reference'],reference_replaced=False,
        research_goal_achieved=state.get('acceptance',{}).get('passed',False),
        outcome_semantics=state.get('outcome_semantics'),
        resume_validation=state['resume_validation'],preparation=parents,models=models,
        learning_trends=trends,budget_by_run=budget,budget_totals=totals,model_parameter_identity=identities,
        rows='reports/ddqn_stabilization/episodes.jsonl',learning_curves='reports/ddqn_stabilization/learning_curves.png',
        behavior=state.get('behavior_analysis'),conditional_input_history=state.get('conditional_input_history'),
        input_history_resume_validation=state.get('input_history_pretraining_validation'),
        notebook_execution=state.get('notebook_execution'),
        final_evaluation=state.get('final_evaluation'),independent_reload=state.get('independent_reload'),
        frozen_reporting=state.get('frozen_reporting'),stability_assessment=state.get('stability_assessment'),
        legacy_input_feature_compatibility=state.get('legacy_input_feature_compatibility'),
        conclusions=state['conclusions'],recommendation=state['recommendation'],videos=videos,video_stills=stills,
        tests=state.get('tests'),process_cleanup=cleanup,
        acceptance=state.get('acceptance'),
        transfer_status=state.get('transfer_status'),
        official_arena='Not executed; no course scanning, packing or Arena validation path is used.',
        scope='Fixed147 training conditions:96CPU character/star cases,48rule profiles,3frozen neural opponents. Known40case development coverage is24CPU endpoint-star cases and16PvP cases. It is not96-condition KO success, unseen-opponent robustness or formal promotion.')
    write(report/'summary.json',compact);return compact


def dispatch(config,args,artifact,report,update,completed,interface=None):
    artifact.mkdir(parents=True,exist_ok=True)
    if args.stage in ('stabilization-short','stabilization-resume-check'):return resume_check(config,artifact,update)
    if args.stage=='stabilization-prepare':return prepare(config,artifact,update)
    if args.stage=='stabilization-run':return paired_training(config,artifact,update,completed)
    if args.stage=='stabilization-input-history':return input_history_training(config,artifact,report,update,completed)
    if args.stage=='stabilization-analyze':return analyze(config,artifact,report,update)
    if args.stage in ('export','video'):
        if interface is None:raise ValueError('Use the shared study interface')
        args.agent=args.agent or str(artifact/'resume_validation/restored/agent.pt')
        return interface(config,args)
    if args.stage=='evaluate':
        source=args.agent or str(artifact/'resume_validation/restored/agent.pt')
        rivals=read(args.pool) if args.pool else config['development_pool']
        destination=artifact/'evaluation'
        if destination.exists():raise FileExistsError('Use a fresh evaluation output root')
        actor=load_actor(source)
        try:return evaluate(actor,rivals,destination,cpu=False,offsets=tuple(args.offsets),ports=tuple(args.players),
            arena_call_sequence=True,conditions=None if args.conditions is None else read(args.conditions))
        finally:actor.close()
    raise ValueError('This configuration permits ordinary fixed-Guile stabilization stages only')
