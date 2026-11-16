"""Full-network recurrent PPO with actual complete-game coverage quotas."""
import argparse
import copy
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.league_adaptive.runtime import ROOT,setup, write
from experiments.ryu_direct_league.environment import RyuEnv,CoverageEnv,CoverageWorker
from experiments.ryu_direct_league.policy import save
from experiments.ryu_adaptive_curriculum.recurrent_direct import initialize, construct_recurrent, initialize_durations,initialize_health
from experiments.league_adaptive.characters import advance_input_durations
from experiments.ryu_counterq.training_state import restore_optimizer


def recurrent_step(policy,features,hidden,previous,starts,positions,remaining,health,left,elapsed,durations):
    """One training/inference recurrent API, with explicit legal input history."""
    if getattr(policy,'probability_policy',False):
        return policy.sequence_step(features,hidden,previous,starts,positions[:,4]<0,input_durations=durations)
    return policy.sequence_step(features,hidden,previous,starts,positions[:,4]<0,remaining,health,positions,left,elapsed)


def input_durations_after(policy,durations,actions):
    return torch.as_tensor(np.stack([advance_input_durations(before,policy.physical_commands[int(action)],
        policy.input_history['cap_native_frames']) for before,action in zip(durations.numpy(),actions.numpy())]))


def refresh_successes(policy,definition,replay,spec,shared,out,frame,event,selected=None,cycle_event=None):
    """Independent complete greedy self-games, separate from PPO rollouts."""
    from experiments.league_adaptive.evaluate import evaluate
    from experiments.league_adaptive.runtime import Actor
    from experiments.ryu_counterq.replay import TrajectoryRecorder
    from experiments.ryu_counterq.training_state import random_state,restore_random_state
    conditions=spec['conditions'];rivals={r['id']:r for r in shared['pool']}
    groups=[[c for c in conditions if ('pve' if rivals[c['opponent']]['kind']=='cpu' else f"p{c['player']}")==g]
        for g in ('pve','p1','p2')]
    if spec['bank']['refresh_conditions_per_event']!=3 or any(not g for g in groups):
        raise ValueError('Success refresh needs one complete condition per PvE/P1/P2 group')
    cycle=event if cycle_event is None else cycle_event
    chosen=[g[cycle%len(g)] for g in groups] if selected is None else selected
    version=out/f'success_source_{event}.pt'
    save(version,policy,definition,dict(algorithm='PPO',native_frames=frame,purpose='Frozen independent self-success collection'))
    observer=TrajectoryRecorder(shared['pool'],version.resolve().relative_to(ROOT).as_posix())
    saved=random_state()
    try:
        collection_pool=[r for r in shared['pool'] if r['id'] in {c['opponent'] for c in chosen}]
        offsets=tuple(sorted({c['offset'] for c in chosen}))
        result=evaluate(Actor(copy.deepcopy(policy),copy.deepcopy(definition)),collection_pool,
            out/f'success_collection_{event}',cpu=False,offsets=offsets,conditions=chosen,
            arena_call_sequence=True,progress=False,reward_kind=shared['learner']['reward'],transition_observer=observer)
    finally:restore_random_state(saved)
    changes=replay.refresh_self_successes(observer.episodes,frame+result['native_frames'],
        spec['bank']['maximum_episodes'],out.resolve().relative_to(ROOT).as_posix())
    replay.note_complete_results(result['rows'],prioritize_failures=spec.get('prioritize_current_failures',False))
    return dict(native_frames=result['native_frames'],environment_calls=sum(len(e['actions']) for e in observer.episodes),
        changes=changes,outcomes=result['rows'])




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initial', required=True); parser.add_argument('--pool', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=294912); parser.add_argument('--seed', type=int, default=2231)
    parser.add_argument('--memory', choices=('on','off'), default='on')
    parser.add_argument('--lr', type=float, default=1e-4); parser.add_argument('--family-entropy', type=float, default=.01)
    parser.add_argument('--entropy', type=float, default=.005); parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--checkpoint-every', type=int, default=49152); parser.add_argument('--max-seconds', type=float, default=14400)
    parser.add_argument('--reward', choices=('potential','damage'), default='potential')
    parser.add_argument('--learn-durations', action='store_true')
    parser.add_argument('--damage-weight',type=float,default=.25)
    parser.add_argument('--health-reader');parser.add_argument('--mask-health',action='store_true')
    parser.add_argument('--temperature-initial',type=float);parser.add_argument('--temperature-final',type=float)
    parser.add_argument('--entropy-final',type=float);parser.add_argument('--family-entropy-final',type=float)
    parser.add_argument('--config',help='Shared Guile condition/reward configuration; retains the existing PPO learner')
    parser.add_argument('--native-frames',type=int,default=0);parser.add_argument('--checkpoints',type=int,nargs='*')
    parser.add_argument('--max-updates',type=int,default=0,help='Optional diagnostic optimizer-step limit; zero keeps normal training')
    parser.add_argument('--capture-updates',type=int,nargs='*',default=[0,128],help='A few actual pre/post-update diagnostic snapshots')
    args=parser.parse_args(); setup(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.max_updates<0 or any(x<0 for x in args.capture_updates):raise ValueError('Diagnostic update indices must be nonnegative')
    rng=np.random.default_rng(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.initial,map_location='cpu',weights_only=False)
    sequence_ppo=source['definition']['kind']=='sequence_ppo'
    shared=None
    if sequence_ppo:
        from experiments.ryu_counterq.policy import construct_sequence_q
        if not args.config or args.family_entropy!=0 or args.learn_durations or args.health_reader:
            raise ValueError('Matched sequence PPO requires shared config, ordinary entropy, and fixed six-frame commands')
        shared=json.loads(Path(args.config).read_text(encoding='utf-8'))
        if shared['learner']['gamma']!=1.:raise ValueError('Declare a separate common discounted-reward experiment first')
        definition=copy.deepcopy(source['definition']);policy=construct_sequence_q(definition)
        policy.load_state_dict(source['state_dict'],strict=True)
        if 'optimizer_state' in source:restore_optimizer(policy.optimizer,source['optimizer_state'])
    elif source['definition']['kind']=='recurrent_direct_ppo':
        definition=copy.deepcopy(source['definition'])
        if definition['memory_enabled']!=(args.memory=='on'): raise ValueError('Resume the same recurrent condition')
        if args.learn_durations and len(definition.get('duration_choices',(1,)))==1:
            policy,definition=initialize_durations(source)
        else:
            policy=construct_recurrent(definition); policy.load_state_dict(source['state_dict'],strict=True)
            if 'optimizer_state' in source: restore_optimizer(policy.optimizer,source['optimizer_state'])
    else: policy,definition=initialize(source,args.memory=='on')
    if (len(definition.get('duration_choices',(1,)))>1)!=args.learn_durations: raise ValueError('Declare the duration-aware training condition')
    if args.health_reader:
        reader=torch.load(args.health_reader,map_location='cpu',weights_only=False)
        policy,definition=initialize_health(dict(definition=definition,state_dict=policy.state_dict()),reader,args.mask_health)
    temperature_initial=policy.action_temperature if args.temperature_initial is None else args.temperature_initial
    temperature_final=temperature_initial if args.temperature_final is None else args.temperature_final
    if min(temperature_initial,temperature_final)<=0:raise ValueError('Temperatures must remain positive')
    policy.action_temperature=float(temperature_initial);definition['action_temperature']=float(temperature_initial)
    for group in policy.optimizer.param_groups: group['lr']=args.lr
    if not all(parameter.requires_grad for parameter in policy.parameters()): raise RuntimeError('All inference weights must learn')
    pool=json.loads(Path(args.pool).read_text(encoding='utf-8'))
    ppo_settings=(shared or {}).get('ppo',{})
    worker_count=int(ppo_settings.get('n_envs',4))
    rollout=int(ppo_settings.get('n_steps',512))
    length=int(ppo_settings.get('sequence_length',32))
    batch_sequences=int(ppo_settings.get('batch_sequences',8))
    epochs=int(ppo_settings.get('n_epochs',4))
    clip_range=float(ppo_settings.get('clip_range',.1))
    target_kl=float(ppo_settings.get('target_kl',.04))
    if min(worker_count,rollout,length,batch_sequences,epochs)<1 or rollout%length or not 0<clip_range<1 or target_kl<=0:
        raise ValueError('Invalid recurrent PPO rollout, sequence or update settings')
    cpu=[dict(id=f'CPU_Ryu_{s}',kind='cpu',character='Ryu',stars=s) for s in range(1,9)]
    if sequence_ppo:
        from experiments.league_adaptive.environment import TrainingWorker
        if pool!=shared['pool']:raise ValueError('PPO opponent versions differ from its common configuration')
        cpu=[r for r in pool if r['kind']=='cpu']
        workers=[TrainingWorker(definition,pool,shared['learner'],args.seed+i*317) for i in range(worker_count)]
    else:
        workers=[CoverageWorker(definition,pool,i,args.seed+i*317,args.reward,args.damage_weight) for i in range(4)]
        workers.extend(CoverageWorker(definition,cpu,i,args.seed+i*317,args.reward,args.damage_weight) for i in (4,5))
    success_spec=shared.get('success_consolidation') if sequence_ppo else None
    success_replay=None;success_frames=0;success_calls=0;success_events=[]
    auxiliary_samples=0;auxiliary_updates=0;auxiliary_seconds=0.;auxiliary_losses=[];gradient_audits=[]
    success_clock_base=0;assessment_seconds=0.;assessments=[];success_sampling_initial={}
    if success_spec is not None:
        from experiments.ryu_counterq.replay import SequenceReplay
        from experiments.ryu_direct_league.rehearsal import recurrent_success_loss
        bank=success_spec['bank']
        success_replay=SequenceReplay(capacity=bank['capacity_transitions'],byte_capacity=bank['byte_capacity'],
            seed=args.seed+95003,burn=0,learn=bank['learn_length'],nstep=1)
        saved_bank=torch.load(ROOT/success_spec['initial_bank'],map_location='cpu',weights_only=False)
        success_replay.load_state_dict(saved_bank)
        if success_spec.get('retemperature_initial_references',False):
            from experiments.ryu_direct_league.rehearsal import retemperature_success_references
            if temperature_initial!=temperature_final:raise ValueError('Reference retemperature requires a constant policy temperature')
            transformation=retemperature_success_references(success_replay.episodes.values(),policy.action_temperature)
            success_replay.bytes=sum(success_replay.episode_bytes(e) for e in success_replay.episodes.values())
            if success_replay.bytes>success_replay.byte_capacity:raise ValueError('Probability references exceed bounded replay storage')
            write(out/'reference_temperature_initialization.json',transformation)
        success_sampling_initial=copy.deepcopy(success_replay.auxiliary_sampling_totals)
        success_clock_base=int(saved_bank.get('native_frames',success_spec.get('initial_bank_clock',0)))
        if not success_spec.get('resume_bank_sampling',False):success_replay.rng=np.random.default_rng(args.seed+95003)
        success_replay.note_complete_results(success_spec.get('initial_outcomes',[]),
            prioritize_failures=success_spec.get('prioritize_current_failures',False))
        success_due=bank['refresh_every_native_frames'];success_event=0;periodic_round=0
    env=SubprocVecEnv(workers,start_method='spawn'); env.seed(args.seed)
    nenv=len(workers)
    started=time.monotonic(); steps=0; last_save=0; history=[]; rounds=Counter(); scenes=Counter(); inputs=Counter(); conditions=Counter()
    native_frames=0;optimizer_updates=0;optimization_samples=0;optimization_seconds=0.;checkpoints=list(args.checkpoints or [])
    frozen_evaluations=[];evaluation_native_frames=0;evaluation_seconds=0.;exposure=Counter();condition_frames=Counter()
    initial_parameters={name:value.detach().clone() for name,value in policy.named_parameters()}
    write(out/'config.json',dict(arguments=vars(args),pool=pool,cpu=cpu,n_envs=nenv,rollout=rollout,sequence_length=length,batch_sequences=batch_sequences,epochs=epochs,
        optimizer='Adam; all visual, coordinate, direct action/value and GRU parameters trainable',clip_range=clip_range,target_kl=target_kl,
        shared_config=shared,coverage=('Fixed exogenous distribution shared with the fresh DDQN comparator' if sequence_ppo else 'Completed-game coverage with hardness tie breaks'),
        exploration=('Ordinary categorical policy sampling and entropy only; all125 physical actions remain available' if sequence_ppo else 'Categorical and action-family entropy; all42 actions remain available'),
        objective='Native complete-game outcome and declared HP reward; no opponent identity, star, elapsed script phase or simulator state enters the actor.',
        durations=definition.get('duration_choices',[1]),
        temporal_objective='Actor likelihood is the joint chosen physical input and conditional duration at option starts only. The augmented critic observes pending duration and learns at every six-frame call. Opponents continue every six frames. Native frame budgets include all committed calls.',
        inference=f"{len(definition['actions'])} physical actions; own executed previous input, recurrent memory and declared legal history; no teacher or opponent identity."))
    save(out/'initial.pt',policy,definition,dict(initial=args.initial,algorithm='PPO',
        objective='Fresh on-policy GAE and clipped categorical actor/critic updates; no TD replay targets'))
    observation=env.reset(); hidden=torch.zeros(nenv,policy.memory_hidden); previous=torch.zeros(nenv,dtype=torch.long); starts=np.ones(nenv,bool)
    remaining=torch.zeros(nenv,dtype=torch.long); committed=torch.zeros(nenv,dtype=torch.long); option_decisions=0; duration_counts=Counter()
    controller_left=torch.as_tensor(observation['position'][:,4]>=0);elapsed=torch.zeros(nenv)
    durations=torch.zeros(nenv,3,dtype=torch.uint8)
    def report_success_learning():
        if success_spec is None:return
        write(out/'success_learning.json',dict(coefficient=success_spec['coefficient'],disagreement_scope=success_spec.get('disagreement_scope','failed'),
            use_soft_physical_references=success_spec.get('use_soft_references',False),
            hard_failed_conditions=success_spec.get('hard_failed_conditions',False),
            prioritize_current_failures=success_spec.get('prioritize_current_failures',False),
            within_group_priority_fraction=success_spec.get('priority_fraction',0.),native_frames=success_frames,
            environment_calls=success_calls,collection=success_events,episodes=len(success_replay.episodes),transitions=success_replay.transitions,
            bytes=success_replay.bytes,auxiliary_updates=auxiliary_updates,auxiliary_samples=auxiliary_samples,auxiliary_seconds=auxiliary_seconds,
            recent_auxiliary_loss=float(np.mean(auxiliary_losses[-32:])) if auxiliary_losses else None,
            actual_use=success_replay.retention_usage,gradient_audits=gradient_audits,
            current_run_sampling_by_group={g:{k:value-success_sampling_initial.get(g,{}).get(k,0)
                for k,value in counts.items()} for g,counts in success_replay.auxiliary_sampling_totals.items()},
            bank_clock=success_clock_base+native_frames,assessment_seconds=assessment_seconds,assessments=assessments))
    try:
        while (native_frames<args.native_frames if args.native_frames else steps<args.steps) and (not args.max_updates or optimizer_updates<args.max_updates) and (args.max_seconds<=0 or time.monotonic()-started<args.max_seconds) and not (out/'STOP').exists():
            fraction=min(1.,native_frames/args.native_frames if args.native_frames else steps/args.steps)
            policy.action_temperature=temperature_initial+(temperature_final-temperature_initial)*fraction
            entropy_coefficient=args.entropy if args.entropy_final is None else args.entropy+(args.entropy_final-args.entropy)*fraction
            family_entropy_coefficient=args.family_entropy if args.family_entropy_final is None else args.family_entropy+(args.family_entropy_final-args.family_entropy)*fraction
            buffer={key:[] for key in ('image','position','hidden','previous','starts','remaining','option_start','duration','input_durations','action','log_prob','value','reward','done','controller_left','elapsed','health_label')}
            policy.set_training_mode(False)
            for _ in range(rollout):
                tensor,_=policy.obs_to_tensor(observation)
                controller_left=torch.where(torch.as_tensor(starts),tensor['position'][:,4]>=0,controller_left)
                elapsed=torch.where(torch.as_tensor(starts),torch.zeros_like(elapsed),elapsed)
                durations[starts]=0
                with torch.no_grad():
                    features=policy.extract_features(tensor); flip=tensor['position'][:,4]<0
                    health=None if sequence_ppo else policy.read_health(tensor)
                    distribution,value,next_hidden=recurrent_step(policy,features,hidden,previous,torch.as_tensor(starts),tensor['position'],remaining,
                        health,controller_left,elapsed,durations)
                    option_start=remaining==0; proposed=distribution.sample(); action=torch.where(option_start,proposed,committed)
                    duration_distribution=None if sequence_ppo else policy.duration_distribution(next_hidden,action,flip)
                    duration_index=duration_distribution.sample() if args.learn_durations else torch.zeros(nenv,dtype=torch.long)
                    selected_duration=torch.ones(nenv,dtype=torch.long) if sequence_ppo else torch.as_tensor(policy.duration_choices)[duration_index]
                    next_remaining=torch.where(option_start,selected_duration-1,remaining-1)
                    log_prob=distribution.log_prob(action)+(0 if sequence_ppo else duration_distribution.log_prob(duration_index))
                    option_decisions+=int(option_start.sum())
                    duration_counts.update(selected_duration[option_start].tolist())
                next_observation,reward,done,infos=env.step(action.numpy())
                next_durations=input_durations_after(policy,durations,action) if sequence_ppo else durations
                if sequence_ppo:
                    # A recording frame cap is a truncation, never an invented loss or terminal zero bootstrap.
                    for e,info in enumerate(infos):
                        if info.get('TimeLimit.truncated',False):
                            with torch.no_grad():
                                terminal,_=policy.obs_to_tensor(info['terminal_observation']);terminal_features=policy.extract_features(terminal)
                                _,terminal_value,_=recurrent_step(policy,terminal_features,next_hidden[e:e+1],action[e:e+1],torch.zeros(1,dtype=torch.bool),
                                    terminal['position'],remaining[e:e+1],None,controller_left[e:e+1],elapsed[e:e+1]+1,next_durations[e:e+1])
                                reward[e]+=float(terminal_value[0])
                data=dict(image=observation['image'],position=observation['position'],hidden=hidden.numpy(),previous=previous.numpy(),starts=starts.copy(),
                          remaining=remaining.numpy(),option_start=option_start.numpy(),duration=duration_index.numpy(),
                          input_durations=durations.numpy(),controller_left=controller_left.numpy(),elapsed=elapsed.numpy(),health_label=np.asarray([info.get('pre_action_absolute_health',[0.,0.]) for info in infos],np.float32),
                          action=action.numpy(),log_prob=log_prob.numpy(),value=value.numpy(),reward=reward,done=done)
                for key,value_array in data.items(): buffer[key].append(value_array.copy())
                for info in infos:
                    name=definition['actions'][info['action']]['name']; inputs[name]+=1
                    if sequence_ppo:
                        group='pve' if info['opponent_kind']=='cpu' else f"p{info['player']}"
                        exposure[group]+=info['elapsed_native_frames']
                        condition_frames[f"P{info['player']}|{info['opponent']}|{info['offset']}"]+=info['elapsed_native_frames']
                    if 'scene' in info: scenes[f"P{info['player']}|{info['opponent']}|{info['scene']}"]+=1
                    if 'episode' in info:
                        rounds[('CPU' if info['opponent_kind']=='cpu' else 'P'+str(info['player']))+'_'+info['result']]+=1
                        if info['opponent_kind']=='cpu' and info['result']=='win' and info['end_reason'] in ('signed_hp_ko','round_counter') and info['opponent_hp']<=0:rounds['CPU_KO']+=1
                        conditions[f"P{info['player']}|{info['opponent']}|{info['offset']}"]+=1
                        if sequence_ppo:
                            recorded={key:info[key] for key in ('opponent','opponent_kind','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')}
                            recorded.update(opponent_config=next(r for r in pool if r['id']==info['opponent']),
                                algorithm='PPO',collection='fresh_stochastic_on_policy',temperature=policy.action_temperature,
                                optimizer_updates_before_collection=optimizer_updates,environment_calls=steps+nenv)
                            with (out/'training_games.jsonl').open('a',encoding='utf-8') as handle:
                                handle.write(json.dumps(recorded)+'\n')
                hidden=next_hidden.detach(); previous=policy.canonical_actions(action,flip).detach(); starts=done.copy()
                remaining=torch.where(torch.as_tensor(done),torch.zeros_like(next_remaining),next_remaining).detach(); committed=action.detach()
                elapsed=elapsed+1
                durations=next_durations
                observation=next_observation; steps+=nenv
            data={key:np.asarray(value) for key,value in buffer.items()}
            with torch.no_grad():
                tensor,_=policy.obs_to_tensor(observation); features=policy.extract_features(tensor)
                bootstrap_left=torch.where(torch.as_tensor(starts),tensor['position'][:,4]>=0,controller_left)
                bootstrap_elapsed=torch.where(torch.as_tensor(starts),torch.zeros_like(elapsed),elapsed)
                _,bootstrap,_=recurrent_step(policy,features,hidden,previous,torch.as_tensor(starts),tensor['position'],remaining,
                    None if sequence_ppo else policy.read_health(tensor),bootstrap_left,bootstrap_elapsed,durations)
            advantages=np.zeros_like(data['reward']); gae=np.zeros(nenv,np.float32)
            for t in range(rollout-1,-1,-1):
                next_value=bootstrap.numpy() if t==rollout-1 else data['value'][t+1]
                live=1-data['done'][t].astype(np.float32)
                delta=data['reward'][t]+live*next_value-data['value'][t]
                gae=delta+args.gae_lambda*live*gae; advantages[t]=gae
            data['advantage']=advantages; data['returns']=advantages+data['value']
            tensors={key:torch.as_tensor(value) for key,value in data.items()}
            sequence_ids=[(e,s) for e in range(nenv) for s in range(0,rollout,length)]
            metrics=[]; early=False; policy.set_training_mode(True);optimization_started=time.monotonic()
            for epoch in range(epochs):
                rng.shuffle(sequence_ids)
                for begin in range(0,len(sequence_ids),batch_sequences):
                    chosen=sequence_ids[begin:begin+batch_sequences]; batch=len(chosen)
                    def sequence(key): return torch.stack([tensors[key][s:s+length,e] for e,s in chosen],1)
                    images=sequence('image'); positions=sequence('position')
                    features=policy.extract_features(dict(image=images.flatten(0,1),position=positions.flatten(0,1))).reshape(length,batch,-1)
                    health_predictions=None if sequence_ppo else policy.read_health(dict(image=images.flatten(0,1)))
                    if health_predictions is not None:health_predictions=health_predictions.reshape(length,batch,2)
                    states=torch.stack([tensors['hidden'][s,e] for e,s in chosen])
                    actions=sequence('action'); starts_sequence=sequence('starts'); previous_sequence=sequence('previous')
                    old_log=sequence('log_prob'); adv=sequence('advantage'); returns=sequence('returns')
                    option_mask=sequence('option_start'); selected_advantage=adv[option_mask]
                    selected_advantage=(selected_advantage-selected_advantage.mean())/(selected_advantage.std()+1e-8)
                    remaining_sequence=sequence('remaining'); duration_sequence=sequence('duration')
                    left_sequence=sequence('controller_left');elapsed_sequence=sequence('elapsed')
                    duration_history=sequence('input_durations')
                    logs=[]; values=[]; entropies=[]; family_entropies=[]
                    for offset in range(length):
                        flip=positions[offset,:,4]<0
                        distribution,value,states=recurrent_step(policy,features[offset],states,previous_sequence[offset],starts_sequence[offset],positions[offset],remaining_sequence[offset],
                            None if health_predictions is None else health_predictions[offset],left_sequence[offset],elapsed_sequence[offset],duration_history[offset])
                        duration_distribution=None if sequence_ppo else policy.duration_distribution(states,actions[offset],flip)
                        logs.append(distribution.log_prob(actions[offset])+(0 if sequence_ppo else duration_distribution.log_prob(duration_sequence[offset]))); values.append(value)
                        entropies.append(distribution.entropy() if sequence_ppo else policy.joint_entropy(distribution,states,flip))
                        family_entropies.append(value*0 if sequence_ppo else policy.family_entropy(distribution))
                    new_log=torch.stack(logs); log_change=(new_log-old_log)[option_mask]; ratio=log_change.exp()
                    actor_loss=-torch.minimum(selected_advantage*ratio,selected_advantage*ratio.clamp(1-clip_range,1+clip_range)).mean()
                    value_loss=(returns-torch.stack(values)).square().mean()
                    entropy=torch.stack(entropies)[option_mask].mean(); family_entropy=torch.stack(family_entropies)[option_mask].mean()
                    health_loss=actor_loss.new_zeros(())
                    if health_predictions is not None:health_loss=(health_predictions-sequence('health_label')).square().mean()
                    loss=actor_loss+.5*value_loss-entropy_coefficient*entropy-family_entropy_coefficient*family_entropy+10.*health_loss
                    with torch.no_grad(): kl=float(((ratio-1)-log_change).mean())
                    if kl>target_kl: early=True; break
                    if optimizer_updates==0:
                        # One actual on-policy minibatch for causal update
                        # diagnosis; no additional rollout or learning step.
                        torch.save(dict(images=images.detach().cpu(),positions=positions.detach().cpu(),
                            hidden=torch.stack([tensors['hidden'][s,e] for e,s in chosen]).detach().cpu(),
                            actions=actions.cpu(),starts=starts_sequence.cpu(),previous=previous_sequence.cpu(),
                            input_durations=duration_history.cpu(),old_log=old_log.cpu(),
                            normalized_advantage=selected_advantage.cpu(),option_mask=option_mask.cpu(),returns=returns.cpu(),
                            optimizer=policy.optimizer.state_dict(),entropy_coefficient=entropy_coefficient,
                            replay_rng=None if success_replay is None else copy.deepcopy(success_replay.rng.bit_generator.state),
                            source=args.initial,scope='Actual first fresh on-policy batch; separate auxiliary bank is not PPO advantage data.'),
                            out/'first_update_batch.pt')
                    auxiliary=loss.new_zeros(())
                    if success_spec is not None and success_spec['coefficient']:
                        interval=success_spec.get('assessment_every_optimizer_updates',0)
                        if interval and optimizer_updates%interval==0:
                            assessment_started=time.monotonic()
                            assessments.append(success_replay.assess_success_disagreements(policy,optimizer_updates))
                            assessment_seconds+=time.monotonic()-assessment_started
                        auxiliary_started=time.monotonic()
                        auxiliary,n=recurrent_success_loss(policy,success_replay,success_spec['bank']['sample_sequences'],
                            success_spec['bank']['learn_length'],success_spec.get('disagreement_fraction',0.),
                            success_spec.get('disagreement_scope','failed'),priority_fraction=success_spec.get('priority_fraction',0.),
                            use_soft_references=success_spec.get('use_soft_references',False),
                            hard_failed_conditions=success_spec.get('hard_failed_conditions',False),
                            disagreement_selection=success_spec.get('disagreement_selection','random'))
                        if optimizer_updates%128==0 or optimizer_updates in args.capture_updates:
                            parameters=[p for p in policy.parameters() if p.requires_grad]
                            ppo_gradient=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                            replay_gradient=torch.autograd.grad(success_spec['coefficient']*auxiliary,parameters,retain_graph=True,allow_unused=True)
                            pnorm=sum(float(g.square().sum()) for g in ppo_gradient if g is not None)**.5
                            rnorm=sum(float(g.square().sum()) for g in replay_gradient if g is not None)**.5
                            dot=sum(float(a.mul(b).sum()) for a,b in zip(ppo_gradient,replay_gradient) if a is not None and b is not None)
                            actor_gradient=torch.autograd.grad(actor_loss-entropy_coefficient*entropy,parameters,retain_graph=True,allow_unused=True)
                            critic_gradient=torch.autograd.grad(.5*value_loss,parameters,retain_graph=True,allow_unused=True)
                            anorm=sum(float(g.square().sum()) for g in actor_gradient if g is not None)**.5
                            vnorm=sum(float(g.square().sum()) for g in critic_gradient if g is not None)**.5
                            def gradient_cosine(first,second,first_norm,second_norm):
                                product=sum(float(a.mul(b).sum()) for a,b in zip(first,second) if a is not None and b is not None)
                                return product/max(1e-30,first_norm*second_norm)
                            gradient_audits.append(dict(optimizer_update=optimizer_updates,ppo_norm=pnorm,weighted_auxiliary_norm=rnorm,
                                cosine=dot/max(1e-30,pnorm*rnorm),actor_norm=anorm,critic_norm=vnorm,
                                actor_critic_cosine=gradient_cosine(actor_gradient,critic_gradient,anorm,vnorm),
                                actor_replay_cosine=gradient_cosine(actor_gradient,replay_gradient,anorm,rnorm),
                                critic_replay_cosine=gradient_cosine(critic_gradient,replay_gradient,vnorm,rnorm)))
                            if optimizer_updates in args.capture_updates:
                                names=[name for name,p in policy.named_parameters() if p.requires_grad]
                                torch.save(dict(definition=dict(definition,action_temperature=policy.action_temperature),
                                    state_dict=policy.state_dict(),optimizer=policy.optimizer.state_dict(),
                                    gradients={label:{name:None if g is None else g.detach().cpu() for name,g in zip(names,values)}
                                        for label,values in [('ppo',ppo_gradient),('auxiliary',replay_gradient),('actor',actor_gradient),('critic',critic_gradient)]},
                                    optimizer_update=optimizer_updates,native_frames=native_frames,
                                    scope='Actual fresh-rollout gradients before this update; diagnostic only, no additional optimization.'),
                                    out/f'update_diagnostic_{optimizer_updates}.pt')
                        loss=loss+success_spec['coefficient']*auxiliary
                        auxiliary_samples+=n;auxiliary_updates+=1;auxiliary_losses.append(float(auxiliary.detach()))
                        auxiliary_seconds+=time.monotonic()-auxiliary_started
                    policy.optimizer.zero_grad(); loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
                    if not bool(torch.isfinite(gradient)): raise RuntimeError('Nonfinite full-network recurrent gradient')
                    policy.optimizer.step()
                    if optimizer_updates==0 or optimizer_updates in args.capture_updates:
                        name='first_update_after.pt' if optimizer_updates==0 else f'update_after_{optimizer_updates}.pt'
                        save(out/name,policy,definition,
                            dict(purpose='Actual single update for diagnostic replay verification; not a selected candidate'),policy.optimizer)
                    optimizer_updates+=1;optimization_samples+=int(actions.numel())
                    metrics.append(dict(actor_loss=float(actor_loss.detach()),value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),family_entropy=float(family_entropy.detach()),kl=kl,
                                        health_mse=float(health_loss.detach())))
                    if args.max_updates and optimizer_updates>=args.max_updates:break
                if early or (args.max_updates and optimizer_updates>=args.max_updates): break
            optimization_seconds+=time.monotonic()-optimization_started
            if len(history)==0:
                changes={name:float((parameter-initial_parameters[name]).abs().max()) for name,parameter in policy.named_parameters()}
                action_weight='action_net.2.weight' if definition.get('action_readout')=='relu' else 'action_net.weight'
                required=(('features_extractor.visual.0.0.weight','features_extractor.coordinates.0.weight',action_weight,'value_net.weight','sequence_gru.weight_ih_l0','input_duration_projection.weight') if sequence_ppo else
                    ('features_extractor.image_net.0.0.weight','features_extractor.position_net.0.weight','action_net.weight','value_net.weight','history_gru.weight_ih','history_action.weight','history_value.weight'))
                if any(changes[name]<=0 for name in required): raise RuntimeError('A required trainable component did not update')
                if args.learn_durations and changes['history_duration.weight']<=0: raise RuntimeError('The duration choice did not learn')
                write(out/'first_update_contract.json',dict(max_parameter_change=changes,all_inference_parameters_trainable=True))
            fresh_frames=sum(env.get_attr('native_frames'));native_frames=fresh_frames+success_frames
            if success_spec is not None and native_frames>=success_due and args.native_frames-native_frames>=10000:
                collected=refresh_successes(policy,definition,success_replay,success_spec,shared,out,
                    success_clock_base+native_frames,success_event,cycle_event=periodic_round)
                success_frames+=collected['native_frames'];success_calls+=collected['environment_calls']
                success_events.append(dict(frame=native_frames,reason='periodic',**collected));success_event+=1;periodic_round+=1
                success_due=native_frames+success_spec['bank']['refresh_every_native_frames'];native_frames=fresh_frames+success_frames
                torch.save(dict(success_replay.state_dict(),native_frames=success_clock_base+native_frames),out/'success_bank.pt')
            report_success_learning()
            row=dict(steps=steps,seconds=time.monotonic()-started,native_frames=native_frames,completed_rounds=dict(rounds),
                     optimizer_updates=optimizer_updates,optimization_samples=optimization_samples,optimization_seconds=optimization_seconds,
                     decision_native_frames_by_group=dict(exposure),reset_offset_native_frames=fresh_frames-sum(exposure.values()) if sequence_ppo else None,
                     success_collection_native_frames=success_frames,auxiliary_samples=auxiliary_samples,
                     temperature=policy.action_temperature,entropy_coefficient=entropy_coefficient,family_entropy_coefficient=family_entropy_coefficient,
                     option_decisions=option_decisions,duration_choices=dict(duration_counts),
                     optimizer={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]} if metrics else {},early_kl_stop=early,
                     minimum_pvp_condition_rounds=min(conditions[f"P{port}|{r['id']}|{offset}"] for port in (1,2) for r in pool for offset in (0,12)))
            history.append(row); write(out/'live.json',row); write(out/'condition_exposure.json',dict(completed_rounds=dict(conditions),decision_native_frames=dict(condition_frames)))
            print('DIRECT_SEQUENCE_PPO',json.dumps(row),flush=True)
            due=[p for p in checkpoints if native_frames>=p]
            if (bool(due) if args.native_frames else steps-last_save>=args.checkpoint_every or steps>=args.steps):
                definition['action_temperature']=policy.action_temperature
                save(out/f'checkpoint_{steps}.pt',policy,definition,dict(algorithm='PPO',steps=steps,native_frames=native_frames,
                    seed=args.seed,condition_counts=dict(conditions),optimizer_updates=optimizer_updates,optimization_samples=optimization_samples),policy.optimizer)
                last_save=steps
                if due:
                    write(out/f'checkpoint_{steps}.json',dict(native_frames=native_frames,requested_native_frames=due,optimizer_updates=optimizer_updates,optimization_samples=optimization_samples))
                    checkpoints=[p for p in checkpoints if p not in due]
                    if sequence_ppo and shared.get('development_pool'):
                        from experiments.league_adaptive.runtime import Actor
                        from experiments.league_adaptive.evaluate import evaluate
                        from experiments.ryu_counterq.training_state import random_state,restore_random_state
                        saved_rng=random_state();evaluation_started=time.monotonic()
                        try:
                            tested=evaluate(Actor(copy.deepcopy(policy),definition),shared['development_pool'],out/f'development_{due[-1]}',
                                cpu=False,offsets=(0,),arena_call_sequence=True,progress=False,reward_kind=shared['learner']['reward'])
                        finally:restore_random_state(saved_rng)
                        evaluation_seconds+=time.monotonic()-evaluation_started
                        evaluation_native_frames+=tested['native_frames']
                        frozen_evaluations.append(dict(requested_native_frames=due,actual_native_frames=native_frames,
                            checkpoint=(out/f'checkpoint_{steps}.pt').resolve().relative_to(ROOT).as_posix(),development=tested))
                        write(out/'frozen_evaluations.json',frozen_evaluations)
                        if success_spec is not None:
                            missing=success_replay.note_complete_results(tested['rows'],
                                self_success_owner=out.resolve().relative_to(ROOT).as_posix(),
                                prioritize_failures=success_spec.get('prioritize_current_failures',False))
                            if success_spec.get('capture_missing_checkpoint_successes',False):
                                if missing:
                                    collected=refresh_successes(policy,definition,success_replay,success_spec,shared,out,
                                        success_clock_base+native_frames,success_event,
                                        selected=[{k:r[k] for k in ('opponent','player','offset')} for r in missing])
                                    expected={(r['opponent'],r['player'],r['offset']):r for r in missing}
                                    for r in collected['outcomes']:
                                        before=expected[(r['opponent'],r['player'],r['offset'])]
                                        if any(r[k]!=before[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')):
                                            raise RuntimeError('New success capture did not reproduce its complete discovery game')
                                    success_frames+=collected['native_frames'];success_calls+=collected['environment_calls']
                                    success_events.append(dict(frame=native_frames,reason='new_checkpoint_success',**collected));success_event+=1
                                    native_frames=sum(env.get_attr('native_frames'))+success_frames
                                    torch.save(dict(success_replay.state_dict(),native_frames=success_clock_base+native_frames),out/'success_bank.pt')
                                    report_success_learning()
        if last_save!=steps:
            definition['action_temperature']=policy.action_temperature
            save(out/f'checkpoint_{steps}.pt',policy,definition,dict(algorithm='PPO',steps=steps,native_frames=native_frames,
                seed=args.seed,condition_counts=dict(conditions),optimizer_updates=optimizer_updates,optimization_samples=optimization_samples),policy.optimizer)
        write(out/'scene_coverage.json',dict(decisions=dict(scenes),input_counts=dict(inputs)))
        if success_spec is not None:torch.save(dict(success_replay.state_dict(),native_frames=success_clock_base+native_frames),out/'success_bank.pt')
        report_success_learning()
        write(out/'result.json',dict(status='TRAINED_REQUIRES_COMPLETE_GAMES',arguments=vars(args),steps=steps,seconds=time.monotonic()-started,
                                     native_frames=sum(env.get_attr('native_frames'))+success_frames,completed_rounds=dict(rounds),history=history,
                                     optimizer_updates=optimizer_updates,optimization_samples=optimization_samples,optimization_seconds=optimization_seconds,
                                     environment_calls=steps,decision_native_frames_by_group=dict(exposure),
                                     reset_offset_native_frames=sum(env.get_attr('native_frames'))-sum(exposure.values()) if sequence_ppo else None,
                                     success_collection_native_frames=success_frames,success_collection_environment_calls=success_calls,
                                     auxiliary_updates=auxiliary_updates,auxiliary_samples=auxiliary_samples,auxiliary_seconds=auxiliary_seconds,
                                     replay_assessment_seconds=assessment_seconds,
                                     evaluation_native_frames=evaluation_native_frames,evaluation_seconds=evaluation_seconds,frozen_evaluations=frozen_evaluations))
    finally: env.close()


if __name__=='__main__': main()
