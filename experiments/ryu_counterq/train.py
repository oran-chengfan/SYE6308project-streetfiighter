"""Semi-Markov double DQN using real and counterfactual native transitions."""
import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.league_adaptive.runtime import setup,write
from experiments.ryu_direct_league.policy import save
from experiments.ryu_counterq.policy import initialize,construct_counterq,make_extras,expand_multiscale
from experiments.ryu_counterq.replay import Replay,DemonstrationMix,double_q_target,demonstration_objective
from experiments.ryu_counterq.environment import Worker


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initial',required=True);parser.add_argument('--reader',required=True)
    parser.add_argument('--pool',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--steps',type=int,default=196608);parser.add_argument('--seed',type=int,default=2231)
    parser.add_argument('--branch-every',type=int,default=8);parser.add_argument('--context',type=int,default=16)
    parser.add_argument('--max-seconds',type=float,default=14400);parser.add_argument('--checkpoint-every',type=int,default=24576)
    parser.add_argument('--demonstrations',nargs='+');parser.add_argument('--pretrain-updates',type=int,default=0)
    parser.add_argument('--gamma',type=float,default=.999);parser.add_argument('--exploration-call-fraction',type=float)
    parser.add_argument('--demo-objective',choices=('lower_bound','nstep'),default='lower_bound')
    parser.add_argument('--reward-objective',choices=('graded','win'),default='graded')
    parser.add_argument('--history-layout',choices=('dense','multiscale'),default='dense');parser.add_argument('--replay-nodes',type=int,default=150000)
    parser.add_argument('--prefix-priority',action='store_true')
    args=parser.parse_args();setup();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    action_rng=np.random.default_rng(args.seed);branch_rng=np.random.default_rng(args.seed+1987)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=torch.load(args.initial,map_location='cpu',weights_only=False);reader=torch.load(args.reader,map_location='cpu',weights_only=False)
    if source['definition']['kind']=='context_double_dqn':
        definition=copy.deepcopy(source['definition']);policy=construct_counterq(definition);policy.load_state_dict(source['state_dict'],strict=True)
        if 'optimizer_state' in source:policy.optimizer.load_state_dict(source['optimizer_state'])
    else:policy,definition=initialize(source,reader,args.context)
    if args.history_layout=='multiscale' and definition.get('context_offsets') is None:policy,definition=expand_multiscale(policy,definition)
    elif args.history_layout=='dense' and definition.get('context_offsets') is not None:raise ValueError('Declare the saved history layout')
    target=construct_counterq(definition);target.load_state_dict(policy.state_dict(),strict=True)
    for p in target.parameters():p.requires_grad_(False)
    target.optimizer=None;target.set_training_mode(False)
    context=definition['context_length'];offsets=definition.get('context_offsets')
    gamma=args.gamma;demo=DemonstrationMix(args.demonstrations,context=context,seed=args.seed+799,objective=args.reward_objective,offsets=offsets,prefix_priority=args.prefix_priority) if args.demonstrations else None
    if args.reward_objective=='win' and gamma!=1.:raise ValueError('Win-probability potential objective requires gamma=1')
    updates=0;last_demo_metrics={}

    def optimize(data,demonstration_data=None):
        nonlocal updates,last_demo_metrics
        policy.set_training_mode(True)
        with torch.no_grad():
            online_next=policy.context_values(*data['next_observation']);target_next=target.context_values(*data['next_observation'])
            desired=double_q_target(online_next,target_next,data['reward'],data['done'],data['duration'],gamma).clamp(-1.5,1.5)
        values=policy.context_values(*data['observation']);actual=values.gather(1,data['action'][:,None]).flatten()
        loss=F.smooth_l1_loss(actual,desired)
        if demonstration_data is not None:
            if demonstration_data is data:demo_values=values
            else:
                with torch.no_grad():
                    online=policy.context_values(*demonstration_data['next_observation']);fixed=target.context_values(*demonstration_data['next_observation'])
                    demo_target=double_q_target(online,fixed,demonstration_data['reward'],demonstration_data['done'],demonstration_data['duration'],gamma).clamp(-1.5,1.5)
                demo_values=policy.context_values(*demonstration_data['observation'])
                loss=loss+.5*F.smooth_l1_loss(demo_values.gather(1,demonstration_data['action'][:,None]).flatten(),demo_target)
            auxiliary,last_demo_metrics=demonstration_objective(demo_values,demonstration_data,return_anchor=args.demo_objective=='lower_bound')
            loss=loss+auxiliary
            if args.demo_objective=='nstep':
                if gamma!=1.:raise ValueError('Stored native demonstration n-step rewards are undiscounted')
                with torch.no_grad():
                    online=policy.context_values(*demonstration_data['nstep_next_observation']);fixed=target.context_values(*demonstration_data['nstep_next_observation'])
                    multistep=double_q_target(online,fixed,demonstration_data['nstep_reward'],demonstration_data['nstep_done'],demonstration_data['nstep_duration'],gamma).clamp(-1.5,1.5)
                multistep_loss=F.smooth_l1_loss(demo_values.gather(1,demonstration_data['action'][:,None]).flatten(),multistep)
                loss=loss+multistep_loss;last_demo_metrics['nstep_td']=float(multistep_loss.detach())
                loss=loss+1e-5*sum(p.square().sum() for p in policy.parameters() if p.requires_grad)
        policy.optimizer.zero_grad();loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],10.)
        if not bool(torch.isfinite(gradient)):raise RuntimeError('Nonfinite double-Q gradient')
        policy.optimizer.step();updates+=1
        if updates%500==0:target.load_state_dict(policy.state_dict(),strict=True)
        return float(loss.detach())

    if args.pretrain_updates:
        if demo is None or gamma!=1.:raise ValueError('Undiscounted demonstration returns require demonstrations and gamma=1')
        save(out/'before_demonstrations.pt',policy,definition,dict(stage='before_native_demonstrations'))
        for iteration in range(args.pretrain_updates):
            batch=demo.sample(64);loss_value=optimize(batch,batch)
            if (iteration+1)%128==0:
                report=dict(updates=updates,seconds=time.monotonic()-started,loss=loss_value,**last_demo_metrics)
                write(out/'pretraining_live.json',report);print('NATIVE_DEMONSTRATION_PRETRAIN',json.dumps(report),flush=True)
            if (iteration+1)%512==0 or iteration+1==args.pretrain_updates:
                save(out/f'pretrained_{iteration+1}.pt',policy,definition,dict(stage='native_demonstration_pretraining',updates=updates),policy.optimizer)
        target.load_state_dict(policy.state_dict(),strict=True)
    pool=json.loads(Path(args.pool).read_text());cpu=[dict(id=f'CPU_Ryu_{s}',kind='cpu',character='Ryu',stars=s) for s in range(1,9)]
    env=SubprocVecEnv([Worker(definition,pool,0,args.seed,args.reward_objective),Worker(definition,pool,1,args.seed+317,args.reward_objective),Worker(definition,cpu,2,args.seed+634,args.reward_objective)],start_method='spawn')
    env.seed(args.seed);replay=Replay(nodes=args.replay_nodes,context=context,seed=args.seed+321,offsets=offsets)
    durations=policy.context_durations;k=len(durations);names={a['name']:i for i,a in enumerate(definition['actions'])}
    observation=env.reset();role=observation['position'][:,4]>=0;elapsed=np.zeros(3,np.int64);pending=np.zeros(3,np.int64)
    committed=np.zeros(3,np.int64);previous=np.zeros(3,np.int64);records=[None]*3;seen_options=np.zeros(3,np.int64)
    current=[];steps=0;last_save=0;rounds=Counter();conditions=Counter();family_counts=Counter();history=[];branch_transitions=0
    exploring=np.zeros(3,bool);exploration_calls=0;epsilon_sum=0.;exploration_choices=0
    initial_parameters={n:p.detach().clone() for n,p in policy.named_parameters()}

    def encoded(observations,roles,times,remaining):
        tensor,_=policy.obs_to_tensor(observations)
        with torch.no_grad():
            maps=policy.encode_maps(tensor).cpu().numpy();health=policy.health_reader(tensor['image'])
            extra=make_extras(tensor['position'],health,torch.as_tensor(roles),torch.as_tensor(times),torch.as_tensor(remaining)).cpu().numpy()
        return maps,extra

    maps,extra=encoded(observation,role,elapsed,pending)
    for i in range(3):current.append(replay.add_node(-1,maps[i],observation['position'][i],extra[i],0))

    def proposal(i):
        dx=observation['position'][i,4];forward='RIGHT' if dx>=0 else 'LEFT';back='LEFT' if dx>=0 else 'RIGHT';face='R' if dx>=0 else 'L'
        choice=int(branch_rng.integers(4))
        if choice==0:return [(names['UP_'+forward],1),(names['C_tap'],1)]
        if choice==1:return [(names[forward],4),(names['CROUCH_KICK_C'],1)]
        if choice==2:return [(names[back],4),(names[f'HADOUKEN_{face}_Y'],1)]
        return [(int(branch_rng.integers(42)),int(branch_rng.choice(durations))) for _ in range(2)]

    def retain_branch(i,branch):
        nonlocal branch_transitions
        group=f"{branch['opponent']}|P{branch['player']}|{branch['offset']}"
        for segments in branch['plans']:
            parent=current[i];clock=int(elapsed[i])
            for segment in segments:
                start=parent;physical=segment['action'];planned=segment['duration'];reward=0.;ticks=segment['ticks']
                start_flip=replay.positions[start%replay.capacity,4]<0
                canonical=int(policy.physical_mirror[physical]) if start_flip and policy.canonical else physical
                observations={key:np.stack([tick['observation'][key] for tick in ticks]) for key in ('image','position')}
                remaining=np.maximum(0,planned-np.arange(1,len(ticks)+1));times=clock+np.arange(1,len(ticks)+1)
                branch_maps,branch_extra=encoded(observations,np.full(len(ticks),role[i]),times,remaining)
                for j,tick in enumerate(ticks):
                    flip=replay.positions[parent%replay.capacity,4]<0
                    actual_previous=int(policy.physical_mirror[physical]) if flip and policy.canonical else physical
                    parent=replay.add_node(parent,branch_maps[j],observations['position'][j],branch_extra[j],actual_previous)
                    reward+=(gamma**j)*tick['reward']
                replay.add_transition(start,canonical*k+durations.index(planned),reward,parent,ticks[-1]['done'],len(ticks),group)
                branch_transitions+=1;clock+=len(ticks)

    write(out/'config.json',dict(arguments=vars(args),pool=pool,cpu=cpu,algorithm='Recurrent double DQN on action-duration choices, with target-network evaluation and online-network selection.',
        gamma_per_six_frame_call=gamma,reward='win+1, all other complete outcomes-1; telescoping0.25HP potential with terminal potential0' if args.reward_objective=='win' else 'signed terminal outcome plus0.25*net damage/176',batch=64,train_every_actual_calls=12,target_every_updates=500,
        demonstration_replay=None if demo is None else dict(path=args.demonstrations,transitions=demo.transition_count,pretrain_updates=args.pretrain_updates,
            objective=args.demo_objective,
            prefix_priority=args.prefix_priority,
            method='Permanent group-balanced native successes and failures. Double-Q TD on both; success-only joint-action margin. Explicit objective chooses observed-return anchor or32-call native n-step Double-Q TD and L2. Not an oracle or a frozen teacher policy.'),
        exploration='Original option-epsilon schedule' if args.exploration_call_fraction is None else 'Desired random-call fraction rho; option epsilon=rho*greedy_duration/(mean_random_duration*(1-rho)+rho*greedy_duration). Measured actual random calls reported separately.',
        replay='RAM-only graph; half uniform transitions and half uniform opponent/port/offset groups. Context rebuilt from actual shared prefixes, never stale stored GRU states.',
        representation='Fixed pretrained convolutions and screen-only health reader; trainable spatial projection, position encoder, recurrent belief and every Q head. Frozen modules are perception, not combat policies.',
        temporal_memory=dict(calls_retained=context,sampled_ages=offsets,explicit_gap_feature=offsets is not None),
        counterfactual='Restore one real state and both histories, execute a sampled one/two-option plan against the acting opponent, retain every actual transition and restore the factual stream. Planning proposals are training exploration only.',
        deployment='One weight set, four legal images and six supplied coordinates. Bounded16-observation history, actual previous action, visible health and own call count. Greedy action-duration value maximization; no runtime simulator/search or opponent ID.'))
    save(out/'initial.pt',policy,definition,dict(stage='double_dqn_initialization',source=args.initial))
    try:
        while steps<args.steps and time.monotonic()-started<args.max_seconds and not (out/'STOP').exists():
            policy.set_training_mode(False);free=np.flatnonzero(pending==0)
            epsilon=.4+(.05-.4)*min(steps/150000.,1.)
            if len(free):
                with torch.no_grad():q=policy.context_values(*replay.contexts([current[i] for i in free]))
                options=q.argmax(1).numpy()
                for j,i in enumerate(free):
                    if args.exploration_call_fraction is not None:
                        rho=args.exploration_call_fraction+(.03-args.exploration_call_fraction)*min(steps/150000.,1.)
                        greedy_duration=durations[int(options[j])%k]
                        epsilon=rho*greedy_duration/(np.mean(durations)*(1-rho)+rho*greedy_duration)
                    epsilon_sum+=epsilon;exploration_choices+=1
                    exploring[i]=action_rng.random()<epsilon
                    if exploring[i]:options[j]=int(action_rng.integers(42*k))
                    canonical=int(options[j])//k;duration=int(durations[int(options[j])%k]);flip=observation['position'][i,4]<0
                    committed[i]=int(policy.physical_mirror[canonical]) if flip and policy.canonical else canonical
                    pending[i]=duration;seen_options[i]+=1
                    records[i]=dict(start=current[i],action=int(options[j]),reward=0.,spent=0)
                    if args.branch_every and seen_options[i]%args.branch_every==0:
                        branch=env.env_method('counterfactual_plans',[proposal(i)],indices=int(i))[0]
                        retain_branch(int(i),branch)
            before_flip=observation['position'][:,4]<0
            exploration_calls+=int(exploring.sum())
            next_observation,reward,done,infos=env.step(committed.copy())
            pending_next=pending-1;elapsed_next=elapsed+1
            previous_next=np.asarray([int(policy.physical_mirror[a]) if before_flip[i] and policy.canonical else int(a) for i,a in enumerate(committed)],np.int64)
            next_role=np.where(done,next_observation['position'][:,4]>=0,role);next_elapsed=np.where(done,0,elapsed_next);next_pending=np.where(done,0,pending_next)
            next_maps,next_extra=encoded(next_observation,next_role,next_elapsed,next_pending)
            for i,info in enumerate(infos):
                record=records[i];record['reward']+=(gamma**record['spent'])*float(reward[i]);record['spent']+=1
                group=f"{info['opponent']}|P{info['player']}|{info['offset']}"
                family_counts[definition['actions'][int(committed[i])]['name']]+=1
                if done[i]:
                    terminal={key:info['terminal_observation'][key][None] for key in ('image','position')}
                    terminal_maps,terminal_extra=encoded(terminal,np.asarray([role[i]]),np.asarray([elapsed_next[i]]),np.asarray([pending_next[i]]))
                    end=replay.add_node(current[i],terminal_maps[0],terminal['position'][0],terminal_extra[0],previous_next[i])
                else:end=replay.add_node(current[i],next_maps[i],next_observation['position'][i],next_extra[i],previous_next[i])
                if done[i] or pending_next[i]==0:
                    replay.add_transition(record['start'],record['action'],record['reward'],end,bool(done[i]),record['spent'],group);records[i]=None
                if done[i]:
                    rounds[('CPU' if info['opponent_kind']=='cpu' else 'P'+str(info['player']))+'_'+info['result']]+=1
                    conditions[group]+=1
                    current[i]=replay.add_node(-1,next_maps[i],next_observation['position'][i],next_extra[i],0)
                else:current[i]=end
            observation=next_observation;role=next_role;elapsed=next_elapsed;pending=next_pending;previous=previous_next;steps+=3
            loss_value=None
            if steps>=3072 and (steps//3)%4==0:
                data=replay.sample(48 if demo is not None else 64)
                if data is not None:
                    loss_value=optimize(data,demo.sample(16) if demo is not None else None)
                    if updates==args.pretrain_updates+32:
                        changes={name:float((parameter-initial_parameters[name]).abs().max()) for name,parameter in policy.named_parameters()}
                        required=('features_extractor.image_net.1.weight','features_extractor.position_net.0.weight','action_net.weight','value_net.weight','context_gru.weight_ih_l0','context_advantage.weight','context_value.weight')
                        if any(changes[name]<=0 for name in required):raise RuntimeError('A trainable Q component did not update')
                        if any(value!=0 for name,value in changes.items() if name.startswith(('features_extractor.image_net.0.','health_reader.'))):raise RuntimeError('A fixed perception sensor changed')
                        write(out/'gradient_contract.json',dict(parameter_changes=changes,updates=updates))
            if steps%3072==0:
                total_frames=sum(env.get_attr('native_frames'));branch_frames=sum(env.get_attr('branch_native_frames'))
                row=dict(steps=steps,seconds=time.monotonic()-started,epsilon=epsilon,updates=updates,replay_transitions=replay.transition_count,
                         mean_option_epsilon=epsilon_sum/max(1,exploration_choices),actual_random_call_fraction=exploration_calls/max(1,steps),demonstration=last_demo_metrics,
                         counterfactual_transitions=branch_transitions,native_frames=total_frames,additional_branch_native_frames=branch_frames,
                         completed_rounds=dict(rounds),loss=loss_value,minimum_pvp_condition_rounds=min(conditions[f"{r['id']}|P{p}|{o}"] for r in pool for p in (1,2) for o in (0,12)))
                history.append(row);write(out/'live.json',row);write(out/'condition_exposure.json',dict(completed_rounds=dict(conditions)))
                print('COUNTERQ',json.dumps(row),flush=True)
            if steps-last_save>=args.checkpoint_every or steps>=args.steps:
                save(out/f'checkpoint_{steps}.pt',policy,definition,dict(algorithm='counterfactual_double_dqn',steps=steps,updates=updates,condition_counts=dict(conditions)),policy.optimizer);last_save=steps
        if last_save!=steps:save(out/f'checkpoint_{steps}.pt',policy,definition,dict(algorithm='counterfactual_double_dqn',steps=steps,updates=updates),policy.optimizer)
        write(out/'result.json',dict(status='TRAINED_REQUIRES_COMPLETE_MATCHES',arguments=vars(args),steps=steps,seconds=time.monotonic()-started,
                                     updates=updates,native_frames=sum(env.get_attr('native_frames')),additional_branch_native_frames=sum(env.get_attr('branch_native_frames')),
                                     completed_rounds=dict(rounds),input_counts=dict(family_counts),history=history))
    finally:env.close()


if __name__=='__main__':main()
