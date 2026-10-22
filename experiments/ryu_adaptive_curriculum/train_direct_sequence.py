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

from experiments.league_adaptive.runtime import setup, write
from experiments.ryu_direct_league.environment import RyuEnv
from experiments.ryu_direct_league.policy import save
from experiments.ryu_adaptive_curriculum.recurrent_direct import initialize, construct_recurrent, initialize_durations,initialize_health


class CoverageEnv(RyuEnv):
    def __init__(self, definition, pool, rank, seed, randomize=True, reward_kind='potential',damage_weight=.25):
        super().__init__(definition, pool, rank, gamma=1.)
        self.scheduler_rng = np.random.default_rng(seed)
        self.completed_conditions = Counter()
        self.condition_results = defaultdict(lambda: deque(maxlen=16))
        self.pending_condition = None
        self.track_scenes = True
        self.training_reward_kind = reward_kind
        self.damage_weight=float(damage_weight)
        if randomize: self.enable_behavior_randomization(seed+10000)

    def reset(self, *, seed=None, options=None):
        if options is None:
            conditions = [(i, offset) for i, rival in enumerate(self.pool)
                          for offset in ((0,) if rival['kind']=='cpu' else (0,12))]
            least = min(self.completed_conditions[c] for c in conditions)
            candidates = [c for c in conditions if self.completed_conditions[c]==least]
            weights = np.asarray([(1-np.mean(self.condition_results[c]))**2+.1 if self.condition_results[c] else .35
                                  for c in candidates])
            chosen = candidates[int(self.scheduler_rng.choice(len(candidates), p=weights/weights.sum()))]
            options = dict(opponent=chosen[0], offset=chosen[1])
            self.pending_condition = chosen
        else: self.pending_condition = None
        return super().reset(seed=seed, options=options)

    def step(self, action, **kwargs):
        pre_health=[max(0,self.game.raw['p1_hp'])/176.,max(0,self.game.raw['p2_hp'])/176.]
        observation, reward, terminated, truncated, info = super().step(action, **kwargs)
        info['pre_action_absolute_health']=pre_health
        if self.training_reward_kind == 'damage':
            reward = self.damage_weight*(info['damage_dealt']-info['damage_taken'])/176. + float(info['result']=='win')-float(info['result']=='loss')
            self.episode_return += reward-info['step_reward']; info['step_reward']=reward
            if 'episode' in info: info['episode']['r']=self.episode_return
        if 'episode' in info and self.pending_condition is not None:
            self.completed_conditions[self.pending_condition] += 1
            self.condition_results[self.pending_condition].append(float(info['result']=='win'))
        return observation, reward, terminated, truncated, info


class CoverageWorker:
    def __init__(self, definition, pool, rank, seed, reward_kind,damage_weight=.25):
        self.definition, self.pool, self.rank, self.seed, self.reward_kind = definition, pool, rank, seed, reward_kind
        self.damage_weight=float(damage_weight)
    def __call__(self):
        setup()
        return CoverageEnv(self.definition, self.pool, self.rank, self.seed, reward_kind=self.reward_kind,damage_weight=self.damage_weight)


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
    args=parser.parse_args(); setup(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.initial,map_location='cpu',weights_only=False)
    if source['definition']['kind']=='recurrent_direct_ppo':
        definition=copy.deepcopy(source['definition'])
        if definition['memory_enabled']!=(args.memory=='on'): raise ValueError('Resume the same recurrent condition')
        if args.learn_durations and len(definition.get('duration_choices',(1,)))==1:
            policy,definition=initialize_durations(source)
        else:
            policy=construct_recurrent(definition); policy.load_state_dict(source['state_dict'],strict=True)
            if 'optimizer_state' in source: policy.optimizer.load_state_dict(source['optimizer_state'])
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
    pool=json.loads(Path(args.pool).read_text())
    cpu=[dict(id=f'CPU_Ryu_{s}',kind='cpu',character='Ryu',stars=s) for s in range(1,9)]
    workers=[CoverageWorker(definition,pool,i,args.seed+i*317,args.reward,args.damage_weight) for i in range(4)]
    workers.extend(CoverageWorker(definition,cpu,i,args.seed+i*317,args.reward,args.damage_weight) for i in (4,5))
    env=SubprocVecEnv(workers,start_method='spawn'); env.seed(args.seed)
    nenv=6; rollout=512; length=32; batch_sequences=8
    started=time.monotonic(); steps=0; last_save=0; history=[]; rounds=Counter(); scenes=Counter(); inputs=Counter(); conditions=Counter()
    initial_parameters={name:value.detach().clone() for name,value in policy.named_parameters()}
    write(out/'config.json',dict(arguments=vars(args),pool=pool,cpu=cpu,rollout=rollout,sequence_length=length,epochs=4,
        optimizer='Adam; all visual, coordinate, direct action/value and GRU parameters trainable',clip_range=.1,
        coverage='Each worker selects only conditions with its lowest completed-game count; hardness breaks ties. Offsets are scheduled explicitly and are not correlated with randomly selected opponents.',
        exploration='Ordinary categorical entropy plus action-family entropy. This is an exploration regularizer, not a reward for walking or an action ban. All42 actions remain available.',
        objective='Native complete-game outcome and declared HP reward; no opponent identity, star, elapsed script phase or simulator state enters the actor.',
        durations=definition.get('duration_choices',[1]),
        temporal_objective='Actor likelihood is the joint chosen physical input and conditional duration at option starts only. The augmented critic observes pending duration and learns at every six-frame call. Opponents continue every six frames. Native frame budgets include all committed calls.',
        inference='Direct42-action network with own actual previous input and recurrent memory. No frozen expert, teacher prior or tactical proposal routing.'))
    save(out/'initial.pt',policy,definition,dict(initial=args.initial))
    observation=env.reset(); hidden=torch.zeros(nenv,policy.memory_hidden); previous=torch.zeros(nenv,dtype=torch.long); starts=np.ones(nenv,bool)
    remaining=torch.zeros(nenv,dtype=torch.long); committed=torch.zeros(nenv,dtype=torch.long); option_decisions=0; duration_counts=Counter()
    controller_left=torch.as_tensor(observation['position'][:,4]>=0);elapsed=torch.zeros(nenv)
    try:
        while steps<args.steps and time.monotonic()-started<args.max_seconds and not (out/'STOP').exists():
            fraction=min(1.,steps/args.steps)
            policy.action_temperature=temperature_initial+(temperature_final-temperature_initial)*fraction
            entropy_coefficient=args.entropy if args.entropy_final is None else args.entropy+(args.entropy_final-args.entropy)*fraction
            family_entropy_coefficient=args.family_entropy if args.family_entropy_final is None else args.family_entropy+(args.family_entropy_final-args.family_entropy)*fraction
            buffer={key:[] for key in ('image','position','hidden','previous','starts','remaining','option_start','duration','action','log_prob','value','reward','done','controller_left','elapsed','health_label')}
            policy.set_training_mode(False)
            for _ in range(rollout):
                tensor,_=policy.obs_to_tensor(observation)
                controller_left=torch.where(torch.as_tensor(starts),tensor['position'][:,4]>=0,controller_left)
                elapsed=torch.where(torch.as_tensor(starts),torch.zeros_like(elapsed),elapsed)
                with torch.no_grad():
                    features=policy.extract_features(tensor); flip=tensor['position'][:,4]<0
                    distribution,value,next_hidden=policy.sequence_step(features,hidden,previous,torch.as_tensor(starts),flip,remaining,
                        policy.read_health(tensor),tensor['position'],controller_left,elapsed)
                    option_start=remaining==0; proposed=distribution.sample(); action=torch.where(option_start,proposed,committed)
                    duration_distribution=policy.duration_distribution(next_hidden,action,flip)
                    duration_index=duration_distribution.sample() if args.learn_durations else torch.zeros(nenv,dtype=torch.long)
                    selected_duration=torch.as_tensor(policy.duration_choices)[duration_index]
                    next_remaining=torch.where(option_start,selected_duration-1,remaining-1)
                    log_prob=distribution.log_prob(action)+duration_distribution.log_prob(duration_index)
                    option_decisions+=int(option_start.sum())
                    duration_counts.update(selected_duration[option_start].tolist())
                next_observation,reward,done,infos=env.step(action.numpy())
                data=dict(image=observation['image'],position=observation['position'],hidden=hidden.numpy(),previous=previous.numpy(),starts=starts.copy(),
                          remaining=remaining.numpy(),option_start=option_start.numpy(),duration=duration_index.numpy(),
                          controller_left=controller_left.numpy(),elapsed=elapsed.numpy(),health_label=np.asarray([info['pre_action_absolute_health'] for info in infos],np.float32),
                          action=action.numpy(),log_prob=log_prob.numpy(),value=value.numpy(),reward=reward,done=done)
                for key,value_array in data.items(): buffer[key].append(value_array.copy())
                for info in infos:
                    name=definition['actions'][info['action']]['name']; inputs[name]+=1
                    if 'scene' in info: scenes[f"P{info['player']}|{info['opponent']}|{info['scene']}"]+=1
                    if 'episode' in info:
                        rounds[('CPU' if info['opponent_kind']=='cpu' else 'P'+str(info['player']))+'_'+info['result']]+=1
                        conditions[f"P{info['player']}|{info['opponent']}|{info['offset']}"]+=1
                hidden=next_hidden.detach(); previous=policy.canonical_actions(action,flip).detach(); starts=done.copy()
                remaining=torch.where(torch.as_tensor(done),torch.zeros_like(next_remaining),next_remaining).detach(); committed=action.detach()
                elapsed=elapsed+1
                observation=next_observation; steps+=nenv
            data={key:np.asarray(value) for key,value in buffer.items()}
            with torch.no_grad():
                tensor,_=policy.obs_to_tensor(observation); features=policy.extract_features(tensor)
                bootstrap_left=torch.where(torch.as_tensor(starts),tensor['position'][:,4]>=0,controller_left)
                bootstrap_elapsed=torch.where(torch.as_tensor(starts),torch.zeros_like(elapsed),elapsed)
                _,bootstrap,_=policy.sequence_step(features,hidden,previous,torch.as_tensor(starts),tensor['position'][:,4]<0,remaining,
                    policy.read_health(tensor),tensor['position'],bootstrap_left,bootstrap_elapsed)
            advantages=np.zeros_like(data['reward']); gae=np.zeros(nenv,np.float32)
            for t in range(rollout-1,-1,-1):
                next_value=bootstrap.numpy() if t==rollout-1 else data['value'][t+1]
                live=1-data['done'][t].astype(np.float32)
                delta=data['reward'][t]+live*next_value-data['value'][t]
                gae=delta+args.gae_lambda*live*gae; advantages[t]=gae
            data['advantage']=advantages; data['returns']=advantages+data['value']
            tensors={key:torch.as_tensor(value) for key,value in data.items()}
            sequence_ids=[(e,s) for e in range(nenv) for s in range(0,rollout,length)]
            metrics=[]; early=False; policy.set_training_mode(True)
            for epoch in range(4):
                rng.shuffle(sequence_ids)
                for begin in range(0,len(sequence_ids),batch_sequences):
                    chosen=sequence_ids[begin:begin+batch_sequences]; batch=len(chosen)
                    def sequence(key): return torch.stack([tensors[key][s:s+length,e] for e,s in chosen],1)
                    images=sequence('image'); positions=sequence('position')
                    features=policy.extract_features(dict(image=images.flatten(0,1),position=positions.flatten(0,1))).reshape(length,batch,-1)
                    health_predictions=policy.read_health(dict(image=images.flatten(0,1)))
                    if health_predictions is not None:health_predictions=health_predictions.reshape(length,batch,2)
                    states=torch.stack([tensors['hidden'][s,e] for e,s in chosen])
                    actions=sequence('action'); starts_sequence=sequence('starts'); previous_sequence=sequence('previous')
                    old_log=sequence('log_prob'); adv=sequence('advantage'); returns=sequence('returns')
                    option_mask=sequence('option_start'); selected_advantage=adv[option_mask]
                    selected_advantage=(selected_advantage-selected_advantage.mean())/(selected_advantage.std()+1e-8)
                    remaining_sequence=sequence('remaining'); duration_sequence=sequence('duration')
                    left_sequence=sequence('controller_left');elapsed_sequence=sequence('elapsed')
                    logs=[]; values=[]; entropies=[]; family_entropies=[]
                    for offset in range(length):
                        flip=positions[offset,:,4]<0
                        distribution,value,states=policy.sequence_step(features[offset],states,previous_sequence[offset],starts_sequence[offset],flip,remaining_sequence[offset],
                            None if health_predictions is None else health_predictions[offset],positions[offset],left_sequence[offset],elapsed_sequence[offset])
                        duration_distribution=policy.duration_distribution(states,actions[offset],flip)
                        logs.append(distribution.log_prob(actions[offset])+duration_distribution.log_prob(duration_sequence[offset])); values.append(value)
                        entropies.append(policy.joint_entropy(distribution,states,flip)); family_entropies.append(policy.family_entropy(distribution))
                    new_log=torch.stack(logs); log_change=(new_log-old_log)[option_mask]; ratio=log_change.exp()
                    actor_loss=-torch.minimum(selected_advantage*ratio,selected_advantage*ratio.clamp(.9,1.1)).mean()
                    value_loss=(returns-torch.stack(values)).square().mean()
                    entropy=torch.stack(entropies)[option_mask].mean(); family_entropy=torch.stack(family_entropies)[option_mask].mean()
                    health_loss=actor_loss.new_zeros(())
                    if health_predictions is not None:health_loss=(health_predictions-sequence('health_label')).square().mean()
                    loss=actor_loss+.5*value_loss-entropy_coefficient*entropy-family_entropy_coefficient*family_entropy+10.*health_loss
                    with torch.no_grad(): kl=float(((ratio-1)-log_change).mean())
                    if kl>.04: early=True; break
                    policy.optimizer.zero_grad(); loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
                    if not bool(torch.isfinite(gradient)): raise RuntimeError('Nonfinite full-network recurrent gradient')
                    policy.optimizer.step()
                    metrics.append(dict(actor_loss=float(actor_loss.detach()),value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),family_entropy=float(family_entropy.detach()),kl=kl,
                                        health_mse=float(health_loss.detach())))
                if early: break
            if len(history)==0:
                changes={name:float((parameter-initial_parameters[name]).abs().max()) for name,parameter in policy.named_parameters()}
                required=('features_extractor.image_net.0.0.weight','features_extractor.position_net.0.weight','action_net.weight','value_net.weight','history_gru.weight_ih','history_action.weight','history_value.weight')
                if any(changes[name]<=0 for name in required): raise RuntimeError('A required trainable component did not update')
                if args.learn_durations and changes['history_duration.weight']<=0: raise RuntimeError('The duration choice did not learn')
                write(out/'first_update_contract.json',dict(max_parameter_change=changes,all_inference_parameters_trainable=True))
            row=dict(steps=steps,seconds=time.monotonic()-started,native_frames=sum(env.get_attr('native_frames')),completed_rounds=dict(rounds),
                     temperature=policy.action_temperature,entropy_coefficient=entropy_coefficient,family_entropy_coefficient=family_entropy_coefficient,
                     option_decisions=option_decisions,duration_choices=dict(duration_counts),
                     optimizer={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]} if metrics else {},early_kl_stop=early,
                     minimum_pvp_condition_rounds=min(conditions[f"P{port}|{r['id']}|{offset}"] for port in (1,2) for r in pool for offset in (0,12)))
            history.append(row); write(out/'live.json',row); write(out/'condition_exposure.json',dict(completed_rounds=dict(conditions)))
            print('DIRECT_SEQUENCE_PPO',json.dumps(row),flush=True)
            if steps-last_save>=args.checkpoint_every or steps>=args.steps:
                definition['action_temperature']=policy.action_temperature
                save(out/f'checkpoint_{steps}.pt',policy,definition,dict(steps=steps,seed=args.seed,condition_counts=dict(conditions)),policy.optimizer)
                last_save=steps
        if last_save!=steps:
            definition['action_temperature']=policy.action_temperature
            save(out/f'checkpoint_{steps}.pt',policy,definition,dict(steps=steps,seed=args.seed,condition_counts=dict(conditions)),policy.optimizer)
        write(out/'scene_coverage.json',dict(decisions=dict(scenes),input_counts=dict(inputs)))
        write(out/'result.json',dict(status='TRAINED_REQUIRES_COMPLETE_GAMES',arguments=vars(args),steps=steps,seconds=time.monotonic()-started,
                                     native_frames=sum(env.get_attr('native_frames')),completed_rounds=dict(rounds),history=history))
    finally: env.close()


if __name__=='__main__': main()
