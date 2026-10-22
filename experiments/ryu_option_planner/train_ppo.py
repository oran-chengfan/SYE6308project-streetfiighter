"""On-policy semi-Markov PPO over complete plans and a retained PPO controller.

Every recorded choice completes before the collecting policy is updated.
Workers may pause their independent simulator while other options finish;
both fighters pause together and paused workers contribute no frames/data.
"""
import argparse,copy,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.league_adaptive.runtime import setup,write
from experiments.ryu_direct_league.policy import save
from experiments.ryu_counterq.policy import make_extras
from experiments.ryu_counterq.replay import Replay
from experiments.ryu_option_planner.policy import construct_planner
from experiments.ryu_option_planner.options import PLANS,plan_input
from experiments.ryu_option_planner.environment import Worker
from experiments.ryu_option_planner.train import load_roots,context


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--pool',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=196608);p.add_argument('--seed',type=int,default=4311);p.add_argument('--max-seconds',type=float,default=7200)
    p.add_argument('--checkpoint-every',type=int,default=12288);p.add_argument('--cf-data');p.add_argument('--cf-weight',type=float,default=0.)
    a=p.parse_args();setup();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);cf_rng=np.random.default_rng(a.seed+1801)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=torch.load(a.initial,map_location='cpu',weights_only=False);definition=copy.deepcopy(source['definition']);definition.update(planner_control='ppo',planner_semantics=True)
    policy=construct_planner(definition);state=policy.state_dict()
    for key,value in source['state_dict'].items():
        if key.startswith('planner_heads.') and key.endswith('.0.weight') and value.shape!=state[key].shape:
            state[key].zero_();state[key][:,:value.shape[1]]=value
        elif key in state and value.shape==state[key].shape:state[key]=value.clone()
    policy.load_state_dict(state,strict=True)
    if source['definition'].get('planner_control')=='ppo' and 'optimizer_state' in source:policy.optimizer.load_state_dict(source['optimizer_state'])
    for group in policy.optimizer.param_groups:group['lr']=3e-4
    pool=json.loads(Path(a.pool).read_text());cpu=[dict(id=f'CPU_Ryu_{s}',kind='cpu',character='Ryu',stars=s) for s in range(1,9)]
    env=SubprocVecEnv([Worker(definition,pool,0,a.seed),Worker(definition,pool,1,a.seed+317),Worker(definition,cpu,2,a.seed+634)],start_method='spawn');env.seed(a.seed)
    replay=Replay(nodes=60000,context=definition['context_length'],offsets=definition['context_offsets'],seed=a.seed)
    observation=env.reset();role=observation['position'][:,4]>=0;elapsed=np.zeros(3,np.int64);previous=np.zeros(3,np.int64)
    queues=[[] for _ in range(3)];records=[None]*3;current=[];steps=0;last_save=0;rounds=Counter();conditions=Counter();choices=Counter();history=[];paused_calls=0
    names={v['name']:i for i,v in enumerate(definition['actions'])};cf_data=None;cf_files=0;cf_groups=None

    def encoded(observation,roles,times):
        tensor,_=policy.obs_to_tensor(observation)
        with torch.no_grad():
            maps=policy.encode_maps(tensor).numpy();extra=make_extras(tensor['position'],policy.health_reader(tensor['image']),torch.as_tensor(roles),torch.as_tensor(times),torch.zeros(len(roles))).numpy()
        return maps,extra

    maps,extra=encoded(observation,role,elapsed)
    for i in range(3):current.append(replay.add_node(-1,maps[i],observation['position'][i],extra[i],0))
    write(out/'config.json',dict(arguments=vars(a),pool=pool,cpu=cpu,gamma=1.,gae_lambda=1.,epochs=4,batch_size=64,rollout_actual_calls=1536,
        reward='Victory+1; every other complete outcome-1;0.25HP potential with terminal0. Full-start objective2*P(win)-1.',
        algorithm='Semi-Markov PPO over9 complete motor plans. Actor likelihood and critic at free option boundaries only; native rewards sum over every actual six-frame call.',
        retention='Established learned PPO controller is frozen and available as option0. Initial greedy actor chooses it exactly. New plan selection is trained by actual on-policy whole-game rewards.',
        history='Recompute a23-token legal history spanning96 calls for every sampled boundary; no stored stale GRU hidden states.',
        auxiliary='Optional paired native option-outcome prediction is trained jointly and read by the actor. It never overrides the PPO action. Conditional old-controller outcomes are not a global oracle.',
        pausing='Independent simulators pause both fighters while other workers finish options before an update; no synthetic game waiting, no paused frames or histories counted.'))
    save(out/'initial.pt',policy,definition,dict(stage='retained_PPO_high_level_initialization'))
    try:
        while steps<a.steps and time.monotonic()-started<a.max_seconds and not (out/'STOP').exists():
            if a.cf_data and a.cf_weight>0:
                count=len(list(Path(a.cf_data).glob('shard_*/condition_*.npz')))
                if cf_data is None or (count>cf_files and (not history or steps-history[-1].get('last_cf_reload',0)>=6144)):
                    cf_data,_=load_roots(a.cf_data);cf_files=count;last_cf_reload=steps
                    cf_groups={str(g):np.flatnonzero(cf_data['opponent']==g) for g in np.unique(cf_data['opponent'])}
            else:last_cf_reload=0
            policy.set_training_mode(False);rollout=[[] for _ in range(3)];begin_steps=steps
            while steps-begin_steps<1536 or any(queues):
                collecting=steps-begin_steps<1536;active=np.asarray([collecting or bool(q) for q in queues]);free=np.asarray([active[i] and not queues[i] for i in range(3)])
                if not active.any():break
                free_ids=np.flatnonzero(free);actions=np.full(3,-1,np.int64)
                if len(free_ids):
                    with torch.no_grad():distribution,values=policy.decisions(*replay.contexts([current[i] for i in free_ids]));selected=distribution.sample();logs=distribution.log_prob(selected)
                    baseline=[]
                    for j,i in enumerate(free_ids):
                        option=int(selected[j]);choices[PLANS[option][0]]+=1;queues[i]=[mode for mode,n in PLANS[option][1] for _ in range(n)]
                        records[i]=dict(start=current[i],option=option,log_prob=float(logs[j]),value=float(values[j]),reward=0.,duration=0)
                        if option==0:baseline.append(int(i))
                    if baseline:
                        policy.incumbent.set_training_mode(False)
                        with torch.no_grad():selected_baseline=policy.incumbent.predict({key:value[baseline] for key,value in observation.items()},deterministic=True)[0]
                        for i,action in zip(baseline,np.asarray(selected_baseline).reshape(-1)):actions[i]=int(action)
                for i in np.flatnonzero(active):
                    if queues[i]:actions[i]=plan_input(queues[i].pop(0),float(observation['position'][i,4]),names)
                    if actions[i]<0:raise RuntimeError('An active option did not supply a physical input')
                before_flip=observation['position'][:,4]<0
                next_observation,rewards,dones,infos=env.step(actions);paused_calls+=int((~active).sum())
                next_elapsed=np.where(active,elapsed+1,elapsed);next_role=np.where(dones,next_observation['position'][:,4]>=0,role)
                next_elapsed=np.where(dones,0,next_elapsed);next_maps,next_extra=encoded(next_observation,next_role,next_elapsed)
                for i in np.flatnonzero(active):
                    info=infos[i]
                    if info['paused']:raise RuntimeError('An active game was paused')
                    previous[i]=int(policy.physical_mirror[actions[i]]) if before_flip[i] and policy.canonical else int(actions[i])
                    record=records[i];record['reward']+=float(rewards[i]);record['duration']+=1
                    if dones[i]:
                        terminal={key:info['terminal_observation'][key][None] for key in ('image','position')};terminal_map,terminal_extra=encoded(terminal,np.asarray([role[i]]),np.asarray([elapsed[i]+1]))
                        end=replay.add_node(current[i],terminal_map[0],terminal['position'][0],terminal_extra[0],previous[i])
                    else:end=replay.add_node(current[i],next_maps[i],next_observation['position'][i],next_extra[i],previous[i])
                    if dones[i] or not queues[i]:
                        record.update(end=end,done=bool(dones[i]));rollout[i].append(record);records[i]=None
                    if dones[i]:
                        queues[i]=[];rounds[('CPU' if info['opponent_kind']=='cpu' else 'P'+str(info['player']))+'_'+info['result']]+=1
                        conditions[f"{info['opponent']}|P{info['player']}|{info['offset']}"]+=1;current[i]=replay.add_node(-1,next_maps[i],next_observation['position'][i],next_extra[i],0);previous[i]=0
                    else:current[i]=end
                observation=next_observation;elapsed=next_elapsed;role=next_role;steps+=int(active.sum())
            if any(record is not None for record in records):raise RuntimeError('A collecting option crossed a policy update')
            with torch.no_grad():_,bootstrap=policy.decisions(*replay.contexts(current))
            items=[]
            for i,rows in enumerate(rollout):
                next_value=float(bootstrap[i]);advantage=0.
                for row in reversed(rows):
                    live=float(not row['done']);delta=row['reward']+live*next_value-row['value'];advantage=delta+live*advantage
                    row['advantage']=advantage;row['returns']=advantage+row['value'];next_value=row['value']
                items.extend(rows)
            advantages=np.asarray([row['advantage'] for row in items]);advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            for row,value in zip(items,advantages):row['normalized_advantage']=value
            metrics=[];policy.set_training_mode(True);early=False
            for epoch in range(4):
                order=rng.permutation(len(items))
                for begin in range(0,len(order),64):
                    batch=[items[i] for i in order[begin:begin+64]];distribution,values=policy.decisions(*replay.contexts([row['start'] for row in batch]))
                    option=torch.as_tensor([row['option'] for row in batch]);old_log=torch.as_tensor([row['log_prob'] for row in batch]);adv=torch.as_tensor([row['normalized_advantage'] for row in batch],dtype=torch.float32)
                    returns=torch.as_tensor([row['returns'] for row in batch],dtype=torch.float32);log_change=distribution.log_prob(option)-old_log;ratio=log_change.exp()
                    actor=-torch.minimum(adv*ratio,adv*ratio.clamp(.9,1.1)).mean();critic=(returns-values).square().mean();entropy=distribution.entropy().mean()
                    auxiliary=actor.new_zeros(())
                    if cf_data is not None:
                        groups=list(cf_groups);ids=np.asarray([int(cf_rng.choice(cf_groups[groups[int(cf_rng.integers(len(groups)))]])) for _ in range(16)])
                        pred=policy.option_values(*context(cf_data,ids,definition['context_offsets']));utility=cf_data['utility'][ids]
                        desired=torch.as_tensor(np.concatenate((utility[:,:1],utility[:,1:]-utility[:,:1]),1),dtype=torch.float32)
                        auxiliary=(pred-desired[:,None,:]).square().mean()
                    loss=actor+.5*critic-.003*entropy+a.cf_weight*auxiliary
                    with torch.no_grad():kl=float(((ratio-1)-log_change).mean())
                    if kl>.04:early=True;break
                    policy.optimizer.zero_grad();loss.backward();gradient=torch.nn.utils.clip_grad_norm_([v for v in policy.parameters() if v.requires_grad],.5)
                    if not bool(torch.isfinite(gradient)):raise RuntimeError('Nonfinite high-level PPO gradient')
                    policy.optimizer.step();metrics.append(dict(actor=float(actor.detach()),critic=float(critic.detach()),entropy=float(entropy.detach()),kl=kl,auxiliary=float(auxiliary.detach())))
                if early:break
            row=dict(steps=steps,seconds=time.monotonic()-started,native_frames=sum(env.get_attr('native_frames')),completed_rounds=dict(rounds),high_level_options=dict(choices),
                     options_this_update=len(items),paused_worker_calls=paused_calls,last_cf_reload=last_cf_reload,cf_roots=0 if cf_data is None else len(cf_data['lengths']),
                     optimizer={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]} if metrics else {},early_kl_stop=early,
                     minimum_pvp_condition_rounds=min(conditions[f"{r['id']}|P{port}|{offset}"] for r in pool for port in (1,2) for offset in (0,12)))
            history.append(row);write(out/'live.json',row);write(out/'condition_exposure.json',dict(completed_rounds=dict(conditions)));print('HIGH_LEVEL_PPO',json.dumps(row),flush=True)
            if steps-last_save>=a.checkpoint_every or steps>=a.steps:
                save(out/f'checkpoint_{steps}.pt',policy,definition,dict(algorithm='on_policy_complete_option_PPO',steps=steps,cf_weight=a.cf_weight),policy.optimizer);last_save=steps
        if last_save!=steps:save(out/f'checkpoint_{steps}.pt',policy,definition,dict(algorithm='on_policy_complete_option_PPO',steps=steps,cf_weight=a.cf_weight),policy.optimizer)
        write(out/'result.json',dict(status='TRAINED_REQUIRES_COMPLETE_MATCHES',arguments=vars(a),steps=steps,history=history,native_frames=sum(env.get_attr('native_frames'))))
    finally:env.close()


if __name__=='__main__':main()
