"""Compact root histories and paired complete option/continuation outcomes."""
import argparse,copy,json
from collections import deque
from pathlib import Path
import numpy as np
import torch
from experiments.league_adaptive.runtime import load_actor,Side,compiled,setup,write
from experiments.league_adaptive.environment import FightEnv
from experiments.ryu_counterq.policy import make_extras
from experiments.ryu_option_planner.options import PLANS,plan_input


def main():
    p=argparse.ArgumentParser();p.add_argument('--sensor',required=True);p.add_argument('--out',required=True)
    p.add_argument('--teacher',default='models/counter_ryu_final/agent.pt');p.add_argument('--pool',default='artifacts/ryu_adaptive_curriculum_20261021/main_pool.json')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--roots',default='16,48,96')
    a=p.parse_args();setup();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    sensor=load_actor(a.sensor);teacher=load_actor(a.teacher);definition=sensor.definition
    if teacher.definition['kind'] not in ('direct_ppo','neural_mixture') or getattr(teacher.policy,'inference_hold_steps',0):raise ValueError('Learned continuation must be stateless')
    pool=json.loads(Path(a.pool).read_text())+[dict(id=f'CPU_Ryu_{s}',kind='cpu',character='Ryu',stars=s) for s in range(1,9)]
    env=FightEnv(definition,pool);names={v['name']:i for i,v in enumerate(definition['actions'])}
    commands={tuple(tuple(sorted(keys)) for keys in c):i for i,c in enumerate(compiled(definition['actions']))}
    mapping={i:commands[tuple(tuple(sorted(keys)) for keys in c)] for i,c in enumerate(compiled(teacher.definition['actions']))}
    requested=[int(v) for v in a.roots.split(',')];conditions=[(i,p,o) for i,r in enumerate(pool) for p in ((1,) if r['kind']=='cpu' else (1,2)) for o in ((0,) if r['kind']=='cpu' else (0,12))]
    reports=[];cache={}

    def choose(side):
        observation=side.observe(env.game.raw);key=observation['image'].tobytes()+observation['position'].tobytes()
        if key not in cache:
            if len(cache)>=3000:cache.pop(next(iter(cache)))
            cache[key]=mapping[int(teacher.predict(observation,True)[0])]
        return cache[key]

    def tick(action,side):
        observation,_,t,tr,info=env.step(int(action));side.update(env.game.rgb)
        return observation,info

    def encoded(observation,role,elapsed,previous):
        tensor,_=sensor.policy.obs_to_tensor(observation)
        with torch.no_grad():
            feature=sensor.policy.encode_maps(tensor).numpy()[0]
            extra=make_extras(tensor['position'],sensor.policy.health_reader(tensor['image']),torch.tensor([role]),torch.tensor([elapsed]),torch.tensor([0])).numpy()[0]
        return dict(map=feature.copy(),position=observation['position'].copy(),extras=extra.copy(),previous=previous)

    def result(info):return {k:info[k] for k in ('result','self_hp','opponent_hp','native_frames','end_reason')}

    try:
        for condition,(index,player,offset) in enumerate(conditions):
            if condition%a.shards!=a.shard:continue
            cache.clear();observation,_=env.reset(options=dict(opponent=index,player=player,offset=offset));teacher.reset()
            side=Side(teacher.definition,player);side.reset(env.game.rgb);role=bool(observation['position'][4]>=0);previous=0;history=[];roots=[]
            for decision in range(max(requested)+1):
                history.append(encoded(observation,role,decision,previous))
                if decision in requested:roots.append(dict(saved=env.save(),frames=[f.copy() for f in side.frames],history=copy.deepcopy(history[-96:]),decision=decision))
                if decision==max(requested):break
                action=choose(side);previous=int(sensor.policy.physical_mirror[action]) if observation['position'][4]<0 and sensor.policy.canonical else action
                observation,info=tick(action,side)
                if env.ended:break
            arrays={key:[] for key in ('maps','positions','extras','previous','lengths','utility','margin','opponent','player','offset','root_decision')}
            for root in roots:
                branches=[]
                for label,plan in PLANS:
                    observation=env.restore(root['saved']);side.frames=deque([f.copy() for f in root['frames']],maxlen=4)
                    for mode,count in plan:
                        for _ in range(count):
                            observation,info=tick(plan_input(mode,float(observation['position'][4]),names),side)
                            if env.ended:break
                        if env.ended:break
                    while not env.ended:observation,info=tick(choose(side),side)
                    branches.append(dict(plan=label,**result(info)))
                observation=env.restore(root['saved']);side.frames=deque([f.copy() for f in root['frames']],maxlen=4)
                while not env.ended:observation,info=tick(choose(side),side)
                if result(info)!={k:branches[0][k] for k in result(info)}:raise RuntimeError('Paired complete baseline did not reproduce')
                history=root['history'];n=len(history);maps=np.zeros((96,3584),np.float16);positions=np.zeros((96,6),np.float32);extras=np.zeros((96,12),np.float32);previous=np.zeros(96,np.int64)
                for t,row in enumerate(history):maps[t]=row['map'];positions[t]=row['position'];extras[t]=row['extras'];previous[t]=row['previous']
                row=dict(maps=maps,positions=positions,extras=extras,previous=previous,lengths=n,
                         utility=np.asarray([1. if r['result']=='win' else -1. for r in branches],np.float32),
                         margin=np.asarray([(max(0,r['self_hp'])-max(0,r['opponent_hp']))/176. for r in branches],np.float32),
                         opponent=pool[index]['id'],player=player,offset=offset,root_decision=root['decision'])
                for key in arrays:arrays[key].append(row[key])
                reports.append(dict(opponent=pool[index]['id'],player=player,offset=offset,root=root['decision'],branches=branches,baseline_reproduced=True))
                print('OPTION_ROOT',pool[index]['id'],player,offset,root['decision'],[(b['plan'],b['result']) for b in branches],flush=True)
            if roots:
                temporary=out/f'writing_condition_{condition:03d}.npz'
                np.savez_compressed(temporary,**{k:np.asarray(v) for k,v in arrays.items()});temporary.replace(out/f'condition_{condition:03d}.npz')
            write(out/'progress.json',dict(shard=a.shard,roots=reports,native_frames=env.native_frames,teacher=a.teacher,sensor=a.sensor))
        write(out/'summary.json',dict(arguments=vars(a),roots=reports,native_frames=env.native_frames,
             scope='Compact legal histories at factual learned-policy states. Every motor option followed by the same learned continuation through KO/time; baseline replayed exactly. No opponent identity is intended for actor input.'))
    finally:env.close();teacher.close();sensor.close()


if __name__=='__main__':main()
