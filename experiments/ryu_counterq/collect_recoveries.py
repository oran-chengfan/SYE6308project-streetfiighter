"""Native counterfactual prefixes followed by an achievable learned continuation.

The whole prefix and every continuation transition are retained. A winning
return is an observed achievable outcome, not an oracle label for an isolated
first input. Defeats remain in the transition data.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch

from experiments.league_adaptive.runtime import Side,compiled,load_actor,setup,write
from experiments.league_adaptive.environment import FightEnv
from experiments.ryu_counterq.policy import make_extras


def main():
    p=argparse.ArgumentParser();p.add_argument('--student',required=True);p.add_argument('--out',required=True)
    p.add_argument('--teacher',default='models/counter_ryu_final/agent.pt')
    p.add_argument('--pool',default='artifacts/ryu_adaptive_curriculum_20261021/main_pool.json')
    p.add_argument('--roots',default='0,16,48');p.add_argument('--limit-conditions',type=int,default=0)
    p.add_argument('--include-frames',action='store_true')
    args=p.parse_args();setup();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    student=load_actor(args.student);teacher=load_actor(args.teacher);definition=student.definition
    if teacher.definition['kind'] not in ('direct_ppo','neural_mixture') or getattr(teacher.policy,'inference_hold_steps',0):
        raise ValueError('Use a stateless learned continuation or explicitly reconstruct its forced-action memory')
    pool=[r for r in json.loads(Path(args.pool).read_text()) if r['id'] in ('stress_heldout_fireball_timing','current_strong_response','curriculum_bait_counter')]
    env=FightEnv(definition,pool);names={a['name']:i for i,a in enumerate(definition['actions'])}
    commands={tuple(tuple(sorted(keys)) for keys in c):i for i,c in enumerate(compiled(definition['actions']))}
    teacher_map={i:commands[tuple(tuple(sorted(keys)) for keys in c)] for i,c in enumerate(compiled(teacher.definition['actions']))}
    node={key:[] for key in ('maps','positions','extras','previous','parent')}
    if args.include_frames:node['frame']=[]
    transition={key:[] for key in ('start','end','action','reward','done','duration','returns','win','group')}
    reports=[];roots=[int(v) for v in args.roots.split(',')];conditions=0

    def add(observation,role,elapsed,previous,parent):
        tensor,_=student.policy.obs_to_tensor(observation)
        with torch.no_grad():
            feature=student.policy.encode_maps(tensor).numpy()[0]
            extra=make_extras(tensor['position'],student.policy.health_reader(tensor['image']),torch.tensor([role]),torch.tensor([elapsed]),torch.tensor([0])).numpy()[0]
        number=len(node['maps'])
        for key,value in dict(maps=feature,positions=observation['position'],extras=extra,previous=previous,parent=parent).items():node[key].append(np.array(value,copy=True))
        if args.include_frames:node['frame'].append(observation['image'][-1].copy())
        return number

    def chosen(side):return teacher_map[int(teacher.predict(side.observe(env.game.raw),True)[0])]

    def tick(observation,current,action,role,elapsed,side):
        canonical=int(student.policy.physical_mirror[action]) if observation['position'][4]<0 and student.policy.canonical else action
        nxt,_,t,tr,info=env.step(action);side.update(env.game.rgb);elapsed+=1
        end=add(nxt,role,elapsed,canonical,current)
        reward=.25*(info['damage_dealt']-info['damage_taken'])/176.+float(info['result']=='win')-float(info['result']=='loss')
        return nxt,end,elapsed,dict(start=current,end=end,action=canonical*4,reward=reward,done=bool(t or tr),duration=1),info

    def prefix_action(mode):
        raw=env.game.raw;p=env.player;forward='RIGHT' if raw[f'p{3-p}_x']>=raw[f'p{p}_x'] else 'LEFT';back='LEFT' if forward=='RIGHT' else 'RIGHT'
        return names[forward if mode=='toward' else back if mode=='away' else 'UP_'+forward if mode=='jump' else 'DOWN_'+back if mode=='guard' else 'C_tap' if mode=='kick' else 'CROUCH_KICK_C' if mode=='sweep' else f'HADOUKEN_{"R" if forward=="RIGHT" else "L"}_Y' if mode=='fireball' else 'NO_OP']

    plans=[('baseline',[]),('walk4',[('toward',4)]),('walk8',[('toward',8)]),('retreat8',[('away',8)]),('retreat16',[('away',16)]),
           ('jump_strike',[('jump',3),('kick',1)]),('walk_sweep',[('toward',4),('sweep',1)]),
           ('guard_fireball',[('guard',8),('fireball',1)]),('wait8',[('neutral',8)])]
    try:
        for index,rival in enumerate(pool):
            for player in (1,2):
                for offset in (0,12):
                    if args.limit_conditions and conditions>=args.limit_conditions:break
                    conditions+=1;observation,_=env.reset(options=dict(opponent=index,player=player,offset=offset))
                    teacher.reset();side=Side(teacher.definition,player);side.reset(env.game.rgb)
                    role=bool(observation['position'][4]>=0);elapsed=0;current=add(observation,role,elapsed,0,-1);factual=[];saved=[]
                    for decision in range(max(roots)+1):
                        if decision in roots:saved.append(dict(game=env.save(),frames=[f.copy() for f in side.frames],elapsed=elapsed,node=current,prefix=list(factual),decision=decision))
                        if decision==max(roots):break
                        observation,current,elapsed,row,info=tick(observation,current,chosen(side),role,elapsed,side);factual.append(row)
                        if row['done']:break
                    for root in saved:
                        root_reports=[]
                        for name,plan in plans:
                            observation=env.restore(root['game']);side.frames=deque([f.copy() for f in root['frames']],maxlen=4)
                            current=root['node'];elapsed=root['elapsed'];episode=list(root['prefix'])
                            for mode,repetitions in plan:
                                for _ in range(repetitions):
                                    observation,current,elapsed,row,info=tick(observation,current,prefix_action(mode),role,elapsed,side);episode.append(row)
                                    if env.ended:break
                                if env.ended:break
                            while not env.ended:
                                observation,current,elapsed,row,info=tick(observation,current,chosen(side),role,elapsed,side);episode.append(row)
                            result={k:info[k] for k in ('result','self_hp','opponent_hp','native_frames','end_reason')}
                            group=f"{rival['id']}|P{player}|{offset}";future=0.
                            for row in reversed(episode):
                                future=row['reward']+future;item=dict(row,returns=future,win=info['result']=='win',group=group)
                                for key in transition:transition[key].append(item[key])
                            root_reports.append(dict(plan=name,**result))
                        # One exact unforced baseline replay, with restored opponent and observation histories.
                        observation=env.restore(root['game']);side.frames=deque([f.copy() for f in root['frames']],maxlen=4)
                        while not env.ended:
                            observation,_,_,_,info=env.step(chosen(side));side.update(env.game.rgb)
                        replay={k:info[k] for k in ('result','self_hp','opponent_hp','native_frames','end_reason')}
                        if replay!={k:root_reports[0][k] for k in replay}:raise RuntimeError('Native recovery baseline did not reproduce')
                        report=dict(opponent=rival['id'],player=player,offset=offset,root=root['decision'],branches=root_reports,baseline_reproduced=True)
                        reports.append(report);write(out/'progress.json',dict(roots=reports,nodes=len(node['maps']),transitions=len(transition['start']),native_frames=env.native_frames))
                        print('NATIVE_RECOVERY',rival['id'],player,offset,root['decision'],[(v['plan'],v['result']) for v in root_reports],flush=True)
        arrays={key:np.asarray(value) for key,value in node.items()};arrays.update({'transition_'+key:np.asarray(value) for key,value in transition.items()})
        arrays['maps']=arrays['maps'].astype(np.float16);arrays['positions']=arrays['positions'].astype(np.float32);arrays['extras']=arrays['extras'].astype(np.float32)
        np.savez_compressed(out/'data.npz',**arrays)
        write(out/'summary.json',dict(arguments=vars(args),roots=reports,nodes=len(node['maps']),transitions=len(transition['start']),native_frames=env.native_frames,
             scope='Training-only native complete prefixes and learned continuations. Every baseline replayed exactly. Both successes and defeats retained; branch outcomes are achievable conditional returns, not a global oracle.'))
    finally:env.close();teacher.close();student.close()


if __name__=='__main__':main()
