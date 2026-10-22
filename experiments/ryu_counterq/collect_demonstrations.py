"""Actual full native Ryu games from retained learned teachers.

All transitions, including defeats, are retained. Teacher identities are data
provenance only; the deployed student receives legal pixels and coordinates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.league_adaptive.runtime import ROOT,Side,compiled,load_actor,setup,write
from experiments.league_adaptive.environment import FightEnv
from experiments.ryu_counterq.policy import make_extras


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--student',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--pool',default='artifacts/ryu_adaptive_curriculum_20261021/main_pool.json')
    parser.add_argument('--all-training',action='store_true');parser.add_argument('--teacher',choices=('all','retained','memory_counter','duration_zone'),default='all')
    parser.add_argument('--include-frames',action='store_true')
    args=parser.parse_args();setup()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);student=load_actor(args.student);definition=student.definition
    pool=json.loads(Path(args.pool).read_text());targets=[r for r in pool if r['id'] in ('current_strong_response','stress_heldout_fireball_timing','curriculum_zone_guard')]
    cpu=[dict(id=f'CPU_Ryu_{s}',kind='cpu',character='Ryu',stars=s) for s in range(1,9)]
    teachers=[('retained',ROOT/'models/counter_ryu_final/agent.pt',cpu+targets),
              ('memory_counter',ROOT/'models/counter_ryu_memory_candidate/agent.pt',[r for r in targets if r['id']=='current_strong_response']),
              ('duration_zone',ROOT/'artifacts/ryu_movement_20261022/learned_duration/checkpoint_24576.pt',[r for r in targets if r['id']=='curriculum_zone_guard'])]
    if args.all_training:teachers=[(name,path,cpu+pool) for name,path,_ in teachers]
    if args.teacher!='all':teachers=[row for row in teachers if row[0]==args.teacher]
    node={key:[] for key in ('maps','positions','extras','previous','parent')};transitions={key:[] for key in ('start','end','action','reward','done','duration','returns','win','group')}
    if args.include_frames:node['frame']=[]
    reports=[];native_frames=0
    student_commands=compiled(definition['actions'])
    commands={tuple(tuple(sorted(keys)) for keys in command):i for i,command in enumerate(student_commands)}

    def add_node(observation,role_left,elapsed,previous,parent):
        tensor,_=student.policy.obs_to_tensor(observation)
        with torch.no_grad():
            maps=student.policy.encode_maps(tensor).numpy()[0]
            health=student.policy.health_reader(tensor['image'])
            extra=make_extras(tensor['position'],health,torch.tensor([role_left]),torch.tensor([elapsed]),torch.tensor([0])).numpy()[0]
        number=len(node['maps']);node['maps'].append(maps.copy());node['positions'].append(observation['position'].copy())
        node['extras'].append(extra.copy());node['previous'].append(previous);node['parent'].append(parent)
        if args.include_frames:node['frame'].append(observation['image'][-1].copy())
        return number

    try:
        for teacher_name,path,rivals in teachers:
            teacher=load_actor(path);teacher_commands=compiled(teacher.definition['actions'])
            mapping={i:commands[tuple(tuple(sorted(keys)) for keys in command)] for i,command in enumerate(teacher_commands)}
            env=FightEnv(definition,rivals)
            try:
                for index,rival in enumerate(rivals):
                    for player in ((1,) if rival['kind']=='cpu' else (1,2)):
                        for offset in ((0,) if rival['kind']=='cpu' else (0,12)):
                            observation,_=env.reset(options=dict(opponent=index,player=player,offset=offset));teacher.reset()
                            teacher_side=Side(teacher.definition,player);teacher_side.reset(env.game.rgb)
                            role=bool(observation['position'][4]>=0);elapsed=0;current=add_node(observation,role,elapsed,0,-1);episode=[]
                            while not env.ended:
                                teacher_action=int(teacher.predict(teacher_side.observe(env.game.raw),True)[0]);physical=mapping[teacher_action]
                                canonical=int(student.policy.physical_mirror[physical]) if observation['position'][4]<0 and student.policy.canonical else physical
                                next_observation,_,t,tr,info=env.step(physical);teacher_side.update(env.game.rgb);elapsed+=1
                                end=add_node(next_observation,role,elapsed,canonical,current)
                                reward=.25*(info['damage_dealt']-info['damage_taken'])/176.+float(info['result']=='win')-float(info['result']=='loss')
                                episode.append(dict(start=current,end=end,action=canonical*4,reward=reward,done=bool(t or tr),duration=1))
                                current=end;observation=next_observation
                            total_return=0.;group=f"{rival['id']}|P{player}|{offset}"
                            for row in reversed(episode):
                                total_return=row['reward']+total_return;row['returns']=total_return;row['win']=info['result']=='win';row['group']=group
                            for row in episode:
                                for key in transitions:transitions[key].append(row[key])
                            report=dict(teacher=teacher_name,teacher_path=str(path),opponent=rival,player=player,offset=offset,
                                        **{k:info[k] for k in ('result','self_hp','opponent_hp','native_frames')},transitions=len(episode))
                            reports.append(report);write(out/'progress.json',dict(games=reports,nodes=len(node['maps']),transitions=len(transitions['start'])))
                            print('NATIVE_EXPERIENCE',teacher_name,rival['id'],player,offset,info['result'],info['self_hp'],info['opponent_hp'],flush=True)
                native_frames+=env.native_frames
            finally:env.close();teacher.close()
        arrays={key:np.asarray(value) for key,value in node.items()}
        arrays.update({'transition_'+key:np.asarray(value) for key,value in transitions.items()})
        arrays['maps']=arrays['maps'].astype(np.float16);arrays['positions']=arrays['positions'].astype(np.float32);arrays['extras']=arrays['extras'].astype(np.float32)
        np.savez_compressed(out/'data.npz',**arrays)
        write(out/'summary.json',dict(student_perception=args.student,games=reports,native_frames=native_frames,nodes=len(node['maps']),transitions=len(transitions['start']),
             scope='Fresh native trajectories from learned teachers on training-only Ryu conditions. Both victories and defeats retained. Student uses identical frozen convolution/health sensors; teacher actions are mapped by exact six-frame key sequences. No teacher or opponent identity enters student inference.'))
    finally:student.close()


if __name__=='__main__':main()
