"""Frozen complete-round evaluation; all action statistics describe inputs."""
import argparse,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.league_adaptive.runtime import OUT,TARGET,load_actor,write,summary,setup
from experiments.league_adaptive.environment import FightEnv,family,scheduled_style


def pool(scope):
    if scope=='target':return [dict(id='frozen_current',kind='agent',path=str(TARGET))]
    paths={'focus':'focus_pool.json','train':'train_pool.json'}
    return json.loads((ROOT/'configs'/paths[scope]).read_text(encoding='utf-8'))


def evaluate(actor,rivals,out,*,cpu=True,offsets=(0,12),ports=(1,2),progress=True,arena_call_sequence=False):
    started=time.monotonic();rows=[];rivals=list(rivals)
    present={(int(r.get('stars',1)),r.get('character','Ryu')) for r in rivals if r['kind']=='cpu'}
    if cpu:rivals.extend(dict(id=f'cpu_{s}',kind='cpu',stars=s) for s in range(1,9) if (s,'Ryu') not in present)
    env=FightEnv(actor.definition,rivals)
    if arena_call_sequence:
        import numpy as np
        config=actor.definition['observation_config']
        actor.predict(dict(image=np.zeros((4,config['height'],config['width']),np.uint8),
            position=np.asarray([100,192,200,192,100,0],np.float32)/1024.),True)
    try:
        for index,rival in enumerate(rivals):
            for port in ((1,) if rival['kind']=='cpu' else ports):
                for offset in ((0,) if rival['kind']=='cpu' else offsets):
                    obs,_=env.reset(options=dict(opponent=index,player=port,offset=offset));counts=Counter();contexts=defaultdict(Counter);sides=Counter();skills=Counter();neural_routes=Counter();phases={};outside_proposals=0;after_switch={};duration_options=Counter();duration_contexts=defaultdict(Counter)
                    if not arena_call_sequence:actor.reset()
                    before_memory_resets=getattr(actor.policy,'inference_history_resets',0)
                    while True:
                        a=int(actor.predict(obs,True)[0]);name=actor.actions[a]['name'];f=family(name);raw=env.game.raw;dx=raw[f'p{3-port}_x']-raw[f'p{port}_x']
                        context=('close' if abs(dx)<70 else 'middle' if abs(dx)<130 else 'far')+('_air' if raw[f'p{3-port}_y']<184 else '_ground')
                        counts[f]+=1;contexts[context][f]+=1;sides['opponent_right' if dx>0 else 'opponent_left']+=1
                        if hasattr(actor.policy,'last_option_starts') and bool(actor.policy.last_option_starts[0]):
                            duration=int(actor.policy.last_duration_choices[0]);key=f'{f}|{duration}'
                            duration_options[key]+=1;duration_contexts[context][key]+=1
                        if hasattr(actor,'last_skill_indices'):skills[actor.definition['option_names'][int(actor.last_skill_indices[0])]]+=1
                        if hasattr(actor,'last_neural_route_probabilities'):neural_routes[str(int(actor.last_neural_route_probabilities[0].argmax()))]+=1
                        if hasattr(actor.policy,'last_neural_proposals') and a not in actor.policy.last_neural_proposals[0].tolist():outside_proposals+=1
                        phase=scheduled_style(rival,env.controller.count+1)
                        if phase is not None:
                            values=phases.setdefault(phase,dict(input_families=Counter(),skill_microsteps=Counter(),damage_dealt_hp=0,damage_taken_hp=0,native_frames=0));values['input_families'][f]+=1
                            if hasattr(actor,'last_skill_indices'):values['skill_microsteps'][actor.definition['option_names'][int(actor.last_skill_indices[0])]]+=1
                        obs,_,t,tr,info=env.step(a)
                        if phase is not None:
                            values['damage_dealt_hp']+=info['damage_dealt'];values['damage_taken_hp']+=info['damage_taken'];values['native_frames']+=info['elapsed_native_frames']
                            if (env.controller.count-1)//int(rival['switch_every'])>0:
                                elapsed=(env.controller.count-1)%int(rival['switch_every'])
                                bucket='decisions_0_9' if elapsed<10 else 'decisions_10_29' if elapsed<30 else 'decisions_30_plus'
                                slot=after_switch.setdefault(bucket,dict(decisions=0,damage_dealt_hp=0,damage_taken_hp=0,input_families=Counter()))
                                slot['decisions']+=1;slot['damage_dealt_hp']+=info['damage_dealt'];slot['damage_taken_hp']+=info['damage_taken'];slot['input_families'][f]+=1
                        if t or tr:break
                    row={k:info[k] for k in ('opponent','opponent_kind','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')}
                    row.update(action_families=dict(counts),contexts={k:dict(v) for k,v in contexts.items()},relative_side_decisions=dict(sides))
                    if hasattr(actor.policy,'inference_history_resets'):row['model_memory_resets']=actor.policy.inference_history_resets-before_memory_resets
                    if skills:row['skill_choices']=dict(skills)
                    if neural_routes:row['dominant_neural_routes']=dict(neural_routes)
                    if hasattr(actor.policy,'last_neural_proposals'):row['inputs_outside_neural_proposals']=outside_proposals
                    if phases:row['opponent_phase_profile']=phases
                    if after_switch:row['after_opponent_switch']=after_switch
                    if duration_options:
                        row['learned_input_durations']=dict(duration_options)
                        row['duration_choices_by_context']={k:dict(v) for k,v in duration_contexts.items()}
                    rows.append(row)
                    write(Path(out)/'partial.json',dict(completed=len(rows),rows=rows))
            if progress:print('EVAL',rival['id'],[(r['player'],r['offset'],r['result']) for r in rows if r['opponent']==rival['id']],flush=True)
        pvp=[r for r in rows if r['opponent_kind']!='cpu'];pve=[r for r in rows if r['opponent_kind']=='cpu']
        result=dict(overall=summary(pvp),by_port={str(p):summary([r for r in pvp if r['player']==p]) for p in (1,2)},
            pve=summary(pve),by_opponent={q['id']:summary([r for r in rows if r['opponent']==q['id']]) for q in rivals},
            rows=rows,seconds=time.monotonic()-started,native_frames=env.native_frames,
            inference_call_sequence='Synthetic zero-frame warmup; no explicit model reset between completed matches.' if arena_call_sequence else 'Explicit local model reset at each match.',
            scope='Complete frozen matches. Rule families and neural checkpoint identities remain separate; input counts are not actual move detection.')
        write(Path(out)/'summary.json',result);return result
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--agent',default=str(TARGET));p.add_argument('--scope',default='focus',choices=['focus','train','target'])
    p.add_argument('--out',required=True);p.add_argument('--pool');p.add_argument('--no-cpu',action='store_true')
    p.add_argument('--wait-for-agent',type=float,default=0);p.add_argument('--arena-call-sequence',action='store_true');a=p.parse_args();setup()
    if Path(a.out).exists():raise FileExistsError(a.out)
    waiting=time.monotonic()
    while not Path(a.agent).is_file():
        if time.monotonic()-waiting>a.wait_for_agent:raise FileNotFoundError(a.agent)
        time.sleep(5)
    actor=load_actor(a.agent);r=evaluate(actor,json.loads(Path(a.pool).read_text()) if a.pool else pool(a.scope),a.out,cpu=not a.no_cpu,arena_call_sequence=a.arena_call_sequence)
    print('SUMMARY',json.dumps({k:r[k] for k in ('overall','by_port','pve')}),flush=True)
