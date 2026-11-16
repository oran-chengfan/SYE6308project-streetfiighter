"""Frozen complete-round evaluation; all action statistics describe inputs."""
import argparse,copy,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.league_adaptive.runtime import OUT,TARGET,load_actor,write,summary,setup,same_state
from experiments.league_adaptive.environment import FightEnv,family,scheduled_style


def exact_opponent_pool(*pools):
    """Merge recorded conditions without conflating equal IDs and versions."""
    result={}
    for pool in pools:
        for opponent in pool:
            name=opponent['id']
            if name in result and result[name]!=opponent:
                raise ValueError(f'Opponent ID {name!r} has different configurations; assign an explicit alias')
            result.setdefault(name,copy.deepcopy(opponent))
    return list(result.values())


class ReferenceActionAudit:
    """Read-only observer of deployed choices against frozen physical proposals.

    This observer never supplies anything to the acting policy. Associations
    with later damage do not attribute a move or establish causal improvement.
    """
    def __init__(self, reference_path=TARGET):
        self.reference=load_actor(reference_path)
        if self.reference.definition['kind']!='neural_mixture' or self.reference.definition.get('inference_hold_steps',0):
            raise ValueError('Proposal auditing requires the stateless retained mixture')
        self.rows={}; self.seconds=0.

    def __call__(self, actor, observation, info):
        import torch
        started=time.monotonic()
        key=tuple(info[k] for k in ('opponent','opponent_kind','player','offset'))
        slot=self.rows.setdefault(key,Counter())
        with torch.no_grad():
            tensor,_=self.reference.policy.obs_to_tensor(observation)
            probabilities=self.reference.policy.get_distribution(tensor).distribution.probs[0]
            proposals=self.reference.policy.last_neural_proposals[0].cpu().numpy()
        action=int(info['action'])
        outside=not np.any(proposals==action)
        slot['decisions']+=1
        slot['reference_greedy_changes']+=int(action!=int(probabilities.argmax()))
        slot['outside_reference_proposals']+=int(outside)
        if outside:
            slot['outside_associated_damage_dealt']+=info['damage_dealt']
            slot['outside_associated_damage_taken']+=info['damage_taken']
            slot['outside_family|'+family(actor.actions[action]['name'])]+=1
        self.seconds+=time.monotonic()-started

    def result(self):
        return dict(rows=[dict(condition=list(k),**v) for k,v in self.rows.items()],seconds=self.seconds,
            scope='Actual deterministic deployed inputs against the three retained physical proposals in the same legal observation. Damage associations are not causal move labels.')

    def close(self):self.reference.close()


class InferenceIdentityAudit:
    """Compare a separately loaded policy on every actual deployment input.

    Use with ``arena_call_sequence=True``: both models receive one synthetic
    warmup and then the same legal history, without host resets between games.
    The observer supplies neither actions nor observations to the acting model.
    """
    def __init__(self, source_path):
        self.reference=load_actor(source_path);self.rows={};self.seconds=0.
        config=self.reference.definition['observation_config']
        self.reference.predict(dict(image=np.zeros((4,config['height'],config['width']),np.uint8),
            position=np.asarray([100,192,200,192,100,0],np.float32)/1024.),True)

    def __call__(self, actor, observation, info):
        started=time.monotonic()
        mapping=actor.definition.get('deployment_action_indices') if actor.actions!=self.reference.actions else None
        if mapping is not None:
            if (actor.definition.get('internal_action_count')!=len(self.reference.actions)
                    or actor.actions!=[self.reference.actions[i] for i in mapping]):
                raise ValueError('Explicit deployment catalog does not preserve physical commands')
        elif actor.actions!=self.reference.actions:raise ValueError('Deployment action catalogs differ')
        expected=int(self.reference.predict(observation,True)[0])
        key=tuple(info[k] for k in ('opponent','opponent_kind','player','offset'))
        row=self.rows.setdefault(key,dict(decisions=0,action_differences=0,max_q_error=0.,memory_state_differences=0))
        actual_action=int(info['action']);actual_action=actual_action if mapping is None else mapping[actual_action]
        row['decisions']+=1;row['action_differences']+=int(expected!=actual_action)
        if hasattr(actor.policy,'last_context_q_values'):
            q=actor.policy.last_context_q_values.detach().cpu().numpy()
            reference_q=self.reference.policy.last_context_q_values.detach().cpu().numpy()
            if mapping is not None:reference_q=reference_q[:,mapping]
            if q.shape!=reference_q.shape or not np.isfinite(q).all() or not np.isfinite(reference_q).all():
                raise ValueError('Invalid deployment Q vector')
            row['max_q_error']=max(row['max_q_error'],float(np.max(np.abs(q-reference_q))))
        if actor.definition['kind']=='sequence_ppo':
            for name,attribute in (('probability','last_memory_probabilities'),('critic','last_memory_values')):
                actual=getattr(actor.policy,attribute).detach().cpu().numpy()
                other=getattr(self.reference.policy,attribute).detach().cpu().numpy()
                if mapping is not None and name=='probability':
                    other=other[:,mapping];other=other/other.sum(1,keepdims=True)
                if actual.shape!=other.shape or not np.isfinite(actual).all() or not np.isfinite(other).all():
                    raise ValueError('Invalid deployed PPO probability or critic output')
                key_error='max_'+name+'_error'
                row[key_error]=max(row.get(key_error,0.),float(np.max(np.abs(actual-other))))
        if actor.definition['kind'] in ('sequence_double_q','sequence_ppo'):
            actual_state=actor.get_state()
            if mapping is not None and actual_state is not None:
                actual_state=copy.deepcopy(actual_state)
                actual_state['previous']=np.asarray(mapping)[actual_state['previous']]
            row['memory_state_differences']+=int(not same_state(actual_state,self.reference.get_state()))
        self.seconds+=time.monotonic()-started
        if (row['action_differences'] or row['max_q_error']>1e-6 or row['memory_state_differences']
                or row.get('max_probability_error',0.)>1e-6 or row.get('max_critic_error',0.)>1e-6):
            raise RuntimeError(f'Export changed actual inference: {key}, {row}')

    def result(self):
        return dict(rows=[dict(condition=list(k),**v) for k,v in self.rows.items()],seconds=self.seconds,
            scope='Read-only comparison on actual complete-game legal histories; includes warmup and no explicit cross-game reset. Technical identity, not a strength certificate.')

    def close(self):self.reference.close()


def paired_regression(reference, candidate):
    """Strict old-win gate over exactly matching complete conditions."""
    def index(rows):
        result={}
        for row in rows:
            key=tuple(row[k] for k in ('opponent','opponent_kind','player','offset'))
            if key in result:raise ValueError('Duplicate complete evaluation condition')
            result[key]=row
        return result
    old,new=index(reference),index(candidate)
    if old.keys()!=new.keys():raise ValueError('Incomplete or unmatched regression panel')
    lost=[dict(condition=list(k),reference=old[k],candidate=new[k]) for k in old if old[k]['result']=='win' and new[k]['result']!='win']
    gained=[dict(condition=list(k),reference=old[k],candidate=new[k]) for k in old if old[k]['result']!='win' and new[k]['result']=='win']
    def cpu_ko(row):
        return (row['opponent_kind']=='cpu' and row['result']=='win' and row['opponent_hp']<=0
                and row.get('end_reason') in ('signed_hp_ko','round_counter'))
    old_ko=[k for k in old if cpu_ko(old[k])]
    lost_ko=[dict(condition=list(k),reference=old[k],candidate=new[k]) for k in old_ko if not cpu_ko(new[k])]
    return dict(all_old_wins_kept=not lost,reference_wins=sum(r['result']=='win' for r in old.values()),
        reference_pve_ko=len(old_ko),pve_ko_kept=len(old_ko)-len(lost_ko),pve_ko_lost=lost_ko,
        inheritance_gate_passed=not lost and not lost_ko,
        old_wins_kept=sum(old[k]['result']=='win' and new[k]['result']=='win' for k in old),
        old_wins_lost=lost,new_wins=gained,by_port={str(p):dict(reference=summary([r for r in reference if r['player']==p and r['opponent_kind']!='cpu']),
            candidate=summary([r for r in candidate if r['player']==p and r['opponent_kind']!='cpu'])) for p in (1,2)})


def pool(scope):
    if scope=='target':return [dict(id='frozen_current',kind='agent',path=str(TARGET))]
    paths={'focus':'focus_pool.json','train':'train_pool.json'}
    return json.loads((ROOT/'configs'/paths[scope]).read_text(encoding='utf-8'))


def evaluate(actor,rivals,out,*,cpu=True,offsets=(0,12),ports=(1,2),progress=True,arena_call_sequence=False,history_mode='actual',transition_observer=None,conditions=None,reward_kind='win_probability',deterministic=True,completed_rows=()):
    if history_mode not in ('actual','latest_repeated'):raise ValueError('Unknown history ablation')
    if history_mode!='actual' and (actor.definition['kind']!='neural_mixture' or actor.definition.get('inference_hold_steps',0)):
        raise ValueError('History ablation is restricted to stateless neural mixtures')
    started=time.monotonic();rows=list(completed_rows);rivals=list(rivals)
    present={(int(r.get('stars',1)),r.get('character','Ryu')) for r in rivals if r['kind']=='cpu'}
    if cpu:rivals.extend(dict(id=f'cpu_{s}',kind='cpu',stars=s) for s in range(1,9) if (s,'Ryu') not in present)
    selected=None if conditions is None else {(r['opponent'],int(r['player']),int(r['offset'])) for r in conditions}
    if selected is not None and len(selected)!=len(conditions):raise ValueError('Duplicate selected complete condition')
    available={(r['id'],p,o) for r in rivals for p in ((1,) if r['kind']=='cpu' else ports)
               for o in ((0,) if r['kind']=='cpu' else offsets)}
    if selected is not None and (not selected or not selected<=available):raise ValueError('Invalid selected complete conditions')
    completed={(r['opponent'],r['player'],r['offset']) for r in rows}
    if len(completed)!=len(rows) or not completed<=(available if selected is None else selected):
        raise ValueError('Resumed full-game records do not match the declared conditions')
    configurations={r['id']:r for r in rivals}
    if any(r['opponent_config']!=configurations[r['opponent']] or r.get('character')!=actor.definition.get('character','Ryu')
           or r['result'] not in ('win','loss','draw') for r in rows):
        raise ValueError('Resumed games differ in opponent configuration, own role or completion status')
    previous_native_frames=sum(r['native_frames'] for r in rows)
    env=FightEnv(actor.definition,rivals,reward=reward_kind)
    if arena_call_sequence:
        config=actor.definition['observation_config']
        actor.predict(dict(image=np.zeros((4,config['height'],config['width']),np.uint8),
            position=np.asarray([100,192,200,192,100,0],np.float32)/1024.),True)
    try:
        for index,rival in enumerate(rivals):
            for port in ((1,) if rival['kind']=='cpu' else ports):
                for offset in ((0,) if rival['kind']=='cpu' else offsets):
                    if selected is not None and (rival['id'],port,offset) not in selected:continue
                    if (rival['id'],port,offset) in completed:continue
                    obs,_=env.reset(options=dict(opponent=index,player=port,offset=offset));counts=Counter();contexts=defaultdict(Counter);sides=Counter();skills=Counter();neural_routes=Counter();phases={};outside_proposals=0;after_switch={};duration_options=Counter();duration_contexts=defaultdict(Counter)
                    initial_readings=dict(env.game.raw)
                    physical_counts=Counter()
                    motion=Counter();damage_by_input=defaultdict(Counter);opponent_motion=Counter();opponent_inputs=Counter();context_damage=defaultdict(Counter);outside_retained=0
                    if not arena_call_sequence:actor.reset()
                    before_memory_resets=getattr(actor.policy,'inference_history_resets',0)
                    while True:
                        actor_obs=obs if history_mode=='actual' else dict(image=np.repeat(obs['image'][-1:],4,axis=0),position=obs['position'])
                        a=int(actor.predict(actor_obs,deterministic)[0]);name=actor.actions[a]['name'];f=family(name);raw=env.game.raw;dx=raw[f'p{3-port}_x']-raw[f'p{port}_x']
                        context=('close' if abs(dx)<70 else 'middle' if abs(dx)<130 else 'far')+('_air' if raw[f'p{3-port}_y']<184 else '_ground')
                        counts[f]+=1;physical_counts[name]+=1;contexts[context][f]+=1;sides['opponent_right' if dx>0 else 'opponent_left']+=1
                        if hasattr(actor.policy,'last_option_starts') and bool(actor.policy.last_option_starts[0]):
                            duration=int(actor.policy.last_duration_choices[0]);key=f'{f}|{duration}'
                            duration_options[key]+=1;duration_contexts[context][key]+=1
                        if hasattr(actor,'last_skill_indices'):skills[actor.definition['option_names'][int(actor.last_skill_indices[0])]]+=1
                        if hasattr(actor,'last_neural_route_probabilities'):neural_routes[str(int(actor.last_neural_route_probabilities[0].argmax()))]+=1
                        if hasattr(actor.policy,'last_neural_proposals') and a not in actor.policy.last_neural_proposals[0].tolist():outside_proposals+=1
                        retained=actor.definition.get('retained_expert_count')
                        if retained and a not in actor.policy.last_neural_proposals[0,:retained].tolist():outside_retained+=1
                        phase=scheduled_style(rival,env.controller.count+1)
                        if phase is not None:
                            values=phases.setdefault(phase,dict(input_families=Counter(),skill_microsteps=Counter(),damage_dealt_hp=0,damage_taken_hp=0,native_frames=0));values['input_families'][f]+=1
                            if hasattr(actor,'last_skill_indices'):values['skill_microsteps'][actor.definition['option_names'][int(actor.last_skill_indices[0])]]+=1
                        obs,_,t,tr,info=env.step(a)
                        if transition_observer is not None:
                            if hasattr(transition_observer,'record_transition'):
                                transition_observer.record_transition(actor,actor_obs,obs,info)
                            else:transition_observer(actor,actor_obs,info)
                        after=env.game.raw;self_dx=after[f'p{port}_x']-raw[f'p{port}_x']
                        other_dx=after[f'p{3-port}_x']-raw[f'p{3-port}_x']
                        opponent_motion['horizontal_travel']+=abs(other_dx)
                        if raw[f'p{3-port}_y']==192 and after[f'p{3-port}_y']==192:
                            opponent_motion['ground_horizontal_travel']+=abs(other_dx)
                        if info['opponent_action']>=0:
                            other_name=env.controller.definition['actions'][info['opponent_action']]['name']
                            opponent_inputs[family(other_name)]+=1
                        gap_change=abs(dx)-abs(after[f'p{3-port}_x']-after[f'p{port}_x'])
                        motion['horizontal_travel']+=abs(self_dx)
                        motion['toward_displacement_signed']+=self_dx*(1 if dx>=0 else -1)
                        motion['gap_closing_signed']+=gap_change
                        if raw[f'p{port}_y']==192 and after[f'p{port}_y']==192:
                            motion['ground_horizontal_travel']+=abs(self_dx)
                            if info['damage_taken']==0:
                                motion['ground_horizontal_travel_without_hp_loss']+=abs(self_dx)
                                signed=self_dx*(1 if dx>=0 else -1)
                                motion['ground_toward_travel_without_hp_loss']+=max(0,signed)
                                motion['ground_away_travel_without_hp_loss']+=max(0,-signed)
                                if name in ('LEFT','RIGHT'):
                                    motion['walking_ground_travel_without_hp_loss']+=abs(self_dx)
                                motion['ground_gap_closing_without_hp_loss']+=max(0,gap_change)
                                motion['ground_gap_opening_without_hp_loss']+=max(0,-gap_change)
                        if name in ('LEFT','RIGHT'):
                            motion['directional_input_calls']+=1
                            motion['directional_input_horizontal_travel']+=abs(self_dx)
                            motion['directional_input_stationary_calls']+=int(self_dx==0)
                        damage_by_input[f]['damage_dealt_hp']+=info['damage_dealt']
                        damage_by_input[f]['damage_taken_hp']+=info['damage_taken']
                        context_damage[context]['damage_dealt_hp']+=info['damage_dealt']
                        context_damage[context]['damage_taken_hp']+=info['damage_taken']
                        context_damage[context]['native_frames']+=info['elapsed_native_frames']
                        if phase is not None:
                            values['damage_dealt_hp']+=info['damage_dealt'];values['damage_taken_hp']+=info['damage_taken'];values['native_frames']+=info['elapsed_native_frames']
                            values['horizontal_travel']=values.get('horizontal_travel',0)+abs(self_dx)
                            values['gap_closing_signed']=values.get('gap_closing_signed',0)+gap_change
                            if (env.controller.count-1)//int(rival['switch_every'])>0:
                                elapsed=(env.controller.count-1)%int(rival['switch_every'])
                                bucket='decisions_0_9' if elapsed<10 else 'decisions_10_29' if elapsed<30 else 'decisions_30_plus'
                                slot=after_switch.setdefault(bucket,dict(decisions=0,damage_dealt_hp=0,damage_taken_hp=0,input_families=Counter()))
                                slot['decisions']+=1;slot['damage_dealt_hp']+=info['damage_dealt'];slot['damage_taken_hp']+=info['damage_taken'];slot['input_families'][f]+=1
                                slot['horizontal_travel']=slot.get('horizontal_travel',0)+abs(self_dx)
                                slot['gap_closing_signed']=slot.get('gap_closing_signed',0)+gap_change
                        if t or tr:break
                    row={k:info[k] for k in ('opponent','opponent_kind','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')}
                    row.update(character=actor.definition.get('character','Ryu'),opponent_character=rival.get('character','Ryu'),
                               opponent_config=dict(rival),actual_character_ids=[env.game.raw['p1_char'],env.game.raw['p2_char']],
                               initial_readings=initial_readings)
                    row.update(action_families=dict(counts),contexts={k:dict(v) for k,v in contexts.items()},relative_side_decisions=dict(sides))
                    row['physical_input_counts']=dict(physical_counts)
                    ko = info['result']=='win' and info['end_reason'] in ('signed_hp_ko','round_counter') and info['opponent_hp']<=0
                    row.update(ko_win=bool(ko), perfect_ko=bool(ko and info['self_hp']==176),
                        time_up=info['end_reason']=='native_time_up', native_fps=env.game.fps,
                        elapsed_seconds=info['native_frames']/env.game.fps,
                        cpu_stars=int(rival['stars']) if rival['kind']=='cpu' else None)
                    row.update(motion=dict(motion),damage_by_requested_input={k:dict(v) for k,v in damage_by_input.items()},
                        opponent_motion=dict(opponent_motion),opponent_requested_input_families=dict(opponent_inputs),
                        damage_by_decision_start_context={k:dict(v) for k,v in context_damage.items()},
                        motion_scope='Decision-endpoint displacement; includes knockback and airborne motion. Damage during an input is association, not move attribution.')
                    if hasattr(actor.policy,'inference_history_resets'):row['model_memory_resets']=actor.policy.inference_history_resets-before_memory_resets
                    if skills:row['skill_choices']=dict(skills)
                    if neural_routes:row['dominant_neural_routes']=dict(neural_routes)
                    if hasattr(actor.policy,'last_neural_proposals'):row['inputs_outside_neural_proposals']=outside_proposals
                    if actor.definition.get('retained_expert_count'):row['inputs_outside_retained_proposals']=outside_retained
                    if phases:row['opponent_phase_profile']=phases
                    if after_switch:row['after_opponent_switch']=after_switch
                    if duration_options:
                        row['learned_input_durations']=dict(duration_options)
                        row['duration_choices_by_context']={k:dict(v) for k,v in duration_contexts.items()}
                    rows.append(row)
                    write(Path(out)/'partial.json',dict(completed=len(rows),rows=rows))
            if progress:print('EVAL',rival['id'],[(r['player'],r['offset'],r['result']) for r in rows if r['opponent']==rival['id']],flush=True)
        if selected is not None and {(r['opponent'],r['player'],r['offset']) for r in rows}!=selected:
            raise ValueError('Selected complete conditions are missing from the supplied pool/ports/offsets')
        pvp=[r for r in rows if r['opponent_kind']!='cpu'];pve=[r for r in rows if r['opponent_kind']=='cpu']
        result=dict(overall=summary(pvp),by_port={str(p):summary([r for r in pvp if r['player']==p]) for p in (1,2)},
            pve=summary(pve),by_opponent={q['id']:summary([r for r in rows if r['opponent']==q['id']]) for q in rivals},
            rows=rows,seconds=time.monotonic()-started,native_frames=previous_native_frames+env.native_frames,setup_native_frames=env.game.setup_frames,
            observation_history_mode=history_mode,inference_deterministic=bool(deterministic),
            inference_call_sequence='Synthetic zero-frame warmup; no explicit model reset between completed matches.' if arena_call_sequence else 'Explicit local model reset at each match.',
            scope='Complete frozen matches. Rule families and neural checkpoint identities remain separate; input counts are not actual move detection.')
        if completed:result.update(previously_completed_native_frames=previous_native_frames,
            newly_simulated_native_frames=env.native_frames,
            resume_scope='Frozen checkpoint and exact completed conditions reused by caller; new process begins with the same legal warmup. Previous setup and interrupted incomplete-game costs are separate.')
        write(Path(out)/'summary.json',result);return result
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--agent',default=str(TARGET));p.add_argument('--scope',default='focus',choices=['focus','train','target'])
    p.add_argument('--out',required=True);p.add_argument('--pool');p.add_argument('--no-cpu',action='store_true')
    p.add_argument('--conditions',help='Optional explicit complete opponent/port/offset list, using the same evaluator.')
    p.add_argument('--reward',default='win_probability',help='Training-reward bookkeeping only; native game results retain their original rules.')
    p.add_argument('--wait-for-agent',type=float,default=0);p.add_argument('--arena-call-sequence',action='store_true')
    p.add_argument('--history-mode',choices=('actual','latest_repeated'),default='actual');a=p.parse_args();setup()
    if Path(a.out).exists():raise FileExistsError(a.out)
    waiting=time.monotonic()
    while not Path(a.agent).is_file():
        if time.monotonic()-waiting>a.wait_for_agent:raise FileNotFoundError(a.agent)
        time.sleep(5)
    actor=load_actor(a.agent);rivals=json.loads(Path(a.pool).read_text()) if a.pool else pool(a.scope)
    write(Path(a.out)/'config.json',dict(arguments=vars(a),pool=rivals,ports=[1,2],offsets=[0,12],cpu=not a.no_cpu))
    chosen=None if a.conditions is None else json.loads(Path(a.conditions).read_text(encoding='utf-8'))
    r=evaluate(actor,rivals,a.out,cpu=not a.no_cpu,arena_call_sequence=a.arena_call_sequence,history_mode=a.history_mode,conditions=chosen,reward_kind=a.reward)
    print('SUMMARY',json.dumps({k:r[k] for k in ('overall','by_port','pve')}),flush=True)
