"""Outcome-driven capability studies; learning and matches use shared entries.

This module owns experiment decisions and diagnostic evidence, not a second
DDQN/PPO trainer or evaluator. Full replay stays in local artifacts.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch

from experiments.league_adaptive.runtime import ROOT,Actor,load_actor,setup,write,compiled,save_fight,restore_fight
from experiments.league_adaptive.environment import FightEnv
from experiments.league_adaptive.evaluate import evaluate
from experiments.ryu_counterq.policy import construct_sequence_q
from experiments.ryu_counterq.replay import SequenceReplay,TrajectoryRecorder,episode_values,sequence_double_q_targets,successful_episode,full_values,same_episode_condition
from experiments.ryu_counterq.training_state import learner_components,contract,write_checkpoint,restore_optimizer
from experiments.ryu_counterq.train import optimize,sample_sequence_batch
from experiments.ryu_counterq.collect_demonstrations import continue_episode,terminal
from experiments.ryu_counterq.stabilization import read,run_training_jobs
from experiments.ryu_direct_league.policy import save
from experiments.ryu_counterq.skill_data import tactical_skill_catalog,collect_coherent_skill_bank,stochastic_skill_execution


def update_report(root,**values):
    path=ROOT/'reports'/root.name/'summary.json'
    state=read(path) if path.exists() else {};state.update(values);write(path,state)


def prefix_hidden(policy,episode,index):
    hidden=None
    with torch.no_grad():
        for first in range(0,index,32):
            _,hidden=episode_values(policy,episode,first,min(first+32,index),hidden)
    return torch.zeros(1,1,policy.memory_hidden) if hidden is None else hidden


def probe_collection(spec,base,root):
    destination=root/'diagnostic/probes.pt'
    if destination.exists():return torch.load(destination,map_location='cpu',weights_only=False)
    records=[];frames=0
    for number,case in enumerate(spec['diagnostic']['cases']):
        source=case.get('historical_weights',f"models/ddqn_stabilization/input_real_{case['seed']}.pt")
        rival=next(r for r in base['pool'] if r['id']==case['opponent'])
        recorder=TrajectoryRecorder([rival],source);actor=load_actor(source)
        try:
            outcome=evaluate(actor,[rival],root/f'diagnostic/collection_{number}',cpu=False,
                offsets=(case['offset'],),ports=(case['player'],),arena_call_sequence=True,
                transition_observer=recorder,reward_kind=base['learner']['reward'])
        finally:actor.close()
        if len(recorder.episodes)!=1:raise ValueError('A probe is one complete independently controlled game')
        frames+=outcome['native_frames'];records.append(dict(case=case,source=source,
            episode=recorder.episodes[0],result=outcome['rows'][0]))
    destination.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(records=records,native_frames=frames),destination)
    return dict(records=records,native_frames=frames)


def replay_view(state,nstep):
    raw=dict(state['replay']);settings=list(raw['settings']);settings[4]=nstep;raw['settings']=tuple(settings)
    replay=SequenceReplay(capacity=settings[0],byte_capacity=settings[1],burn=settings[2],
        learn=settings[3],nstep=nstep,alpha=settings[5]);replay.load_state_dict(raw)
    return replay


def panel_rivals(spec,base):
    wanted=set(spec['panel']['cpu']+spec['panel']['pvp'])
    rows=[copy.deepcopy(r) for r in base['pool'] if r['id'] in wanted]
    if len(rows)!=len(wanted):raise ValueError('The fixed panel contains an unknown opponent')
    return rows


def prepare_accumulation(spec,base,root):
    """Same complete learner state per pair, plus identical verified anchors."""
    results={};rivals=panel_rivals(spec,base)
    for seed in spec['seeds']:
        directory=root/'parents'/str(seed);summary=directory/'summary.json'
        if summary.exists():results[str(seed)]=read(summary);continue
        state=torch.load(ROOT/spec['parents'][str(seed)],map_location='cpu',weights_only=False)
        source=f'models/ddqn_stabilization/input_real_{seed}.pt';actor=load_actor(source)
        if any(not torch.equal(actor.policy.state_dict()[k],v) for k,v in state['online'].items()):
            raise ValueError('Independent collection weights differ from the complete parent')
        recorder=TrajectoryRecorder(rivals,source)
        try:initial=evaluate(actor,rivals,directory/'initial_panel',cpu=False,offsets=(0,),
            ports=(1,2),arena_call_sequence=True,transition_observer=recorder,reward_kind=base['learner']['reward'])
        finally:actor.close()
        replay=replay_view(state,8);references={};verification=[]
        for number in sorted(replay.history_ids):
            episode=replay.episodes[number]
            episode['retention_role']='original_success' if successful_episode(episode) else 'difficult_failure'
            version=episode.get('policy_version')
            if not version or not (ROOT/version).is_file():continue
            if version not in references:references[version]=load_actor(version)
            greedy=full_values(references[version].policy,episode)[:-1].argmax(-1).numpy()
            matches=np.array_equal(greedy,episode['actions'])
            verification.append(dict(episode=number,policy_version=version,greedy_history_verified=matches))
            if matches:
                old_bytes=replay.episode_bytes(episode);episode['greedy_actions']=greedy
                replay.bytes+=replay.episode_bytes(episode)-old_bytes
        for ref in references.values():ref.close()
        for episode in recorder.episodes:
            episode.update(retention_role='starting_success' if successful_episode(episode) else 'difficult_failure',
                actor_update_version=state['progress']['optimizer_updates'],verified_frame=state['progress']['native_frames'])
            if episode.get('validated_complete'):
                replay.insert_verified_episode(episode,state['progress']['native_frames'],require_success=False)
        bank=replay.refresh_history(state['progress']['native_frames'],spec['history_refresh'])
        cfg=copy.deepcopy(base);cfg.update(experiment=spec['experiment'],protocol='capability_accumulation',
            history_refresh=copy.deepcopy(spec['history_refresh']),development_pool=rivals)
        cfg['network']=copy.deepcopy(state['contract']['network'])
        cfg['learner'].update(nstep=8,backup_steps=1,backup_rule='one_step')
        cfg.pop('stale_history_audit',None);cfg['comparison']['checkpoints']=spec['ddqn_comparison']['checkpoints']
        state['replay']=replay.state_dict();state['purpose']='constructed_common_parent'
        state['history_seed_ids']=sorted(replay.history_ids);state['progress']['pending_success_refresh']=[]
        state['migration_origin']=dict(complete_parent=spec['parents'][str(seed)],
            old_backup_steps=1,new_replay_read_span=8,preparation_SGD_updates=0,
            changed='Bookkeeping and verified replay only; online,target,Adam,RNG andclocks are preserved. This explicitly changes the learning contract.')
        paths={}
        for arm in spec['ddqn_comparison']['arms']:
            resolved=copy.deepcopy(cfg);resolved['learner'].update(backup_steps=1 if arm=='one_step' else 8,
                backup_rule='one_step' if arm=='one_step' else 'collector_cut')
            state['contract']=contract(resolved,resolved['pool'],'Guile')
            state['progress']['lineage_id']=f"{spec['experiment']}/{seed}/{arm}"
            target=directory/arm/'training_state.pt';write_checkpoint(target,state)
            write(directory/arm/'configuration.json',resolved);paths[arm]=target.relative_to(ROOT).as_posix()
        value=dict(source=spec['parents'][str(seed)],initial_panel={k:initial[k] for k in ('overall','by_port','pve','native_frames')},
            useful_starting_conditions=[dict(opponent=r['opponent'],player=r['player'],offset=r['offset'])
                for r in initial['rows'] if (r['ko_win'] if r['opponent_kind']=='cpu' else r['result']=='win')],
            preparation_native_frames=initial['native_frames'],optimizer_updates=0,
            learner_states_preserved=True,history_bank=bank,original_collector_verification=verification,parents=paths)
        write(summary,value);results[str(seed)]=value
    update_report(root,preparation=results,stage='shared_success_bank_ready');return results


def paired_ddqn(spec,base,root):
    prepare_accumulation(spec,base,root)
    prepared_trace_profile(spec,root)
    for segment,budget,points in [('pilot',60000,[60000]),
                                  ('initial',spec['ddqn_comparison']['first_native_frame_budget']-60000,[140000,540000]),
                                  ('continuation',spec['ddqn_comparison']['extension_native_frames'],[300000,600000])]:
        jobs=[]
        for seed in spec['seeds']:
            for arm in spec['ddqn_comparison']['arms']:
                out=root/'ddqn'/str(seed)/arm/segment
                parent=(root/'parents'/str(seed)/arm if segment=='pilot' else root/'ddqn'/str(seed)/arm/('pilot' if segment=='initial' else 'initial'))/'training_state.pt'
                if (out/'summary.json').exists():continue
                cfg=root/'parents'/str(seed)/arm/'configuration.json'
                jobs.append(dict(name=f'{arm}_{seed}_{segment}',log=root/f'{arm}_{seed}_{segment}.log',command=[
                    sys.executable,'-m','experiments.ryu_counterq.train','--config',str(cfg),
                    '--character','Guile','--seed',str(seed),'--arm','ordinary','--frames',str(budget),'--out',str(out),
                    '--resume',str(parent),'--checkpoints',*map(str,points)]))
        update_report(root,stage=f'ddqn_{segment}_training',ddqn_method=spec['ddqn_comparison'])
        run_training_jobs(jobs,root/'ddqn_workers.json')
        update_report(root,stage=f'ddqn_{segment}_complete')


def condition_key(row):
    """Exact frozen opponent configuration, port and offset; names are insufficient."""
    return (json.dumps(row['opponent_config'],sort_keys=True),int(row['player']),int(row['offset']))


def useful_conditions(rows):
    return {condition_key(r) for r in rows if (r['ko_win'] if r['opponent_kind']=='cpu' else r['result']=='win')}


def capability_point(evaluation,initial,frames,checkpoint,updates=None):
    before=useful_conditions(initial['rows']);now=useful_conditions(evaluation['rows'])
    lookup={condition_key(r):r for r in initial['rows']+evaluation['rows']}
    def named(keys):
        return [dict(opponent=lookup[k]['opponent'],opponent_config=lookup[k]['opponent_config'],player=k[1],offset=k[2]) for k in sorted(keys)]
    return dict(native_frames=frames,checkpoint=checkpoint,optimizer_updates=updates,
        cpu_ko=sum(r['ko_win'] for r in evaluation['rows'] if r['opponent_kind']=='cpu'),
        cpu_time_wins=sum(r['result']=='win' and r['time_up'] for r in evaluation['rows'] if r['opponent_kind']=='cpu'),
        pvp_wins={str(p):sum(r['result']=='win' for r in evaluation['rows'] if r['opponent_kind']!='cpu' and r['player']==p) for p in (1,2)},
        starting_useful=len(before),starting_kept=len(before&now),old_lost=named(before-now),new_useful=named(now-before),
        useful=named(now),evaluation_native_frames=evaluation['native_frames'])


def persistent_capability_gate(points,spec):
    criteria=spec['capability_gate'];passing=[]
    for i,a in enumerate(points):
        for b in points[-1:] if i<len(points)-1 else []:
            if b['native_frames']-a['native_frames']<criteria['minimum_continuation_gap_native_frames']:continue
            def qualifies(p):
                return (p['starting_kept']==p['starting_useful'] and p['cpu_ko']>=criteria['minimum_CPU_KOs_per_seed']
                    and all(v>=criteria['minimum_PvP_wins_per_port_per_seed'] for v in p['pvp_wins'].values())
                    and p.get('fixed_focus_passed',True))
            shared=set.intersection(*(useful_conditions_from_labels(p['new_useful']) for p in points[i:]))
            if all(qualifies(p) for p in points[i:]) and len(shared)>=criteria['minimum_new_useful_conditions_per_seed']:
                passing.append(dict(first=a['native_frames'],later=b['native_frames'],persistent_new=len(shared),checkpoint=b['checkpoint']))
    return dict(passed=bool(passing),passing_pairs=passing)


def useful_conditions_from_labels(rows):
    return {condition_key(r) for r in rows}


def annotate_accumulation(points):
    """Describe newly learned results disappearing; never add hard win locks."""
    seen=set();previous=set();first=None
    for p in points:
        current=useful_conditions_from_labels(p['new_useful'])
        if first is None:first=current.copy()
        p['new_gain_tracking']=dict(first_checkpoint_new=len(first),first_checkpoint_new_still_kept=len(first&current),
            first_time_new_results=len(current-seen),previous_checkpoint_new=len(previous),
            previous_checkpoint_new_still_kept=len(previous&current),previous_checkpoint_new_lost=len(previous-current))
        seen|=current;previous=current


def analyze_ddqn(spec,root):
    results={};complete=True
    for arm in spec['ddqn_comparison']['arms']:
        runs={}
        for seed in spec['seeds']:
            initial=read(root/'parents'/str(seed)/'initial_panel/summary.json');points=[];costs={}
            base_frames=0;usage={}
            for segment in ('pilot','initial','continuation'):
                path=root/'ddqn'/str(seed)/arm/segment/'summary.json'
                if not path.exists():complete=False;continue
                run=read(path);costs[segment]=run['costs'];usage.update(run['history_usage'])
                for p in run['curve']:
                    points.append(capability_point(p['development'],initial,base_frames+p['actual_frames'],p['checkpoint'],
                        p['learning_clock']['optimizer_updates']))
                base_frames+=run['actual_training_frames']
            new=[dict(episode=n,**v) for n,v in usage.items() if v['role']=='discovered_success']
            annotate_accumulation(points)
            covered={condition_key(r) for r in initial['rows']};final_useful=useful_conditions_from_labels(points[-1]['useful']) if points else set()
            failed_protected=[]
            for item in new:
                k=condition_key(dict(opponent_config=item['opponent'],player=item['player'],offset=item['offset']))
                if k in covered and k not in final_useful and item['retired_frame'] is None and item['history_TD_transitions']>0:
                    failed_protected.append(item)
            runs[str(seed)]=dict(points=points,gate=persistent_capability_gate(points,spec),costs=costs,
                newly_verified_history=dict(trajectories=len(new),actual_history_TD=sum(v['history_TD_transitions'] for v in new),
                    trajectories_sampled=sum(v['history_TD_transitions']>0 for v in new),
                    protected_and_sampled_but_failed_at_latest_panel=failed_protected,details=new))
        results[arm]=dict(seeds=runs,passed_on_both_seeds=all(r['gate']['passed'] for r in runs.values()))
    result=dict(complete=complete,arms=results,
        passed=complete and any(r['passed_on_both_seeds'] for r in results.values()),
        scope='Consumed 19-condition development panel. CPU time wins never count as KO. Each point is one independently acting checkpoint, not a union of different models.')
    write(root/'ddqn_analysis.json',result);update_report(root,ddqn_comparison=result)
    return result


def prepare_algorithms(spec,base,root,backup,*,phase='algorithms',learner_overrides=None):
    """Fresh paired initialization, not transfer of learned Q-values into PPO."""
    from experiments.ryu_counterq.policy import random_sequence_q
    root=Path(root).resolve()
    prepared={};rivals=panel_rivals(spec,base);character=spec.get('own_character','Guile')
    for seed in spec['algorithm_comparison']['seeds']:
        out=root/phase/str(seed);path=out/'initialization.json'
        if path.exists():prepared[str(seed)]=read(path);continue
        cfg=copy.deepcopy(base);cfg.update(protocol='capability_accumulation',experiment=spec['experiment'],
            development_pool=rivals,history_refresh=copy.deepcopy(spec['history_refresh']))
        cfg.pop('stale_history_audit',None);cfg.pop('active_replay_policy',None)
        cfg['network']=copy.deepcopy(spec['network']) if 'network' in spec else read(root/'parents'/str(spec['seeds'][0])/backup/'configuration.json')['network']
        cfg['learner'].update(sampling_mode='fixed_groups',coverage_offsets=[0,12],save_training_state=True,
            nstep=8,backup_steps=1 if backup=='one_step' else 8,backup_rule='one_step' if backup=='one_step' else 'collector_cut')
        cfg['learner'].update(copy.deepcopy(learner_overrides or {}))
        torch.manual_seed(seed);policy,definition=random_sequence_q(character,cfg['network'])
        ppo_definition=copy.deepcopy(definition);ppo_definition['kind']='sequence_ppo'
        ppo=construct_sequence_q(ppo_definition);ppo.load_state_dict(policy.state_dict(),strict=True)
        if any(not torch.equal(policy.state_dict()[k],v) for k,v in ppo.state_dict().items()):
            raise ValueError('Fresh algorithms do not have the declared shared initial parameters')
        save(out/'initial_ddqn.pt',policy,definition,dict(algorithm='DDQN',seed=seed,initialization='fresh random'))
        save(out/'initial_ppo.pt',ppo,ppo_definition,dict(algorithm='PPO',seed=seed,initialization='same fresh random tensors; no learned Q transfer'))
        write(out/'configuration.json',cfg);write(out/'pool.json',cfg['pool'])
        initial_frames=0;reused=None
        if phase!='algorithms':
            original=root/'algorithms'/str(seed)
            original_parameters=torch.load(original/'initial_ddqn.pt',map_location='cpu',weights_only=False)['state_dict']
            if any(not torch.equal(policy.state_dict()[k],v) for k,v in original_parameters.items()):
                raise ValueError('Task-distribution comparison changed its shared initial parameters')
            initial=read(original/'initial_panel/summary.json');reused=(original/'initial_panel/summary.json').relative_to(ROOT).as_posix()
            write(out/'initial_panel/summary.json',initial)
        else:
            initial=evaluate(Actor(policy,definition),rivals,out/'initial_panel',cpu=False,offsets=(0,),arena_call_sequence=True,progress=False)
            initial_frames=initial['native_frames']
        value=dict(seed=seed,character=character,parameters=sum(p.numel() for p in policy.parameters()),identical_initial_tensors=True,
            initial_panel_native_frames=initial_frames,reused_initial_panel=reused,backup=backup,
            information='Same trainable legal encoder, GRU, physical125 actions, signed coordinates, executed previous action and real issued-duration counters.',
            differences='DDQN epsilon/charge exploration and off-policy replay vs PPO categorical on-policy GAE/clipped updates. Algorithm-specific samples and costs are reported; no claim isolates only the loss formula.')
        write(path,value);prepared[str(seed)]=value
    update_report(root,**{('algorithm_initialization' if phase=='algorithms' else phase+'_initialization'):prepared});return prepared


def algorithm_jobs(spec,root,*,phase='algorithms',budget=None,points=None,algorithms=('DDQN','PPO'),starts=None,ppo_gae_lambda=None,dispatch=True):
    jobs=[];study=spec['algorithm_comparison']
    budget=budget if budget is not None else study['native_frames_per_algorithm_seed']+study['continuation_segment_native_frames']
    points=points if points is not None else study['checkpoints'][1:]+[budget]
    for seed in study['seeds']:
        parent=root/phase/str(seed);cfg=parent/'configuration.json'
        for algorithm in algorithms:
            out=parent/algorithm
            if (out/('summary.json' if algorithm=='DDQN' else 'result.json')).exists():continue
            if algorithm=='DDQN':
                source=(starts or {}).get(str(seed),{})
                command=[sys.executable,'-m','experiments.ryu_counterq.train','--config',str(cfg),'--character',spec.get('own_character','Guile'),'--seed',str(seed),
                    '--arm','ordinary','--frames',str(budget),'--out',str(out),
                    '--resume' if 'resume' in source else '--start',str(source.get('resume',source.get('initial',parent/'initial_ddqn.pt'))),
                    '--checkpoints',*map(str,points)]
            else:
                start=(starts or {}).get(str(seed),{});initial=start.get('initial',parent/'initial_ppo.pt')
                command=[sys.executable,'-m','experiments.ryu_adaptive_curriculum.train_direct_sequence','--config',str(cfg),
                    '--pool',str(parent/'pool.json'),'--initial',str(initial),'--seed',str(seed),'--out',str(out),
                    '--native-frames',str(budget),'--checkpoints',*map(str,points),'--max-seconds','0','--family-entropy','0']
                if ppo_gae_lambda is not None:command.extend(['--gae-lambda',str(ppo_gae_lambda)])
                if 'temperature' in start:
                    command.extend(['--temperature-initial',str(start['temperature']),'--temperature-final',str(start['temperature'])])
            prefix='' if phase=='algorithms' else phase+'_'
            jobs.append(dict(name=f'{prefix}{algorithm}_{seed}',command=command,log=root/f'{prefix}{algorithm}_{seed}.log'))
    if not dispatch:return jobs
    if not jobs:return
    update_report(root,stage='matched_fresh_algorithms_training' if phase=='algorithms' else phase+'_training',own_character=spec.get('own_character','Guile'))
    workers_file=root/('algorithm_workers.json' if phase=='algorithms' else phase+'_workers.json')
    batches=([job] for job in jobs) if study.get('serial_training_jobs',False) else [jobs]
    for batch in batches:run_training_jobs(batch,workers_file)


def algorithm_records(run,algorithm):
    """One checkpoint-record reader shared by initial and continued studies."""
    if algorithm=='DDQN':
        return [dict(requested=p['threshold'],native_frames=p['actual_frames'],checkpoint=p['checkpoint'],
            evaluation=p['development'],optimizer_updates=p['learning_clock']['optimizer_updates']) for p in run['curve']]
    return [dict(requested=p['requested_native_frames'][-1],native_frames=p['actual_native_frames'],checkpoint=p['checkpoint'],
        evaluation=p['development'],optimizer_updates=None) for p in run['frozen_evaluations']]


def continued_records(root,phase,seed,algorithm):
    records=algorithm_records(read(root/phase/str(seed)/algorithm/('summary.json' if algorithm=='DDQN' else 'result.json')),algorithm)
    provenance=root/(phase+'_provenance.json')
    previous=[]
    if provenance.exists():
        origin=read(provenance)[str(seed)]['source_phase']
        previous=continued_records(root,origin,seed,algorithm)
    offset=previous[-1]['native_frames'] if previous else 0
    if previous:
        parent_result=read(root/origin/str(seed)/algorithm/('summary.json' if algorithm=='DDQN' else 'result.json'))
        parent_records=algorithm_records(parent_result,algorithm)
        actual=parent_result['actual_training_frames'] if algorithm=='DDQN' else parent_result['native_frames']
        offset+=max(0,actual-parent_records[-1]['native_frames'])
    return previous+[dict(r,native_frames=offset+r['native_frames'],phase=phase) for r in records]


def focus_keys(spec):
    opponents={r['id']:r for r in read(ROOT/spec['base_config'])['pool']}
    return {condition_key(dict(**c,opponent_config=opponents[c['opponent']])) for c in spec['representative_curriculum']['focus_conditions']}


def learned_anchor(initial,learned):
    old={condition_key(r):r for r in initial['rows']};new={condition_key(r):r for r in learned['rows']}
    initial_useful=useful_conditions(initial['rows']);required=initial_useful|useful_conditions(learned['rows'])
    if old.keys()!=new.keys():raise ValueError('An accumulation anchor requires the same complete panel')
    result=dict(rows=[old[k] if k in initial_useful else new[k] for k in old])
    if useful_conditions(result['rows'])!=required:raise RuntimeError('Learned retention anchor was weakened')
    return result


def analyze_algorithms(spec,root,*,phase='algorithms'):
    result={}
    for algorithm in ('DDQN','PPO'):
        seeds={}
        for seed in spec['algorithm_comparison']['seeds']:
            out=root/phase/str(seed);initial=read(out/'initial_panel/summary.json');points=[];evaluations=[]
            path=out/algorithm/('summary.json' if algorithm=='DDQN' else 'result.json')
            if not path.exists():continue
            run=read(path)
            for p in algorithm_records(run,algorithm):
                points.append(capability_point(p['evaluation'],initial,p['native_frames'],p['checkpoint'],p['optimizer_updates']))
                points[-1]['requested_native_frames']=p['requested'];evaluations.append((p['requested'],p['evaluation']))
            if algorithm=='DDQN':
                cost=run['costs']
            else:
                cost={k:run[k] for k in ('native_frames','environment_calls','optimizer_updates','optimization_samples','optimization_seconds','seconds',
                    'decision_native_frames_by_group','reset_offset_native_frames','evaluation_native_frames','evaluation_seconds')}
            annotate_accumulation(points)
            anchor_limit=spec['algorithm_comparison'].get('learned_anchor_native_frames',1000000)
            anchor=next((ev for threshold,ev in evaluations if threshold==anchor_limit),None);accumulation=[]
            if anchor is not None:
                combined=learned_anchor(initial,anchor);required=useful_conditions(combined['rows'])
                for p,(threshold,ev) in zip(points,evaluations):
                    if threshold>=anchor_limit:accumulation.append(capability_point(ev,combined,p['native_frames'],p['checkpoint'],p['optimizer_updates']))
                annotate_accumulation(accumulation)
            seeds[str(seed)]=dict(points=points,random_start_gate=persistent_capability_gate(points,spec),
                learned_anchor_native_frames=anchor_limit,learned_anchor_useful=len(required) if anchor is not None else None,
                accumulation_points=accumulation,gate=persistent_capability_gate(accumulation,spec),costs=cost)
        result[algorithm]=dict(seeds=seeds,complete=len(seeds)==len(spec['algorithm_comparison']['seeds']),
            passed_on_both_seeds=len(seeds)==2 and all(r['gate']['passed'] for r in seeds.values()))
    write(root/('algorithm_analysis.json' if phase=='algorithms' else phase+'_analysis.json'),result)
    update_report(root,**{('algorithm_comparison' if phase=='algorithms' else phase+'_comparison'):result})
    return result


def selected_backup(comparison):
    def score(arm):
        points=[r['points'][-1] for r in comparison['arms'][arm]['seeds'].values()]
        return (min(p['starting_kept']/max(1,p['starting_useful']) for p in points),
            sum(p['starting_kept'] for p in points),sum(len(p['new_useful']) for p in points),
            sum(p['cpu_ko']+sum(p['pvp_wins'].values()) for p in points),arm=='one_step')
    return max(comparison['arms'],key=score)


def late_recovery_protocols(result,spec):
    eligible=[];rule=spec['capability_gate']
    for name,study in result.items():
        if not study.get('complete') or study['passed_on_both_seeds']:continue
        points=[s['accumulation_points'][-1] for s in study['seeds'].values()]
        if all(p['starting_kept']==p['starting_useful'] and len(p['new_useful'])>=rule['minimum_new_useful_conditions_per_seed']
            and p['cpu_ko']>=rule['minimum_CPU_KOs_per_seed']
            and all(v>=rule['minimum_PvP_wins_per_port_per_seed'] for v in p['pvp_wins'].values()) for p in points):eligible.append(name)
    return eligible


def representative_study(spec,base,root):
    """Authorized common-task follow-through, never a third learner."""
    comparison=analyze_algorithms(spec,root)
    if not all(s['complete'] for s in comparison.values()):
        raise RuntimeError('Finish the same-budget full-pool comparison before changing task allocation')
    if any(s['passed_on_both_seeds'] for s in comparison.values()):
        raise RuntimeError('A passing protocol proceeds to coverage; representative retraining is unnecessary')
    recovery=late_recovery_protocols(comparison,spec)
    if recovery:raise RuntimeError(f'Apply the declared late-recovery continuation first:{recovery}')
    study=spec['representative_curriculum'];phase='representative'
    backup=read(root/'selected_backup.json')['backup'] if (root/'selected_backup.json').exists() else selected_backup(read(root/'ddqn_analysis.json'))
    prepare_algorithms(spec,base,root,backup,phase=phase,learner_overrides=dict(
        training_focus=dict(opponents=study['focus_opponents'],fraction=1.)))
    algorithm_jobs(spec,root,phase=phase,budget=study['first_native_frames_per_algorithm_seed'],points=study['checkpoints'])
    result=analyze_algorithms(spec,root,phase=phase);local={}
    required=focus_keys(spec)
    for name,method in result.items():
        seeds={}
        for seed,run in method['seeds'].items():
            points=[p for p in run['points'] if p['requested_native_frames'] in (1000000,2000000)]
            seeds[seed]=dict(counts=[dict(native_frames=p['native_frames'],focus_results=len(required&useful_conditions_from_labels(p['useful']))) for p in points],
                passed=len(points)==2 and all(required<=useful_conditions_from_labels(p['useful']) for p in points))
        local[name]=dict(seeds=seeds,passed_on_both_seeds=len(seeds)==2 and all(v['passed'] for v in seeds.values()))
    write(root/'representative_local_gate.json',local)
    update_report(root,representative_local_gate=local,
        stage='representative_requires_expansion' if any(v['passed_on_both_seeds'] for v in local.values()) else 'representative_requires_targeted_diagnosis')
    return result,local


def continue_algorithms(spec,root,algorithm,*,source_phase,phase,budget,points,focus_fraction=None,focus_opponents=None,ppo_gae_lambda=None,pending_jobs=None,success_disagreement_scope=None):
    """Continue existing learners across a declared curriculum boundary."""
    starts={};provenance={};source_lambdas=set()
    for seed in spec['algorithm_comparison']['seeds']:
        source=root/source_phase/str(seed);destination=root/phase/str(seed)
        cfg=read(source/'configuration.json');original=copy.deepcopy(cfg)
        result=read(source/algorithm/('summary.json' if algorithm=='DDQN' else 'result.json'))
        if algorithm=='PPO' and cfg.get('success_consolidation') is not None:
            auxiliary=cfg['success_consolidation'];usage=read(source/algorithm/'success_learning.json')
            auxiliary.update(initial_bank=(source/algorithm/'success_bank.pt').relative_to(ROOT).as_posix(),
                initial_bank_clock=usage.get('bank_clock',auxiliary.get('initial_bank_clock',0)+result['native_frames']),
                resume_bank_sampling=True,initial_outcomes=algorithm_records(result,algorithm)[-1]['evaluation']['rows'])
            if success_disagreement_scope is not None:auxiliary['disagreement_scope']=success_disagreement_scope
        if focus_fraction is not None:cfg['learner']['training_focus']['fraction']=focus_fraction
        if focus_opponents is not None:cfg['learner']['training_focus']['opponents']=list(focus_opponents)
        cfg_path=destination/'configuration.json'
        if cfg_path.exists() and read(cfg_path)!=cfg:raise ValueError('Continuation configuration changed after declaration')
        write(cfg_path,cfg);write(destination/'pool.json',cfg['pool'])
        write(destination/'initial_panel/summary.json',read(source/'initial_panel/summary.json'))
        if algorithm=='DDQN':
            parent=destination/'parent_state.pt'
            if not parent.exists():
                state=torch.load(source/algorithm/'training_state.pt',map_location='cpu',weights_only=False)
                if state['contract']!=contract(original,original['pool'],spec.get('own_character','Guile')):
                    raise ValueError('Source continuation state does not match its declared contract')
                state['purpose']='constructed_common_parent';state['contract']=contract(cfg,cfg['pool'],spec.get('own_character','Guile'))
                state['migration_origin']=dict(source=(source/algorithm/'training_state.pt').relative_to(ROOT).as_posix(),
                    change='Only the predeclared sampling curriculum changes. Online,target,Adam,replay/priorities,RNG and all clocks are preserved.')
                write_checkpoint(parent,state)
            starts[str(seed)]=dict(resume=parent)
            restoration='Complete DDQN boundary state with explicit sampling-contract migration; no optimizer, target, replay or exploration restart.'
        else:
            source_lambdas.add(float(result['arguments'].get('gae_lambda',.95)))
            parent=ROOT/algorithm_records(result,algorithm)[-1]['checkpoint']
            starts[str(seed)]=dict(initial=parent)
            restoration='PPO weights and Adam continue; fresh on-policy workers and random streams start at this phase boundary. This is not an exact simulator/sampler resume.'
        provenance[str(seed)]=dict(source=parent.relative_to(ROOT).as_posix(),source_phase=source_phase,restoration=restoration,
            source_native_frames=result['actual_training_frames'] if algorithm=='DDQN' else result['native_frames'])
    if algorithm=='PPO' and ppo_gae_lambda is None:
        if len(source_lambdas)!=1:raise ValueError('PPO continuation sources have different GAE settings')
        ppo_gae_lambda=source_lambdas.pop()
    write(root/(phase+'_provenance.json'),provenance)
    update_report(root,**{phase+'_protocol':dict(algorithm=algorithm,budget_per_seed=budget,checkpoints=points,
        focus_fraction=focus_fraction,focus_opponents=focus_opponents,ppo_gae_lambda=ppo_gae_lambda,provenance=provenance)})
    jobs=algorithm_jobs(spec,root,phase=phase,budget=budget,points=points,algorithms=(algorithm,),starts=starts,
        ppo_gae_lambda=ppo_gae_lambda,dispatch=pending_jobs is None)
    if pending_jobs is not None:pending_jobs.extend(jobs)
    return provenance


def representative_recovery(spec,root):
    """One fixed confirmation budget for observed two-seed late recovery."""
    required=focus_keys(spec);rule=spec['representative_curriculum']['recovery_confirmation']
    path=root/'representative_recovery_gate.json';result=read(path) if path.exists() else {}
    for algorithm in ('DDQN','PPO'):
        parents=[root/'representative'/str(seed)/algorithm/('summary.json' if algorithm=='DDQN' else 'result.json') for seed in spec['algorithm_comparison']['seeds']]
        if not all(p.exists() for p in parents):continue
        last=[algorithm_records(read(p),algorithm)[-1] for p in parents]
        if not all(required<=useful_conditions(p['evaluation']['rows']) for p in last):
            result[algorithm]=dict(eligible=False,reason='Both2m parents do not pass all four focus conditions',passed_on_both_seeds=False)
            write(path,result);continue
        phase=rule['phase_prefix']+algorithm
        continue_algorithms(spec,root,algorithm,source_phase='representative',phase=phase,
            budget=rule['additional_native_frames'],points=rule['checkpoints'])
        seeds={}
        for seed in spec['algorithm_comparison']['seeds']:
            points=continued_records(root,phase,seed,algorithm)[-3:]
            seeds[str(seed)]=dict(points=[dict(native_frames=p['native_frames'],checkpoint=p['checkpoint'],
                focus_results=len(required&useful_conditions(p['evaluation']['rows']))) for p in points],
                passed=len(points)==3 and all(required<=useful_conditions(p['evaluation']['rows']) for p in points))
        result[algorithm]=dict(eligible=True,source_phase=phase,seeds=seeds,
            passed_on_both_seeds=all(s['passed'] for s in seeds.values()))
        write(path,result);update_report(root,representative_recovery=result,stage='representative_recovery_analyzed')
    return result


def representative_expansion(spec,root):
    """Expand only after both seeds establish the declared local capability."""
    local=read(root/'representative_local_gate.json');comparison=analyze_algorithms(spec,root,phase='representative')
    recovery=read(root/'representative_recovery_gate.json') if (root/'representative_recovery_gate.json').exists() else {}
    if not all(method['complete'] for method in comparison.values()):raise RuntimeError('Complete both declared2m parent studies first')
    required=focus_keys(spec)
    for name,method in comparison.items():
        eligible_for_confirmation=all(required<=useful_conditions_from_labels(s['points'][-1]['useful']) for s in method['seeds'].values())
        if eligible_for_confirmation and not local[name]['passed_on_both_seeds'] and len(recovery.get(name,{}).get('seeds',{}))!=2:
            raise RuntimeError(f'Complete the declared recovery confirmation before protocol selection:{name}')
    eligible=[name for name in ('DDQN','PPO') if local.get(name,{}).get('passed_on_both_seeds') or recovery.get(name,{}).get('passed_on_both_seeds')]
    if not eligible:raise RuntimeError('Diagnose the failed local gate before expanding the task distribution')
    sources={name:recovery[name]['source_phase'] if recovery.get(name,{}).get('passed_on_both_seeds') else 'representative' for name in eligible}
    def score(name):
        p=[capability_point(continued_records(root,sources[name],seed,name)[-1]['evaluation'],
            read(root/'representative'/str(seed)/'initial_panel/summary.json'),0,'selection') for seed in spec['algorithm_comparison']['seeds']]
        useful=[v['cpu_ko']+sum(v['pvp_wins'].values()) for v in p]
        return min(useful),sum(useful),min(min(v['pvp_wins'].values()) for v in p),sum(v['cpu_ko'] for v in p),name=='DDQN'
    selected=max(eligible,key=score);rule=spec['representative_curriculum']['expansion'];phase='representative_expansion'
    focus=set(spec['representative_curriculum']['focus_opponents'])
    if rule.get('rehearsal_focus'):
        for seed in spec['algorithm_comparison']['seeds']:
            anchor=next(p['evaluation'] for p in continued_records(root,'representative',seed,selected) if p['requested']==1000000)
            won=useful_conditions(anchor['rows'])
            focus.update(r['opponent'] for r in anchor['rows'] if condition_key(r) in won)
    focus=sorted(focus)
    continue_algorithms(spec,root,selected,source_phase=sources[selected],phase=phase,
        budget=rule['additional_native_frames'],points=rule['checkpoints'],focus_fraction=rule['focus_sampling_fraction'],focus_opponents=focus)
    return analyze_coverage(spec,root,selected,phase)


def analyze_coverage(spec,root,selected,phase,*,coverage_origin=None):
    """One outcome audit for expanded training and its declared continuations."""
    coverage_origin=coverage_origin or phase;required=focus_keys(spec)
    provenance=read(root/(coverage_origin+'_provenance.json'))
    seeds={}
    for seed in spec['algorithm_comparison']['seeds']:
        source=root/'representative'/str(seed)
        records=continued_records(root,phase,seed,selected)
        parent=continued_records(root,provenance[str(seed)]['source_phase'],seed,selected)[-1]
        parent_useful=useful_conditions(parent['evaluation']['rows'])
        anchor=next(r['evaluation'] for r in records if r['phase']=='representative' and r['requested']==1000000)
        initial=learned_anchor(read(source/'initial_panel/summary.json'),anchor)
        combined=[];expansion_points=[];in_coverage=False
        for r in records:
            if r['phase']=='representative' and r['requested']<1000000:continue
            point=capability_point(r['evaluation'],initial,r['native_frames'],r['checkpoint'],r['optimizer_updates'])
            point['study_phase']=r['phase'];combined.append(point)
            in_coverage=in_coverage or r['phase']==coverage_origin
            if in_coverage:
                current_useful=useful_conditions(r['evaluation']['rows'])
                point['coverage_parent_useful']=len(parent_useful)
                point['coverage_parent_kept']=len(parent_useful&current_useful)
                point['coverage_parent_lost']=[v for v in parent['evaluation']['rows'] if condition_key(v) in parent_useful-current_useful]
                point['new_since_coverage_parent']=[v for v in point['new_useful'] if condition_key(v) not in parent_useful]
                point['fixed_focus_kept']=len(required&useful_conditions(r['evaluation']['rows']))
                point['fixed_focus_passed']=point['fixed_focus_kept']==len(required)
                extra=[v for v in point['new_useful'] if condition_key(v) not in required]
                point['new_after_expansion_outside_original_focus']=[v for v in extra if condition_key(v) not in parent_useful]
                expansion_points.append(dict(point,new_useful=extra))
        annotate_accumulation(combined)
        seeds[str(seed)]=dict(points=combined,small_panel_gate=persistent_capability_gate(combined,spec),
            gate=persistent_capability_gate(expansion_points,spec),expansion_parent=parent['checkpoint'])
    value=dict(algorithm=selected,coverage_origin=coverage_origin,seeds=seeds,passed_on_both_seeds=all(s['gate']['passed'] for s in seeds.values()),
        scope='Consumed development capability gate after curriculum expansion; not96CPU KO, unknown robustness or formal promotion.')
    write(root/(phase+'_analysis.json'),value);update_report(root,**{phase:value},
        stage='capability_gate_passed_requires_coverage' if value['passed_on_both_seeds'] else 'expansion_requires_targeted_diagnosis')
    return value


def coverage_recovery_eligible(result,spec):
    if result['passed_on_both_seeds']:return False
    required=focus_keys(spec)
    return len(result['seeds'])==2 and all(p['starting_kept']==p['starting_useful'] and p.get('fixed_focus_passed',False)
        and bool(useful_conditions_from_labels(p['new_useful'])-required)
        for p in (s['points'][-1] for s in result['seeds'].values()))


def confirm_coverage(spec,root,origin='representative_expansion'):
    """Confirm a measured final recovery once; never extend a missing skill."""
    if origin in ('priority','eligibility') or origin.startswith(('priority_','eligibility_')):
        kind='eligibility' if origin.startswith('eligibility') else 'priority'
        plan=spec['prospective_success_comparison' if kind=='eligibility' else 'priority_success_comparison']
        rule=plan['confirmation' if kind=='eligibility' else 'late_confirmation'];arms=plan['arms']
        compared={arm:audit_shared_parent(spec,root,kind+'_'+arm) for arm in arms}
        if any(v['passed_on_both_seeds'] for v in compared.values()):
            raise RuntimeError('A trial already passes persistently; freeze and validate it')
        for arm in arms:
            for seed in spec['algorithm_comparison']['seeds']:
                if not (root/(kind+'_'+arm)/str(seed)/'PPO/result.json').exists():
                    raise RuntimeError('Finish both matched trials before confirming an endpoint')
        eligible=[kind+'_'+arm for arm,result in compared.items()
            if all(s['points'][-1]['starting_kept']==s['points'][-1]['starting_useful']
                   and s['points'][-1]['new_useful'] for s in result['seeds'].values())]
        if origin!=kind and origin not in eligible or not eligible:
            raise RuntimeError('The stronger shared-parent floor and genuine gain must recover on BOTH seeds')
        origins=eligible if origin==kind else [origin];jobs=[]
        for source in origins:
            continue_algorithms(spec,root,'PPO',source_phase=source,phase=source+'_confirmation',
                budget=rule['maximum_additional_native_frames'],points=rule['checkpoints'],pending_jobs=jobs)
        if jobs:
            update_report(root,stage=kind+'_unchanged_recovery_confirmation_training',
                **{kind+'_confirmation':dict(origins=origins,budget_per_seed=rule['maximum_additional_native_frames'],
                    checkpoints=rule['checkpoints'],rule='One unchanged continuation after both endpoints restore all shared-parent results and genuine gains.')})
            run_training_jobs(jobs,root/(kind+'_confirmation_workers.json'))
        return {source:dict(original_anchor=analyze_coverage(spec,root,'PPO',source+'_confirmation',coverage_origin=source),
                           shared_parent=audit_shared_parent(spec,root,source+'_confirmation')) for source in origins}
    algorithm=read(root/(origin+'_analysis.json'))['algorithm']
    result=analyze_coverage(spec,root,algorithm,origin)
    if not coverage_recovery_eligible(result,spec):raise RuntimeError('No two-seed final recovery eligible for an unchanged continuation')
    rule=spec['coverage_confirmation'];phase=origin+'_confirmation'
    continue_algorithms(spec,root,algorithm,source_phase=origin,phase=phase,
        budget=rule['maximum_additional_native_frames'],points=rule['checkpoints'])
    return analyze_coverage(spec,root,algorithm,phase,coverage_origin=origin)


def credit_compare(spec,root):
    """A single diagnosed credit-horizon comparison in the existing PPO."""
    rule=spec['PPO_credit_comparison'];origin=rule['A_phase']
    ordinary=analyze_coverage(spec,root,'PPO',origin)
    if ordinary['passed_on_both_seeds']:raise RuntimeError('Ordinary PPO passed; proceed to coverage rather than another variant')
    if coverage_recovery_eligible(ordinary,spec):
        phase=origin+'_confirmation';path=root/(phase+'_analysis.json')
        if not path.exists():raise RuntimeError('Confirm the measured ordinary-PPO recovery first')
        if analyze_coverage(spec,root,'PPO',phase,coverage_origin=origin)['passed_on_both_seeds']:
            raise RuntimeError('The unchanged continuation passed; a credit variant is unnecessary')
    audit=read(root/'on_policy_credit/summary.json')
    if audit['games_count']!=8 or audit['optimizer_updates']!=0:
        raise RuntimeError('Complete the registered frozen on-policy diagnostic first')
    common=None;parent=rule['parent_phase']
    provenance=read(root/(origin+'_provenance.json'))
    for seed in rule['seeds']:
        cfg=read(root/origin/str(seed)/'configuration.json');expected=read(root/parent/str(seed)/'configuration.json')
        focus=cfg['learner']['training_focus']
        expected['learner']['training_focus']=copy.deepcopy(focus)
        if cfg!=expected or provenance[str(seed)]['source_phase']!=parent:
            raise ValueError('The longer-credit branch no longer has the matched3m parent and common curriculum')
        if common is not None and common!=focus:raise ValueError('Seeds use different curriculum distributions')
        common=focus
        if read(root/origin/str(seed)/'PPO/result.json')['arguments']['gae_lambda']!=rule['A_GAE_lambda']:
            raise ValueError('Ordinary control has a different GAE setting')
    update_report(root,credit_comparison_started=dict(hypothesis=rule['hypothesis'],common_parent=parent,
        A_phase=origin,B_phase=rule['B_phase'],native_frames_per_seed=rule['native_frames_per_seed'],
        changed_only='GAE lambda .95 to1; existing on-policy PPO, unchanged gamma/reward/network/curriculum'))
    continue_algorithms(spec,root,'PPO',source_phase=parent,phase=rule['B_phase'],
        budget=rule['native_frames_per_seed'],points=rule['checkpoints'],
        focus_fraction=common['fraction'],focus_opponents=common['opponents'],ppo_gae_lambda=rule['B_GAE_lambda'])
    return analyze_coverage(spec,root,'PPO',rule['B_phase'])


def prepare_success_consolidation(spec,root,rule=None):
    """A shared, verified self-success bank; no training or new opponents."""
    rule=spec['PPO_success_consolidation'] if rule is None else rule;results={}
    for seed in rule['seeds']:
        out=root/rule.get('preparation_phase','success_preparation')/str(seed);summary=out/'summary.json'
        if summary.exists():results[str(seed)]=read(summary);continue
        source=root/rule['parent_phase']/str(seed)
        record=algorithm_records(read(source/'PPO/result.json'),'PPO')[-1]
        cfg=read(source/'configuration.json')
        actual_source=rule.get('source_overrides',{}).get(str(seed),record['checkpoint'])
        baseline=read(ROOT/rule['source_evaluations'][str(seed)]) if rule.get('source_evaluations') else record['evaluation']
        actor=load_actor(ROOT/actual_source)
        if rule.get('optimizer_sources'):
            parent=torch.load(ROOT/rule['optimizer_sources'][str(seed)],map_location='cpu',weights_only=False)
            if any(not torch.equal(v,parent['state_dict'][k]) for k,v in actor.policy.state_dict().items() if k!='physical_mirror'):
                raise ValueError('A reconfigured parent changed trainable parameters before collection')
            trainable=construct_sequence_q(copy.deepcopy(actor.definition))
            trainable.load_state_dict(actor.policy.state_dict(),strict=True)
            restore_optimizer(trainable.optimizer,parent['optimizer_state'])
            actual_source=(out/'initial_actor.pt').relative_to(ROOT).as_posix()
            save(ROOT/actual_source,trainable,actor.definition,dict(purpose='Same parameters/Adam; explicit representation change, fresh workers/RNG, not full simulator resume',
                optimizer_source=rule['optimizer_sources'][str(seed)]),trainable.optimizer)
            del trainable
        observer=TrajectoryRecorder(cfg['pool'],actual_source)
        selected=rule.get('collection_conditions')
        if rule.get('collect_successes_only'):
            selected=[{k:r[k] for k in ('opponent','player','offset')} for r in baseline['rows']
                if (r['ko_win'] if r['opponent_kind']=='cpu' else r['result']=='win')]
        offsets=tuple(sorted({c['offset'] for c in selected})) if selected is not None else (0,)
        try:
            result=evaluate(actor,cfg['pool'] if selected is not None else cfg['development_pool'],out/'collection',
                cpu=False,offsets=offsets,conditions=selected,arena_call_sequence=True,progress=False,
                reward_kind=cfg['learner']['reward'],transition_observer=observer)
        finally:actor.close()
        fields=('opponent','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')
        previous={condition_key(r):r for r in baseline['rows']}
        repeated=[r for r in result['rows'] if condition_key(r) in previous]
        if any(tuple(r[k] for k in fields)!=tuple(previous[condition_key(r)][k] for k in fields) for r in repeated):
            raise RuntimeError('Independent parent success collection failed to reproduce its scored games')
        if selected is None and len(repeated)!=len(baseline['rows']):
            raise RuntimeError('The original parent success panel is incomplete')
        if result['native_frames']>rule.get('preparation_native_frame_cap_per_seed',float('inf')):
            raise RuntimeError('Independent success preparation exceeded its declared native-frame budget')
        bank=rule['bank'];replay=SequenceReplay(capacity=bank['capacity_transitions'],byte_capacity=bank['byte_capacity'],
            seed=seed+95003,burn=0,learn=bank['learn_length'],nstep=1)
        if rule.get('initial_bank_states'):
            saved=torch.load(ROOT/rule['initial_bank_states'][str(seed)],map_location='cpu',weights_only=False)
            replay.load_state_dict(saved['bank'] if 'bank' in saved else saved)
            replay.refresh_self_successes(observer.episodes,0,bank['maximum_episodes'],out.relative_to(ROOT).as_posix())
        else:
            for episode in observer.episodes:
                if successful_episode(episode):
                    if rule.get('soft_reference_collection'):
                        if 'policy_probabilities' not in episode:raise ValueError('Soft reference collection requires a real PPO distribution')
                        episode['retention_target']='physical_distribution'
                    replay.replace_verified_success(episode,0,bank['maximum_episodes'])
        replay.note_complete_results(baseline['rows'],prioritize_failures=rule.get('prioritize_current_failures',False))
        torch.save(replay.state_dict(),out/'bank.pt')
        value=dict(source=actual_source,native_frames=result['native_frames'],optimizer_updates=0,
            environment_calls=sum(len(e['actions']) for e in observer.episodes),episodes=len(replay.episodes),
            transitions=replay.transitions,bytes=replay.bytes,rows=result['rows'],
            purpose='Shared independent parent capability bank; consumed training conditions, not new discovery or holdout evidence.')
        write(summary,value);results[str(seed)]=value
    update_report(root,success_consolidation_preparation={seed:{k:v for k,v in value.items() if k!='rows'}
        for seed,value in results.items()});return results


def skill_curriculum(spec,root,name):
    """A declared role curriculum using the shared collector and PPO learner."""
    rule=spec['skill_curricula'][name];local=copy.deepcopy(spec)
    local['own_character']=rule['character'];local['algorithm_comparison']['seeds']=rule['seeds']
    prepared=prepare_success_consolidation(local,root,rule)
    starts={}
    for seed in rule['seeds']:
        parent=root/rule['parent_phase']/str(seed);out=root/rule['phase']/str(seed)
        source=(prepared[str(seed)]['source'] if rule.get('initial_source_from_preparation') else
            algorithm_records(read(parent/'PPO/result.json'),'PPO')[-1]['checkpoint'])
        cfg=read(parent/'configuration.json');cfg['own_character']=rule['character']
        cfg['network'].update(rule.get('network_updates',{}))
        cfg['learner']['training_focus']=dict(fraction=rule['focus_fraction'],opponents=rule['focus_opponents'])
        selected=rule['development_conditions'];ids={c['opponent'] for c in selected}
        cfg['development_pool']=[r for r in cfg['pool'] if r['id'] in ids]
        # The learner's checkpoint evaluator uses both ports and offset0 for
        # every selected rival. Require exactly that contract, not a partial
        # panel accidentally scored against the larger collection panel.
        expected={(r['id'],p,0) for r in cfg['development_pool'] for p in ((1,) if r['kind']=='cpu' else (1,2))}
        if expected!={(c['opponent'],c['player'],c['offset']) for c in selected}:
            raise ValueError('The declared curriculum development panel differs from the shared evaluator contract')
        baseline_rows=(read(ROOT/rule['source_evaluations'][str(seed)])['rows'] if rule.get('source_evaluations') else prepared[str(seed)]['rows'])
        rows=[r for r in baseline_rows if (r['opponent'],r['player'],r['offset']) in expected]
        if len(rows)!=len(expected):raise ValueError('The curriculum baseline misses a development condition')
        cfg['success_consolidation']=dict(bank=copy.deepcopy(rule['bank']),conditions=rule['collection_conditions'],
            coefficient=rule['auxiliary_coefficient'],initial_bank=(root/rule['preparation_phase']/str(seed)/'bank.pt').relative_to(ROOT).as_posix(),
            initial_outcomes=baseline_rows,capture_missing_checkpoint_successes=True,
            assessment_every_optimizer_updates=128,disagreement_fraction=.5,disagreement_scope='verified_success',
            **({'priority_fraction':rule['priority_fraction']} if rule.get('priority_fraction') else {}))
        if rule.get('initial_bank_overrides'):
            cfg['success_consolidation']['initial_bank']=rule['initial_bank_overrides'][str(seed)]
        if 'use_soft_references' in rule:cfg['success_consolidation']['use_soft_references']=rule['use_soft_references']
        for key in ('hard_failed_conditions','prioritize_current_failures','retemperature_initial_references'):
            if key in rule:cfg['success_consolidation'][key]=rule[key]
        if rule.get('additional_experience'):
            bank_spec=rule['bank'];bank=SequenceReplay(capacity=bank_spec['capacity_transitions'],byte_capacity=bank_spec['byte_capacity'],
                seed=seed+95003,burn=0,learn=bank_spec['learn_length'],nstep=1)
            bank.load_state_dict(torch.load(ROOT/cfg['success_consolidation']['initial_bank'],map_location='cpu',weights_only=False))
            old={condition_key(e) for e in bank.episodes.values()};added=[]
            extra=torch.load(ROOT/rule['additional_experience'],map_location='cpu',weights_only=False)['episodes']
            extra_conditions=rule.get('additional_conditions_by_seed',{}).get(str(seed),rule.get('additional_conditions'))
            for episode in extra:
                if extra_conditions is not None:
                    key=(episode['opponent_config']['id'],episode['player'],episode['offset'])
                    if key not in {(c['opponent'],c['player'],c['offset']) for c in extra_conditions}:continue
                elif episode['opponent_config']['id'] not in rule['additional_opponents']:continue
                expected_rival=next(r for r in cfg['pool'] if r['id']==episode['opponent_config']['id'])
                if episode['opponent_config']!=expected_rival:raise ValueError('Supplemental opponent version differs from training')
                if episode['character']!=rule['character']:raise ValueError('A curriculum demonstration uses the wrong own role')
                if not successful_episode(episode):raise ValueError('A priority tactic is not a complete independently won game')
                episode=copy.deepcopy(episode);episode['auxiliary_priority']=True
                bank.replace_verified_success(episode,0,bank_spec['maximum_episodes'])
                added.append(dict(opponent=episode['opponent_config'],source=episode['policy_version'],decisions=len(episode['actions'])))
            expected_count=len(extra_conditions) if extra_conditions is not None else len(rule['additional_opponents'])
            if len(added)!=expected_count:raise ValueError('The declared tactical experience is incomplete')
            if old-{condition_key(e) for e in bank.episodes.values()}:raise ValueError('Tactical merge removed an old success condition')
            bank_path=out/'initial_bank.pt';bank_path.parent.mkdir(parents=True,exist_ok=True);torch.save(bank.state_dict(),bank_path)
            cfg['success_consolidation']['initial_bank']=bank_path.relative_to(ROOT).as_posix()
            write(out/'tactical_preparation.json',dict(added=added,all_original_conditions_kept=True,episodes=len(bank.episodes),
                transitions=bank.transitions,bytes=bank.bytes,native_frames=0,source=rule['additional_experience'],
                scope='Reuse already independently learned full games; their acquisition and source-policy updates are accounted separately. Not on-policy PPO data.'))
        path=out/'configuration.json'
        if path.exists() and read(path)!=cfg:raise ValueError('The declared curriculum configuration changed after preparation')
        write(path,cfg);write(out/'pool.json',cfg['pool']);write(out/'initial_panel/summary.json',dict(rows=rows))
        starts[str(seed)]=dict(initial=ROOT/source)
        if 'action_temperature' in rule:starts[str(seed)]['temperature']=rule['action_temperature']
    update_report(root,**{name+'_decision':dict(character=rule['character'],hypothesis=rule['hypothesis'],
        reason=rule['reason'],budget_per_seed=rule['native_frames_per_seed'],checkpoints=rule['checkpoints'],
        source_phase=rule['parent_phase'],formal_promotion=False)},goal_achieved=False)
    algorithm_jobs(local,root,phase=rule['phase'],budget=rule['native_frames_per_seed'],points=rule['checkpoints'],
        algorithms=('PPO',),starts=starts,ppo_gae_lambda=.95)
    results={}
    for seed in rule['seeds']:
        out=root/rule['phase']/str(seed);run=read(out/'PPO/result.json');initial=read(out/'initial_panel/summary.json')
        points=[capability_point(p['evaluation'],initial,p['native_frames'],p['checkpoint']) for p in algorithm_records(run,'PPO')]
        annotate_accumulation(points)
        results[str(seed)]=dict(points=points,native_frames=run['native_frames'],optimizer_updates=run['optimizer_updates'],
            optimization_samples=run['optimization_samples'],seconds=run['seconds'],success_collection_native_frames=run['success_collection_native_frames'])
    write(root/(rule['phase']+'_analysis.json'),results);update_report(root,**{name+'_results':results},goal_achieved=False)
    return results


def success_consolidation(spec,root,prepare_only=False):
    rule=spec['PPO_success_consolidation'];credit=root/'long_credit_analysis.json'
    if not credit.exists():raise RuntimeError('Complete the matched longer-credit comparison first')
    longer=analyze_coverage(spec,root,'PPO','long_credit')
    if longer['passed_on_both_seeds']:raise RuntimeError('The current protocol passed; proceed to readiness rather than a new variant')
    if coverage_recovery_eligible(longer,spec):
        path=root/'long_credit_confirmation_analysis.json'
        if not path.exists() or analyze_coverage(spec,root,'PPO','long_credit_confirmation',coverage_origin='long_credit')['passed_on_both_seeds']:
            raise RuntimeError('Finish an eligible longer-credit confirmation first, or use its passing protocol')
    prepare_success_consolidation(spec,root)
    if prepare_only:return
    conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in
        read(root/rule['parent_phase']/str(rule['seeds'][0])/'initial_panel/summary.json')['rows']]
    for arm in rule['arms']:
        phase='success_'+arm
        auxiliary=dict(bank=rule['bank'],conditions=conditions,coefficient=0. if arm=='ordinary' else .1)
        # Run both arms together through the common job dispatcher. Preparing
        # configs is separated from launching so no branch receives extra CPU.
        for seed in rule['seeds']:
            parent=root/rule['parent_phase']/str(seed);destination=root/phase/str(seed)
            cfg=read(parent/'configuration.json');cfg['success_consolidation']=dict(auxiliary,
                initial_bank=(root/'success_preparation'/str(seed)/'bank.pt').relative_to(ROOT).as_posix())
            write(destination/'configuration.json',cfg);write(destination/'pool.json',cfg['pool'])
            write(destination/'initial_panel/summary.json',read(parent/'initial_panel/summary.json'))
        provenance={str(seed):dict(source_phase=rule['parent_phase'],source=algorithm_records(
            read(root/rule['parent_phase']/str(seed)/'PPO/result.json'),'PPO')[-1]['checkpoint'],
            restoration='Same frozen PPO weights and Adam, fresh workers/RNG; not exact simulator resume.') for seed in rule['seeds']}
        write(root/(phase+'_provenance.json'),provenance)
    jobs=[]
    for arm in rule['arms']:
        starts={str(seed):dict(initial=ROOT/algorithm_records(read(root/rule['parent_phase']/str(seed)/'PPO/result.json'),'PPO')[-1]['checkpoint']) for seed in rule['seeds']}
        jobs.extend(algorithm_jobs(spec,root,phase='success_'+arm,budget=rule['native_frames_per_arm_seed'],
            points=rule['checkpoints'],algorithms=('PPO',),starts=starts,ppo_gae_lambda=.95,dispatch=False))
    if jobs:run_training_jobs(jobs,root/'success_workers.json')
    return {arm:analyze_coverage(spec,root,'PPO','success_'+arm) for arm in rule['arms']}


def success_update_probe(source,bank,out,updates,cfg,*,conditions=None,loss_options=None,sample_sequences=4,retain_optimizer=False,transition_observer=None,trainable_scope='all',optimizer_start='saved'):
    """One bounded auxiliary diagnostic runner, separate from PPO advantages."""
    import time
    from experiments.ryu_direct_league.rehearsal import recurrent_success_loss
    policy=construct_sequence_q(copy.deepcopy(source['definition']));policy.load_state_dict(source['state_dict'])
    if optimizer_start not in ('saved','fresh'):raise ValueError('Unknown auxiliary optimizer initialization')
    if optimizer_start=='saved':restore_optimizer(policy.optimizer,source['optimizer_state'])
    if trainable_scope not in ('all','readout'):raise ValueError('Unknown auxiliary trainable scope')
    if trainable_scope=='readout':
        for name,parameter in policy.named_parameters():parameter.requires_grad_(name.startswith('action_net.'))
    gradient_groups={group:[p for name,p in policy.named_parameters() if name.startswith(prefix)]
        for group,prefix in (('visual','features_extractor.visual'),('coordinates','features_extractor.coordinates'),
            ('input_history','input_duration_projection'),('recurrent','sequence_gru'),('readout','action_net'))}
    gradient_totals={group:0. for group in gradient_groups}
    losses=[];norms=[];samples=0;started=time.monotonic();assessment_seconds=0.
    sampling_before=copy.deepcopy(bank.auxiliary_sampling_totals)
    for step in range(updates):
        if (loss_options or {}).get('disagreement_fraction',0.) and step%128==0:
            before=time.monotonic();bank.assess_success_disagreements(policy,step)
            assessment_seconds+=time.monotonic()-before
        policy.set_training_mode(True);loss,n=recurrent_success_loss(policy,bank,sample_sequences,16,**(loss_options or {}))
        policy.optimizer.zero_grad();(.1*loss).backward()
        for group,parameters in gradient_groups.items():
            gradient_totals[group]+=sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None)**.5
        norms.append(float(torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)));policy.optimizer.step()
        losses.append(float(loss.detach()));samples+=n
        if (step+1)%128==0 or step+1==updates:
            write(out/'learning_progress.json',dict(optimizer_updates=step+1,declared_updates=updates,
                auxiliary_samples=samples,recent_mean_loss=float(np.mean(losses[-128:])),
                seconds=time.monotonic()-started,scope='Optimization progress; complete games determine capability.'))
    optimization_seconds=time.monotonic()-started
    save(out/'agent.pt',policy,source['definition'],dict(purpose='Separate action-supervision stage; no PPO/TD updates',
         updates=updates,trainable_scope=trainable_scope,optimizer_start=optimizer_start),
         policy.optimizer if retain_optimizer else None)
    actor=Actor(policy,source['definition'])
    try:
        tested=evaluate(actor,cfg['development_pool'],out/'complete_games',cpu=False,offsets=(0,),conditions=conditions,
            arena_call_sequence=True,progress=False,reward_kind=cfg['learner']['reward'],transition_observer=transition_observer)
    finally:actor.close()
    return dict(updates=updates,auxiliary_samples=samples,
        actual_sampling_by_group={group:{key:value-sampling_before.get(group,{}).get(key,0)
            for key,value in counts.items()} for group,counts in bank.auxiliary_sampling_totals.items()},
        first_loss=losses[0] if losses else None,last_loss=losses[-1] if losses else None,
        mean_gradient_norm=float(np.mean(norms)) if norms else None,maximum_gradient_norm=max(norms) if norms else None,
        optimization_seconds=optimization_seconds,evaluation_seconds=tested['seconds'],
        replay_assessment_seconds=assessment_seconds,
        independent_optimizer_snapshot=True,trainable_scope=trainable_scope,optimizer_start=optimizer_start,
        mean_preclip_gradient_norm_by_component={g:value/max(1,updates) for g,value in gradient_totals.items()},
        rows=tested['rows'],evaluation_native_frames=tested['native_frames'])


def demonstration_learnability(spec,root,variant=None):
    """Use verified search experience in the existing auxiliary learning loop."""
    from experiments.ryu_counterq.replay import full_values,same_episode_condition,audit_success_history_labels
    from experiments.ryu_counterq.training_state import random_state,restore_random_state
    rule=copy.deepcopy(spec['demonstration_learning'])
    if variant is not None:rule.update(rule['variants'][variant])
    phase=rule.get('phase','demonstration_learning');out=root/phase;out.mkdir(exist_ok=True)
    prepared=None
    if not rule.get('initial_bank_state') or rule.get('append_verified_demonstrations',False):
        prepared=torch.load(root/rule.get('verified_directory','verified_demonstrations')/'verified_episodes.pt',map_location='cpu',weights_only=False)
        if rule.get('demonstration_conditions'):
            prepared=dict(prepared,episodes=[e for e in prepared['episodes'] if any(
                all((e['opponent_config']['id'] if key=='opponent' else e.get(key))==value for key,value in case.items())
                for case in rule['demonstration_conditions'])])
        if not prepared['episodes']:raise RuntimeError('No independently replayed winning control sequences to learn')
    bank_spec=rule['bank'];bank=SequenceReplay(capacity=bank_spec['capacity_transitions'],byte_capacity=bank_spec['byte_capacity'],
        seed=rule['seed'],burn=0,learn=16,nstep=1)
    if rule.get('initial_bank_state'):
        saved=torch.load(ROOT/rule['initial_bank_state'],map_location='cpu',weights_only=False)
        bank.load_state_dict(saved['bank'] if 'bank' in saved else saved)
        if 'bank' in saved and 'rng' in saved:restore_random_state(saved['rng'])
        if rule.get('initial_sampling_seed') is not None:
            bank.rng=np.random.default_rng(rule['initial_sampling_seed'])
            write(out/'sampling_initialization.json',dict(seed=rule['initial_sampling_seed'],
                scope='Explicit new independent auxiliary stream; same dataset, not an exact training resume.'))
        if rule.get('append_verified_demonstrations',False):
            for episode in prepared['episodes']:
                duplicate=any(same_episode_condition(episode,e) and all(np.array_equal(episode[k],e[k]) for k in ('actions','frames','positions')) for e in bank.episodes.values())
                if not duplicate:
                    episode=copy.deepcopy(episode)
                    if rule.get('prioritize_appended_demonstrations',False):episode['auxiliary_priority']=True
                    bank.add_verified_demonstration(episode,replace_condition=rule.get('replace_appended_conditions',False))
    else:
        for episode in prepared['episodes']:bank.add_verified_demonstration(episode)
    if 'priority_opponents' in rule:
        for number,episode in bank.episodes.items():
            episode['auxiliary_priority']=episode['opponent_config']['id'] in rule['priority_opponents']
            bank.retention_usage[number]['auxiliary_priority']=episode['auxiliary_priority']
    for episode in bank.episodes.values():
        if episode['opponent_config']['id'] not in rule.get('full_verified_prefix_opponents',[]):continue
        if (not successful_episode(episode) or not episode.get('controlled_replay_equal')
                or not episode.get('validated_complete')):
            raise ValueError('Full-prefix supervision requires a physically replayed complete winning branch')
        episode['original_supervision_start']=episode.get('supervision_start',0)
        episode['supervision_start']=0
        episode['prefix_label_origin']='Actual source-policy prefix of the same physically verified complete win; not teacher/oracle actions.'
    if rule.get('reset_phase_usage',False):
        for use in bank.retention_usage.values():
            use['phase_auxiliary_transitions']=0;use['phase_actual_disagreements_trained']=0
            use['phase_assessed_frontier_windows']=0
    if rule.get('initial_self_episodes'):
        actual=torch.load(ROOT/rule['initial_self_episodes'],map_location='cpu',weights_only=False)['episodes']
        if any(e['policy_version']!=rule['initial'] for e in actual):raise ValueError('Initial self-data uses a different frozen actor')
        changes=bank.refresh_self_successes(actual,0,rule.get('maximum_success_episodes',112),phase)
        write(out/'initial_self_refresh.json',dict(changes=changes,native_frames=0,
            acquisition=rule['initial_self_episodes'],scope='Reuse already recorded autonomous complete source games; auxiliary bank native clock stays zero.'))
    loss_options=rule.get('loss_options',{})
    preflight=audit_success_history_labels(bank.episodes.values(),
        use_soft_references=loss_options.get('use_soft_references',False),
        hard_failed_conditions=loss_options.get('hard_failed_conditions',False))
    write(out/'label_preflight.json',preflight)
    if preflight['both_hard_target_conflicts']:
        raise ValueError('Conflicting hard labels after identical legal histories; verify a coherent full-game response before fitting')
    from experiments.league_adaptive.evaluate import exact_opponent_pool
    cfg=read(ROOT/spec['base_config'])
    cfg['pool']=exact_opponent_pool(cfg['pool'],[e['opponent_config'] for e in bank.episodes.values()])
    cfg['development_pool']=[r for r in cfg['pool'] if r['id'] in rule['evaluation_opponents']]
    initial=rule.get('initial',spec['source'])
    source=torch.load(ROOT/initial,map_location='cpu',weights_only=False);previous=rule.get('starting_updates',0);points=[]
    if rule.get('visual_normalization') is not None:
        if rule['visual_normalization']!='layer':raise ValueError('Unknown declared visual conditioning')
        from experiments.ryu_counterq.policy import initialize_visual_normalization
        rng=random_state()
        policy,definition,conversion=initialize_visual_normalization(source)
        save(out/'initial_actor.pt',policy,definition,dict(purpose='Explicit visual-conditioning research initialization',source=initial),policy.optimizer)
        write(out/'initialization.json',dict(source=initial,**conversion))
        source=torch.load(out/'initial_actor.pt',map_location='cpu',weights_only=False)
        restore_random_state(rng)
    for threshold in rule['optimizer_checkpoints']:
        directory=out/str(threshold);path=directory/'summary.json'
        if path.exists():
            record=read(path);saved=torch.load(directory/'bank_state.pt',map_location='cpu',weights_only=False)
            bank.load_state_dict(saved['bank']);restore_random_state(saved['rng'])
        else:
            directory.mkdir(exist_ok=True)
            observer=TrajectoryRecorder(cfg['development_pool'],directory.relative_to(ROOT).as_posix()+'/agent.pt')
            record=success_update_probe(source,bank,directory,threshold-previous,cfg,
                sample_sequences=min(4,len(bank.episodes)),retain_optimizer=True,transition_observer=observer,
                loss_options=rule.get('loss_options'),trainable_scope=rule.get('trainable_scope','all'),
                optimizer_start=rule.get('optimizer_start','saved') if previous==rule.get('starting_updates',0) else 'saved')
            learned=torch.load(directory/'agent.pt',map_location='cpu',weights_only=False)
            policy=construct_sequence_q(learned['definition']);policy.load_state_dict(learned['state_dict']);policy.set_training_mode(False);fits=[]
            for episode in bank.episodes.values():
                scores=full_values(policy,episode)[:-1];actions=torch.as_tensor(episode['actions'])
                minimum=episode.get('supervision_start',0)
                wrong=torch.nonzero(scores.argmax(1).cpu()!=actions).flatten();wrong=wrong[wrong>=minimum]
                actual=next((e for e in observer.episodes if same_episode_condition(e,episode)),None)
                differences=(np.empty(0,np.int64) if actual is None else
                    np.flatnonzero(actual['actions'][:min(len(actual['actions']),len(actions))]!=episode['actions'][:min(len(actual['actions']),len(actions))]))
                first=int(differences[0]) if len(differences) else None
                fits.append(dict(opponent=episode['opponent_config']['id'],teacher_decisions=len(actions)-minimum,
                    opponent_config=episode['opponent_config'],player=episode['player'],offset=episode['offset'],
                    history_total_decisions=len(actions),supervision_start=minimum,
                    fitted_actions=len(actions)-minimum-len(wrong),first_teacher_history_mismatch=int(wrong[0]) if len(wrong) else None,
                    teacher_history_cross_entropy=float(torch.nn.functional.cross_entropy(scores[minimum:],actions[minimum:])),
                    first_independent_action_difference=first,
                    same_legal_prefix_before_difference=None if first is None else bool(np.array_equal(actual['frames'][:first+4],episode['frames'][:first+4]) and np.array_equal(actual['positions'][:first+1],episode['positions'][:first+1])),
                    independent_result=None if actual is None else actual['result'],independent_ko=None if actual is None else actual['ko_win'],
                    source=episode['policy_version'],auxiliary_priority=bool(episode.get('auxiliary_priority',False))))
            record.update(cumulative_updates=threshold,teacher_history_fit=fits,actual_auxiliary_use=copy.deepcopy(bank.retention_usage),
                declaration='Pure action-supervision learnability, no TD/return loss or PPO advantages. Research only; old skills may change.')
            if rule.get('refresh_current_successes',False):
                record['self_success_refresh']=bank.refresh_self_successes(observer.episodes,0,
                    rule.get('maximum_success_episodes',112),phase)
                bank.note_complete_results(record['rows'])
                record['self_refresh_scope']='Only autonomous full greedy wins replace coaches; physical soft targets can change after verified wins. Fits above describe the pre-refresh teachers.'
            torch.save(dict(bank=bank.state_dict(),rng=random_state()),directory/'bank_state.pt')
            torch.save(dict(episodes=observer.episodes),directory/'actual_episodes.pt')
            write(path,record)
        points.append(record);previous=threshold
        source=torch.load(directory/'agent.pt',map_location='cpu',weights_only=False)
        write(out/'progress.json',dict(points=points,complete_updates=threshold))
        print('DEMONSTRATION_LEARNING',threshold,[(f['opponent'],f['fitted_actions'],f['teacher_decisions'],f['independent_ko']) for f in record['teacher_history_fit']],flush=True)
    value=dict(points=points,source=initial,total_optimizer_updates=previous-rule.get('starting_updates',0),
        evaluation_native_frames=sum(p['evaluation_native_frames'] for p in points),
        success_source='Verified continuous experience with per-episode provenance: autonomous wins and explicitly marked controlled suffixes; not oracle labels',goal_complete=False)
    if value['evaluation_native_frames']>rule['evaluation_native_frame_cap']:raise RuntimeError('Learning evaluation budget exceeded')
    write(out/'summary.json',value)
    if variant is None:update_report(root,**{phase:value},goal_achieved=False)
    return value


def compare_demonstration_learning(spec,root,comparison=None):
    """Dispatch one declared loss-weighting pair through the same learner."""
    jobs=[]
    rule=spec['demonstration_learning']
    names=rule['comparisons'][comparison] if comparison is not None else rule.get('comparison_variants',list(rule['variants']))
    for name in names:
        rule=spec['demonstration_learning']['variants'][name]
        if (root/rule['phase']/'summary.json').exists():continue
        jobs.append(dict(name=name,command=[sys.executable,'-m','experiments.ryu_counterq.capability',
            '--config',str(ROOT/'configs'/f"{spec['experiment']}.json"),'--stage','demonstration-learning',
            '--experiment',name],log=root/(rule['phase']+'.log')))
    batches=([job] for job in jobs) if spec['algorithm_comparison'].get('serial_training_jobs',False) else [jobs]
    for batch in batches:
        if batch:run_training_jobs(batch,root/'demonstration_comparison_workers.json')
    results={name:read(root/spec['demonstration_learning']['variants'][name]['phase']/'summary.json') for name in names}
    field='demonstration_learning_comparison' if comparison is None else comparison+'_comparison'
    full_record=root/(field+'.json');write(full_record,results)
    compact={name:{key:result[key] for key in ('source','total_optimizer_updates','evaluation_native_frames')}
        for name,result in results.items()}
    update_report(root,**{field:dict(variants=compact,full_record=full_record.relative_to(ROOT).as_posix())},goal_achieved=False)


def prepare_student_consolidation(spec,root,variant=None):
    """Refresh the actual learned skills, not just their original coach labels."""
    from experiments.ryu_counterq.replay import same_episode_condition
    rule=spec['student_success_consolidation'] if variant is None else spec['independent_skill_refresh'][variant]
    out=root/rule.get('phase','student_consolidation_preparation')
    if (out/'summary.json').exists():return read(out/'summary.json')
    out.mkdir(exist_ok=True)
    original_directory=rule.get('original_directory','verified_aggregated_demonstrations')
    originals=[] if original_directory is None else torch.load(root/original_directory/'verified_episodes.pt',map_location='cpu',weights_only=False)['episodes']
    base=read(ROOT/spec['base_config']);episodes=[];rows=[];spent=0;duplicates=[]
    replacements={(json.dumps(next(r for r in base['pool'] if r['id']==case['opponent']),sort_keys=True),int(case['player']),int(case['offset']))
        for case in rule['independent_successes']}
    retained=[e for e in originals if condition_key(e) not in replacements or e.get('supervision_start',0)>0]
    for case in rule['independent_successes']:
        pool=[next(r for r in base['pool'] if r['id']==case['opponent'])]
        actor=load_actor(ROOT/case['agent']);observer=TrajectoryRecorder(pool,case['agent'])
        try:
            tested=evaluate(actor,pool,out/f"{case['opponent']}_P{case['player']}_{case['offset']}",cpu=False,ports=(case['player'],),offsets=(case['offset'],),
                arena_call_sequence=True,progress=False,reward_kind=base['learner']['reward'],transition_observer=observer)
        finally:actor.close()
        row=tested['rows'][0]
        if any(row[k]!=value for k,value in case['expected'].items()):raise RuntimeError('Learned success did not independently reproduce')
        episode=observer.episodes[0]
        if not successful_episode(episode) or not np.array_equal(episode['actions'],episode['greedy_actions']):
            raise RuntimeError('Only genuine independent greedy success can refresh the skill bank')
        episode.update(independent_learned_control=True,supervision_only=True,supervision_start=0)
        if any(same_episode_condition(episode,e) and all(np.array_equal(episode[k],e[k]) for k in ('actions','frames','positions')) for e in episodes):
            raise RuntimeError('The same deterministic trajectory cannot be counted again')
        episodes.append(episode);rows.append(row);spent+=tested['native_frames']
    for episode in retained:
        previous=next((e for e in episodes if same_episode_condition(episode,e) and all(np.array_equal(episode[k],e[k]) for k in ('actions','frames','positions'))),None)
        if previous is not None:
            duplicates.append(dict(source=episode['policy_version'],same_trajectory_as=previous['policy_version'],
                supervision_start=episode.get('supervision_start',0)))
        else:episodes.append(episode)
    if spent>rule['maximum_native_frames']:raise RuntimeError('Independent success refresh budget exceeded')
    torch.save(dict(episodes=episodes),out/'verified_episodes.pt')
    value=dict(rows=rows,native_frames=spent,episodes=len(episodes),dependent_duplicate_trajectories_not_added=duplicates,
        episode_sources=[dict(opponent=e['opponent_config'],source=e['policy_version'],supervision_start=e.get('supervision_start',0),
            independent_learned_control=e.get('independent_learned_control',False),decisions=len(e['actions'])) for e in episodes],
        rule='Prefer actual learned independent wins over original same-condition root0scripts; retain verified student-visited suffixes and their real unsupervised prefixes.')
    write(out/'summary.json',value)
    update_report(root,**{('student_success_preparation' if variant is None else rule['phase']):value})
    return value


def guile_tactical_online(spec,root):
    """Return discovered skills to strong parents via the existing PPO loop."""
    rule=spec['guile_tactical_online'];local=copy.deepcopy(spec);local['own_character']='Guile'
    local['algorithm_comparison']['seeds']=rule['seeds'];starts={};preparations={}
    for seed in rule['seeds']:
        out=root/rule['phase']/str(seed);proof=out/'preparation.json'
        if proof.exists():preparations[str(seed)]=read(proof)
        else:
            source_phase=ROOT/rule['old_phase']/str(seed)
            original=read(source_phase/'configuration.json');bank_spec=original['success_consolidation']['bank']
            saved=torch.load(source_phase/'PPO/success_bank.pt',map_location='cpu',weights_only=False)
            clock=int(saved.get('native_frames',0))
            bank=SequenceReplay(capacity=bank_spec['capacity_transitions'],byte_capacity=bank_spec['byte_capacity'],
                seed=seed+95003,burn=0,learn=bank_spec['learn_length'],nstep=1)
            bank.load_state_dict(saved)
            required_conditions={condition_key(e) for e in bank.episodes.values()}
            curated=torch.load(root/'student_consolidation_preparation/verified_episodes.pt',map_location='cpu',weights_only=False)['episodes']
            for episode in curated:
                if episode.get('independent_learned_control',False):
                    bank.replace_verified_success(episode,clock,bank_spec['maximum_episodes'])
                elif not any(same_episode_condition(episode,e) and all(np.array_equal(episode[k],e[k]) for k in ('actions','frames','positions')) for e in bank.episodes.values()):
                    bank.add_verified_demonstration(episode,clock)
            retired_coaches=[]
            for number,episode in list(bank.episodes.items()):
                controlled=episode.get('controlled_replay_equal') and not episode.get('independent_learned_control',False)
                mastered=any(same_episode_condition(episode,e) and e.get('independent_learned_control',False) for e in bank.episodes.values())
                if controlled and mastered:
                    bank.retention_usage[number]['retired_frame']=clock;bank._remove(number);retired_coaches.append(number)
            if required_conditions-{condition_key(e) for e in bank.episodes.values()}:
                raise ValueError('Initial tactical merge silently lost prior success conditions')
            if len(bank.episodes)>bank_spec['maximum_episodes']:raise ValueError('Merged tactical bank exceeds its declared trajectory cap')
            for use in bank.retention_usage.values():
                use['phase_auxiliary_transitions']=0;use['phase_actual_disagreements_trained']=0
            out.mkdir(parents=True,exist_ok=True);bank_path=out/'initial_bank.pt'
            torch.save(dict(bank.state_dict(),native_frames=clock),bank_path)
            cfg={k:copy.deepcopy(original[k]) for k in ('network','learner','pool')}
            cfg.update(experiment=spec['experiment'],protocol='tactical_online_PPO',own_character='Guile')
            cfg['learner']['training_focus']=dict(fraction=rule['focus_fraction'],opponents=rule['focus_opponents'])
            cfg['development_pool']=[r for r in cfg['pool'] if r['id'] in spec['demonstration_learning']['evaluation_opponents']]
            initial=read(root/'demonstration_student_visited/3072/complete_games/summary.json')
            # That diagnostic student is not this strong initialization. Its
            # result is only the common condition list, never the baseline.
            conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in initial['rows']]
            old_conditions=original['success_consolidation']['conditions']
            collect=conditions+[c for c in old_conditions if c not in conditions]
            prioritized=[c for name in rule['refresh_priority'] for c in collect if c['opponent']==name]
            collect=prioritized+[c for c in collect if c not in prioritized]
            cfg['success_consolidation']=dict(bank=copy.deepcopy(bank_spec),conditions=collect,coefficient=.1,
                initial_bank=bank_path.relative_to(ROOT).as_posix(),initial_bank_clock=clock,resume_bank_sampling=True,
                initial_outcomes=[],capture_missing_checkpoint_successes=True,assessment_every_optimizer_updates=128,
                disagreement_fraction=.5,disagreement_scope='verified_success')
            cfg['success_consolidation']['bank']['refresh']='Bounded independent successes from training conditions; actual mastery consolidates conditional coach versions. Collection frames enter the training budget.'
            write(out/'configuration.json',cfg);write(out/'pool.json',cfg['pool'])
            source=ROOT/rule['parents'][str(seed)];actor=load_actor(source)
            try:
                baseline=evaluate(actor,cfg['development_pool'],out/'initial_panel',cpu=False,offsets=(0,),
                    arena_call_sequence=True,progress=False,reward_kind=cfg['learner']['reward'])
            finally:actor.close()
            value=dict(source=rule['parents'][str(seed)],seed=seed,character='Guile',
                baseline_native_frames=baseline['native_frames'],bank_episodes=len(bank.episodes),bank_transitions=bank.transitions,
                bank_bytes=bank.bytes,bank_clock=clock,coached_episodes=sum(bool(e.get('controlled_replay_equal')) for e in bank.episodes.values()),
                redundant_conditional_coaches_retired=retired_coaches,all_prior_bank_conditions_retained=True,
                initial_useful_conditions=len(useful_conditions(baseline['rows'])),
                lineage='Strong35KOparent weights/Adam initialize real PPO; offline probes supply data only. Fresh workers/RNG; no exact simulator resume claim.')
            write(proof,value);preparations[str(seed)]=value
        starts[str(seed)]=dict(initial=ROOT/rule['parents'][str(seed)])
    update_report(root,guile_online_preparation=preparations,goal_achieved=False)
    algorithm_jobs(local,root,phase=rule['phase'],budget=rule['native_frames_per_seed'],points=rule['checkpoints'],
        algorithms=('PPO',),starts=starts,ppo_gae_lambda=.95)
    results={}
    for seed in rule['seeds']:
        directory=root/rule['phase']/str(seed);initial=read(directory/'initial_panel/summary.json')
        run=read(directory/'PPO/result.json')
        points=[capability_point(p['evaluation'],initial,p['native_frames'],p['checkpoint']) for p in algorithm_records(run,'PPO')]
        annotate_accumulation(points)
        results[str(seed)]=dict(points=points,native_frames=run['native_frames'],optimizer_updates=run['optimizer_updates'],
            optimization_samples=run['optimization_samples'],seconds=run['seconds'],success_collection_native_frames=run['success_collection_native_frames'])
    write(root/'guile_online_analysis.json',results);update_report(root,guile_online_results=results,goal_achieved=False)


def character_algorithm_panel(spec,root,name=None):
    """One shared consumed panel for registered frozen or untrained cells."""
    rule=spec['primary_matrix'] if name is None else spec['comparative_panels'][name]
    output=rule.get('phase','primary_matrix');algorithms=rule.get('algorithms',('DDQN','PPO'))
    base=read(ROOT/spec['base_config']);conditions=[]
    for rival in base['pool']:
        if rival['kind']=='cpu':conditions.append(dict(opponent=rival['id'],player=1,offset=0))
        elif rival['id'] in rule['pvp_opponents']:
            conditions.extend(dict(opponent=rival['id'],player=p,offset=o) for p in (1,2) for o in (0,12))
    if len(conditions)!=112:raise ValueError('The registered matrix is not the exact112condition panel')
    write(root/(output+'_conditions.json'),conditions);results=[];pending=[]
    for character,source in rule['source_roots'].items():
        for algorithm in algorithms:
            for seed in spec['algorithm_comparison']['seeds']:
                directory=ROOT/source/str(seed)/algorithm
                delivered=next((c for c in rule.get('delivered_controls',[])
                    if (c['character'],c['algorithm'],c['seed'])==(character,algorithm,seed)),None)
                if delivered is not None:
                    path=ROOT/delivered['delivered']
                    if not path.is_file():raise FileNotFoundError(path)
                    record=dict(checkpoint=delivered['delivered'],native_frames=delivered['actual_native_frames'])
                elif algorithm=='untrained':
                    path=directory.parent/'initial_ddqn.pt'
                    if not path.exists():pending.append([character,algorithm,seed]);continue
                    record=dict(checkpoint=path.relative_to(ROOT).as_posix(),native_frames=0)
                else:
                    history=directory/('checkpoints.json' if algorithm=='DDQN' else 'frozen_evaluations.json')
                    if history.exists():
                        raw=read(history);records=algorithm_records(dict(curve=raw) if algorithm=='DDQN' else dict(frozen_evaluations=raw),algorithm)
                    else:
                        completed=directory/('summary.json' if algorithm=='DDQN' else 'result.json')
                        if not completed.exists():pending.append([character,algorithm,seed]);continue
                        records=algorithm_records(read(completed),algorithm)
                    record=next((r for r in records if r['requested']==rule['native_frame_checkpoint']),None)
                    if record is None:pending.append([character,algorithm,seed]);continue
                out=root/output/character/algorithm/str(seed)
                if (out/'summary.json').exists():tested=read(out/'summary.json')
                else:
                    actor=load_actor(ROOT/record['checkpoint'])
                    try:
                        if actor.definition.get('character')!=character:raise ValueError('A matrix cell has the wrong actual own role')
                        kind='sequence_ppo' if algorithm=='PPO' else 'sequence_double_q'
                        if actor.definition['kind']!=kind or len(actor.actions)!=125:
                            raise ValueError('The primary cell is not the registered fresh learner/action protocol')
                        tested=evaluate(actor,base['pool'],out,cpu=False,ports=(1,2),offsets=(0,12),conditions=conditions,
                            arena_call_sequence=True,progress=False,reward_kind=base['learner']['reward'])
                    finally:actor.close()
                if tested['native_frames']>rule['evaluation_native_frame_cap_per_model']:raise RuntimeError('Matrix evaluation budget exceeded')
                cpu=[r for r in tested['rows'] if r['opponent_kind']=='cpu']
                value=dict(character=character,algorithm=algorithm,seed=seed,source=record['checkpoint'],
                    actual_inherited_training_frames=record['native_frames'],nominal_checkpoint=rule['native_frame_checkpoint'],
                    CPU_KO=sum(r['ko_win'] for r in cpu),CPU_conditions=len(cpu),CPU_time_wins=sum(r['time_up'] and r['result']=='win' for r in cpu),
                    CPU_KO_by_star=[sum(r['ko_win'] for r in cpu if r['cpu_stars']==s) for s in range(1,9)],
                    PvP=tested['overall'],by_port=tested['by_port'],evaluation_native_frames=tested['native_frames'],
                    complete_records=(out/'summary.json').relative_to(ROOT).as_posix())
                results.append(value);print('PRIMARY_MATRIX',character,algorithm,seed,value['CPU_KO'],tested['overall']['wins'],flush=True)
                write(root/(output+'_progress.json'),dict(results=results,pending=pending))
    value=dict(results=results,pending=pending,complete=len(results)==len(rule['source_roots'])*len(algorithms)*len(spec['algorithm_comparison']['seeds']),conditions=112,
        scope=rule.get('scope','Previously consumed training/development conditions. Same fresh3m protocol per algorithm/role, two paired seeds. HistoricalRyu and laterGuile supervised candidates are not cells in this causal table.'))
    write(root/(output+'_analysis.json'),value);update_report(root,**{output:value},goal_achieved=False)
    return value


def representation_probe(spec,root):
    """Frozen weight/input-view comparison through the existing evaluator."""
    import time
    rule=spec['facing_normalization_probe'];base=read(ROOT/spec['base_config'])
    out=root/'facing_normalization_probe';out.mkdir(exist_ok=True)
    results=[];spent=cached_frames=0;pending=[];started=time.monotonic()
    for item in rule['sources']:
        original=torch.load(ROOT/item['source'],map_location='cpu',weights_only=False)
        for enabled in rule['variants']:
            directory=out/item['name']/('canonical' if enabled else 'absolute');path=directory/'agent.pt'
            complete=directory/'complete_games/summary.json'
            if complete.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['training']['source']!=item['source'] or saved['definition'].get('facing_canonical',False)!=enabled:
                    raise ValueError('A cached representation has different provenance')
                tested=read(complete);cached_frames+=tested['native_frames']
            else:
                if spent+7200*len(rule['conditions'])>rule['native_frame_cap']:
                    pending.append(dict(source=item['name'],facing_canonical=enabled));continue
                definition=copy.deepcopy(original['definition']);definition['facing_canonical']=enabled
                policy=construct_sequence_q(definition);state=copy.deepcopy(original['state_dict'])
                state['physical_mirror']=policy.physical_mirror.clone();policy.load_state_dict(state,strict=True)
                save(path,policy,definition,dict(purpose='Zero-SGD facing representation probe',source=item['source'],facing_canonical=enabled))
                actor=load_actor(path)
                try:tested=evaluate(actor,base['pool'],directory/'complete_games',cpu=False,offsets=(0,12),conditions=rule['conditions'],
                    arena_call_sequence=True,progress=False,reward_kind=base['learner']['reward'])
                finally:actor.close()
            spent+=tested['native_frames']
            results.append(dict(source=item['name'],original=item['source'],view=path.relative_to(ROOT).as_posix(),
                facing_canonical=enabled,rows=tested['rows'],native_frames=tested['native_frames']))
            write(out/'summary.json',dict(results=results,not_completed=pending,native_frames=spent,
                cached_native_frames=cached_frames,additional_native_frames=spent-cached_frames,
                seconds=time.monotonic()-started,optimizer_updates=0,scope=rule['preserves']))
            print('FACING_PROBE',item['name'],enabled,[(r['opponent'],r['player'],r['offset'],r['result'],r['self_hp'],r['opponent_hp']) for r in tested['rows']],flush=True)
    if spent>rule['native_frame_cap']:raise RuntimeError('Facing representation budget exceeded')
    return read(out/'summary.json')


def success_freshness_probe(spec,root):
    """Controlled use of the shared auxiliary loss, not another RL learner."""
    rule=spec['success_freshness_probe'];seed=rule['seed'];out=root/'success_freshness_probe_independent'
    if (out/'summary.json').exists():return read(out/'summary.json')
    run=root/'success_success_consolidation'/str(seed)/'PPO';cfg=read(run.parent/'configuration.json')
    records=read(run/'frozen_evaluations.json');student=next(e for e in records if e['requested_native_frames'][-1]==rule['student_checkpoint'])
    fresh=next(e for e in records if e['requested_native_frames'][-1]==rule['fresh_checkpoint'])
    source_initial=torch.load(root/'success_preparation'/str(seed)/'bank.pt',map_location='cpu',weights_only=False)
    data={};frames=0
    for name,path in [('earlier',run/rule['earlier_source']),('fresh',ROOT/fresh['checkpoint'])]:
        actor=load_actor(path);observer=TrajectoryRecorder(cfg['pool'],path.relative_to(ROOT).as_posix())
        result=evaluate(actor,cfg['development_pool'],out/name/'collection',cpu=False,offsets=(0,),
            conditions=[dict(opponent=rule['opponent'],player=1,offset=0)],arena_call_sequence=True,
            progress=False,reward_kind=cfg['learner']['reward'],transition_observer=observer)
        frames+=result['native_frames'];episode=observer.episodes[0]
        if not successful_episode(episode):raise RuntimeError('The selected self-success did not reproduce')
        data[name]=episode;actor.close()
    results={};source=torch.load(ROOT/student['checkpoint'],map_location='cpu',weights_only=False)
    for name in ('earlier','fresh'):
        bank=SequenceReplay(capacity=10000,byte_capacity=100663296,burn=0,learn=16,nstep=1,seed=seed+95003)
        for episode in source_initial['episodes'].values():
            if (episode['opponent_config']['id'],episode['player']) in {('cpu_Guile_1',1),('pressure_ChunLi',1),('pressure_ChunLi',2)}:
                bank.replace_verified_success(episode,0,4)
        bank.replace_verified_success(data[name],0,4)
        if len(bank.episodes)!=4:raise ValueError('The fixed four-trajectory diagnostic bank is incomplete')
        measured=success_update_probe(source,bank,out/name,rule['updates'],cfg,conditions=rule['evaluation_conditions'])
        frames+=measured['evaluation_native_frames'];results[name]=dict(source_trajectory=data[name]['policy_version'],**measured)
    value=dict(discarded=True,native_frames=frames,on_policy_samples=0,optimizer_updates=2*rule['updates'],results=results,
        independent_optimizer_snapshots=True,
        scope='Corrected same frozen1m weights and independently owned Adam tensors. Only the fourth real self-success history changes. Original non-independent snapshot probe remains recorded separately and is not a fair comparison. Discarded auxiliary diagnostic, not a PPO continuation.')
    if frames>rule['maximum_native_frames']:raise RuntimeError('The diagnostic exceeded its fixed frame budget')
    write(out/'summary.json',value);update_report(root,success_freshness_probe=value);return value


def sparse_supervision_probe(spec,root):
    """Compare loss weighting on one measured rare continuation failure."""
    rule=spec['sparse_supervision_probe'];phase=rule['source_phase'];seed=rule['seed']
    for arm in spec['prospective_success_comparison']['arms']:
        for trained_seed in spec['algorithm_comparison']['seeds']:
            if not (root/('eligibility_'+arm)/str(trained_seed)/'PPO/result.json').exists():
                raise RuntimeError('Complete the current matched budget before running the conditional probe')
    if any(audit_shared_parent(spec,root,'eligibility_'+arm)['passed_on_both_seeds']
           for arm in spec['prospective_success_comparison']['arms']):
        raise RuntimeError('The capability gate passed; do frozen readiness rather than more diagnostics')
    out=root/'sparse_supervision_probe'
    if (out/'summary.json').exists():return read(out/'summary.json')
    parent=root/phase/str(seed);cfg=read(parent/'configuration.json')
    records=algorithm_records(read(parent/'PPO/result.json'),'PPO')
    source_record=next(r for r in records if r['requested']==rule['source_requested_checkpoint'])
    source=torch.load(ROOT/source_record['checkpoint'],map_location='cpu',weights_only=False)
    results={};frames=0;settings=cfg['success_consolidation']['bank']
    frozen=construct_sequence_q(source['definition']);frozen.load_state_dict(source['state_dict'])
    for arm in rule['arms']:
        bank=SequenceReplay(capacity=settings['capacity_transitions'],byte_capacity=settings['byte_capacity'],burn=0,learn=16,nstep=1)
        bank.load_state_dict(torch.load(ROOT/rule['bank'],map_location='cpu',weights_only=False))
        bank.assess_success_disagreements(frozen,0)
        results[arm]=success_update_probe(source,bank,out/arm,rule['updates_per_arm'],cfg,
            loss_options=dict(disagreement_fraction=.5,disagreement_scope='verified_success',error_balance=0. if arm=='mean' else .5))
        frames+=results[arm]['evaluation_native_frames']
    if frames>rule['maximum_native_frames']:raise RuntimeError('Sparse supervision probe exceeded its declared frame bound')
    value=dict(discarded=True,native_frames=frames,optimizer_updates=2*rule['updates_per_arm'],on_policy_samples=0,
        source=source_record['checkpoint'],source_bank=rule['bank'],independent_optimizer_snapshots=True,results=results,
        scope='Auxiliary-only data-fit diagnosis with the same Adam/data/RNG, not PPO continuation or a candidate. Complete outcomes use standard starts and independent greedy control. Mean/error-balanced results do not establish relative efficacy under joint PPO updates.')
    write(out/'summary.json',value);update_report(root,sparse_supervision_probe=value);return value


def audit_shared_parent(spec,root,phase):
    """Keep the shared5m cohort separate from the earlier1m-anchor audit."""
    seeds={}
    for seed in spec['algorithm_comparison']['seeds']:
        reference=read(root/'success_preparation'/str(seed)/'summary.json')
        points=[]
        for record in continued_records(root,phase,seed,'PPO'):
            if record['phase']==phase:
                points.append(capability_point(record['evaluation'],reference,record['native_frames'],record['checkpoint']))
        seeds[str(seed)]=dict(points=points,gate=persistent_capability_gate(points,spec))
    value=dict(reference='Fixed shared5m14/12 useful results; no formal Ryu protection change',seeds=seeds,
        passed_on_both_seeds=all(v['gate']['passed'] for v in seeds.values()))
    write(root/(phase+'_shared_parent.json'),value);update_report(root,**{phase+'_shared_parent':value});return value


def prepare_priority_bank(spec,root):
    rule=spec['priority_success_comparison'];prepared={}
    for seed in rule['seeds']:
        out=root/'priority_preparation'/str(seed);summary=out/'summary.json'
        if summary.exists():prepared[str(seed)]=read(summary);continue
        source=root/rule['source_phase']/str(seed);cfg=read(source/'configuration.json');result=read(source/'PPO/result.json')
        records=continued_records(root,rule['source_phase'],seed,'PPO');latest=records[-1]
        bank=cfg['success_consolidation']['bank']
        replay=SequenceReplay(capacity=bank['capacity_transitions'],byte_capacity=bank['byte_capacity'],
            seed=seed+95003,burn=0,learn=bank['learn_length'],nstep=1)
        state=torch.load(source/'PPO/success_bank.pt',map_location='cpu',weights_only=False);replay.load_state_dict(state)
        clock=int(state.get('native_frames',result['native_frames']))
        # Original collection stamps were taken just before the independent
        # capture. Normalize data timestamps explicitly; parameters are untouched.
        end_stamps={e['frame']:e['frame']+e['native_frames'] for e in read(source/'PPO/success_learning.json')['collection']}
        for use in replay.retention_usage.values():
            for key in ('first_protected_frame','retired_frame'):
                if use[key] in end_stamps:use[key]=end_stamps[use[key]]
            use['phase_auxiliary_transitions']=0;use['phase_actual_disagreements_trained']=0
        available={}
        for record in records:
            won=useful_conditions(record['evaluation']['rows'])
            for row in record['evaluation']['rows']:
                if condition_key(row) in won:available[condition_key(row)]=(record['checkpoint'],row)
        missing={}
        for key,(path,row) in available.items():
            if not any(e['opponent_config']==row['opponent_config'] and e['player']==row['player'] and e['offset']==row['offset'] for e in replay.episodes.values()):
                missing.setdefault(path,[]).append(row)
        frames=0;captures=[]
        for i,(path,rows) in enumerate(missing.items()):
            actor=load_actor(path);observer=TrajectoryRecorder(cfg['pool'],path)
            captured=evaluate(actor,cfg['development_pool'],out/f'recovered_{i}',cpu=False,offsets=(0,),
                conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in rows],arena_call_sequence=True,
                progress=False,reward_kind=cfg['learner']['reward'],transition_observer=observer)
            actor.close();frames+=captured['native_frames']
            expected={condition_key(r):r for r in rows}
            for row in captured['rows']:
                if any(row[k]!=expected[condition_key(row)][k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')):
                    raise RuntimeError('Recovery of a missing self-success did not reproduce its scored game')
            for episode in observer.episodes:replay.replace_verified_success(episode,clock+frames,bank['maximum_episodes'])
            captures.append(dict(source=path,native_frames=captured['native_frames'],rows=captured['rows']))
        replay.note_complete_results(latest['evaluation']['rows'])
        torch.save(dict(replay.state_dict(),native_frames=clock+frames),out/'bank.pt')
        value=dict(source_checkpoint=latest['checkpoint'],source_phase=rule['source_phase'],shared_bank=(out/'bank.pt').relative_to(ROOT).as_posix(),
            bank_clock=clock+frames,preparation_native_frames=frames,optimizer_updates=0,captures=captures,
            episodes=len(replay.episodes),bytes=replay.bytes,initial_outcomes=latest['evaluation']['rows'],
            restoration='Identical completed B weights/Adam and actual bounded bank/RNG for both arms. Fresh PPO workers/RNG remain disclosed. Missing consumed-condition self-successes are independently recovered; no fabricated labels or future test data.')
        write(summary,value);prepared[str(seed)]=value
    update_report(root,priority_preparation=prepared);return prepared


def priority_success(spec,root,prepare_only=False):
    rule=spec['priority_success_comparison'];source=rule['source_phase']
    # Require complete original arms so a budget is never silently abandoned.
    for phase in ('success_ordinary',source):
        for seed in rule['seeds']:
            if not (root/phase/str(seed)/'PPO/result.json').exists():raise RuntimeError('Finish the current matched trial first')
    earlier=audit_shared_parent(spec,root,source)
    if earlier['passed_on_both_seeds']:raise RuntimeError('The stronger accumulation gate passed; perform frozen readiness instead')
    prepared=prepare_priority_bank(spec,root)
    if prepare_only:return prepared
    jobs=[]
    for arm in rule['arms']:
        phase='priority_'+arm;provenance={};starts={}
        for seed in rule['seeds']:
            parent=root/source/str(seed);destination=root/phase/str(seed);cfg=read(parent/'configuration.json')
            auxiliary=cfg['success_consolidation'];auxiliary.update(initial_bank=prepared[str(seed)]['shared_bank'],
                initial_bank_clock=prepared[str(seed)]['bank_clock'],resume_bank_sampling=True,
                initial_outcomes=prepared[str(seed)]['initial_outcomes'],capture_missing_checkpoint_successes=True,
                assessment_every_optimizer_updates=rule['failure_disagreements']['assessment_every_optimizer_updates'],
                disagreement_fraction=0. if arm=='uniform' else rule['failure_disagreements']['fraction'])
            if auxiliary['coefficient']!=.1:raise ValueError('Auxiliary coefficient changed in a sampler comparison')
            write(destination/'configuration.json',cfg);write(destination/'pool.json',cfg['pool'])
            write(destination/'initial_panel/summary.json',read(parent/'initial_panel/summary.json'))
            checkpoint=prepared[str(seed)]['source_checkpoint'];starts[str(seed)]=dict(initial=ROOT/checkpoint)
            provenance[str(seed)]=dict(source_phase=source,source=checkpoint,restoration=prepared[str(seed)]['restoration'])
        write(root/(phase+'_provenance.json'),provenance)
        jobs.extend(algorithm_jobs(spec,root,phase=phase,budget=rule['native_frames_per_arm_seed'],points=rule['checkpoints'],
            algorithms=('PPO',),starts=starts,ppo_gae_lambda=.95,dispatch=False))
    if jobs:run_training_jobs(jobs,root/'priority_workers.json')
    return {arm:dict(original_anchor=analyze_coverage(spec,root,'PPO','priority_'+arm),
        shared_parent=audit_shared_parent(spec,root,'priority_'+arm)) for arm in rule['arms']}


def compare_success_eligibility(spec,root):
    """One diagnosed sampler-predicate correction; existing learner unchanged."""
    rule=spec['prospective_success_comparison'];diagnosis=read(root/'prospective_diagnosis/summary.json')
    if not diagnosis['current_Ken_disagreements'] or diagnosis['sampling']['verified_success']['Ken_covered_current_disagreements']<=diagnosis['sampling']['failed']['Ken_covered_current_disagreements']:
        raise RuntimeError('The measured eligibility gap did not reproduce; do not start a speculative trial')
    if audit_shared_parent(spec,root,rule['source_phase'])['passed_on_both_seeds']:
        raise RuntimeError('The prior trial already passes; freeze instead of adding another test')
    jobs=[]
    for arm in rule['arms']:
        continue_algorithms(spec,root,'PPO',source_phase=rule['source_phase'],phase='eligibility_'+arm,
            budget=rule['native_frames_per_arm_seed'],points=rule['checkpoints'],pending_jobs=jobs,
            success_disagreement_scope='failed' if arm=='reactive' else 'verified_success')
    if jobs:
        update_report(root,stage='prospective_success_eligibility_training')
        run_training_jobs(jobs,root/'eligibility_workers.json')
    return {arm:dict(original_anchor=analyze_coverage(spec,root,'PPO','eligibility_'+arm),
                     shared_parent=audit_shared_parent(spec,root,'eligibility_'+arm)) for arm in rule['arms']}


def freeze_capability(spec,root):
    """Freeze only a repeated two-seed capability result, before readiness."""
    from datetime import datetime,timezone
    from experiments.ryu_counterq.report import retain_weights,model_identity
    state=read(ROOT/'reports'/root.name/'summary.json')
    if state.get('capability_final',{}).get('frozen'):return state['capability_final']
    trials={};sources={};points={}
    selection=spec.get('capability_selection',{})
    phases=selection.get('phases',{arm:'priority_'+arm for arm in spec['priority_success_comparison']['arms']})
    for arm,origin in phases.items():
        phase=origin+'_confirmation'
        if not all((root/phase/str(seed)/'PPO/result.json').exists() for seed in spec['algorithm_comparison']['seeds']):
            if any((root/phase/str(seed)).exists() for seed in spec['algorithm_comparison']['seeds']):
                raise RuntimeError('An authorized confirmation is still incomplete')
            phase=origin
        if not all((root/phase/str(seed)/'PPO/result.json').exists() for seed in spec['algorithm_comparison']['seeds']):
            raise RuntimeError('Finish both paired trials before selecting a model')
        weak=analyze_coverage(spec,root,'PPO',phase,coverage_origin=origin)
        strong=audit_shared_parent(spec,root,phase)
        trials[arm]=dict(phase=phase,old1m_gate=weak['passed_on_both_seeds'],shared5m_gate=strong['passed_on_both_seeds'],
            seeds={seed:dict(gate=v['gate'],checkpoints=[{k:p[k] for k in ('native_frames','checkpoint','starting_kept',
                'starting_useful','cpu_ko','pvp_wins','new_useful')} for p in v['points']]) for seed,v in strong['seeds'].items()})
        sources[arm]={str(seed):continued_records(root,phase,seed,'PPO')[-1] for seed in spec['algorithm_comparison']['seeds']}
        points[arm]=[s['points'][-1] for s in strong['seeds'].values()]
    eligible=[arm for arm,t in trials.items() if t['old1m_gate'] and t['shared5m_gate']]
    if not eligible:raise RuntimeError('No persistent two-seed accumulation result; diagnose and continue research')
    def score(arm):
        p=points[arm];useful=[v['cpu_ko']+sum(v['pvp_wins'].values()) for v in p]
        return min(useful),sum(useful),min(min(v['pvp_wins'].values()) for v in p),sum(v['cpu_ko'] for v in p),arm==selection.get('tie_preference','uniform')
    selected=max(eligible,key=score)
    def seed_score(seed):
        rows=sources[selected][str(seed)]['evaluation']['rows'];useful=useful_conditions(rows)
        wins=[sum(r['result']=='win' and r['opponent_kind']!='cpu' and r['player']==p for r in rows) for p in (1,2)]
        return len(useful),min(wins),sum(r['ko_win'] for r in rows if r['opponent_kind']=='cpu'),-seed
    representative=max(spec['algorithm_comparison']['seeds'],key=seed_score)
    models={};identities=[];lineages={}
    for arm in trials:
        models[arm]={};lineages[arm]={}
        for seed,record in sources[arm].items():
            destination=f'models/{spec["experiment"]}/ppo_{arm}_seed{seed}.pt'
            models[arm][seed]=retain_weights(record['checkpoint'],destination,keep_optimizer=True)
            identities.append(model_identity(record['checkpoint'],destination))
            chain=continued_records(root,trials[arm]['phase'],int(seed),'PPO');phases=list(dict.fromkeys(r['phase'] for r in chain))
            lineage=[]
            for phase in phases:
                run=read(root/phase/seed/'PPO/result.json')
                lineage.append(dict(phase=phase,native_frames=run['native_frames'],optimizer_updates=run['optimizer_updates'],
                    optimization_samples=run['optimization_samples'],auxiliary_samples=run.get('auxiliary_samples',0),
                    success_collection_native_frames=run.get('success_collection_native_frames',0)))
            lineages[arm][seed]=dict(phases=lineage,weight_lineage_native_frames=sum(r['native_frames'] for r in lineage),
                scope='Only actual inherited phases. Independent full-pool and discarded diagnostic learners are excluded. Initial independent bank preparations and evaluation are separate costs; PPO phase boundaries resume weights/Adam/bank, not worker/global RNG state.')
    value=dict(frozen=True,frozen_at_utc=datetime.now(timezone.utc).isoformat(),stronger_gate_passed_on_both_seeds=True,
        technical_checks_passed=False,selected_arm=selected,representative_seed=representative,
        representative_model=models[selected][str(representative)],models=models,selection=trials,
        model_identity=identities,lineages=lineages,readiness_pending=True,formal_reference_replaced=False,
        scope='Finite consumed development capability gate only. No96-condition KO, unknown robustness or official Arena claim.')
    update_report(root,capability_final=value,stage='capability_frozen_requires_readiness')
    return value


def breakthrough_readiness(spec,root,selection_name=None):
    """Gate the registered consolidated endpoints, using shared game entries.

    The exact90 protection cases are already inside each registered endpoint
    panel. Broad204 games are measured once per substantive successor; they
    remain consumed development, and never open the final holdout here.
    """
    from experiments.ryu_counterq.study import evaluated,protection_from_evaluation,old_protection
    from experiments.ryu_counterq.report import retain_weights,model_identity
    study_root=Path(root).resolve()
    selection=(spec['protected_repertoire_learning_declaration'] if selection_name is None
        else spec['qualification_rounds'][selection_name])
    root=study_root/selection.get('output_subdirectory','')
    if not root.is_relative_to(study_root):raise ValueError('Qualification output must remain in this study')
    root.mkdir(parents=True,exist_ok=True)
    rules=spec['readout_feasibility']['variants'];manifest=read(ROOT/'reports/selective_retention/protection_manifest.json')['rows']
    names=selection['endpoint_variants']
    breadth=spec['broader_known_PvP'];pool=read(ROOT/breadth['pool'])
    floors=breadth['prospective_successor_breadth_check'];results={};pending=[]
    for seed,name in names.items():
        seed=int(seed)
        rule=rules[name];directory=study_root/rule['phase'];path=directory/'summary.json'
        if not path.exists():pending.append(seed);continue
        complete=read(path);protection=protection_from_evaluation(complete,manifest)
        core=(complete['CPU_KO']==complete['CPU_conditions']==96 and protection['required']==90
            and protection['passed'] and complete['PvP_wins']==complete['PvP_conditions']==selection['expected_PvP_conditions'])
        if any(r['character']!=selection['own_character'] for r in complete['rows']):
            raise ValueError('Consolidated endpoint has a different own character')
        record=dict(seed=seed,source=(directory/'agent.pt').relative_to(ROOT).as_posix(),
            endpoint_record=path.relative_to(ROOT).as_posix(),CPU_KO=complete['CPU_KO'],
            PvP_wins=complete['PvP_wins'],PvP_conditions=complete['PvP_conditions'],core_passed=core,
            protection={k:v for k,v in protection.items() if k not in ('evaluation','lost')},
            failed_protected_conditions=[dict(opponent=x['result']['opponent'],player=x['result']['player'],
                offset=x['result']['offset'],result=x['result']['result']) for x in protection['lost']])
        if core:
            model=retain_weights(record['source'],f'models/{spec["experiment"]}/{selection["public_prefix"]}_seed{seed}.pt')
            record.update(model=model,parameter_identity=model_identity(record['source'],model))
            tested=evaluated(model,pool,root/'successor_breadth'/str(seed),ports=(1,2),offsets=(0,12))
            if len(tested['rows'])!=204 or tested['native_frames']>breadth['native_frame_cap_per_model']:
                raise RuntimeError('Successor breadth is incomplete or exceeds its fixed budget')
            record['breadth']=dict(overall=tested['overall'],by_port=tested['by_port'],native_frames=tested['native_frames'],
                complete_record=(root/'successor_breadth'/str(seed)/'summary.json').relative_to(ROOT).as_posix(),
                passed=tested['by_port']['1']['wins']>=floors['reference_P1_wins']
                    and tested['by_port']['2']['wins']>=floors['reference_P2_wins'])
            known=old_protection({str(seed):model},publish=False,full_known=True,
                output=root/'successor_known176')[str(seed)]
            if (known['evaluated_consumed_conditions']!=selection['full_known_conditions']
                    or known['evaluation']['native_frames']>selection['full_known_native_frame_cap_per_model']):
                raise RuntimeError('The complete known manifest is incomplete or exceeds its fixed budget')
            ledger=root/'successor_known176_results.json';all_known=read(ledger) if ledger.exists() else {}
            all_known[str(seed)]=known;write(ledger,all_known)
            record['complete_known_manifest']=dict(
                conditions=known['evaluated_consumed_conditions'],wins=known['all_consumed_wins'],
                protected_kept=known['protected_wins_kept'],passed=known['passed'],
                overall_PvP=known['evaluation']['overall'],by_port=known['evaluation']['by_port'],
                native_frames=known['evaluation']['native_frames'],
                complete_record=(root/'successor_known176'/str(seed)/'summary.json').relative_to(ROOT).as_posix())
        results[str(seed)]=record
        write(root/'breakthrough_readiness.json',dict(results=results,pending=pending,
            eligible=False,goal_complete=False,final_holdout_used=False))
    eligible=(not pending and all(r['core_passed'] and r.get('breadth',{}).get('passed')
        and r.get('complete_known_manifest',{}).get('passed') for r in results.values()))
    value=dict(results=results,pending=pending,eligible=eligible,selected_seed=selection['primary_seed'],
        selected_model=results.get(str(selection['primary_seed']),{}).get('model'),goal_complete=False,final_holdout_used=False,
        scope=selection.get('qualification_scope','Both independentbackbones with commonverifieddata; primaryseed fixed before outcomes. Core mastery, fixed90 and prospective204port checks are development gates, not unknown robustness or official Arena acceptance.'))
    write(root/'breakthrough_readiness.json',value)
    field='breakthrough_readiness' if selection_name is None else 'breakthrough_readiness_'+selection_name
    update_report(study_root,**{field:value},goal_achieved=False)
    return value


def breakthrough_final(spec,root,configuration_path,round_name=None,freeze_only=False):
    """Freeze the qualified primary, then open registered final conditions.

    Only orchestration lives here. The established evaluator, PPO learner,
    matched counter search and inference-weight writer own their operations.
    """
    from datetime import datetime,timezone
    from experiments.ryu_counterq.study import evaluated,matched_counter_search
    from experiments.ryu_counterq.report import retain_weights,model_identity
    study_root=Path(root).resolve();plan=None
    if round_name is None:
        root=study_root;held=spec['final_holdout'];execution=spec['frozen_counter_execution'];phase='frozen_final'
        readiness=read(root/'breakthrough_readiness.json')
    else:
        plan=spec['frozen_evaluation_rounds'][round_name]
        if plan.get('development_primary_review',{}).get('status')=='pending':
            raise RuntimeError('Resolve the registered consumed-development selection before final freeze')
        root=study_root/plan['output_subdirectory']
        if not root.is_relative_to(study_root):raise ValueError('Validation must remain within this study')
        root.mkdir(parents=True,exist_ok=True)
        held=plan['final_holdout'];execution=plan['counter_execution'];phase=plan['counter_phase']
        readiness=read(ROOT/plan['qualification_record'])
        if readiness['selected_seed']!=plan['selected_parent_seed']:
            raise ValueError('Prospective candidate differs from the declared development selection')
    if not readiness['eligible']:raise RuntimeError('Complete the declared development gates before final conditions')
    smoke=ROOT/execution['smoke_record'] if execution.get('smoke_record') else study_root/(execution['smoke_phase']+'_counter_runs.json')
    smoke_runs=read(smoke)['runs'] if smoke.exists() else {}
    if len(smoke_runs)!=1 or not all(r['algorithm']=='PPO' and r['costs']['optimizer_updates']>0
            and r['target']==execution['smoke_target'] for r in smoke_runs.values()):
        raise RuntimeError('Verify the shared PPO counter entry on its consumed smoke condition first')
    reference_name=held.get('reference_label','formal_reference');candidate_name=held.get('candidate_label','frozen_candidate')
    if reference_name==candidate_name:raise ValueError('Comparison models need distinct labels')
    models={reference_name:held.get('reference_model') or held['formal_reference'],candidate_name:readiness['selected_model']}
    freeze=root/'final_freeze.json'
    if freeze.exists():
        frozen=read(freeze)
        if frozen['models']!=models:raise ValueError('Final models differ from the retained freeze')
    else:
        if held['used']:raise RuntimeError('These final conditions are already consumed without a compatible freeze')
        identity=model_identity(readiness['results'][str(readiness['selected_seed'])]['source'],models[candidate_name])
        frozen=dict(models=models,selected_seed=readiness['selected_seed'],
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),parameter_identity=identity,
            final_conditions=copy.deepcopy(held),main_training_after_freeze=False,validation_round=round_name or 'original')
        write(freeze,frozen)
        held.update(used=True,used_at_utc=frozen['frozen_at_utc'])
        if plan is not None:
            plan['selected_seed']=readiness['selected_seed']
            write(ROOT/plan['selection_record'],dict(selected_seed=readiness['selected_seed'],
                model=models[candidate_name],qualification=plan['qualification_record'],
                selection_fixed_before_prospective_outcomes=True,earlier_final_conditions_consumed=True))
        write(ROOT/configuration_path,spec)
    write(root/'final_rule_pool.json',held['rules'])
    if freeze_only:
        write(root/'final_progress.json',dict(stage='models_frozen_waiting_registered_counter_jobs',goal_complete=False))
        return frozen
    floors=read(root/'final_reference_floors.json') if (root/'final_reference_floors.json').exists() else {}
    results={name:{} for name in models}
    cfg=read(ROOT/spec['base_config']);cfg['network']=copy.deepcopy(execution['network'])
    write(root/'final_progress.json',dict(stage='training_registered_counters',goal_complete=False))
    counter_runs,snapshots=matched_counter_search(cfg,models,execution,phase,artifact_root=root)
    expected=len(models)*len(execution['counter_roles'])*len(execution['counter_seeds'])*len(execution['counter_snapshots'])
    if len(snapshots)!=expected:raise RuntimeError('Final counter snapshot set is incomplete')
    counter_pool=[]
    for rival in snapshots:
        destination=f'models/{spec["experiment"]}/final_counters/{rival["id"]}.pt'
        delivered=retain_weights(rival['path'],destination)
        identity=model_identity(rival['path'],delivered)
        counter_pool.append(dict(rival,path=delivered,source_path=rival['path'],parameter_identity=identity))
    write(root/'final_counter_pool.json',counter_pool)
    pools=dict(rules=held['rules'],counters=counter_pool)
    # The outer model loop fixes both reference panels before any candidate
    # evaluation. Counter training above cannot update the frozen main weights.
    for name,path in models.items():
        for panel,pool in pools.items():
            write(root/'final_progress.json',dict(stage='evaluating_'+panel,model=name,goal_complete=False))
            directory=root/'final_evaluations'/panel/name
            source_checkpoint=(plan or {}).get('source_checkpoints',{}).get(name)
            receipt=directory/'source_inference_identity.json';audit=None
            if source_checkpoint and not receipt.exists():
                from experiments.league_adaptive.evaluate import InferenceIdentityAudit
                if (directory/'summary.json').exists():
                    raise RuntimeError('A completed frozen panel is missing its source identity receipt; account for a technical replay explicitly')
                audit=InferenceIdentityAudit(source_checkpoint)
            try:
                tested=evaluated(path,pool,directory,ports=tuple(held['ports']),offsets=tuple(held['offsets']),
                    **({} if audit is None else dict(transition_observer=audit)))
                if audit is not None:write(receipt,dict(source=source_checkpoint,runtime=path,**audit.result()))
            finally:
                if audit is not None:audit.close()
            count=len(pool)*len(held['ports'])*len(held['offsets'])
            if len(tested['rows'])!=count or tested['native_frames']>held['native_frame_cap_per_model']:
                raise RuntimeError('Final panel incomplete or over its declared native-frame cap')
            results[name][panel]=dict(overall=tested['overall'],by_port=tested['by_port'],
                native_frames=tested['native_frames'],complete_record=(root/'final_evaluations'/panel/name/'summary.json').relative_to(ROOT).as_posix())
            if source_checkpoint:results[name][panel]['source_inference_identity']=receipt.relative_to(ROOT).as_posix()
            if name==reference_name:
                measured={str(p):tested['by_port'][str(p)]['wins'] for p in held['ports']}
                if panel in floors and floors[panel]!=measured:raise RuntimeError('Frozen reference port floors changed')
                floors[panel]=measured;write(root/'final_reference_floors.json',floors)
            write(root/'final_results_progress.json',dict(results=results,floors=floors,main_weights_frozen=True))
    gates={panel:all(results[candidate_name][panel]['by_port'][str(p)]['wins']>=floors[panel][str(p)]
        for p in held['ports']) for panel in ('rules','counters')}
    improvement=sum(results[candidate_name][p]['overall']['wins'] for p in gates)>sum(
        results[reference_name][p]['overall']['wins'] for p in gates)
    value=dict(frozen=frozen,results=results,reference_port_floors=floors,
        port_gates=gates,new_complete_wins_improved=improvement,
        strength_gates_passed=all(gates.values()) and (improvement or not held.get('require_fresh_win_gain',True)),counter_runs=counter_runs,
        comparison_labels=dict(reference=reference_name,candidate=candidate_name),
        counter_pool=(root/'final_counter_pool.json').relative_to(ROOT).as_posix(),
        final_holdout_used=True,goal_complete=False,technical_reload_pending=True,behavior_review_pending=True,
        official_arena='not_executed',
        scope='Registered new combinations and fresh equally budgeted PPO searches. All compared models face all snapshots; mid/end snapshots are dependent. Limited coverage, no universal unknown-opponent guarantee. Final results never tune these frozen weights.')
    write(root/'frozen_final_results.json',value)
    field='frozen_final' if round_name is None else 'frozen_final_'+round_name
    update_report(study_root,**{field:value},goal_achieved=False)
    return value


def breakthrough_delivery(spec,root,known_only=False,round_name=None):
    """Audit the frozen candidate and record fixed full-game comparisons.

    Export, inference comparison and native movies remain owned by their
    existing shared implementations. This stage cannot train or pick cases.
    """
    import subprocess
    from experiments.ryu_counterq.study import validate_exported_policy
    study_root=Path(root).resolve();root=study_root;plan=None
    if round_name is not None:
        plan=spec['frozen_evaluation_rounds'][round_name];root=study_root/plan['output_subdirectory']
        if not root.is_relative_to(study_root):raise ValueError('Delivery must remain within this study')
    frozen=read(root/'final_freeze.json');rules=spec['delivery_validation'] if plan is None else plan['delivery_validation']
    final=None if known_only else read(root/'frozen_final_results.json')
    if not known_only and not final['strength_gates_passed']:
        raise RuntimeError('Inspect failed strength gates before a candidate delivery')
    readiness=read(study_root/'breakthrough_readiness.json' if plan is None else ROOT/plan['qualification_record'])
    models=frozen['models']
    labels=(plan or {}).get('final_holdout',{}) if final is None else final.get('comparison_labels',{})
    reference_name=labels.get('reference_label','formal_reference') if final is None else labels.get('reference','formal_reference')
    candidate_name=labels.get('candidate_label','frozen_candidate') if final is None else labels.get('candidate','frozen_candidate')
    source=models[candidate_name];seed=str(readiness['selected_seed'])
    if plan is not None:source=plan.get('source_checkpoints',{}).get(candidate_name,source)
    core=read(ROOT/readiness['results'][seed]['endpoint_record'])['rows']
    for record in (plan or {}).get('additional_core_records',[]):core+=read(ROOT/record)['rows']
    panels={candidate_name:dict(core=core),reference_name:dict(core=[])}
    if not known_only:
        for name in models:
            for panel in ('rules','counters'):
                panels[name][panel]=read(ROOT/final['results'][name][panel]['complete_record'])['rows']
    if plan is not None and plan.get('reference_core_records'):
        reference_cpu=[row for path in plan['reference_core_records'] for row in read(ROOT/path)['rows']]
        reference_manifest=[]
    else:
        reference_cpu=read(study_root/'reference_Ryu_cpu96/summary.json')['rows']
        reference_manifest=read(ROOT/'reports/selective_retention/protection_manifest.json')['rows']
    same_configuration=lambda r:{k:v for k,v in r.items() if k!='id'}
    for row in core:
        matches=[r for r in reference_cpu+reference_manifest
            if r['player']==row['player'] and r['offset']==row['offset']
            and same_configuration(r.get('opponent_config',r.get('opponent_configuration',{})))
                ==same_configuration(row['opponent_config'])]
        if matches:
            signatures={tuple(r[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')) for r in matches}
            if len(signatures)!=1:raise ValueError('Reference aliases disagree in a delivery condition')
            panels[reference_name]['core'].append(dict(matches[0],opponent=row['opponent'],
                opponent_config=row['opponent_config']))
    def selected(name,case):
        found=[r for r in panels[name][case['panel']]
            if all(r[k]==case[k] for k in ('opponent','player','offset'))]
        if len(found)!=1:raise ValueError(f'Incomplete fixed delivery case: {name}, {case}')
        return found[0]
    identity=None
    if not known_only:
        identity=validate_exported_policy(source,root/'exports/frozen_candidate',
            [selected(candidate_name,case) for case in rules['export_cases']])
        if identity['native_frames']>rules['export_native_frame_cap']:
            raise RuntimeError('Export audit exceeded its declared frame cap')
    movies=[];terminal_keys=('player','offset','result','self_hp','opponent_hp','native_frames','end_reason')
    for number,case in enumerate(rules['movie_cases']):
        if known_only and case['panel']!='core':continue
        rival=selected(candidate_name,case)['opponent_config']
        pool=root/'delivery_behavior'/f'{number}_pool.json';write(pool,[rival])
        for name in rules['movie_models']:
            out=root/'delivery_behavior'/f'{number}_{name}'
            if not (out/'summary.json').exists():
                subprocess.run([sys.executable,'-m','experiments.league_adaptive.record',
                    '--agent',models[name],'--opponent-pool',str(pool),'--opponent-id',rival['id'],
                    '--player',str(case['player']),'--offset',str(case['offset']),
                    '--out',str(out),'--label',name],cwd=ROOT,check=True)
            recorded=read(out/'summary.json');expected=selected(name,case)
            if Path(recorded['agent']).resolve()!=(ROOT/models[name]).resolve() or recorded['opponent']!=rival:
                raise ValueError('Frozen movie uses another policy or opponent version')
            if any(recorded[k]!=expected[k] for k in terminal_keys):
                raise RuntimeError(f'Complete movie differs from its frozen score: {name}, {case}')
            movies.append(dict(case=case,model=name,out=out.relative_to(ROOT).as_posix(),
                result={k:recorded[k] for k in terminal_keys},terminal_fields_equal=True,
                video_bytes=(out/'replay.mp4').stat().st_size))
            write(root/('known_behavior_progress.json' if known_only else 'delivery_progress.json'),
                dict(export_identity=identity,movies=movies,goal_complete=False))
    movie_frames=sum(r['result']['native_frames'] for r in movies)
    if movie_frames>rules['movie_native_frame_cap'] or sum(r['video_bytes'] for r in movies)>rules['maximum_delivered_movie_bytes']:
        raise RuntimeError('Complete movie evidence exceeded the declared delivery budget')
    result=dict(export_identity=identity,movies=movies,movie_native_frames=movie_frames,
        behavior_review_pending=True,goal_complete=False,official_arena='not_executed',
        scope='Frozen technical and full-movie reproducibility; not additional independent victories or a substitute for behavioral review.')
    name='breakthrough_known_behavior' if known_only else 'breakthrough_delivery'
    result['known_conditions_only']=bool(known_only)
    write(root/(name+'.json'),result)
    field=name if round_name is None else name+'_'+round_name
    update_report(study_root,**{field:result},goal_achieved=False)
    return result


def readiness_capability(spec,root):
    """Run the predeclared broad panel through the existing evaluator CLI."""
    from experiments.ryu_counterq.report import retain_weights,model_identity,evaluation_stats
    from experiments.league_adaptive.evaluate import paired_regression
    final=read(ROOT/'reports'/root.name/'summary.json').get('capability_final',{})
    if not final.get('frozen') or not final.get('stronger_gate_passed_on_both_seeds'):
        raise RuntimeError('Freeze a passing two-seed model before broader results')
    if not spec['frozen_readiness'].get('full_comparison'):raise RuntimeError('Declare the complete comparison before evaluating')
    models=copy.deepcopy(final['models']);models['shared5m']={};expected={};identities=[]
    for arm in final['models']:
        phase=final['selection'][arm]['phase']
        for seed in models[arm]:expected[(arm,seed)]=continued_records(root,phase,int(seed),'PPO')[-1]['evaluation']['rows']
    for seed in spec['algorithm_comparison']['seeds']:
        parent=read(root/'success_preparation'/str(seed)/'summary.json')
        target=f'models/{spec["experiment"]}/ppo_shared5m_seed{seed}.pt'
        models['shared5m'][str(seed)]=retain_weights(parent['source'],target)
        identities.append(model_identity(parent['source'],target));expected[('shared5m',str(seed))]=parent['rows']
    pool=read(ROOT/spec['base_config'])['pool'];pool_path=root/'readiness_pool.json';write(pool_path,pool)
    jobs=[]
    for arm,seeds in models.items():
        for seed,model in seeds.items():
            out=root/'readiness'/arm/seed
            if (out/'summary.json').exists():continue
            jobs.append(dict(name=f'readiness_{arm}_{seed}',log=root/f'readiness_{arm}_{seed}.log',
                command=[sys.executable,'-m','experiments.league_adaptive.evaluate','--agent',model,
                    '--pool',str(pool_path),'--no-cpu','--arena-call-sequence','--out',str(out)]))
    if jobs:
        update_report(root,stage='frozen_capability_readiness_running')
        run_training_jobs(jobs,root/'readiness_workers.json')
    results={};raw={};differences=[];frames=0
    for arm,seeds in models.items():
        results[arm]={};raw[arm]={}
        for seed,model in seeds.items():
            out=root/'readiness'/arm/seed;ev=read(out/'summary.json');raw[arm][seed]=ev
            if len(ev['rows'])!=300:raise RuntimeError('Incomplete96CPU/204PvP readiness')
            by_condition={condition_key(r):r for r in ev['rows']}
            for old in expected[(arm,seed)]:
                now=by_condition[condition_key(old)]
                for key in ('result','end_reason','self_hp','opponent_hp','native_frames','ko_win'):
                    if now[key]!=old[key]:differences.append(dict(arm=arm,seed=seed,condition=condition_key(old),field=key,expected=old[key],actual=now[key]))
            if any(r.get('model_memory_resets')!=1 for r in ev['rows']):
                differences.append(dict(arm=arm,seed=seed,field='automatic_memory_reset',expected='one per complete game'))
            results[arm][seed]=dict(model=model,**evaluation_stats(ev));frames+=ev['native_frames']
    write(root/'readiness_overlap.json',dict(terminal_differences=differences,native_frames=frames))
    if differences:raise RuntimeError('Broader call sequence changed a known game; diagnose before strength interpretation')
    if frames>spec['frozen_readiness']['full_comparison']['maximum_game_native_frames']:
        raise RuntimeError('Broad readiness exceeded its declared frame bound')
    paired={}
    for arm in final['models']:
        paired[arm]={}
        for seed in models[arm]:
            change=paired_regression(raw['shared5m'][seed]['rows'],raw[arm][seed]['rows'])
            paired[arm][seed]={k:change[k] for k in ('reference_wins','old_wins_kept','reference_pve_ko','pve_ko_kept','inheritance_gate_passed','by_port')}
            for key in ('old_wins_lost','new_wins','pve_ko_lost'):
                paired[arm][seed][key]=[dict(condition=x['condition'],reference={k:x['reference'][k] for k in ('result','self_hp','opponent_hp','end_reason')},
                    candidate={k:x['candidate'][k] for k in ('result','self_hp','opponent_hp','end_reason')}) for x in change[key]]
    final.update(readiness_pending=False,readiness=dict(results=results,shared5m_changes=paired,
        native_frames=frames,known_overlap_fields=('result','end_reason','self_hp','opponent_hp','native_frames','ko_win'),
        terminal_differences=differences,source_identity=identities,
        scope='All147 existing frozen entries are consumed training conditions, not unknown holdout. Matching final arms share budgets; shared5m parents have shorter lineages and are retention references only.'))
    update_report(root,capability_final=final,stage='frozen_readiness_complete_requires_export_behavior')
    return final


def publish_ddqn(spec,root):
    """Compact exact matches, paired endpoints and a reproducible learning plot."""
    report=ROOT/'reports'/spec['experiment'];comparison=analyze_ddqn(spec,root)
    if not comparison['complete']:raise ValueError('Publish completed paired comparisons only')
    records=[];models=[]
    for seed in spec['seeds']:
        initial=read(root/'parents'/str(seed)/'initial_panel/summary.json')
        records.extend(dict(panel='initial',seed=seed,arm='shared',additional_native_frames=0,**r) for r in initial['rows'])
        for arm in spec['ddqn_comparison']['arms']:
            base_frames=0
            for segment in ('pilot','initial','continuation'):
                source=root/'ddqn'/str(seed)/arm/segment;run=read(source/'summary.json')
                for p in run['curve']:
                    records.extend(dict(panel='development',seed=seed,arm=arm,additional_native_frames=base_frames+p['actual_frames'],**r) for r in p['development']['rows'])
                base_frames+=run['actual_training_frames']
            path=ROOT/'models'/spec['experiment']/f'resumed_{arm}_seed{seed}.pt'
            if not path.exists():
                actor=load_actor(source/'agent.pt')
                try:save(path,actor.policy,actor.definition,dict(algorithm='DDQN',seed=seed,arm=arm,
                    starting_model=f'models/ddqn_stabilization/input_real_{seed}.pt',additional_native_frames=base_frames,
                    protocol='configs/guile_capability_accumulation.json',
                    status='PANEL_GATE_PASSED' if comparison['arms'][arm]['seeds'][str(seed)]['gate']['passed'] else 'FAILED_CAPABILITY_ACCUMULATION_GATE',
                    scope='Research endpoint only; formal reference unchanged; no automatic promotion'))
                finally:actor.close()
            models.append(path.relative_to(ROOT).as_posix())
    fields=('panel','seed','arm','additional_native_frames','opponent','opponent_config','opponent_kind','player','offset',
        'result','end_reason','self_hp','opponent_hp','native_frames','ko_win','time_up','actual_character_ids')
    with (report/'ddqn_matches.jsonl').open('w',encoding='utf-8') as stream:
        for row in records:stream.write(json.dumps({k:row[k] for k in fields if k in row},ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(13,6),squeeze=False)
    for row,seed in enumerate(spec['seeds']):
        initial=read(root/'parents'/str(seed)/'initial_panel/summary.json')
        first=capability_point(initial,initial,0,'shared parent')
        for arm,color in [('one_step','tab:blue'),('multi_step','tab:orange')]:
            points=[first]+comparison['arms'][arm]['seeds'][str(seed)]['points']
            for column,getter in enumerate((lambda p:p['cpu_ko'],lambda p:p['pvp_wins']['1'],lambda p:p['pvp_wins']['2'],lambda p:p['starting_kept'])):
                axes[row,column].plot([p['native_frames']/1e6 for p in points],[getter(p) for p in points],marker='o',color=color,label=arm)
        for column,(name,limit) in enumerate(zip(('CPU KO /9','PvP P1 wins /5','PvP P2 wins /5',f"Starting useful results /{first['starting_useful']}"),(9,5,5,first['starting_useful']))):
            ax=axes[row,column];ax.set_title(f'{seed}: {name}');ax.set_ylim(-.2,limit+.4);ax.grid(alpha=.25);ax.axvline(.6,color='gray',linestyle=':',alpha=.6)
            if row==1:ax.set_xlabel('Additional training native frames (million)')
    axes[0,0].legend();fig.suptitle('Known complete games: refreshed history shared across paired DDQN backups')
    fig.tight_layout();fig.savefig(report/'ddqn_learning.png',dpi=150);plt.close(fig)
    update_report(root,DDQN_frozen_artifacts=dict(matches='reports/'+spec['experiment']+'/ddqn_matches.jsonl',
        plot='reports/'+spec['experiment']+'/ddqn_learning.png',models=models,all_failed=True))


def prepared_trace_profile(spec,root):
    path=root/'prepared_trace_profile.json'
    if path.exists():return read(path)
    results={}
    for seed in spec['seeds']:
        directory=root/'parents'/str(seed)/'multi_step';cfg=read(directory/'configuration.json')
        state=torch.load(directory/'training_state.pt',map_location='cpu',weights_only=False)
        actor=construct_sequence_q(state['definition']);actor.load_state_dict(state['online']);actor.set_training_mode(False)
        replay=replay_view(state,8);lengths=[];parts={'history':[],'recent':[]}
        with torch.no_grad():
            for _ in range(64):
                d=sample_sequence_batch(replay,cfg['learner']);burn=d['burn']
                q,_=actor.observation_sequence_values(d['images'],d['positions'],d['previous'],d['hidden'],burn,
                    input_durations=d['input_durations'])
                mask=torch.where(d['collector_known'],d['collector_greedy'],d['actions']==q.argmax(-1))
                _,values=sequence_double_q_targets(q,q,d['rewards'],d['dones'],burn,16,8,valid=d['valid'],
                    actions=d['actions'],cut_non_greedy=True,trace_mask=mask,return_details=True)
                for i,source in enumerate(d['retention_sources']):
                    valid=d['valid'][i,burn:burn+16];v=values['steps'][i][valid].tolist();lengths.extend(v);parts[source].extend(v)
        results[str(seed)]=dict(roots=len(lengths),mean_steps=float(np.mean(lengths)),
            fraction_gt1=float(np.mean(np.asarray(lengths)>1)),by_source={k:dict(roots=len(v),mean_steps=float(np.mean(v))) for k,v in parts.items() if v})
        if results[str(seed)]['mean_steps']<spec['diagnostic']['minimum_mean_effective_steps']:
            raise RuntimeError(f'Effective propagation still collapsed before training:{seed}:{results[str(seed)]}')
    result=dict(seeds=results,native_frames=0,optimizer_updates=0,
        method='Observed/verified collector-greedy prefixes, current-greedy fallback for unknown labels. Policy-lag bias remains; this is not a proof of optimal-Q targets.')
    write(path,result);update_report(root,prepared_trace_profile=result);return result


def propagation_profile(spec,root):
    path=root/'diagnostic/propagation_profile.json'
    if path.exists():return read(path)
    collection=torch.load(root/'diagnostic/probes.pt',map_location='cpu',weights_only=False)
    episode=next(r['episode'] for r in collection['records'] if r['case']['opponent']=='pressure_ChunLi')
    state=torch.load(ROOT/spec['parents']['56037'],map_location='cpu',weights_only=False)
    actor=construct_sequence_q(state['definition']);actor.load_state_dict(state['online'])
    target=construct_sequence_q(state['definition']);target.load_state_dict(state['target'])
    q=full_values(actor,episode)[None];tq=full_values(target,episode)[None]
    reward=torch.as_tensor(episode['rewards'])[None];done=torch.as_tensor(episode['dones'])[None]
    actions=torch.as_tensor(episode['actions'])[None];known=torch.as_tensor(episode['greedy_actions']==episode['actions'])[None]
    rows=[]
    for index in (11,25,35,36,37,40,67,83):
        variants={}
        for label,length,mask in [('one_step',1,None),('strict_current',8,None),('collector_policy',8,known)]:
            y,d=sequence_double_q_targets(q,tq,reward,done,index,1,length,actions=actions,
                cut_non_greedy=length>1,trace_mask=mask,return_details=True)
            variants[label]=dict(target=float(y[0,0]),steps=int(d['steps'][0,0]),
                reward_sum=float(d['reward_sum'][0,0]),bootstrap=float(d['bootstrap'][0,0]))
        rows.append(dict(decision=index,native_frame=index*6,physical_action=int(actions[0,index]),variants=variants))
    result=dict(rows=rows,native_frames=0,optimizer_updates=0,
        scope='Actual successful source-policy history. Reward propagation is measured; its finite prefix plus Double-Q bootstrap is not an optimalQ or unbiasedcurrent-greedy label.')
    write(path,result);update_report(root,propagation_profile=result);return result


def diagnostic_point(policy,target,reference,records,initial,actions,step):
    result=[]
    for number,record in enumerate(records):
        episode=record['episode'];q=full_values(policy,episode);tq=full_values(target,episode)
        previous=initial[number];executed=torch.as_tensor(episode['actions']);chosen=q[:-1].argmax(-1)
        different=[i for i in range(len(executed)) if actions[int(chosen[i])]!=actions[int(executed[i])]]
        roots=sorted(set(([different[0]] if different else [])+([40,67,83,108] if record['case']['opponent']=='pressure_ChunLi' else [0])))
        details=[]
        for index in roots:
            if index>=len(executed):continue
            h0=prefix_hidden(reference,episode,index);h1=prefix_hidden(policy,episode,index)
            with torch.no_grad():fixed,_=episode_values(policy,episode,index,index+1,h0)
            rewards=torch.as_tensor(episode['rewards'])[None];done=torch.as_tensor(episode['dones'])[None]
            _,backup=sequence_double_q_targets(q[None],tq[None],rewards,done,index,1,1,
                return_details=True)
            old=int(executed[index]);new=int(chosen[index])
            details.append(dict(decision=index,native_frame=index*6,old_action=old,new_action=new,
                physically_different=actions[old]!=actions[new],old_action_Q=float(q[index,old]),new_action_Q=float(q[index,new]),
                initial_old_action_Q=float(previous[index,old]),fixed_initial_hidden_greedy=int(fixed[0,0].argmax()),
                hidden_change_L2=float((h1-h0).norm()),one_step_reward=float(backup['reward_sum'][0,0]),
                one_step_bootstrap=float(backup['bootstrap'][0,0]),one_step_target=float((backup['reward_sum']+backup['bootstrap'])[0,0])))
        result.append(dict(case=record['case'],different_physical_decisions=len(different),
            first_different_decision=different[0] if different else None,roots=details))
    return dict(updates=step,probes=result,history='Identical raw legal observations, executed inputs and duration histories. Reencoded h may change with weights; fixed-initial-h choices are separately reported.')


def same_state_consequences(spec,base,root,record,updated_path,curve,spent,prefix_lengths=(1,8)):
    """Cross old/new choices with frozen old/new tails, using shared restore."""
    case=record['case'];rival=next(r for r in base['pool'] if r['id']==case['opponent'])
    old=load_actor(record['source']);new=load_actor(updated_path);env=FightEnv(old.definition,[rival],reward=base['learner']['reward'])
    maximum=spec['diagnostic']['maximum_native_frames'];results=[]
    try:
        first=curve['first_different_decision'];roots=[] if first is None else [first]
        if case['opponent']=='pressure_ChunLi' and 40 not in roots:roots.append(40)
        for index in roots[:2]:
            if spent+7200>maximum:break
            before=env.native_frames;obs,_=env.reset(options=dict(opponent=0,player=case['player'],offset=case['offset']))
            old.reset();new.reset()
            for _ in range(index):
                action=int(old.predict(obs,True)[0]);new.policy.observe_and_force(obs,action);obs,_,_,_,_=env.step(action)
                if env.ended:raise ValueError('A selected root lies after the real terminal')
            spent+=env.native_frames-before;saved=save_fight(env,old,{'updated':new})
            old_action=int(old.predict(obs,True)[0]);new_action=int(new.predict(obs,True)[0])
            if env.commands[old_action]==env.commands[new_action]:
                results.append(dict(decision=index,skipped='Updated and old physical inputs are identical'));continue
            for prefix in prefix_lengths:
                prefix_baseline=None
                for choice in ('old','new'):
                    for tail in ('old','new'):
                        if spent+7200>maximum:break
                        obs=restore_fight(env,old,saved,{'updated':new});begin=env.native_frames;actual=[]
                        for _ in range(prefix):
                            if env.ended:break
                            ao=int(old.predict(obs,True)[0]);an=int(new.predict(obs,True)[0]);action=ao if choice=='old' else an
                            old.policy.record_executed_actions(action,obs['position'][None]);new.policy.record_executed_actions(action,obs['position'][None])
                            actual.append(action);obs,_,_,_,info=env.step(action)
                        after_prefix=dict(position=obs['position'].tolist(),self_hp=info['self_hp'],opponent_hp=info['opponent_hp'],
                            elapsed_native_frames=env.native_frames-begin)
                        if prefix_baseline is None:
                            prefix_baseline=(obs['image'].copy(),obs['position'].copy(),info['self_hp'],info['opponent_hp'])
                        after_prefix.update(same_visible_history_as_old_choice=bool(np.array_equal(obs['image'],prefix_baseline[0])),
                            same_position_as_old_choice=bool(np.array_equal(obs['position'],prefix_baseline[1])),
                            same_hp_as_old_choice=(info['self_hp'],info['opponent_hp'])==prefix_baseline[2:])
                        if not env.ended:obs,info=continue_episode(env,old if tail=='old' else new,obs)
                        spent+=env.native_frames-begin
                        row=dict(decision=index,prefix_decisions=prefix,choice=choice,
                            continuation_policy=record['source'] if tail=='old' else updated_path.relative_to(ROOT).as_posix(),
                            actual_prefix=actual,after_prefix=after_prefix,result=terminal(info),restore_verified=True)
                        if choice==tail=='old':
                            expected=record['result']
                            if any(info[k]!=expected[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')):
                                raise RuntimeError('Unintervened restoration did not reproduce the recorded complete result')
                            row['unintervened_replay_matches']=True
                        results.append(row)
    finally:env.close();old.close();new.close()
    return dict(rows=results,native_frames_including_previous=spent,
        scope='Conditional prefix plus explicitly named frozen tail. Shared simulator, both observed histories, both actor states, opponent controller and RNG restore are checked. Branches are dependent, not independent wins or regret.')


def diagnose(spec,base,root):
    cached=root/'diagnostic/summary.json'
    if cached.exists():return read(cached)
    collection=probe_collection(spec,base,root);records=[r for r in collection['records'] if r['case']['seed']==56037]
    state=torch.load(ROOT/spec['parents']['56037'],map_location='cpu',weights_only=False)
    policy=construct_sequence_q(state['definition']);policy.load_state_dict(state['online'])
    reference=copy.deepcopy(policy);reference.set_training_mode(False)
    target,optimizer=learner_components(policy,base['learner']['learning_rate'],state['optimizer']);target.load_state_dict(state['target'])
    sampler=replay_view(state,8);learning_replay=replay_view(state,8)
    cfg=copy.deepcopy(base['learner']);cfg.update(nstep=8,backup_steps=1,backup_rule='one_step')
    initial=[full_values(reference,r['episode']) for r in records]
    for r,q in zip(records,initial):
        if not torch.equal(q[:-1].argmax(-1),torch.as_tensor(r['episode']['actions'])):
            raise RuntimeError('The initial current-network full-prefix history does not reproduce the collected actions')
    commands=compiled(state['definition']['actions']);curve=[];journal=[]
    count=state['progress']['optimizer_updates'];next_sync=state['progress']['target_next_sync_update']
    output=root/'diagnostic/updates';output.mkdir(parents=True,exist_ok=True)
    for step in range(spec['diagnostic']['maximum_updates']+1):
        if step in spec['diagnostic']['update_checkpoints']:
            point=diagnostic_point(policy,target,reference,records,initial,commands,step);curve.append(point)
            save(output/f'after_{step}.pt',policy,state['definition'],training=dict(purpose='Frozen-data controlled update diagnostic',updates=step))
            write(root/'diagnostic/live.json',point)
        if step==spec['diagnostic']['maximum_updates']:break
        data=sample_sequence_batch(sampler,cfg)
        measured=optimize(policy,target,optimizer,learning_replay,cfg,data=data,audit=True)
        journal.append(dict(update=step+1,source_episode_conditions=[dict(condition=learning_replay.episodes[k[0]].get('opponent_config'),family=learning_replay.episodes[k[0]]['family']) for k in data['keys']],**measured))
        count+=1
        if count>=next_sync:target.load_state_dict(policy.state_dict());next_sync+=cfg['target_every_updates']
    write(root/'diagnostic/update_journal.json',journal);write(root/'diagnostic/rank_curve.json',curve)
    spent=collection['native_frames'];complete=[];failure=None
    for step in (1,16,spec['diagnostic']['maximum_updates']):
        actor=load_actor(output/f'after_{step}.pt')
        rivals=[next(r for r in base['pool'] if r['id']==x) for x in ('cpu_Guile_1','pressure_ChunLi')]
        try:out=evaluate(actor,rivals,root/f'diagnostic/independent_after_{step}',cpu=False,offsets=(0,),ports=(1,),arena_call_sequence=True)
        finally:actor.close()
        spent+=out['native_frames'];complete.append(dict(updates=step,rows=out['rows'],native_frames=out['native_frames']))
        for number,r in enumerate(records):
            row=next(x for x in out['rows'] if x['opponent']==r['case']['opponent'])
            if r['result']['ko_win'] and not row['ko_win']:
                failure=(step,number);break
        if failure:break
    consequence=None
    if failure:
        step,number=failure;point=next(p for p in curve if p['updates']==step)['probes'][number]
        consequence=same_state_consequences(spec,base,root,records[number],output/f'after_{step}.pt',point,spent)
        spent=consequence['native_frames_including_previous']
    result=dict(frozen_data=True,added_environment_data_during_updates=0,controlled_updates=spec['diagnostic']['maximum_updates'],
        frozen_sampling_priorities=True,actual_Adam_and_target_restored=True,rank_curve=curve,
        complete_games=complete,first_independent_failure=failure,conditional_consequences=consequence,
        native_frames=spent,optimizer_updates=spec['diagnostic']['maximum_updates'],
        journal='artifacts/guile_capability_accumulation/diagnostic/update_journal.json',
        scope='Exact controlled continuation from the complete seed56037 parent, not a claim of reconstructing the historical500k-to1m training stream. No new experience enters the128tracked TD updates.')
    write(cached,result);update_report(root,diagnostic=result,stage='short_diagnosis_complete')
    return result


def diagnostic_fixture(spec,base,root):
    """One actual destructive SGD batch, reviewable without a huge replay file."""
    destination=ROOT/'reports'/spec['experiment']/'diagnostic_update.pt'
    state=torch.load(ROOT/spec['parents']['56037'],map_location='cpu',weights_only=False)
    replay=replay_view(state,8);cfg=copy.deepcopy(base['learner']);cfg.update(nstep=8,backup_steps=1,backup_rule='one_step')
    batch=sample_sequence_batch(replay,cfg);expected=read(root/'diagnostic/update_journal.json')[0]
    if [list(k) for k in batch['keys']]!=expected['audit']['keys']:raise RuntimeError('Review fixture did not reconstruct the observed first batch')
    actor=construct_sequence_q(state['definition']);actor.load_state_dict(state['online'])
    target,optimizer=learner_components(actor,cfg['learning_rate'],copy.deepcopy(state['optimizer']));target.load_state_dict(state['target'])
    measured=optimize(actor,target,optimizer,None,cfg,data=batch,audit=True)
    after=torch.load(root/'diagnostic/updates/after_1.pt',map_location='cpu',weights_only=False)['state_dict']
    error=max(float((actor.state_dict()[k]-v).abs().max()) for k,v in after.items())
    if error>1e-7:raise RuntimeError(f'Fixed-batch update differs from the observed destructive update:{error}')
    torch.save(dict(purpose='One controlled diagnostic update, not a training-resume checkpoint',definition=state['definition'],
        online=state['online'],target=state['target'],optimizer=state['optimizer'],batch=batch,learner=cfg,
        expected_loss=expected['loss'],observed_loss=measured['loss'],max_parameter_error=error),destination)
    update_report(root,diagnostic_fixture=dict(path=destination.relative_to(ROOT).as_posix(),actual_first_batch_verified=True,
        maximum_reapplied_parameter_error=error,native_frames=0,discarded_optimizer_updates=1,
        usage='Construct shared policy and target, restore Adam, call existing optimize with replay=None and data=batch. No training cache or file digest is required.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/guile_capability_accumulation.json')
    parser.add_argument('--stage',choices=['diagnose','prepare','profile','execution-profile','readout-feasibility','ddqn','analyze','algorithms','matched-algorithms','primary-matrix','skill-catalog','coherent-bank','representation','demonstration-learning','demonstration-comparison','prepare-student-consolidation','guile-tactical-online','fixture','short','publish','skill-curriculum','representative','recovery','expansion','confirm-coverage','credit','success-prepare','success-compare','success-freshness','sparse-supervision','priority-prepare','priority','eligibility','freeze-capability','readiness','breakthrough-readiness','breakthrough-final-freeze','breakthrough-final','breakthrough-delivery'],required=True)
    parser.add_argument('--experiment',help='Predeclared learning variant, curriculum, qualification or frozen evaluation round')
    parser.add_argument('--phase',choices=['representative_expansion','long_credit','priority','priority_uniform','priority_failure_disagreements','eligibility'],default='representative_expansion',help='Coverage protocol to confirm, only after measured two-seed recovery')
    parser.add_argument('--out',help='Fresh short-run directory only; formal runs use the registered study directory')
    args=parser.parse_args();setup()
    spec=read(ROOT/args.config);base=read(ROOT/spec['base_config']);root=ROOT/'artifacts'/spec['experiment'];root.mkdir(parents=True,exist_ok=True)
    if args.out and args.stage!='short':raise ValueError('Formal study paths are fixed in the protocol')
    if args.stage=='short':
        from experiments.ryu_counterq.train import train_run
        cfg=copy.deepcopy(base);cfg['learner'].update(nstep=8,backup_steps=8,backup_rule='collector_cut')
        cfg['network']['input_history']=dict(mode='real',cap_native_frames=120,normalization=120)
        cfg.pop('stale_history_audit',None)
        rivals=[r for r in cfg['pool'] if r['id'] in ('cpu_Guile_1','pressure_ChunLi')]
        train_run(cfg,spec.get('own_character','Guile'),57031,'ordinary',12000,Path(args.out) if args.out else root/'short',pool=rivals,dev_pool=[])
    elif args.stage=='diagnose':diagnose(spec,base,root);propagation_profile(spec,root)
    elif args.stage=='prepare':prepare_accumulation(spec,base,root)
    elif args.stage=='profile':prepared_trace_profile(spec,root)
    elif args.stage=='ddqn':paired_ddqn(spec,base,root)
    elif args.stage=='analyze':analyze_ddqn(spec,root)
    elif args.stage=='publish':publish_ddqn(spec,root)
    elif args.stage=='representative':representative_study(spec,base,root)
    elif args.stage=='recovery':representative_recovery(spec,root)
    elif args.stage=='expansion':representative_expansion(spec,root)
    elif args.stage=='confirm-coverage':confirm_coverage(spec,root,args.phase)
    elif args.stage=='credit':credit_compare(spec,root)
    elif args.stage=='success-prepare':
        if args.experiment:prepare_success_consolidation(spec,root,spec['skill_curricula'][args.experiment])
        else:success_consolidation(spec,root,prepare_only=True)
    elif args.stage=='success-compare':success_consolidation(spec,root)
    elif args.stage=='success-freshness':success_freshness_probe(spec,root)
    elif args.stage=='sparse-supervision':sparse_supervision_probe(spec,root)
    elif args.stage=='priority-prepare':priority_success(spec,root,prepare_only=True)
    elif args.stage=='priority':priority_success(spec,root)
    elif args.stage=='eligibility':compare_success_eligibility(spec,root)
    elif args.stage=='freeze-capability':freeze_capability(spec,root)
    elif args.stage=='readiness':readiness_capability(spec,root)
    elif args.stage=='fixture':diagnostic_fixture(spec,base,root)
    elif args.stage=='matched-algorithms':
        prepare_algorithms(spec,base,root,spec['algorithm_comparison']['DDQN_backup'])
        algorithm_jobs(spec,root)
        analyze_algorithms(spec,root)
    elif args.stage=='demonstration-learning':demonstration_learnability(spec,root,args.experiment)
    elif args.stage=='demonstration-comparison':compare_demonstration_learning(spec,root,args.experiment)
    elif args.stage=='prepare-student-consolidation':prepare_student_consolidation(spec,root,args.experiment)
    elif args.stage=='guile-tactical-online':guile_tactical_online(spec,root)
    elif args.stage=='skill-curriculum':skill_curriculum(spec,root,args.experiment)
    elif args.stage=='primary-matrix':character_algorithm_panel(spec,root,args.experiment)
    elif args.stage=='breakthrough-readiness':breakthrough_readiness(spec,root,args.experiment)
    elif args.stage=='breakthrough-final':breakthrough_final(spec,root,args.config,args.experiment)
    elif args.stage=='breakthrough-final-freeze':breakthrough_final(spec,root,args.config,args.experiment,freeze_only=True)
    elif args.stage=='breakthrough-delivery':breakthrough_delivery(spec,root,round_name=args.experiment)
    elif args.stage=='skill-catalog':tactical_skill_catalog(spec,root)
    elif args.stage=='coherent-bank':collect_coherent_skill_bank(spec,root,args.experiment)
    elif args.stage=='readout-feasibility':
        from experiments.ryu_counterq.update_analysis import readout_feasibility,readout_geometry
        result=readout_geometry(spec,root) if args.experiment=='geometry' else readout_feasibility(spec,root,args.experiment)
        field='readout_feasibility' if args.experiment is None else 'readout_feasibility_'+args.experiment
        update_report(root,**{field:{k:v for k,v in result.items() if k!='rows'}},goal_achieved=False)
    elif args.stage=='execution-profile':stochastic_skill_execution(spec,root,args.experiment or '57037')
    elif args.stage=='representation':representation_probe(spec,root)
    elif args.stage=='algorithms':
        comparison=analyze_ddqn(spec,root)
        if not comparison['complete']:raise RuntimeError('Complete the declared DDQN comparison and continuation first')
        if comparison['passed']:raise RuntimeError('A successful DDQN protocol proceeds to coverage; do not start unnecessary algorithm trials')
        backup=selected_backup(comparison)
        update_report(root,automatic_fallback_reason='Both-seed retention and accumulation gate remains unsatisfied after the declared continuation.',selected_DDQN_backup=backup)
        prepare_algorithms(spec,base,root,backup);algorithm_jobs(spec,root);analyze_algorithms(spec,root)


if __name__=='__main__':main()
