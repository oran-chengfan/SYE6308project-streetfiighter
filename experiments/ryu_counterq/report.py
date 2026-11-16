"""Publish compact, reproducible evidence after the research processes finish.

Training and evaluation remain in their existing modules. This module only
selects delivery files and removes duplicated details from the live report.
"""
import argparse
import json
import re
import shutil
from collections import Counter,defaultdict
from pathlib import Path

from experiments.league_adaptive.runtime import ROOT,load_actor,setup,write
from experiments.ryu_direct_league.policy import save
from experiments.ryu_counterq.study import ARTIFACT,REPORT,read,selection_stages


def portable(value,model_map=None):
    """Repository-relative evidence paths, optionally mapped to delivery files."""
    model_map=model_map or {}
    if isinstance(value,dict):
        result={k:portable(v,model_map) for k,v in value.items()}
        if 'path' in value:
            original=result['path'];result['path']=model_map.get(original,original)
            if result['path']!=original:result['source_path']=original
        return result
    if isinstance(value,list):return [portable(v,model_map) for v in value]
    if isinstance(value,str):
        path=Path(value)
        if path.is_absolute() and path.is_relative_to(ROOT):return path.relative_to(ROOT).as_posix()
        if path.parts and path.parts[0] in ('artifacts','models','configs','reports','notebooks'):return path.as_posix()
    return value


def publish_behavior_movies(records,report,*,video_directory='videos',trace_name='behavior_traces.jsonl'):
    """Copy the selected full movies and index their actual decision inputs."""
    behavior=[];report=Path(report);destination_root=report/video_directory
    destination_root.mkdir(parents=True,exist_ok=True)
    with (report/trace_name).open('w',encoding='utf-8',newline='\n') as stream:
        for item in records:
            if not item['terminal_fields_equal']:raise ValueError('Unverified complete behavior movie')
            source=ROOT/item['out'];recorded=read(source/'summary.json');name=source.name
            keys=('result','self_hp','opponent_hp','native_frames','end_reason')
            if any(recorded[k]!=item['result'][k] for k in keys):
                raise ValueError('Behavior source differs from its scored record')
            destination=destination_root/f'{name}.mp4';shutil.copyfile(source/'replay.mp4',destination)
            for decision in read(source/'decisions.json'):
                stream.write(json.dumps(dict(trace_group=name,model=item['model'],offset=item['case']['offset'],
                    **decision),ensure_ascii=False,separators=(',',':'))+'\n')
            behavior.append(dict(case=item['case'],model=item['model'],video=destination.relative_to(ROOT).as_posix(),
                trace_group=name,native_fps=recorded['native_fps'],source_agent=portable(recorded['agent']),
                opponent=recorded['opponent'],result={k:recorded[k] for k in keys},terminal_fields_equal=True,
                input_trace=(report/trace_name).relative_to(ROOT).as_posix()))
    return behavior


def add_evaluation_manifest(manifests,name,evaluation,model_name,model_path,scope):
    """Publish exact replay inputs, without copying an evaluator or outcomes."""
    from experiments.league_adaptive.evaluate import exact_opponent_pool
    rows=evaluation['rows']
    conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in rows]
    pool=exact_opponent_pool([r['opponent_config'] for r in rows])
    if len({(r['opponent'],r['player'],r['offset']) for r in conditions})!=len(conditions):
        raise ValueError('Published panel contains duplicate complete conditions')
    if name in manifests and any(manifests[name][k]!=v for k,v in (('pool',pool),('conditions',conditions))):
        raise ValueError('Compared models do not share the exact published replay panel')
    record=manifests.setdefault(name,dict(pool=pool,conditions=conditions,completed_models={},scope=scope))
    record['completed_models'][model_name]=model_path


def publish_breakthrough(config,artifact,report):
    """Compact live role/protocol evidence without ending the active goal."""
    state=read(report/'summary.json');games=[];curves=[];diagnostics={};replay_panels={}
    fields=('opponent','opponent_kind','opponent_config','opponent_character','cpu_stars','player','offset',
        'result','end_reason','self_hp','opponent_hp','native_frames','ko_win','time_up',
        'physical_input_counts','motion','damage_by_requested_input','actual_character_ids')
    demonstration=config['demonstration_learning']
    variants=list(dict.fromkeys([demonstration.get('phase','demonstration_learning')]+
        [v['phase'] for v in demonstration.get('variants',{}).values()]))
    variant_rules={v['phase']:dict(demonstration,**v) for v in demonstration.get('variants',{}).values()}
    for phase in variants:
        path=artifact/phase/'summary.json'
        complete=path.exists()
        if not complete:path=artifact/phase/'progress.json'
        if not path.exists():continue
        raw=read(path);points=[]
        if not complete:
            rule=variant_rules.get(phase,demonstration)
            raw=dict(raw,source=rule.get('initial',config['source']),
                total_optimizer_updates=raw['complete_updates']-rule.get('starting_updates',0),
                evaluation_native_frames=sum(p['evaluation_native_frames'] for p in raw['points']))
        for point in raw['points']:
            rows=point['rows'];cpu=[r for r in rows if r['opponent_kind']=='cpu'];pvp=[r for r in rows if r['opponent_kind']!='cpu']
            value={k:point[k] for k in ('cumulative_updates','auxiliary_samples','first_loss','last_loss',
                'mean_gradient_norm','maximum_gradient_norm','optimization_seconds','evaluation_seconds',
                'evaluation_native_frames','trainable_scope','optimizer_start',
                'mean_preclip_gradient_norm_by_component','actual_sampling_by_group') if k in point}
            fits=point.get('teacher_history_fit',[])
            focus=set(variant_rules.get(phase,demonstration).get('priority_opponents',[]))|{
                'cpu_ChunLi_8','cpu_Sagat_8','cpu_Guile_8','cpu_Blanka_8'}
            value['history_fit']={
                'trajectories':len(fits),'fully_fitted':sum(f['fitted_actions']==f['teacher_decisions'] for f in fits),
                'actions_fitted':sum(f['fitted_actions'] for f in fits),'actions_total':sum(f['teacher_decisions'] for f in fits),
                'selected_trajectories':[f for f in fits if len(fits)<=4 or f['opponent'] in focus],
                'complete_record':(artifact/phase/str(point['cumulative_updates'])/'summary.json').relative_to(ROOT).as_posix()}
            value.update(CPU_KO=sum(r['ko_win'] for r in cpu),CPU_conditions=len(cpu),
                PvP_wins=sum(r['result']=='win' for r in pvp),PvP_conditions=len(pvp),
                PvP_by_port={str(p):sum(r['result']=='win' for r in pvp if r['player']==p) for p in (1,2)},
                source=(artifact/phase/str(point['cumulative_updates'])/'agent.pt').relative_to(ROOT).as_posix())
            use=point.get('actual_auxiliary_use',{})
            value['actual_auxiliary_transitions']=sum(u['auxiliary_transitions'] for u in use.values())
            points.append(value);curves.append(dict(phase=phase,**value))
            games.extend(dict(phase=phase,checkpoint=value['source'],optimizer_updates=point['cumulative_updates'],
                **{k:r[k] for k in fields if k in r}) for r in rows)
        diagnostics[phase]=dict(points=points,source=raw['source'],optimizer_updates=raw['total_optimizer_updates'],
            evaluation_native_frames=raw['evaluation_native_frames'],full_record=path.relative_to(ROOT).as_posix(),
            complete_declared_budget=complete,
            scope='Auxiliary-only learnability; no TD/PPO advantages. Checkpoints are dependent and wins are never combined across them.')
    for key in ('demonstration_learning_comparison','demonstration_learning','verified_prefix_comparison','history_credit_comparison'):
        state.pop(key,None)
    state['learnability_diagnostics']=diagnostics
    for key,value in list(state.items()):
        if (not key.endswith('_comparison') or not isinstance(value,dict) or not value
                or not all(isinstance(v,dict) and 'points' in v for v in value.values())):continue
        full_record=artifact/(key+'.json');write(full_record,value)
        state[key]=dict(full_record=full_record.relative_to(ROOT).as_posix(),variants={
            name:{k:result[k] for k in ('source','total_optimizer_updates','evaluation_native_frames')}
            for name,result in value.items()})
    preparation=state.get('student_success_preparation')
    if preparation and 'rows' in preparation:
        state['student_success_preparation']={k:v for k,v in preparation.items() if k!='rows'}
        state['student_success_preparation']['complete_records']='artifacts/'+artifact.name+'/student_consolidation_preparation/summary.json'
    preparation={}
    for rule in config.get('skill_curricula',{}).values():
        for seed in rule['seeds']:
            source=artifact/rule['preparation_phase']/str(seed)/'summary.json'
            if source.exists():
                key=rule['preparation_phase']+'/'+str(seed)
                preparation[key]=dict(read(source),seed=seed,preparation_phase=rule['preparation_phase'],
                    complete_records=source.relative_to(ROOT).as_posix())
    state['success_consolidation_preparation']=preparation
    for key,value in list(preparation.items()):
        if 'rows' not in value:continue
        rows=value['rows'];cpu=[r for r in rows if r['opponent_kind']=='cpu'];pvp=[r for r in rows if r['opponent_kind']!='cpu']
        preparation[key]={k:v for k,v in value.items() if k!='rows'}
        preparation[key].update(CPU_KO=sum(r['ko_win'] for r in cpu),CPU_conditions=len(cpu),
            PvP_wins=sum(r['result']=='win' for r in pvp),PvP_conditions=len(pvp),
            complete_records=value['complete_records'])
        games.extend(dict(phase=value['preparation_phase'],seed=value['seed'],checkpoint=value['source'],
            **{k:r[k] for k in fields if k in r}) for r in rows)
    from experiments.ryu_counterq.capability import algorithm_records,capability_point
    main_curves=[];costs=[]
    for name,rule in config.get('skill_curricula',{}).items():
        for seed in rule['seeds']:
            directory=artifact/rule['phase']/str(seed);history=directory/'PPO/frozen_evaluations.json'
            if not history.exists():continue
            initial=read(directory/'initial_panel/summary.json')
            for record in algorithm_records(dict(frozen_evaluations=read(history)),'PPO'):
                point=capability_point(record['evaluation'],initial,record['native_frames'],record['checkpoint'])
                row={k:point[k] for k in ('native_frames','checkpoint','cpu_ko','cpu_time_wins','pvp_wins',
                    'starting_useful','starting_kept','new_useful','old_lost')}
                main_curves.append(dict(protocol=name,seed=seed,**row))
                games.extend(dict(phase=rule['phase'],seed=seed,checkpoint=record['checkpoint'],
                    **{k:r[k] for k in fields if k in r}) for r in record['evaluation']['rows'])
            source=directory/'PPO/result.json'
            if not source.exists():source=directory/'PPO/live.json'
            if source.exists():
                raw=read(source);costs.append(dict(protocol=name,seed=seed,complete=source.name=='result.json',
                    **{k:raw[k] for k in ('native_frames','optimizer_updates','optimization_samples','optimization_seconds',
                        'seconds','auxiliary_samples','auxiliary_seconds','success_collection_native_frames') if k in raw}))
    state['current_skill_curves']=main_curves;state['current_skill_costs']=costs
    readout=config.get('readout_feasibility',{});readout_results={};readout_progress={}
    readout_rules=([readout]+[dict(readout,**v) for v in readout.get('variants',{}).values()]) if readout else []
    for rule in readout_rules:
        path=artifact/rule['phase']/'summary.json'
        if not path.exists():
            progress=artifact/rule['phase']/'fit_progress.json'
            if progress.exists():
                points=read(progress)['points']
                readout_progress[rule['phase']]=dict(source=rule['source'],bank=rule['bank'],last_record=points[-1],
                    complete=False,capability_results_pending=True,
                    scope='Surrogate full-corpus fitting progress only; not independent game mastery or goal completion.')
            continue
        raw=read(path);fit=read(artifact/rule['phase']/'fit.json')
        readout_results[rule['phase']]={k:v for k,v in raw.items() if k!='rows'}
        readout_results[rule['phase']]['fitting']={k:v for k,v in fit.items() if k not in ('initial','final')}
        readout_results[rule['phase']]['fitting'].update({name:{k:v for k,v in fit[name].items() if k!='trajectories'}
            for name in ('initial','final')})
        if rule.get('protection_manifest'):
            from experiments.ryu_counterq.study import protection_from_evaluation
            measured=protection_from_evaluation(raw,read(ROOT/rule['protection_manifest'])['rows'])
            if measured['required']!=90:raise ValueError('Readout protection is not the declared90-condition union')
            readout_results[rule['phase']]['known_protection']={k:v for k,v in measured.items() if k not in ('evaluation','lost')}
            readout_results[rule['phase']]['known_protection'].update(
                manifest=rule['protection_manifest'],additional_native_frames=0,
                lost_conditions=[dict(opponent=x['condition']['original']['opponent'],
                    source_panel=x['condition']['original']['source_panel'],player=x['result']['player'],
                    offset=x['result']['offset'],result=x['result']['result']) for x in measured['lost']],
                scope='Exact configuration/weight-path, port and offset projection from these same completed games; display aliases alone are ignored. No controlled demonstrations or extra repeat wins counted.')
        games.extend(dict(phase=rule['phase'],checkpoint=(artifact/rule['phase']/'agent.pt').relative_to(ROOT).as_posix(),
            **{k:r[k] for k in fields if k in r}) for r in raw['rows'])
    state['readout_fitting_diagnostics']=readout_results
    state['readout_learning_progress']=readout_progress
    probes={};probe_games=[]
    for phase in ('high_star_transfer','high_star_coaches','Blanka5_update_cross','Blanka5_divergence_cross',
                  'Blanka5_physical_input_check','student_recoveries','learned_Blanka_recovery','ryu_reference_recoveries',
                  'ryu_verified_demonstrations','ryu_hard_response_discovery','ryu_verified_exchanges',
                  'ryu_common_retention_cross','ryu_P1_visited_recovery','ryu_Blanka_visited_recovery',
                  'ryu_Blanka_fixed_future','ryu_feedback_response','ryu_verified_feedback',
                  'ryu_historical_highstar_transfer','ryu_verified_historical','ryu_Sagat_timing_probe','ryu_joint_guard_cross',
                  'canonical_missing_diagnosis','canonical_Guile8_finish','canonical_Guile8_verified',
                  'canonical_MBison8_intercept','canonical_MBison8_guarded_exchange','canonical_MBison8_verified',
                  'canonical_common_prefix_recovery','remaining_neural_skill_recovery','common_exchanges_verified',
                  'canonical_MBison_transfer_4','canonical_MBison_transfer_5','canonical_MBison_transfer_6','canonical_MBison_transfer_7',
                  'coherent_opening_recovery','MBison6_neural_followup','ChunLi8_failure_exchange','Sagat8_failure_exchange',
                  'ChunLi8_late_exchange','Sagat8_late_exchange','remaining_finishes_verified',
                  'Sagat8_single_unfitted_cross37','coherent_refresh_recovery','coherent_refresh_verified',
                  'Sagat7_remaining_diagnosis','Sagat7_input_effect_cross',
                  'PvP_gap_diagnosis','PvP_gap_response_recovery','PvP_gap_verified_trajectories',
                  'PvP_common_prefix_recovery','PvP_common_prefix_verified','PvP_shared_reaction_test',
                  'PvP_shared_response_verified','remaining_protection_responses',
                  'remaining_shared_response_families','projectile_common_exchange',
                  'projectile_peer_retention','remaining_protection_verified','seed31_guard_fitting_cross',
                  'frozen_P1_recovery','counter_feedback_recovery','counter_feedback_verified'):
        path=artifact/phase/'summary.json'
        if not path.exists():continue
        raw=read(path);rows=raw.get('rows',[])
        probes[phase]={k:raw[k] for k in ('native_frames','native_frames_including_previous','native_frames_lower_bound','native_frames_upper_bound',
            'restored_roots','seconds','scope','configuration') if k in raw}
        complete=[dict(r,**r['result']) if isinstance(r.get('result'),dict) else r for r in rows if 'result' in r]
        selected=(complete if len(complete)<=4 else [next(r for r in complete if r.get('opponent')==name and r['result']=='win')
            for name in dict.fromkeys(r.get('opponent') for r in complete if r['result']=='win')])
        outcome_fields=('opponent','player','offset','root','prefix','continuation','result','self_hp','opponent_hp',
            'native_frames','end_reason','plan','prefix_source','first_different_actual_input','first_game_difference_from_recorded_path',
            'student','teacher','exact_physical_translation','input_timing_overrides',
            'decision','choice','continuation_policy','actual_prefix','after_prefix','unintervened_replay_matches','source')
        probes[phase].update(complete_games=len(complete),conditional_wins=sum(r['result']=='win' for r in complete),
            not_completed=len(raw.get('not_completed',[])),selected_outcomes=[{k:r[k] for k in outcome_fields if k in r} for r in selected])
        probes[phase]['unknown_examples']=raw.get('not_completed',[])[:2]
        probe_games.extend(dict(probe=phase,**r) for r in rows)
        probe_games.extend(dict(probe=phase,**r) for r in raw.get('not_completed',[]))
        probes[phase]['full_record']=path.relative_to(ROOT).as_posix()
    accepted=(state.get('breakthrough_acceptance',{}).get('passed',False)
        and state.get('process_cleanup',{}).get('training_evaluation_workers_remaining')==0)
    state.update(format='completed_character_algorithm_breakthrough' if accepted else 'live_character_algorithm_breakthrough',
        goal_achieved=bool(accepted),formal_reference_replaced=False,
        conditional_probes=probes,learning_game_records=(report/'learning_games.jsonl').relative_to(ROOT).as_posix(),
        conditional_game_records=(report/'conditional_games.jsonl').relative_to(ROOT).as_posix(),
        fixed_CPU96_completed=any(r.get('CPU_KO')==96 and r.get('CPU_conditions')==96 for r in readout_results.values()),
        complete_development_gate=[phase for phase,r in readout_results.items()
            if r.get('CPU_KO')==96 and r.get('CPU_conditions')==96
            and r.get('known_protection',{}).get('passed') and r['known_protection']['required']==90],
        limitations=('Finite listed conditions only; no universal unknown-opponent guarantee or officialArena acceptance. Formal reference remains unchanged; Mac independently reviews the recommendation.' if accepted else
            'CPU96 and fixed90 protection are separate, checkpoint-specific development gates. Consumed development, frozen new conditions and technical identity remain distinct; no promotion or officialArena acceptance.'))
    report.mkdir(parents=True,exist_ok=True)
    matrix=artifact/'primary_matrix_analysis.json'
    if matrix.exists():state['primary_matrix']=read(matrix)
    delivered_matrix=artifact/'primary_matrix_delivery.json'
    if delivered_matrix.exists():state['primary_matrix_delivery']=read(delivered_matrix)
    for name in ('breakthrough_readiness','frozen_final_results','breakthrough_known_behavior','breakthrough_delivery','readiness_export_probe'):
        path=artifact/(name+'.json')
        if path.exists():state['frozen_final' if name=='frozen_final_results' else name]=portable(read(path))
    movie_records=state.get('breakthrough_delivery',state.get('breakthrough_known_behavior',{})).get('movies',[])
    if movie_records:
        state['frozen_behavior']=publish_behavior_movies(movie_records,report,
            video_directory='frozen_behavior',trace_name='frozen_behavior_inputs.jsonl')
    qualification_rounds={};consolidation_curves=[]
    for name,rule in config.get('qualification_rounds',{}).items():
        path=artifact/rule['output_subdirectory']/'breakthrough_readiness.json'
        if path.exists():qualification_rounds[name]=portable(read(path))
        for seed,variant in rule['endpoint_variants'].items():
            phase=config['readout_feasibility']['variants'][variant]['phase'];directory=artifact/phase
            progress=directory/'fit_progress.json';initial=directory/'initial_fit.json'
            if not progress.exists() or not initial.exists():continue
            curve=dict(qualification=name,seed=int(seed),phase=phase,actions=read(initial)['metrics']['actions'],
                points=read(progress)['points'],endpoint_complete=phase in readout_results,
                scope='Shared continuous-data supervised readout fitting; closures are dependent optimizer evaluations, not PPO updates or independent victories.')
            if phase in readout_results:
                endpoint=readout_results[phase];fit=endpoint['fitting']
                curve['endpoint']={k:endpoint[k] for k in ('CPU_KO','CPU_conditions','PvP_wins','PvP_conditions','PvP_by_port')}
                curve['points'].append(dict(closure_calls=fit['closure_calls'],
                    loss=fit['final']['weighted_cross_entropy'],matching_actions=fit['final']['matching_actions'],
                    seconds=fit['optimization_seconds']))
            consolidation_curves.append(curve)
    state['qualification_rounds']=qualification_rounds
    state['registered_consolidation_learning']=consolidation_curves
    for round_name,ready in [('original',state.get('breakthrough_readiness',{})),*qualification_rounds.items()]:
        for seed,record in ready.get('results',{}).items():
            if record.get('model'):
                add_evaluation_manifest(replay_panels,'core_'+round_name,read(ROOT/record['endpoint_record']),
                    round_name+'_'+seed,record['model'],'Consumed training/development conditions; exact standard starts, no teacher intervention.')
            for panel in ('breadth','complete_known_manifest'):
                entry=record.get(panel)
                if not entry:continue
                tested=read(ROOT/entry['complete_record'])
                add_evaluation_manifest(replay_panels,'known_'+panel,tested,round_name+'_'+seed,record['model'],
                    'Consumed known regression panel; endpoints are different training stages, not independent new tests.')
                games.extend(dict(phase='qualified_'+panel,qualification_round=round_name,seed=int(seed),
                    checkpoint=record['model'],**{k:r[k] for k in fields if k in r}) for r in tested['rows'])
    validation_rounds={}
    for name,rule in config.get('frozen_evaluation_rounds',{}).items():
        directory=artifact/rule['output_subdirectory'];record={}
        for filename,key in [('final_freeze','freeze'),('final_progress','progress'),
                             ('final_reference_floors','reference_port_floors'),
                             ('frozen_final_results','frozen_final'),('breakthrough_known_behavior','known_behavior'),
                             ('breakthrough_delivery','delivery')]:
            path=directory/(filename+'.json')
            if path.exists():record[key]=portable(read(path))
        if not record:continue
        counters=directory/(rule['counter_phase']+'_counter_runs.json')
        if counters.exists():
            record['completed_counter_searches']={name:{k:r[k] for k in
                ('algorithm','actual_training_frames','requested_training_frames','native_frame_overrun','costs') if k in r}
                for name,r in read(counters).get('runs',{}).items()}
        movies=record.get('delivery',record.get('known_behavior',{})).get('movies',[])
        if movies:record['behavior']=publish_behavior_movies(movies,report,
            video_directory=name+'_behavior',trace_name=name+'_inputs.jsonl')
        failure=directory/'failure_review/result.json'
        if failure.exists():
            review=portable(read(failure))
            review['movies']=publish_behavior_movies(review['movies'],report,
                video_directory=name+'_failure_behavior',trace_name=name+'_failure_inputs.jsonl')
            record['failure_review']=review
        if record.get('delivery',{}).get('export_identity'):
            source=directory/'exports/frozen_candidate'
            destination=(ROOT/record['freeze']['models']['frozen_candidate']).parent/'standalone'/name
            destination.mkdir(parents=True,exist_ok=True)
            for filename in ('policy.zip','inference.py','student_model.py','student_actions.py',
                             'student_observation.py','policy_definition.json','runtime_versions.json'):
                shutil.copyfile(source/filename,destination/filename)
            record['standalone_inference']=destination.relative_to(ROOT).as_posix()
        if accepted:
            record['phase_record_scope']='Pending flags in immutable phase records describe creation time; see top-level behavior_acceptance, technical_acceptance and breakthrough_acceptance for completed checks.'
        validation_rounds[name]=record
    state['validation_rounds']=validation_rounds
    finals=[('original',state.get('frozen_final',{}))]+[(name,r.get('frozen_final',{})) for name,r in validation_rounds.items()]
    for round_name,final in finals:
        for name,panels in final.get('results',{}).items():
            for panel,entry in panels.items():
                tested=read(ROOT/entry['complete_record'])
                add_evaluation_manifest(replay_panels,round_name+'_final_'+panel,tested,name,
                    final['frozen']['models'][name],'Conditions first opened after this round freeze; any subsequent use is consumed development.')
                games.extend(dict(phase='frozen_new_'+panel,validation_round=round_name,model=name,
                    checkpoint=final['frozen']['models'][name],**{k:r[k] for k in fields if k in r}) for r in tested['rows'])
    for cell in state.get('primary_matrix',{}).get('results',[]):
        tested=read(ROOT/cell['complete_records'])
        games.extend(dict(phase='fresh3m_role_algorithm_matrix',character=cell['character'],algorithm=cell['algorithm'],
            seed=cell['seed'],checkpoint=cell['source'],actual_training_native_frames=cell['actual_inherited_training_frames'],
            **{k:r[k] for k in fields if k in r}) for r in tested['rows'])
    initial_path=artifact/'untrained_control_analysis.json'
    if initial_path.exists():
        initial=read(initial_path);deltas=[]
        lookup={(c['character'],c['seed']):c for c in initial['results']}
        for cell in state.get('primary_matrix',{}).get('results',[]):
            key=(cell['character'],cell['seed'])
            if key not in lookup:continue
            before=read(ROOT/lookup[key]['complete_records']);after=read(ROOT/cell['complete_records'])
            point=capability_point(after,before,cell['actual_inherited_training_frames'],cell['source'])
            deltas.append(dict(character=cell['character'],algorithm=cell['algorithm'],seed=cell['seed'],
                **{k:point[k] for k in ('starting_useful','starting_kept','new_useful','old_lost')}))
        state['untrained_initialization_audit']=dict(results=initial['results'],learned_changes=deltas,
            interpretation='Initial deterministic policies can already win through repeated inputs. Endpoint scores alone are not learned gains; every delta uses the exact same112-condition configuration. The two seeds differ substantially before learning.')
        for cell in initial['results']:
            delivered=next((r['delivered'] for r in config['comparative_panels']['untrained_control'].get('delivered_controls',[])
                if r['character']==cell['character'] and r['seed']==cell['seed']),None)
            if delivered:
                add_evaluation_manifest(replay_panels,'untrained112',read(ROOT/cell['complete_records']),
                    cell['character']+'_'+str(cell['seed']),delivered,
                    'Original random initializations; deterministic repeated-input wins are not learned capabilities.')
            games.extend(dict(phase='untrained_control',character=cell['character'],seed=cell['seed'],
                **{k:r[k] for k in fields if k in r}) for r in read(ROOT/cell['complete_records'])['rows'])
    endpoints=[]
    for name,protocol in [('hard_endpoints','Ryu_hard_cpu'),('tactic_endpoints','Ryu_verified_tactics')]:
        path=artifact/(name+'_analysis.json')
        if not path.exists():continue
        raw=read(path);rule=config['skill_curricula'][protocol]
        for cell in raw['results']:
            parent=read(artifact/rule['preparation_phase']/str(cell['seed'])/'summary.json')
            tested=read(ROOT/cell['complete_records'])
            point=capability_point(tested,parent,cell['actual_inherited_training_frames'],cell['source'])
            endpoints.append(dict(protocol=protocol,**cell,
                parent_wins_kept=point['starting_kept'],parent_useful=point['starting_useful'],
                new_useful=point['new_useful'],old_lost=point['old_lost'],
                budget_scope='The cell frame count is this2m continuation only; the same4m parent and all separate auxiliary/data-collection costs remain explicit in lineage records.'))
            games.extend(dict(phase=name,seed=cell['seed'],checkpoint=cell['source'],
                **{k:r[k] for k in fields if k in r}) for r in tested['rows'])
    state['full_endpoint_comparison']=endpoints
    frozen_panels=[]
    for name,rule in config.get('comparative_panels',{}).items():
        if name in ('untrained_control','hard_endpoints','tactic_endpoints'):continue
        path=artifact/(rule['phase']+'_analysis.json')
        if not path.exists():continue
        raw=read(path)
        for cell in raw['results']:
            tested=read(ROOT/cell['complete_records'])
            frozen_panels.append(dict(panel=name,**cell,scope=rule.get('scope')))
            games.extend(dict(phase=rule['phase'],checkpoint=cell['source'],seed=cell['seed'],
                **{k:r[k] for k in fields if k in r}) for r in tested['rows'])
    state['frozen_comparative_panels']=frozen_panels
    protection={}
    for name,rule in config.get('known_protection_panels',{}).items():
        path=ROOT/rule['record']
        if not path.exists():continue
        for label,result in read(path).items():
            protection[name+'/'+label]={k:v for k,v in result.items() if k not in ('lost','evaluation')}
            protection[name+'/'+label].update(scope=rule['scope'],complete_record=rule['record'],
                lost_conditions=[dict(opponent=r['condition']['original']['opponent'],
                    player=r['result']['player'],offset=r['result']['offset'],result=r['result']['result'])
                    for r in result['lost']])
            games.extend(dict(phase='known_protection/'+name,model=label,checkpoint=rule['models'][label],
                **{k:r[k] for k in fields if k in r}) for r in result['evaluation']['rows'])
    state['known_protection_panels']=protection
    facts={}
    for name in ('single_exchange_gradient_audit','mixed_success_label_ambiguity','Blanka8_specialist_reload',
                 'remaining_protection_data_preflight/joint_whole_victory_selection','readout_compute_profile',
                 'seed31_guard_fitting_audit','seed31_guard_feature_audit','seed31_guard_before_after_scores',
                 'seed31_guard_actual_history_comparison'):
        path=artifact/(name+'.json')
        if path.exists():
            raw=read(path);facts[name]={k:v for k,v in raw.items() if k!='rows'}
            facts[name]['full_record']=path.relative_to(ROOT).as_posix()
    for name in ('canonical_success_label_audit','tactical_skill_catalog','known_protection_development_analysis'):
        path=artifact/(name+'.json')
        if not path.exists():continue
        raw=read(path)
        if name=='canonical_success_label_audit':
            value=dict(results={k:dict(episodes=v['episodes'],exact_prefix_action_disagreements=v['exact_prefix_action_disagreements'],
                both_hard_target_conflicts=v['both_hard_target_conflicts'],examples=v['examples'][:3]) for k,v in raw['results'].items()},native_frames=0)
        elif name=='tactical_skill_catalog':value={k:v for k,v in raw.items() if k!='selected'}
        else:
            value=dict(native_frames=raw['native_frames'],scope=raw['scope'],results={
                k:{a:b for a,b in v.items() if a!='lost_conditions'} for k,v in raw['results'].items()})
        facts[name]=dict(value,full_record=path.relative_to(ROOT).as_posix())
    path=artifact/'historical_skill_harvest/summary.json'
    if path.exists():
        raw=read(path)
        facts['historical_skill_harvest']=dict(native_frames=raw['native_frames'],
            sources=[dict(name=s['name'],source=s['source'],native_frames=s['native_frames'],
                wins=[r for r in s['rows'] if r['result']=='win'],complete_games=len(s['rows'])) for s in raw['sources']],
            scope=raw['scope'],full_record=path.relative_to(ROOT).as_posix())
        probe_games.extend(dict(probe='historical_skill_harvest',source=s['source'],**r)
            for s in raw['sources'] for r in s['rows'])
    state['current_mechanism_evidence']=facts
    path=artifact/'facing_normalization_probe/summary.json'
    if path.exists():
        raw=read(path);pairs=[]
        for source in dict.fromkeys(v['source'] for v in raw['results']):
            views=[v for v in raw['results'] if v['source']==source]
            if len(views)!=2:continue
            before=next(v for v in views if not v['facing_canonical']);after=next(v for v in views if v['facing_canonical'])
            delta=capability_point(after,before,0,after['view'])
            pairs.append(dict(source=source,original=after['original'],
                old_useful=delta['starting_useful'],kept=delta['starting_kept'],new=delta['new_useful'],lost=delta['old_lost'],
                complete_games_per_view=len(after['rows'])))
        facts['facing_normalization_probe']=dict(pairs=pairs,native_frames=raw['native_frames'],optimizer_updates=0,
            full_record=path.relative_to(ROOT).as_posix(),
            scope='Within-source zero-SGD input/output representation effects. Cross-role source lineages differ; no algorithm training attribution.')
        games.extend(dict(phase='facing_normalization_probe',source=v['source'],facing_canonical=v['facing_canonical'],
            character=r['character'],**{k:r[k] for k in fields if k in r}) for v in raw['results'] for r in v['rows'])
    practical=[]
    for cell in config.get('practical_representative_panels',{}).get('models',[]):
        path=(ROOT/cell['evaluation_directory']/'summary.json' if cell.get('evaluation_directory') else
            artifact/'practical_panels'/cell['name']/'summary.json')
        if not path.exists():continue
        if cell.get('source_result'):
            cell=dict(cell);result_path=ROOT/cell['source_result'];run=read(result_path)
            checkpoint=result_path.parent/f"checkpoint_{run['steps']}.pt"
            if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
            cell['source']=checkpoint.relative_to(ROOT).as_posix()
        raw=read(path);cpu=[r for r in raw['rows'] if r['opponent_kind']=='cpu']
        practical.append(dict(**cell,CPU_KO=sum(r['ko_win'] for r in cpu),CPU_conditions=len(cpu),
            CPU_KO_by_star=[sum(r['ko_win'] for r in cpu if r['cpu_stars']==s) for s in range(1,9)],
            PvP=raw['overall'],by_port=raw['by_port'],native_frames=raw['native_frames'],complete_records=path.relative_to(ROOT).as_posix()))
        games.extend(dict(phase='practical_'+cell['name'],checkpoint=cell['source'],**{k:r[k] for k in fields if k in r}) for r in raw['rows'])
    state['practical_representative_panels']=practical
    path=artifact/'broader_known_PvP_results.json'
    if path.exists():
        broad={}
        for name,raw in read(path).items():
            roles=defaultdict(Counter);kinds=defaultdict(Counter)
            for row in raw['rows']:
                role=row.get('opponent_character',row['opponent_config'].get('character','Ryu'))
                for table,key in ((roles,role),(kinds,row['opponent_kind'])):
                    table[key]['conditions']+=1;table[key][row['result']]+=1
            broad[name]={k:v for k,v in raw.items() if k!='rows'}
            broad[name].update(by_opponent_character=dict(roles),by_controller_kind=dict(kinds))
            games.extend(dict(phase='broader_known_PvP',model=name,checkpoint=raw['model'],
                **{k:r[k] for k in fields if k in r}) for r in raw['rows'])
        state['known_role_breadth']=dict(models=broad,
            pending_models=[name for name in config['broader_known_PvP']['models'] if name not in broad],
            scope='Exact existing204-condition project development panel. Role/style breadth is separate from fixed90 protection and final unknown conditions; formalRyu is a different-lineage practical reference. Rule-controller wins are not Arena strength.')
    for name in variants:
        state.pop(name,None)
    for key in list(state):
        if key.endswith('_results'):state.pop(key)
    path=artifact/'algorithm_analysis.json'
    if path.exists():
        raw=read(path);narrow={}
        metrics=('native_frames','requested_native_frames','checkpoint','cpu_ko','pvp_wins','starting_useful','starting_kept')
        for algorithm,value in raw.items():
            narrow[algorithm]=dict(complete=value['complete'],passed_on_both_seeds=value['passed_on_both_seeds'],seeds={})
            for seed,entry in value['seeds'].items():
                narrow[algorithm]['seeds'][seed]=dict(costs=entry['costs'],
                    points=[dict(**{k:p[k] for k in metrics if k in p},new_useful=len(p['new_useful']),old_lost=len(p['old_lost'])) for p in entry['points']],
                    learned_anchor_retention=[dict(**{k:p[k] for k in metrics if k in p},new_useful=len(p['new_useful']),old_lost=len(p['old_lost'])) for p in entry['accumulation_points']])
        state['algorithm_comparison']=dict(narrow_panel=narrow,full_record=path.relative_to(ROOT).as_posix(),
            scope='Separate19-condition learning curve through4m; full3m96CPU+16PvP comparison is primary_matrix. Historical formalRyu is not a causal training cell.')
    for interrupted in config.get('shared_collection_repair',{}).get('interrupted_runs',[]):
        directory=artifact/interrupted['run']/str(interrupted['seed']);history=directory/'PPO/frozen_evaluations.json'
        if not history.exists():continue
        for record in algorithm_records(dict(frozen_evaluations=read(history)),'PPO'):
            games.extend(dict(phase=interrupted['run'],seed=interrupted['seed'],status='interrupted_not_complete',
                checkpoint=record['checkpoint'],**{k:r[k] for k in fields if k in r}) for r in record['evaluation']['rows'])
    (report/'learning_games.jsonl').write_text(''.join(json.dumps(portable(g),ensure_ascii=False)+'\n' for g in games),encoding='utf-8')
    (report/'conditional_games.jsonl').write_text(''.join(json.dumps(portable(g),ensure_ascii=False)+'\n' for g in probe_games),encoding='utf-8')
    write(report/'evaluation_panels.json',portable(replay_panels))
    state['rerunnable_evaluation_panels']=(report/'evaluation_panels.json').relative_to(ROOT).as_posix()
    for key,value in state.items():
        if not key.endswith('_results') or not isinstance(value,dict):continue
        for run in value.values():
            if isinstance(run,dict) and isinstance(run.get('points'),list):
                run['points']=[{k:v for k,v in p.items() if k not in ('useful','rows')} for p in run['points']]
    plot_breakthrough(state,report)
    write(artifact/'published_learnability_curves.json',portable(curves))
    if accepted:
        # Keep completed evidence readable; live history remains in Git and the
        # local archive, while exact complete games stay in the existing JSONL.
        write(artifact/'published_detailed_state.json',portable(state))
        keys=('format','stage','goal_achieved','formal_reference_replaced','mac_review_read','config',
            'matched_seeds','research_questions','own_character','selected_character','role_protocol_decision',
            'primary_matrix','primary_matrix_delivery','matched_protocol_audit','untrained_initialization_audit',
            'untrained_control_delivery','actual_adjacent_update_diagnosis','readout_feasibility_geometry',
            'registered_consolidation_learning','complete_development_gate','qualification_rounds',
            'counter_feedback_data','complementary_integration_result','published_counter_feedback_models',
            'counter_feedback_breadth','counter_feedback_known176','safe_execution_differences',
            'current_known_neighbor_retention','development_primary_selection','selected_PvE96_quality',
            'selected_model_lineage','first_frozen_validation_result','frozen_final','validation_rounds',
            'prospective_paired_frozen_outcomes','behavior_acceptance','technical_acceptance',
            'breakthrough_acceptance','process_cleanup','unit_tests','learning_game_records',
            'conditional_game_records','rerunnable_evaluation_panels','limitations','official_arena',
            'next_evidence_supported_bottleneck')
        state={k:state[k] for k in keys if k in state}
        state['completed_learning_history']={phase:{k:value[k] for k in
            ('CPU_KO','CPU_conditions','PvP_wins','PvP_conditions','PvP_by_port','known_protection') if k in value}
            for phase,value in readout_results.items()}
        state['completed_auxiliary_history']=[dict(phase=phase,points=[{k:p[k] for k in
            ('cumulative_updates','CPU_KO','CPU_conditions','PvP_wins','PvP_conditions','PvP_by_port',
             'optimization_seconds','evaluation_native_frames') if k in p} for p in history['points']])
            for phase,history in diagnostics.items()]
    write(report/'summary.json',portable(state))
    return dict(stage=state['stage'],active_goal=not bool(accepted),learnability_curves=len(curves),learning_games=len(games),
        full_learning_sources=len(diagnostics),formal_reference_replaced=False)


def plot_breakthrough(state,report):
    """Keep the matched full panel separate from narrower learning panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    consolidation=state.get('registered_consolidation_learning',[])
    if consolidation:
        figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for curve in consolidation:
            points=curve['points'];label=str(curve['seed'])+' / '+curve['qualification']
            axes[0].plot([p['closure_calls'] for p in points],[p['loss'] for p in points],label=label)
            axes[1].plot([p['closure_calls'] for p in points],
                [curve['actions']-p['matching_actions'] for p in points],label=label)
        axes[0].set_yscale('log');axes[0].set_ylabel('Continuous-data cross entropy')
        axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylabel('Recorded actions still different')
        for axis in axes:axis.set_xlabel('LBFGS objective evaluations');axis.legend(fontsize=7)
        figure.suptitle('Independent PPO backbones, common data; full-game ability checked separately')
        figure.savefig(report/'continuous_data_consolidation.png',dpi=140);plt.close(figure)
    matrix=state.get('primary_matrix',{})
    if matrix.get('complete'):
        figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        groups=[('Guile','DDQN'),('Guile','PPO'),('Ryu','DDQN'),('Ryu','PPO')]
        for seed,color,shift in [(57031,'#326a98',-.18),(57037,'#d78532',.18)]:
            values=[next(r for r in matrix['results'] if r['character']==c and r['algorithm']==a and r['seed']==seed) for c,a in groups]
            for axis,metric,total in [(axes[0],'CPU_KO',96),(axes[1],'PvP',16)]:
                heights=[r[metric] if metric=='CPU_KO' else r[metric]['wins'] for r in values]
                bars=axis.bar([i+shift for i in range(4)],heights,.34,label=str(seed),color=color)
                axis.bar_label(bars,padding=2);axis.set_ylim(0,total);axis.set_xticks(range(4),[c+'\n'+a for c,a in groups])
                axis.set_title('CPU KOs /96' if metric=='CPU_KO' else 'PvP wins /16')
        axes[0].legend(title='Independent training seed');figure.suptitle('Fresh3m matched protocols; consumed development conditions')
        figure.savefig(report/'character_algorithm_comparison.png',dpi=140);plt.close(figure)
    rows=state.get('current_skill_curves',[])
    if len({(r['protocol'],r['seed']) for r in rows})<4:
        (report/'skill_accumulation.png').unlink(missing_ok=True)
        return
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,seed in zip(axes,(57031,57037)):
        for name,color in [('Ryu_hard_cpu','#326a98'),('Ryu_verified_tactics','#d78532')]:
            curve=sorted((r for r in rows if r['protocol']==name and r['seed']==seed),key=lambda r:r['native_frames'])
            if not curve:continue
            x=[r['native_frames']/1e6 for r in curve]
            axis.plot(x,[r['starting_kept'] for r in curve],'-o',color=color,label=name+' retained')
            axis.plot(x,[len(r['new_useful']) for r in curve],'--s',color=color,label=name+' new')
        axis.set_title('Seed '+str(seed));axis.set_xlabel('Additional native frames (millions)');axis.set_ylabel('Complete useful outcomes');axis.set_ylim(bottom=0)
    axes[0].legend(fontsize=7);figure.suptitle('Ryu24-condition consumed panel; checkpoints are dependent')
    figure.savefig(report/'skill_accumulation.png',dpi=140);plt.close(figure)
    diagnostics=state.get('learnability_diagnostics',{})
    if 'ryu_single_exchange_fit' in diagnostics:
        figure,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
        solo=diagnostics['ryu_single_exchange_fit']['points']
        fits=[p['history_fit']['selected_trajectories'][0] for p in solo]
        axes[0].plot([p['cumulative_updates'] for p in solo],[100*f['fitted_actions']/f['teacher_decisions'] for f in fits],'-o')
        axes[0].set_title('Blanka8 actual-history label fit');axes[0].set_ylim(0,105);axes[0].set_ylabel('Correct requested actions (%)')
        for axis,phase,names,title in [
            (axes[1],'ryu_single_exchange_fit',['cpu_Ryu_8','cpu_Blanka_8'],'One-skill learning: full-game KO'),
            (axes[2],'ryu_joint_skill_fit',['cpu_Ryu_8','cpu_Blanka_8'],'Joint learning: full-game KO'),
            (axes[3],'ryu_historical_exchange_fit',['cpu_Ryu_8','cpu_Dhalsim_7','cpu_MBison_7'],'Historical tactics: full-game KO')]:
            points=diagnostics.get(phase,{}).get('points',[])
            for name in names:
                curve=[]
                for p in points:
                    raw=read(ROOT/Path(p['source']).parent/'summary.json')
                    row=next((r for r in raw['rows'] if r['opponent']==name),None)
                    if row is not None:curve.append((p['cumulative_updates'],int(row['ko_win'])))
                if curve:axis.plot([x for x,_ in curve],[y for _,y in curve],'-o',label=name.removeprefix('cpu_'))
            axis.set_title(title);axis.set_ylim(-.1,1.15);axis.set_yticks([0,1],['No','Yes']);axis.legend(fontsize=8)
        for axis in axes:axis.set_xlabel('Auxiliary optimizer updates')
        figure.suptitle('Independent standard-start control; consumed conditions, dependent checkpoints')
        figure.savefig(report/'continuous_skill_learning.png',dpi=140);plt.close(figure)


def capability_evidence(config,artifact):
    """Compact actual experiment records; never run a learner or evaluator."""
    from experiments.ryu_counterq.capability import algorithm_records,capability_point,learned_anchor,useful_conditions
    phases=[('algorithms',('DDQN','PPO')),('representative',('DDQN','PPO'))]
    phases += [(name,('PPO',)) for name in ('representative_recovery_PPO','representative_expansion',
        'representative_expansion_confirmation','long_credit','long_credit_confirmation',
        'success_ordinary','success_success_consolidation','priority_uniform','priority_failure_disagreements',
        'priority_uniform_confirmation','priority_failure_disagreements_confirmation','eligibility_reactive','eligibility_prospective',
        'eligibility_reactive_confirmation','eligibility_prospective_confirmation')]
    curves=[];games=[];budgets=[]
    fields=('opponent','opponent_kind','opponent_config','opponent_character','cpu_stars','player','offset',
        'result','end_reason','self_hp','opponent_hp','native_frames','ko_win','time_up','actual_character_ids','model_memory_resets')
    for phase,algorithms in phases:
        for seed in config['algorithm_comparison']['seeds']:
            for algorithm in algorithms:
                directory=artifact/phase/str(seed)/algorithm
                result_file=directory/('summary.json' if algorithm=='DDQN' else 'result.json')
                complete=result_file.exists()
                if complete:result=read(result_file)
                elif algorithm=='PPO' and (directory/'frozen_evaluations.json').exists():
                    result=dict(frozen_evaluations=read(directory/'frozen_evaluations.json'))
                else:continue
                records=algorithm_records(result,algorithm)
                if not records:continue
                family='algorithms' if phase=='algorithms' else 'representative'
                base=artifact/family/str(seed);initial=read(base/'initial_panel/summary.json')
                first_result=read(base/algorithm/('summary.json' if algorithm=='DDQN' else 'result.json'))
                learned=next(r['evaluation'] for r in algorithm_records(first_result,algorithm) if r['requested']==1000000)
                anchor=learned_anchor(initial,learned)
                parent_file=artifact/'success_preparation'/str(seed)/'summary.json'
                parent=read(parent_file) if parent_file.exists() and phase.startswith(('success_','priority_','eligibility_')) else None
                first_new=None
                for record in records:
                    evaluation=record['evaluation'];p=capability_point(evaluation,anchor,record['native_frames'],record['checkpoint'])
                    row=dict(phase=phase,algorithm=algorithm,seed=seed,requested_native_frames=record['requested'],
                        actual_phase_native_frames=record['native_frames'],checkpoint=record['checkpoint'],
                        cpu_ko=p['cpu_ko'],cpu_games=sum(r['opponent_kind']=='cpu' for r in evaluation['rows']),
                        pvp_wins=p['pvp_wins'],fixed1m_kept=p['starting_kept'],fixed1m_required=p['starting_useful'],
                        evaluation_native_frames=evaluation['native_frames'])
                    if parent is not None:
                        useful=useful_conditions(evaluation['rows']);old=useful_conditions(parent['rows']);new=useful-old
                        if first_new is None:first_new=new
                        row.update(shared5m_kept=len(useful&old),shared5m_required=len(old),new_beyond5m=len(new),
                            first_checkpoint_new_cohort=len(first_new),first_new_still_kept=len(first_new&useful))
                    curves.append(row)
                    games.extend(dict(phase=phase,algorithm=algorithm,seed=seed,requested_native_frames=record['requested'],
                        actual_phase_native_frames=record['native_frames'],checkpoint=record['checkpoint'],
                        **{k:r[k] for k in fields if k in r}) for r in evaluation['rows'])
                if complete:
                    cost=(dict(result['costs']) if algorithm=='DDQN' else {k:result[k] for k in
                        ('environment_calls','optimizer_updates','optimization_samples','optimization_seconds','auxiliary_updates',
                         'auxiliary_samples','auxiliary_seconds','replay_assessment_seconds','success_collection_native_frames',
                         'success_collection_environment_calls','evaluation_native_frames','evaluation_seconds','seconds') if k in result})
                    budgets.append(dict(category='training',phase=phase,seed=seed,algorithm=algorithm,
                        actual_native_frames=result['actual_training_frames'] if algorithm=='DDQN' else result['native_frames'],costs=cost))
    for seed in config['seeds']:
        for arm in config['ddqn_comparison']['arms']:
            for segment in ('pilot','initial','continuation'):
                path=artifact/'ddqn'/str(seed)/arm/segment/'summary.json'
                if path.exists():
                    result=read(path);budgets.append(dict(category='training',phase=f'ddqn/{arm}/{segment}',seed=seed,algorithm='DDQN',
                        actual_native_frames=result['actual_training_frames'],costs=result['costs']))
    for path in sorted((artifact/'readiness').glob('*/*/summary.json')):
        evaluation=read(path);arguments=read(path.with_name('config.json'))['arguments']
        arm=path.parents[1].name;seed=int(path.parent.name)
        games.extend(dict(phase='frozen_readiness',arm=arm,seed=seed,checkpoint=arguments['agent'],
            **{k:r[k] for k in fields if k in r}) for r in evaluation['rows'])
        budgets.append(dict(category='evaluation',phase='frozen_readiness',arm=arm,seed=seed,
            actual_native_frames=evaluation['native_frames'],costs=dict(optimizer_updates=0,
                games=len(evaluation['rows']),seconds=evaluation['seconds'],setup_native_frames=evaluation.get('setup_native_frames'))))
    return dict(curves=curves,budgets=budgets,records=games,
        scope='Consumed development conditions only. Counts are per fixed independent training seed; phases and checkpoints in one lineage are dependent. Native frames shown here are additional within that phase. Auxiliary forward/audit times are subsets of total optimization time, not additive costs.')


def plot_capability(evidence,report):
    """Show retained and added whole-game results separately for each trial."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    studies=(('Self-success replay',('success_ordinary','success_success_consolidation')),
             ('Failure-disagreement sampling',('priority_uniform','priority_failure_disagreements')),
             ('Disagreement eligibility',('eligibility_reactive','eligibility_prospective')))
    figure,axes=plt.subplots(len(studies),2,figsize=(11,10),sharex=True)
    colors=('#b45309','#0369a1')
    for row,(title,phases) in enumerate(studies):
        for index,phase in enumerate(phases):
            points=[p for p in evidence['curves'] if p['phase']==phase]
            for seed in sorted({p['seed'] for p in points}):
                series=sorted((p for p in points if p['seed']==seed),key=lambda p:p['actual_phase_native_frames'])
                x=[p['actual_phase_native_frames']/1e6 for p in series]
                if phase.startswith(('priority_','eligibility_')):
                    continuation=sorted((p for p in evidence['curves'] if p['phase']==phase+'_confirmation' and p['seed']==seed),
                        key=lambda p:p['actual_phase_native_frames'])
                    offset=x[-1]
                    x.extend(offset+p['actual_phase_native_frames']/1e6 for p in continuation)
                    series.extend(continuation)
                label=f'{phase.removeprefix("success_").removeprefix("priority_")} / {seed}'
                style='-' if seed==57031 else '--'
                axes[row,0].plot(x,[p['shared5m_kept']/p['shared5m_required'] for p in series],
                    style,marker='o',color=colors[index],label=label)
                axes[row,1].plot(x,[p['new_beyond5m'] for p in series],style,marker='o',color=colors[index],label=label)
        axes[row,0].set_title(title+' — retained fixed parent outcomes')
        axes[row,1].set_title(title+' — additional useful outcomes')
        axes[row,0].set_ylim(0,1.05);axes[row,0].set_ylabel('Retained fraction (14 / 12 per seed)')
        axes[row,1].set_ylabel('CPU KO or native PvP win')
        for axis in axes[row]:
            axis.grid(alpha=.25);axis.legend(fontsize=7)
            axis.set_xlabel('Additional training and collection native frames (millions)')
    figure.suptitle('Guile: consumed 19-condition development panel; checkpoints are dependent')
    figure.tight_layout();figure.savefig(report/'capability_learning.png',dpi=140);plt.close(figure)


def publish_capability(config,artifact,report):
    """Use the existing publication boundary for the capability study too."""
    state=read(report/'summary.json')
    if state.get('format')=='compact_guile_capability_accumulation':state=read(artifact/'full_capability_state.json')
    final=state.get('capability_final',{})
    cleanup=state.get('process_cleanup',{})
    if not final.get('frozen') or not cleanup.get('verified') or cleanup.get('active_related_processes')!=[]:
        raise RuntimeError('Finish the actual stronger capability gate, frozen readiness, export/behavior and process cleanup first')
    if not final.get('stronger_gate_passed_on_both_seeds') or not final.get('technical_checks_passed'):
        raise RuntimeError('An endpoint or load check alone is not completed capability accumulation')
    write(artifact/'full_capability_state.json',state)
    evidence=capability_evidence(config,artifact)
    plot_capability(evidence,report)
    with (report/'complete_games.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for row in evidence.pop('records'):stream.write(json.dumps(portable(row),ensure_ascii=False,separators=(',',':'))+'\n')
    keep=('started_at_utc','configuration','source_review','diagnostic','diagnostic_fixture','diagnostic_videos','causal_update_evidence',
        'propagation_profile','prepared_trace_profile','DDQN_frozen_artifacts','algorithm_interruption','shared_learning_checks',
        'automatic_fallback_reason','selected_DDQN_backup','algorithm_initialization','representative_initialization',
        'representative_2m_behavior','representative_memory_audit','initial_observation_alias_audit',
        'on_policy_credit_diagnosis','success_causal_diagnosis','protected_success_diagnosis','success_freshness_probe',
        'reactive_refresh_proposal_status','success_label_conflict_audit','priority_confirmation_complete','prospective_eligibility_diagnosis',
        'eligibility_initialization_audit','eligibility_early_actual_use',
        'current_success_label_audit','prospective_Ken_fit','optimizer_snapshot_issue','sparse_supervision_probe',
        'priority_initialization_audit','priority_smoke','priority_preparation','priority_prepared_label_audit','priority_relapse_trace')
    selected_evidence={k:state[k] for k in keep if k in state}
    terminal_fields=('opponent','opponent_config','opponent_kind','opponent_character','cpu_stars','player','offset',
                     'result','end_reason','self_hp','opponent_hp','native_frames','ko_win','model_memory_resets')
    for name in ('success_freshness_probe','sparse_supervision_probe'):
        if name in selected_evidence:
            probe=selected_evidence[name]
            selected_evidence[name]=dict(probe,results={arm:dict(result,rows=[{k:r[k] for k in terminal_fields if k in r}
                for r in result['rows']]) for arm,result in probe['results'].items()})
    if 'priority_preparation' in selected_evidence:
        selected_evidence['priority_preparation']={seed:{k:v for k,v in prepared.items()
            if k not in ('initial_outcomes','captures')} | dict(captures=[dict(source=c['source'],
                native_frames=c['native_frames'],conditions=[{k:r[k] for k in
                    ('opponent_config','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')}
                    for r in c['rows']]) for c in prepared['captures']])
            for seed,prepared in selected_evidence['priority_preparation'].items()}
    # Detailed match lists live once in the JSONL; do not repeat them inside
    # every carried-forward phase and every selected-model description.
    ddqn={arm:{seed:dict(gate=value['gate'],newly_verified_history=value['newly_verified_history'],points=[{k:p[k] for k in
        ('native_frames','cpu_ko','cpu_time_wins','pvp_wins','starting_kept','starting_useful') if k in p} for p in value['points']])
        for seed,value in method['seeds'].items()} for arm,method in state['ddqn_comparison']['arms'].items()}
    # Full opponent-by-opponent rows are retained once in complete_games.jsonl.
    # Keep role/star/port aggregates here without another six copies of that list.
    final=dict(final,readiness=dict(final['readiness'],results={arm:{seed:{k:v for k,v in result.items() if k!='by_opponent'}
        for seed,result in seeds.items()} for arm,seeds in final['readiness']['results'].items()}))
    headline=dict(conclusion='有限19条件面板通过持续能力积累门槛；较宽旧胜局仍有退化，不晋级，不宣称完成adaptive/robust。',
        route=final['route'],representative_model=final['representative_model'],representative_seed=final['representative_seed'],
        candidate_selection_before_broad_results=True,
        small_panel={seed:dict(old_kept=v['checkpoints'][-1]['starting_kept'],old_required=v['checkpoints'][-1]['starting_useful'],
            persistent_new=v['gate']['passing_pairs'][-1]['persistent_new'])
            for seed,v in final['selection'][final['selected_arm']]['seeds'].items()},
        broad_results=[dict(arm=arm,seed=seed,cpu_ko=result['cpu_ko'],cpu_conditions=96,
            pvp_wins=result['overall']['wins'],pvp_conditions=204,
            P1_wins=result['by_port']['1']['wins'],P2_wins=result['by_port']['2']['wins'])
            for arm,seeds in final['readiness']['results'].items() for seed,result in seeds.items()],
        broader_parent_retention={seed:{k:v[k] for k in ('reference_wins','old_wins_kept','reference_pve_ko','pve_ko_kept')}
            for seed,v in final['readiness']['shared5m_changes'][final['selected_arm']].items()},
        formal_promotion_eligible=final['nomination_eligible'],remaining_bottleneck=final['next_evidence_supported_bottleneck'])
    compact=dict(format='compact_guile_capability_accumulation',stage='finite_capability_gate_verified',
        goal_achieved=False,stage_gate_achieved=True,formal_reference_replaced=False,
        research_decision=headline,
        setup=dict(character='Guile',game='StreetFighterIISpecialChampionEdition-Genesis',mode='Hyper',speed_stars=2,decision_native_frames=6),
        configuration='configs/guile_capability_accumulation.json',
        evidence=selected_evidence,learning=evidence,resumed_DDQN=ddqn,
        selected=final,process_cleanup=cleanup,complete_game_records='reports/'+config['experiment']+'/complete_games.jsonl',
        earlier_DDQN_records='reports/'+config['experiment']+'/ddqn_matches.jsonl',
        limits=['The19-case capability panel is consumed development evidence, including nine1starCPU conditions, not96-case KO coverage.',
            'The old1m9/6 cohorts and the stronger shared5m14/12 cohorts stay separate. Only the stronger repeated old+new result closes this stage.',
            'Native wins, KO, remaining HP and time-up are distinct. KO never requires full own HP.',
            'Fresh full-pool DDQN/PPO matched starts and the later curriculum/auxiliary protocols are separate comparisons.',
            'The historical160-game Ryu panel is not fully migrated. Ryu76/90 formal protection is not certified by this Guile study.',
            'The96CPU and204PvP readiness pool was already used in training; it is not an unknown holdout or proof of universal robustness.',
            'Source-versus-export identity is technical validation, not tactical ability or official Arena acceptance.'],
        official_arena='Not executed. Original course scanning, packing and validation entries remain unused.')
    write(report/'summary.json',portable(compact))
    print('PUBLISHED_COMPACT_CAPABILITY_EVIDENCE',report,flush=True)


def retain_weights(source,destination,keep_optimizer=False):
    """Preserve one inference model through the shared writer, without caches."""
    source=Path(source);source=source if source.is_absolute() else ROOT/source
    destination=Path(destination);destination=destination if destination.is_absolute() else ROOT/destination
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        if keep_optimizer:shutil.copyfile(source,destination)
        else:
            actor=load_actor(source)
            try:save(destination,actor.policy,actor.definition,
                training=dict(source=source.relative_to(ROOT).as_posix(),purpose='Frozen research evidence; optimizer state omitted'))
            finally:actor.close()
    return destination.relative_to(ROOT).as_posix()


def model_identity(source,delivered):
    """Direct tensors/definitions; explicitly distinguish a deployment catalog."""
    import torch
    original=load_actor(ROOT/source);retained=load_actor(ROOT/delivered)
    try:
        before=original.policy.state_dict();after=retained.policy.state_dict()
        mapping=retained.definition.get('deployment_action_indices') if original.actions!=retained.actions else None
        catalog_verified=False
        if mapping is not None:
            import copy
            restored=copy.deepcopy(retained.definition)
            for field in ('deployment_action_indices','internal_action_count','internal_action_mirror'):restored.pop(field,None)
            restored['actions']=original.actions
            inverse={old:new for new,old in enumerate(mapping)}
            expected_mirror=torch.tensor([inverse[int(before['physical_mirror'][i])] for i in mapping])
            catalog_verified=(not original.definition.get('deployment_action_indices')
                and restored==original.definition and retained.actions==[original.actions[i] for i in mapping]
                and torch.equal(after['physical_mirror'],expected_mirror))
            if not catalog_verified:raise ValueError('Unexplained delivery catalog/definition change')
        different=[key for key in before if not (catalog_verified and key=='physical_mirror')
            and (key not in after or not torch.equal(before[key],after[key]))]
        extra=[key for key in after if key not in before];definition_equal=original.definition==retained.definition
        if different or extra or not (definition_equal or catalog_verified):raise ValueError(f'Delivery changed policy parameters or definition: {delivered}')
        result=dict(source=source,delivered=delivered,tensors=len(before),different_tensors=0,definition_equal=definition_equal)
        if catalog_verified:result.update(explicit_catalog_restriction=True,original_actions=len(original.actions),
            delivered_actions=len(retained.actions),recurrent_and_output_parameters_unchanged=True,
            scope='Full internal tensors retained; external choices are restricted. Deterministic identity requires actual-history tests, not this parameter check alone.')
        return result
    finally:original.close();retained.close()


def development_changes(config):
    from experiments.league_adaptive.evaluate import paired_regression
    def rows(stage,seed,arm):
        if config.get('own_character'):
            if stage=='pretrain':arm='ordinary'
            return read(ARTIFACT/stage/str(seed)/arm/'summary.json')['curve'][-1]['development']['rows']
        base=read(ARTIFACT/stage/str(seed)/arm/'panel/summary.json')['rows']
        return base+read(ARTIFACT/'selection'/stage/f'{arm}_{seed}'/'additional/summary.json')['rows']
    result={}
    phases=(['pretrain'] if config.get('own_character') else [])+selection_stages(config)
    for stage in phases[1:]:
        sources=[phases[phases.index(stage)-1]] if config.get('own_character') else (['comparison'] if stage=='feedback' else ['comparison','feedback'])
        if config.get('own_character') and config.get(stage,{}).get('parent_stage')=='replay_prefill':sources=['pretrain','paired']
        for source in sources:
            for seed in config['comparison']['seeds']:
                for arm in config['comparison']['arms']:
                    old,new=rows(source,seed,arm),rows(stage,seed,arm)
                    key=lambda r:tuple(r[k] for k in ('opponent','opponent_kind','player','offset'))
                    configurations={key(r):r['opponent_config'] for r in old}
                    if any(configurations[key(r)]!=r['opponent_config'] for r in new):raise ValueError('Matched development opponent versions differ')
                    change=paired_regression(old,new)
                    result[f'{source}->{stage}/{arm}_{seed}']=dict(reference_wins=change['reference_wins'],
                        old_wins_kept=change['old_wins_kept'],new_wins=[r['condition'] for r in change['new_wins']],
                        old_wins_lost=[dict(condition=r['condition'],result=r['candidate']['result']) for r in change['old_wins_lost']],
                        reference_pve_ko=change['reference_pve_ko'],pve_ko_kept=change['pve_ko_kept'],
                        pve_ko_lost=[dict(condition=r['condition'],result=r['candidate']['result'],end_reason=r['candidate']['end_reason']) for r in change['pve_ko_lost']],
                        episodes=len(old),scope='Same known cases and exact opponent versions, not the formal90-protection gate or new holdout. Warm-replay runs restart from their recorded original parent, not necessarily the preceding displayed stage.')
    return result


def plot_coverage(config,state,*,report_root=None,pvp_label='novel-rule PvP'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=list(state['final_results']);characters=config.get('opponent_characters',config.get('characters'))
    outcome_labels=dict(win='T',draw='D',loss='L',cutoff='C')
    fig,axes=plt.subplots(1,len(names),figsize=(15,7),squeeze=False)
    for column,name in enumerate(names):
        result=state['final_results'][name];values=np.zeros((len(characters),10));labels={}
        for row,character in enumerate(characters):
            for star in range(1,9):
                case=next(r for r in result['cpu']['rows'] if r['opponent_character']==character and r['cpu_stars']==star)
                text='K' if case['ko_win'] else outcome_labels[case['result']]
                values[row,star-1]=1. if text=='K' else .5 if text=='T' else 0.;labels[row,star-1]=text
            for port in (1,2):
                cases=[r for r in result['rules']['rows'] if r['opponent_character']==character and r['player']==port]
                wins=sum(r['result']=='win' for r in cases);values[row,7+port]=wins/len(cases) if cases else np.nan
                labels[row,7+port]=f'{wins}/{len(cases)}' if cases else '-'
        axis=axes[0,column];axis.imshow(values,vmin=0,vmax=1,cmap='Greens',aspect='auto')
        axis.set_xticks(range(10),[str(s) for s in range(1,9)]+['P1','P2']);axis.set_yticks(range(len(characters)),characters)
        axis.set_title(name+' (own '+('Ryu' if name=='reference' else state['selected_character'])+')')
        axis.set_xlabel('CPU difficulty stars | '+pvp_label)
        axis.axvline(7.5,color='black',linewidth=1)
        for (row,col),text in labels.items():axis.text(col,row,text,ha='center',va='center',fontsize=8,color='white' if values[row,col]>.7 else 'black')
    fig.suptitle('CPU known coverage: K=KO, T=time-up win, L=loss, D=draw, C=cutoff; PvP: '+pvp_label)
    fig.tight_layout();fig.savefig((Path(report_root) if report_root else REPORT)/'frozen_coverage.png',dpi=150);plt.close(fig)


def evaluation_stats(result):
    rows=result['rows'];cpu=[r for r in rows if r['opponent_kind']=='cpu']
    pvp_characters={}
    for character in sorted({r['opponent_character'] for r in rows if r['opponent_kind']!='cpu'}):
        cases=[r for r in rows if r['opponent_kind']!='cpu' and r['opponent_character']==character]
        pvp_characters[character]={str(port):dict(episodes=sum(r['player']==port for r in cases),
            **{label:sum(r['player']==port and r['result']==outcome for r in cases)
               for label,outcome in [('wins','win'),('losses','loss'),('draws','draw'),('cutoffs','cutoff')]})
            for port in (1,2)}
    by_kind={kind:dict(episodes=sum(r['opponent_kind']==kind for r in rows),
        **{label:sum(r['opponent_kind']==kind and r['result']==outcome for r in rows)
           for label,outcome in [('wins','win'),('losses','loss'),('draws','draw'),('cutoffs','cutoff')]})
        for kind in sorted({r['opponent_kind'] for r in rows})}
    return dict(overall=result['overall'],by_port=result['by_port'],pve=result['pve'],
        by_kind=by_kind,by_opponent=result['by_opponent'],pvp_by_opponent_character=pvp_characters,
        episodes=len(rows),native_frames=result['native_frames'],
        setup_native_frames=result.get('setup_native_frames'),
        cpu_ko=sum(r['ko_win'] for r in cpu),cpu_timeup_wins=sum(r['result']=='win' and r['time_up'] for r in cpu),
        cpu_by_character={character:dict(episodes=sum(r['opponent_character']==character for r in cpu),
            ko=sum(r['opponent_character']==character and r['ko_win'] for r in cpu),
            timeup_wins=sum(r['opponent_character']==character and r['result']=='win' and r['time_up'] for r in cpu),
            losses=sum(r['opponent_character']==character and r['result']=='loss' for r in cpu),
            draws=sum(r['opponent_character']==character and r['result']=='draw' for r in cpu))
            for character in sorted({r['opponent_character'] for r in cpu})},
        cpu_by_star={str(s):dict(episodes=sum(r['cpu_stars']==s for r in cpu),
            wins=sum(r['cpu_stars']==s and r['result']=='win' for r in cpu),
            ko=sum(r['cpu_stars']==s and r['ko_win'] for r in cpu),
            timeup_wins=sum(r['cpu_stars']==s and r['result']=='win' and r['time_up'] for r in cpu),
            losses=sum(r['cpu_stars']==s and r['result']=='loss' for r in cpu),
            draws=sum(r['cpu_stars']==s and r['result']=='draw' for r in cpu),
            cutoffs=sum(r['cpu_stars']==s and r['result']=='cutoff' for r in cpu))
            for s in range(1,9) if any(r['cpu_stars']==s for r in cpu)})


def branch_statistics(directory):
    """Real short-window effects, with facing errors separated from labels."""
    training=read(directory/'summary.json');path=directory/'branches.json'
    if not path.exists() or not training['branch_summary']['sequences']:return None,[]
    facts={r['episode']-1:r for r in training['rows']};records=read(path)
    import torch
    definition=torch.load(directory/'agent.pt',map_location='cpu',weights_only=False)['definition']
    from experiments.league_adaptive.characters import PUNCHES
    groups=defaultdict(Counter);kinds=defaultdict(Counter);examples={}
    for b in records:
        episode=int(re.search(r'/episode(\d+)/root',b['family'])[1]);fact=facts[episode]
        p=fact['player'];q=3-p;root=b['root_readings'];context=b.get('context','generic')
        stats=Counter();stats['sequences']+=1;stats['native_frames']+=b['native_frames']
        stats['continuous_back60_at_root']+=bool(b.get('input_history',{}).get('continuous_back_60'))
        stats['continuous_down60_at_root']+=bool(b.get('input_history',{}).get('continuous_down_60'))
        stats['different_inputs']+=bool(b['different_from_factual'])
        stats['same_horizon_comparisons']+=bool(b['factual_same_horizon_complete'])
        stats['unmatched_horizon']+=not b['factual_same_horizon_complete']
        stats['changed_position_or_health']+=bool(b['different_positions_or_health'])
        stats['damage_dealt']+=b['damage_dealt'];stats['damage_taken']+=b['damage_taken']
        dx=root[f'p{q}_x']-root[f'p{p}_x'];front='RIGHT' if dx>=0 else 'LEFT';back='LEFT' if dx>=0 else 'RIGHT'
        root_sign=1 if dx>=0 else -1;previous=root;crossed=False
        index=int(b['family'].rsplit('branch',1)[1])
        if context in ('far_distance','mid_range') and index==0:style='boom'
        elif context=='mid_range' or index==0:style='flash'
        else:style='hold'
        for step in b['trace']:
            before_dx=previous[f'p{q}_x']-previous[f'p{p}_x']
            changed=before_dx*root_sign<0;crossed|=changed
            # The action catalogue is the actual saved learner definition,
            # not a reconstruction from a move label or opponent action.
            name=definition['actions'][step['action']]['name']
            stats['transitions']+=1;stats['not_frozen_actor_greedy']+=step['action']!=step['unforced_greedy']
            is_forced=step.get('forced_input',True)
            stats['forced_input_transitions' if is_forced else 'greedy_continuation_transitions']+=1
            stats['old_back_hold_after_coordinate_cross']+=is_forced and changed and name=='DOWN_'+back
            stats['release_toward_old_side_after_coordinate_cross']+=is_forced and changed and style=='boom' and name in [front+'_'+button+'_tap' for button in PUNCHES]
            previous=step['coordinates']
            crossed|=(previous[f'p{q}_x']-previous[f'p{p}_x'])*root_sign<0
        stats['side_changed_inside_window']+=crossed
        if b['factual_same_horizon_complete']:
            margin=lambda raw:max(raw[f'p{p}_hp'],0)-max(raw[f'p{q}_hp'],0)
            delta=margin(b['final_readings'])-margin(b['factual_final_readings'])
            label='better_window_hp' if delta>0 else 'not_better_window_hp';stats[label]+=1
            # First positive/nonpositive window per context and port; never
            # swap examples after looking at later independent match results.
            key=(context,p,label)
            if key not in examples:
                examples[key]=dict(player=p,offset=fact['offset'],opponent_configuration=fact['opponent_config'],
                    window_hp_margin_difference=delta,record=b,
                    scope='Recorded legal forced prefix and any frozen-greedy continuation under responsive opponent; bootstrap at cut. No independent standard-start victory, global oracle or regret label.')
        groups[f'{context}/P{p}'].update(stats)
        kinds[fact['opponent_config']['kind']].update(stats)
    if sum(g['native_frames'] for g in groups.values())!=training['costs']['counterfactual_training']:
        raise ValueError('Branch native-frame accounting differs from actual training costs')
    return dict(by_context_and_port=dict(groups),by_opponent_kind=dict(kinds),
        scope='One-step off-policy transitions. Coordinate-side changes flag possible stale directions; they do not prove immediate sprite facing, lost charge, move activation or full-game improvement.'),list(examples.values())


def counter_replay_evidence(directory,counter_ids):
    """Recent replay evidence without inventing past sampling counts."""
    path=directory/'replay.pt'
    if not path.exists() or not counter_ids:return None
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)['replay'];examples=[]
    for number,episode in state['episodes'].items():
        rival=episode.get('opponent_config',{})
        if rival.get('id') not in counter_ids:continue
        priorities=[v for (i,_),v in state['priorities'].items() if i==number]
        # add() assigns one common priority to all windows in an episode;
        # only an actual valid TD update can make those priorities differ.
        if len(set(priorities))>1:
            examples.append(dict(opponent=rival,player=episode['player'],source=episode['source'],
                windows=len(priorities),minimum_priority=min(priorities),maximum_priority=max(priorities)))
    return dict(recent_counter_sequences_with_td_evidence=examples,
        scope='Lower-bound evidence from the bounded terminal replay: different within-sequence priorities imply actual valid TD sampling. A sequence can be a factual episode or a short alternative fragment. This does not reconstruct exact past counter sample counts or prove independent match gains.')


def publish(configuration='configs/counterfactual_dqn.json',run_root=None,report_root=None):
    global ARTIFACT,REPORT
    setup();config=read(ROOT/configuration);experiment=config.get('experiment','counterfactual_dqn')
    ARTIFACT=ROOT/(run_root or 'artifacts/'+experiment);REPORT=ROOT/(report_root or 'reports/'+experiment)
    if config.get('protocol')=='character_algorithm_breakthrough':
        return publish_breakthrough(config,ARTIFACT,REPORT)
    if config.get('protocol')=='stabilization':
        from experiments.ryu_counterq.stabilization import publish
        return publish(config,ARTIFACT,REPORT)
    if config.get('protocol')=='capability_accumulation':
        return publish_capability(config,ARTIFACT,REPORT)
    state=read(REPORT/'summary.json');fixed=bool(config.get('own_character'));prefix='models/'+experiment
    archive=ARTIFACT/'full_study_state.json'
    if state.get('format','').startswith('compact_'):state=read(archive)
    cleanup=state.get('process_cleanup',{})
    expected_videos=sum(len(c['models']) for c in config['final']['behavior_cases'])
    if (not state.get('final_frozen') or len(state.get('export_validation',{}))!=2
        or len(state.get('behavior',[]))!=expected_videos or not cleanup.get('verified')
        or cleanup.get('active_related_processes')!=[]):
        raise RuntimeError('Finish frozen evaluation, export identity, movies and actual process cleanup before publishing')
    write(archive,state);model_map={};published_models={}
    for arm,lineage in state['frozen_lineage'].items():
        model_map[lineage['source']]=state['frozen_models'][arm]
        published_models[state['frozen_models'][arm]]=lineage['source']

    def retain_model(source,destination,keep_optimizer=False):
        source=Path(source);source=source if source.is_absolute() else ROOT/source
        relative=source.relative_to(ROOT).as_posix();destination=ROOT/destination
        if relative in model_map:return model_map[relative]
        retain_weights(source,destination,keep_optimizer)
        model_map[relative]=destination.relative_to(ROOT).as_posix();published_models[model_map[relative]]=relative
        return model_map[relative]

    if fixed:
        for seed in config['comparison']['seeds']:
            retain_model(ARTIFACT/'untrained'/str(seed)/'agent.pt',f'{prefix}/controls/untrained/{seed}.pt')
    for stage in config['storage']['public_control_stages']:
        for seed in config['comparison']['seeds']:
            for arm in (['ordinary'] if fixed and stage=='pretrain' else config['comparison']['arms']):
                retain_model(ARTIFACT/stage/str(seed)/arm/'agent.pt',
                    f'{prefix}/controls/{stage}/{arm}_{seed}.pt',keep_optimizer=fixed and stage=='pretrain')
    if fixed and config.get('replay_prefill'):
        for seed in config['comparison']['seeds']:
            origin=(ARTIFACT/'pretrain'/str(seed)/'ordinary/agent.pt').relative_to(ROOT).as_posix()
            unchanged=(ARTIFACT/'replay_prefill'/str(seed)/'ordinary/agent.pt').relative_to(ROOT).as_posix()
            model_map[unchanged]=model_map[origin]
    for rival in state.get('training_counter_snapshots',state['retained_training_counters'])+state['final_counter_pool']:
        retain_model(rival['path'],f"{prefix}/opponents/{rival['id']}.pt")

    # Dropping optimizer state must leave the actual policy unchanged. Compare
    # numerical model parameters and definitions directly, never file digests.
    policy_identity=[model_identity(source,delivered) for delivered,source in published_models.items()]

    first_seed=str(config['comparison']['seeds'][0])
    pools=dict(base=config['pool'] if fixed else read(ARTIFACT/'comparison'/first_seed/'ordinary/config.json')['pool'],
        training=state['league_pool'] if fixed else state['feedback_pool'],final_rules=read(ARTIFACT/'final_rules.json'),
        final_counters=state['final_counter_pool'])
    pools['cpu']=[r['opponent_config'] for r in state['final_results']['reference']['cpu']['rows']]
    for name,pool in pools.items():write(REPORT/f'{name}_pool.json',portable(pool,model_map))

    row_fields=('opponent','opponent_kind','player','offset','result','end_reason','self_hp','opponent_hp',
        'native_frames','character','opponent_character','opponent_config','actual_character_ids',
        'ko_win','time_up','elapsed_seconds','cpu_stars','model_memory_resets')
    patterns=('screen/*/panel/summary.json','screen_refinement/*/panel/summary.json',
        'shortlist/*/*/panel/summary.json','untrained/*/*/panel/summary.json',
        'selection/*/*/additional/summary.json',
        'training_counter_panels/*/summary.json','known_reference_panel/summary.json',
        'final_rules/*/summary.json','cpu_coverage/*/summary.json','known_focus/*/summary.json',
        'final_counter_panels/*/summary.json','old_protection/*/summary.json','exports/*/complete_matches/summary.json')
    patterns+=tuple(f'{stage}/*/*/{folder}/summary.json' for stage in selection_stages(config)
                    for folder in ('panel','light_development_*'))
    if fixed:
        patterns+=('untrained/*/development/summary.json','expanded_development/*/cpu/summary.json')
        patterns+=tuple(f'{stage}/*/*/development_*/summary.json' for stage in ['pretrain']+selection_stages(config))
    sources=sorted({p for pattern in patterns for p in ARTIFACT.glob(pattern)})
    protected={};provenance={}
    reference_rows=state['old_protection']['reference']['evaluation']['rows']
    for original in read(ROOT/config['acceptance']['old_protection'])['rows']:
        if not original['protected'] and not config['acceptance'].get('full_consumed_regression',False):continue
        opponent_configuration=original['opponent_configuration'];matches=[]
        for row in reference_rows:
            restored=dict(row['opponent_config']);restored['id']=opponent_configuration['id']
            if (row['opponent'].split('_',2)[2]==opponent_configuration['id'] and restored==opponent_configuration
                and row['player']==original['player'] and row['offset']==original['offset']):matches.append(row)
        if len(matches)!=1:raise ValueError('Protected condition version cannot be identified uniquely')
        row=matches[0];key=tuple(row[k] for k in ('opponent','player','offset'))
        provenance[key]={k:original[k] for k in ('source_panel','source_config','display_id','opponent_weights',
            'original52','expanded64','reference_result','parent_result','previous_candidate_result')}
        if original['protected']:protected[key]=provenance[key]
    if len(protected)!=90:raise ValueError('The fixed protection union changed')
    panels={}
    with (REPORT/'episodes.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for path in sources:
            result=read(path);panel=path.parent.relative_to(ARTIFACT).as_posix();panels[panel]=evaluation_stats(result)
            exact_pool=[]
            for row in result['rows']:
                if row['opponent_config'] not in exact_pool:exact_pool.append(row['opponent_config'])
            panels[panel].update(pool=exact_pool,
                conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in result['rows']],
                runnable_with_delivered_opponents=all(r['kind']!='agent' or r['path'] in model_map
                    or not r['path'].startswith('artifacts/') for r in exact_pool))
            for row in result['rows']:
                compact=dict(panel=panel,**{k:row[k] for k in row_fields if k in row})
                if panel.startswith('old_protection/'):
                    protection=protected.get(tuple(row[k] for k in ('opponent','player','offset')))
                    compact.update(protected_condition=protection is not None)
                    compact['original_condition']=provenance[tuple(row[k] for k in ('opponent','player','offset'))]
                    if protection is not None:compact['protection']=protection
                if panel.startswith(('final_rules/','final_counter_panels/')):
                    compact.update({k:row[k] for k in ('motion','opponent_phase_profile','after_opponent_switch') if k in row})
                compact=portable(compact,model_map)
                opponent=compact['opponent_config']
                compact['opponent_weight_delivered']=opponent.get('kind')!='agent' or not opponent.get('path','').startswith('artifacts/')
                if panel.startswith(('final_rules/','cpu_coverage/','known_focus/','final_counter_panels/','old_protection/')):
                    if opponent.get('path') and not (ROOT/opponent['path']).is_file():raise FileNotFoundError(opponent['path'])
                    if not compact['opponent_weight_delivered']:raise ValueError('Final panel opponent was not delivered')
                stream.write(json.dumps(compact,ensure_ascii=False,separators=(',',':'))+'\n')
    write(REPORT/'panels.json',portable(panels,model_map))

    runs=[];totals=defaultdict(Counter);run_pools=[];branch_examples=[];recordings={}
    for path in sorted(ARTIFACT.rglob('summary.json')):
        data=read(path)
        if 'video' in data and 'native_frames' in data:
            recordings[path.parent.relative_to(ARTIFACT).as_posix()]={k:data[k] for k in ('native_frames','agent','opponent','player','offset','result')}
        if 'actual_training_frames' not in data or 'costs' not in data:continue
        relative=path.parent.relative_to(ARTIFACT).as_posix();group=relative.split('/')[0]
        invalid_source_metrics=[]
        if group=='branch_feasibility':
            audit=state.get('branch_feasibility',{}).get('source_accounting',{})
            if audit.get('original_pilot_source_counters_valid') is False:
                invalid_source_metrics=audit['invalid_fields']
        costs={k:v for k,v in data['costs'].items() if k not in invalid_source_metrics}
        costs.setdefault('optimizer_updates',0)
        totals[group].update({k:v for k,v in costs.items() if isinstance(v,(float,int))})
        saved=read(path.parent/'config.json');pool=portable(saved['pool'],model_map)
        if pool not in run_pools:run_pools.append(pool)
        runs.append(dict(run=relative,character=data['character'],seed=data['seed'],arm=data['arm'],
            native_frames=data['actual_training_frames'],costs=costs,excluded_invalid_source_metrics=invalid_source_metrics,
            visual_filter_update_l2=data.get('visual_filter_update_l2'),
            branch_summary=data.get('branch_summary'),configuration=(path.parent/'config.json').relative_to(ROOT).as_posix()))
        outcomes=defaultdict(Counter)
        for row in data['rows']:
            observed=outcomes[f"{row['opponent']}/P{row['player']}"]
            observed['episodes']+=1
            observed[row['result'] if row['complete'] else 'incomplete']+=1
            if row['complete'] and row['result']=='win':observed['ko' if row['end_reason']=='signed_hp_ko' else 'other_win']+=1
        runs[-1]['exploratory_training_outcomes']=dict(outcomes)
        if fixed:
            runs[-1]['counter_learning_evidence']=counter_replay_evidence(path.parent,{r['id'] for r in state['retained_training_counters']})
        if fixed and data['arm']=='counterfactual':
            audit,examples=branch_statistics(path.parent);runs[-1]['branch_audit']=audit
            if relative.startswith(('paired/','league_adaptation/')):
                branch_examples.extend(dict(run=relative,**example) for example in examples)
        runs[-1].update(opponent_pool_index=run_pools.index(pool),source_initialization=portable(saved.get('source'),model_map),
            source_replay=portable(saved.get('replay_source'),model_map),initial_replay_transitions=saved.get('initial_replay_transitions',0),
            actual_configuration={k:saved['config'][k] for k in ('learner','network','actions','counterfactual','sampling') if k in saved['config']},
            checkpoints=[{k:c[k] for k in ('threshold','actual_frames') if k in c} for c in data['curve']])
    write(REPORT/'budgets.json',dict(completed_training_runs=runs,opponent_pools=run_pools,training_totals_by_stage=dict(totals),
        recordings_by_physical_output=portable(recordings,model_map),
        evaluation_by_physical_output={k:dict(native_frames=v['native_frames'],setup_native_frames=v['setup_native_frames'],episodes=v['episodes']) for k,v in panels.items()},
        other_measured_costs=dict(
            complete_behavior_recordings_native_frames=sum(x['result']['native_frames'] for x in state['behavior']),
            final_notebook=state.get('technical_checks',{}).get('final_notebook_execution'),
            operation_and_state_restore=state.get('technical_checks',{}).get('operation'),
            physical_side_crossings=state.get('technical_checks',{}).get('crossings'),
            crossed_continuations=state.get('targeted_diagnosis',{}).get('crossed_continuations'),
            initial_notebook_and_interruption=state.get('notebook_execution',state.get('technical_checks',{}).get('notebook_execution')),
            guile_mechanics=state.get('mechanics'),guard_and_recovery=state.get('guard_and_recovery'),
            history_diagnosis=state.get('history_diagnosis_1m'),history_diagnosis_3m=state.get('history_diagnosis_3m'),
            branch_mechanism_audit=state.get('branch_feasibility'),continuation_feasibility=state.get('continuation_feasibility')),
        scope='Actual training frames comprise factual plus alternate simulation only. Per-run costs.evaluation is also indexed under physical evaluation outputs; do not add that overlap twice. Each physical recording is listed once in recordings_by_physical_output; complete_behavior_recordings_native_frames summarizes a subset of those same outputs and must not be added again. Composed selection scores add no simulation. Engineering/menu probes and interrupted unmeasured tails are separate; this is not an exact global native-frame total.'))
    if branch_examples:
        with (REPORT/'branch_examples.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
            for example in branch_examples:stream.write(json.dumps(portable(example,model_map),ensure_ascii=False,separators=(',',':'))+'\n')

    for arm in config['comparison']['arms']:
        destination=ROOT/prefix/'standalone'/arm;destination.mkdir(parents=True,exist_ok=True)
        for name in ('policy.zip','inference.py','student_model.py','student_actions.py','student_observation.py','policy_definition.json','runtime_versions.json'):
            shutil.copyfile(ARTIFACT/'exports'/arm/name,destination/name)
    behavior=publish_behavior_movies(state['behavior'],REPORT)

    comparison={}
    for phase in (['pretrain'] if fixed else [])+selection_stages(config):
        comparison[phase]={}
        for seed in config['comparison']['seeds']:
            arms=['ordinary'] if fixed and phase=='pretrain' else config['comparison']['arms']
            comparison[phase][str(seed)]={}
            for arm in arms:
                if fixed:
                    training=read(ARTIFACT/phase/str(seed)/arm/'summary.json');ev=training['curve'][-1]['development']
                else:
                    value=state[phase][str(seed)][arm];training=value['training'];ev=value['evaluation']
                comparison[phase][str(seed)][arm]=dict(evaluation=evaluation_stats(ev),native_frames=training['actual_training_frames'],
                    costs=training['costs'],branch_summary=training['branch_summary'])
    protection={}
    for model,data in state['old_protection'].items():
        protection[model]={k:data[k] for k in ('protected_wins_kept','required','passed','original52_kept','expanded64_kept','reference76_kept','pve_ko','evaluated_consumed_conditions','all_consumed_wins') if k in data}
        protection[model]['lost']=[dict(condition={k:x['condition'][k] for k in ('opponent','player','offset')},
            panel=x['condition']['original']['source_panel'],result=x['result']['result'],end_reason=x['result']['end_reason']) for x in data['lost']]
        if model=='reference':protection[model]['historical_result_mismatches']=data['historical_result_mismatches']
    lineages={}
    for arm,value in state['frozen_lineage'].items():
        lineage=dict(value);stages=[];source=ROOT/value['source'];chain=[]
        while source.parent.joinpath('summary.json').exists():
            if source.as_posix() in chain:raise ValueError('Training lineage contains a cycle')
            chain.append(source.as_posix());stages.insert(0,read(source.parent/'summary.json'))
            prior=read(source.parent/'config.json').get('source')
            if not prior or prior=='independent random Q initialization':break
            source=Path(prior) if Path(prior).is_absolute() else ROOT/prior
        lineage['training_chain']=[dict(source=Path(p).relative_to(ROOT).as_posix(),
            delivered=model_map.get(Path(p).relative_to(ROOT).as_posix()),
            acquired_native_frames=training['actual_training_frames'],
            optimizer_updates=training['costs'].get('optimizer_updates',0),
            collection_without_parameter_updates=training['costs'].get('optimizer_updates',0)==0)
            for p,training in zip(reversed(chain),stages)]
        lineage.update(actual_inherited_training_native_frames=sum(r['actual_training_frames'] for r in stages),
            collection_without_updates_native_frames=sum(r['actual_training_frames'] for r in stages if not r['costs'].get('optimizer_updates',0)),
            optimizer_update_calls=sum(r['costs'].get('optimizer_updates',0) for r in stages),
            role_screening_weights_inherited=False,
            scope='Only this actual lineage is counted. Alternative continuations and independent role screening are not accumulated into the delivered weight.')
        lineages[arm]=lineage
    guard=state.get('guard_and_recovery')
    if guard:
        guard=dict(guard,rows=[dict(row,probes={name:{k:v for k,v in probe.items() if k!='trace'}
            | dict(trace_decisions=len(probe.get('trace',[]))) for name,probe in row['probes'].items()})
            if 'probes' in row else row for row in guard['rows']],
            raw_trace_location=f'artifacts/{experiment}/guard_and_recovery.json')
    smoke=state.get('smoke')
    if smoke and smoke.get('development'):smoke=dict(smoke,development=evaluation_stats(smoke['development']))
    compact=dict(format='compact_'+experiment,stage=state.get('stage'),selected_character=state['selected_character'],
        research_goal_achieved=state.get('research_goal_achieved',False),
        started_at_utc=state.get('started_at_utc'),process_cleanup_at_utc=cleanup.get('checked_at_utc'),
        reference=config['reference'],reference_replaced=False,configuration=configuration,
        frozen_models=state['frozen_models'],frozen_lineage=lineages,
        final_frozen_at_utc=state['final_frozen_at_utc'],final_frozen=True,
        research_representative=state['research_representative'],representative_model=state['representative_model'],
        character_selection_scores=state.get('character_selection_scores'),
        character_selection_scope=state.get('character_selection_scope','Fixed own Guile by user instruction; no12-role rescreening or claim of global strongest character.'),
        comparison=comparison,development_changes=development_changes(config),selection_measurements=state['selection_measurements'],
        untrained_baselines={str(seed):evaluation_stats(read(ARTIFACT/'untrained'/str(seed)/'development/summary.json')) for seed in config['comparison']['seeds']} if fixed else None,
        training_counter_character=state.get('training_counter_character',config.get('training_counters',{}).get('counter_roles')),
        training_counter_snapshots=state['training_counter_snapshots'],retained_training_counters=state['retained_training_counters'],
        training_counter_scope=state['training_counter_scope'],training_counter_effectiveness=state.get('training_counter_effectiveness'),final_counter_pool=state['final_counter_pool'],
        counter_feedback_changes=state.get('counter_feedback_changes'),
        common_replay_collection=state.get('learning',{}).get('replay_prefill'),
        final_results={name:{panel:evaluation_stats(data) for panel,data in panels.items()} for name,panels in state['final_results'].items()},
        final_floors=state['final_floors'],acceptance=state['acceptance'],old_protection=protection,
        export_validation=state['export_validation'],behavior=behavior,
        interpretation=state.get('interpretation',{}),targeted_diagnosis=state.get('targeted_diagnosis',{}),
        technical_checks=state.get('technical_checks',{}),unit_tests=state.get('unit_tests'),process_cleanup=cleanup,
        mechanics=state.get('mechanics'),guard_and_recovery=guard,
        legacy_inference=state.get('legacy_inference'),smoke=smoke,smoke_independent_reload=state.get('smoke_independent_reload'),
        notebook_execution=state.get('notebook_execution'),literature=state.get('literature'),
        branch_feasibility=state.get('branch_feasibility'),history_diagnosis_1m=state.get('history_diagnosis_1m'),
        history_diagnosis_3m=state.get('history_diagnosis_3m'),continuation_feasibility=state.get('continuation_feasibility'),learning_diagnosis=state.get('learning_diagnosis'),
        published_models=published_models,published_policy_identity=policy_identity,
        episode_records=f'reports/{experiment}/episodes.jsonl',
        behavior_trace_records=f'reports/{experiment}/behavior_traces.jsonl',
        panel_statistics=f'reports/{experiment}/panels.json',budget_records=f'reports/{experiment}/budgets.json',
        branch_example_records=f'reports/{experiment}/branch_examples.jsonl' if branch_examples else None,
        limits=['Only two independent main training seeds; continuations and counter snapshots are dependent.',
            'The former complete160-game historical panel is not fully migrated and was not covered.',
            ('Fresh learned counters cover Ryu,Guile,Sagat. All12-role coverage otherwise uses native CPU and responsive rules; not every possible neural opponent.' if fixed else 'Fresh learned-counter searches used the preselected Ken role. All12-role coverage mainly uses native CPU and responsive rule policies; it does not cover every possible neural opponent.'),
            'Different own character, initialization and training history versus retained Ryu are explicit experimental variables, not a fair PPO/DDQN algorithm comparison.',
            'Short branch outcomes are conditional; inputs, animation complexity and low TD loss are not adaptive/robust success.'],
        official_arena='Not executed; standalone SB3 serialization retains Q inference but is not course Arena certification.')
    write(REPORT/'summary.json',portable(compact,model_map))
    plot_coverage(config,state)
    print('PUBLISHED_COMPACT_RESEARCH_EVIDENCE',REPORT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/counterfactual_dqn.json')
    parser.add_argument('--run-root');parser.add_argument('--report-root');args=parser.parse_args()
    publish(args.config,args.run_root,args.report_root)
