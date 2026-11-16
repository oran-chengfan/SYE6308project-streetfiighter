"""Frozen full-game skill data; never a deployment selector or trainer."""
import copy
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.league_adaptive.runtime import ROOT,load_actor,write
from experiments.league_adaptive.evaluate import evaluate,exact_opponent_pool
from experiments.ryu_counterq.replay import SequenceReplay,TrajectoryRecorder,successful_episode,audit_success_history_labels
from experiments.ryu_counterq.stabilization import read


def stochastic_skill_execution(spec,root,label='57037'):
    """Measure whether a frozen learned skill survives collection sampling.

    Reuses complete native evaluation. Action RNG repeats are not independent
    trained models. Changing temperature changes neither weights nor greedy
    choices; it does not establish that subsequent learning will improve.
    """
    from experiments.league_adaptive.runtime import Actor
    from experiments.ryu_counterq.policy import construct_sequence_q
    rule=spec['stochastic_skill_execution_diagnosis'];run=rule['runs'][str(label)]
    out=root/run['directory'];summary=out/'summary.json'
    if summary.exists():return read(summary)
    out.mkdir(parents=True,exist_ok=False);write(out/'configuration.json',dict(rule,selected=run))
    source=torch.load(ROOT/run['source'],map_location='cpu',weights_only=False)
    if source['definition']['kind']!='sequence_ppo':raise ValueError('Declare a categorical recurrent PPO execution profile')
    pool=read(ROOT/spec['base_config'])['pool'];rows=[];spent=setup_frames=0;incomplete=[]
    for temperature in rule['temperatures']:
        for seed in rule['action_sampling_seeds']:
            for case in rule['conditions']:
                definition=copy.deepcopy(source['definition']);definition['action_temperature']=temperature
                policy=construct_sequence_q(definition);policy.load_state_dict(source['state_dict'],strict=True)
                policy.set_training_mode(False);actor=Actor(policy,definition);torch.manual_seed(seed);previous=0
                def observe(actor,observation,info):
                    nonlocal spent,previous
                    spent+=info['native_frames']-previous;previous=info['native_frames']
                    if spent+6>rule['maximum_native_frames'] and info['result']=='ongoing':
                        raise TimeoutError('Declared stochastic execution budget reached')
                try:
                    result=evaluate(actor,[r for r in pool if r['id']==case['opponent']],
                        out/f'T{temperature}'/str(seed)/case['opponent'],cpu=False,conditions=[case],
                        arena_call_sequence=True,progress=False,deterministic=False,
                        reward_kind=read(ROOT/spec['base_config'])['learner']['reward'],transition_observer=observe)
                except TimeoutError:
                    incomplete.append(dict(temperature=temperature,action_seed=seed,condition=case));break
                finally:actor.close()
                setup_frames+=result['setup_native_frames'];row=result['rows'][0]
                rows.append(dict(temperature=temperature,action_seed=seed,condition=case,**{k:row[k] for k in
                    ('result','end_reason','self_hp','opponent_hp','native_frames','ko_win','physical_input_counts')}))
                write(out/'progress.json',dict(rows=rows,native_frames=spent,setup_native_frames=setup_frames))
                print('STOCHASTIC_EXECUTION',label,temperature,seed,case['opponent'],row['result'],row['self_hp'],row['opponent_hp'],flush=True)
            if incomplete:break
        if incomplete:break
    value=dict(rows=rows,native_frames=spent,setup_native_frames=setup_frames,source=run['source'],weights_unchanged=True,
        parameter_updates=0,not_completed=incomplete,scope=rule['interpretation'])
    write(summary,value);return value


def tactical_skill_catalog(spec,root):
    """Select existing real full-game skills before acquiring more experience.

    Prefer one frozen teacher across a visible character family. This is a
    training-data manifest, never a deployment controller or action oracle.
    Exact-history label compatibility still has to be checked after capture.
    """
    base=read(ROOT/spec['base_config']);rivals={r['id']:r for r in base['pool']}
    rule=spec['skill_catalog_declaration']
    expected_actions=torch.load(ROOT/rule['student_action_source'],map_location='cpu',weights_only=False)['definition']['actions']
    expected={}
    for rival in rivals.values():
        cases=([(1,0)] if rival['kind']=='cpu' else
            [(p,o) for p in (1,2) for o in (0,12)] if rival['id'] in spec['primary_matrix']['pvp_opponents'] else [])
        for player,offset in cases:expected[(rival['id'],player,offset)]=rival
    family_scope=rule.get('family_scope','character')
    if family_scope not in ('character','character_mode'):raise ValueError('Unknown training teacher family scope')
    definitions={};candidates={key:{} for key in expected};families={};source_families={}
    for line in (ROOT/'reports'/spec['experiment']/'learning_games.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);key=(row.get('opponent'),row.get('player'),row.get('offset'))
        if key not in expected or row.get('opponent_config')!=expected[key]:continue
        if not (row.get('ko_win') if expected[key]['kind']=='cpu' else row.get('result')=='win'):continue
        source=row.get('checkpoint')
        if not source and row.get('phase')=='facing_normalization_probe':
            source=(root/'facing_normalization_probe'/row['source']/('canonical' if row['facing_canonical'] else 'absolute')/'agent.pt').relative_to(ROOT).as_posix()
        if not source or 'untrained' in source or Path(source).name in ('initial_ppo.pt','initial_ddqn.pt'):continue
        path=ROOT/source
        if not path.is_file():continue
        if source not in definitions:definitions[source]=torch.load(path,map_location='cpu',weights_only=False)['definition']
        definition=definitions[source]
        if (definition.get('character')!=rule['character'] or not definition.get('physical_actions')
                or definition['kind'] not in ('sequence_ppo','sequence_double_q')):continue
        if definition['actions']!=expected_actions:raise ValueError('A catalog source changes the actual physical input semantics')
        family=expected[key].get('character','Ryu')
        if family_scope=='character_mode':family+='|'+('cpu' if expected[key]['kind']=='cpu' else 'pvp')
        families[key]=family;source_families.setdefault((source,family),set()).add(key)
        candidates[key][source]=dict(source=source,opponent=key[0],player=key[1],offset=key[2],
            opponent_config=expected[key],result=row['result'],end_reason=row['end_reason'],
            self_hp=row['self_hp'],opponent_hp=row['opponent_hp'],native_frames=row['native_frames'],
            algorithm_kind=definition['kind'],recorded_phase=row['phase'])
    preferred={}
    for family in sorted(set(families.values())):
        possible=[(source,keys) for (source,f),keys in source_families.items() if f==family]
        preferred[family]=max(possible,key=lambda item:(len(item[1]),
            min(candidates[k][item[0]]['self_hp'] for k in item[1]),item[0]))[0]
    overrides={(r['opponent'],r['player'],r['offset']):r['source'] for r in rule.get('verified_source_preferences',[])}
    if any(key not in candidates or source not in candidates[key] for key,source in overrides.items()):
        raise ValueError('A preferred training teacher has no matching independently verified full win')
    selected=[];missing=[]
    for key,options in candidates.items():
        if not options:missing.append(dict(opponent=key[0],player=key[1],offset=key[2]));continue
        family=families[key];source=preferred[family]
        if source not in options:
            source=max(options,key=lambda p:(len(source_families[(p,family)]),options[p]['self_hp'],-options[p]['native_frames'],p))
        source=overrides.get(key,source)
        selected.append(dict(options[source],family=family,preferred_family_teacher=source==preferred[family]))
    result=dict(selected=selected,missing=missing,preferred_family_sources=preferred,
        expected_conditions=len(expected),covered_CPU=sum(r['opponent_config']['kind']=='cpu' for r in selected),
        covered_PvP=sum(r['opponent_config']['kind']!='cpu' for r in selected),native_frames=0,optimizer_updates=0,
        family_scope=family_scope,verified_source_preferences=rule.get('verified_source_preferences',[]),
        selection='Maximize observed complete wins per declared training-data family, then worst remainingHP. Fallbacks prefer broad verified coverage then HP/time. No final holdout or branch wins included.',
        scope='Portfolio of frozen learned controllers, not one agent. All cases are consumed training/development evidence; no hidden teacher identity enters deployment.')
    write(ROOT/rule.get('output',(root/'tactical_skill_catalog.json').relative_to(ROOT).as_posix()),result)
    print('TACTICAL_SKILL_CATALOG',result['covered_CPU'],result['covered_PvP'],'missing',missing,flush=True)
    return result


def collect_coherent_skill_bank(spec,root,variant=None):
    """Capture real full-game teachers, reuse exact compatible recordings.

    Teacher selection is frozen before acquisition. Deployment never reads
    this manifest. Data supply separate action/distribution supervision only.
    """
    rule={k:v for k,v in spec['coherent_skill_data'].items() if k!='variants'}
    if variant is not None:rule.update(spec['coherent_skill_data']['variants'][variant])
    out=root/rule['phase']/str(rule['seed'])
    if (out/'summary.json').exists():return read(out/'summary.json')
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/'configuration.json'
    if config_path.exists() and read(config_path)!=rule:raise ValueError('Skill acquisition configuration changed')
    write(config_path,rule)
    selection=out/'selection.json'
    if not selection.exists():
        catalog=rule['catalog']
        document=read(ROOT/(catalog['path'] if isinstance(catalog,dict) else catalog))
        if isinstance(catalog,dict):document=document[catalog['key']]
        write(selection,document['selected'])
    selected=read(selection);base=read(ROOT/spec['base_config'])
    pool=exact_opponent_pool([case['opponent_config'] for case in selected])
    initial=read(ROOT/rule['student_evaluation'])['rows']
    key=lambda e:(e['opponent_config']['id'],e['player'],e['offset'])
    old={key(e) for e in initial if e['opponent_kind']=='cpu' and e['ko_win']}
    old.update(key(e) for e in initial if e['opponent_kind']!='cpu' and e['result']=='win')
    aliases={};equivalences=[]
    for pair in rule.get('equivalent_sources',[]):
        left,right=(torch.load(ROOT/p,map_location='cpu',weights_only=False) for p in pair)
        equal=(left['definition']==right['definition'] and left['state_dict'].keys()==right['state_dict'].keys()
            and all(torch.equal(value,right['state_dict'][name]) for name,value in left['state_dict'].items()))
        if not equal:raise ValueError('A cached teacher is not inference-equivalent to its selected source')
        aliases[pair[0]]=pair[1];equivalences.append(dict(selected=pair[0],recorded=pair[1],same_inference_parameters=True))
    caches=[]
    for path in rule.get('cached_banks',[]):
        saved=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        values=saved['episodes'];caches.extend(values.values() if isinstance(values,dict) else values)
    groups=defaultdict(list)
    for case in selected:groups[case['source']].append(case)
    collected=[];rows=[];costs=[];reused=0
    for index,(source,cases) in enumerate(sorted(groups.items())):
        directory=out/f'collection_{index}';episode_file=directory/'episodes.pt'
        if episode_file.exists():caches.extend(torch.load(episode_file,map_location='cpu',weights_only=False)['episodes'])
        chosen={};pending=[]
        for case in cases:
            identity=(case['opponent'],case['player'],case['offset'])
            found=next((e for e in caches if key(e)==identity and e['opponent_config']==case['opponent_config']
                and e.get('policy_version') in (source,aliases.get(source)) and e.get('independent_learned_control') and successful_episode(e)),None)
            if found is None:pending.append(case)
            else:chosen[identity]=found;reused+=1
        if pending:
            spent=sum(c['native_frames']+c.get('setup_native_frames',0) for c in costs)
            reserve=(sum(c['native_frames'] for c in pending)+int(rule.get('setup_frame_reserve',7200))
                if rule.get('reserve_from_recorded_outcomes',False) else 7200*len(pending))
            if spent+reserve>rule['native_frame_cap']:raise RuntimeError('Coherent acquisition reserve exhausted')
            if (directory/'summary.json').exists():raise RuntimeError('Completed acquisition has no persisted history; preserve and diagnose before repeating')
            actor=load_actor(source);observer=TrajectoryRecorder(pool,source)
            try:
                result=evaluate(actor,pool,directory,cpu=False,conditions=[{k:c[k] for k in ('opponent','player','offset')} for c in pending],
                    offsets=(0,12),arena_call_sequence=True,progress=False,reward_kind=base['learner']['reward'],transition_observer=observer)
            finally:actor.close()
            torch.save(dict(episodes=observer.episodes),episode_file)
            expected={(c['opponent'],c['player'],c['offset']):c for c in pending}
            for row in result['rows']:
                case=expected[(row['opponent'],row['player'],row['offset'])]
                if any(row[k]!=case[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')):
                    raise RuntimeError('Frozen teacher acquisition differs from its complete recorded game')
            for episode in observer.episodes:chosen[key(episode)]=episode
        if (directory/'summary.json').exists():
            result=read(directory/'summary.json');costs.append(dict(source=source,native_frames=result['native_frames'],setup_native_frames=result['setup_native_frames']))
        for case in cases:
            identity=(case['opponent'],case['player'],case['offset']);episode=copy.deepcopy(chosen[identity])
            if (not successful_episode(episode) or any(episode[a]!=case[b] for a,b in
                    (('final_self_hp','self_hp'),('final_opponent_hp','opponent_hp'),('end_reason','end_reason')))):
                raise RuntimeError('A selected full-game teacher did not reproduce its frozen outcome')
            retention=identity in old
            episode.update(catalog_teacher=source,data_phase='coherent_complete_skill_data',supervision_only=True,
                integration_role='capability_retention' if retention else 'known_skill_transfer',auxiliary_priority=not retention,
                retention_target='physical_distribution' if retention and 'policy_probabilities' in episode else 'executed_action')
            collected.append(episode);rows.append(dict(case,integration_role=episode['integration_role'],recorded_policy=episode['policy_version']))
        write(out/'progress.json',dict(episodes=len(collected),sources_completed=len(costs),costs=costs))
    bank_spec=rule['bank'];bank=SequenceReplay(capacity=bank_spec['capacity_transitions'],byte_capacity=bank_spec['byte_capacity'],
        seed=rule['seed']+95003,burn=0,learn=bank_spec['learn_length'],nstep=1)
    for episode in collected:bank.replace_verified_success(episode,0,bank_spec['maximum_episodes'])
    if len(bank.episodes)!=len(selected):raise RuntimeError('A chosen skill disappeared from the bounded bank')
    bank.note_complete_results(initial);torch.save(bank.state_dict(),out/'bank.pt')
    audit=audit_success_history_labels(bank.episodes.values(),use_soft_references=True,maximum_examples=128)
    write(out/'label_audit.json',audit)
    value=dict(source=rule['student_initial'],episodes=len(bank.episodes),transitions=bank.transitions,bytes=bank.bytes,
        rows=rows,native_frames=sum(c['native_frames'] for c in costs),setup_native_frames=sum(c.get('setup_native_frames',0) for c in costs),
        optimizer_updates=0,reused_episodes=reused,source_costs=costs,cache_equivalences=equivalences,
        label_conflicts=audit['exact_prefix_action_disagreements'],hard_label_conflicts=audit['both_hard_target_conflicts'],
        scope='Consumed full-game teacher experience, not one-agent coverage or on-policy PPO advantages. Initialization preserves the separate student weights/Adam.')
    if value['native_frames']+value['setup_native_frames']>rule['native_frame_cap']:raise RuntimeError('Skill acquisition exceeded its actual native-frame budget')
    write(out/'summary.json',value);print('COHERENT_SKILL_BANK',value['episodes'],value['native_frames'],value['hard_label_conflicts'],flush=True)
    return value
