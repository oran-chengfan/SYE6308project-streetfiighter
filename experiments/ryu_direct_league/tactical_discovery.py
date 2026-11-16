"""Bounded native sequence discovery; all deployment remains learned inference.

This module owns search and its continuous legal experience. Native matches,
restoration, demonstration storage, PPO and evaluation use their public entries.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.league_adaptive.environment import FightEnv
from experiments.league_adaptive.runtime import ROOT, compiled, load_actor, restore_fight, save_fight, setup, write
from experiments.ryu_counterq.collect_demonstrations import advance, begin_episode, continue_episode, save_episode, terminal, UnsupportedPhysicalAction
from experiments.ryu_counterq.collect_recoveries import recovery_action


MOTIFS = {
    'source': [('source', 8)],
    'guard_release': [('guard', 6), ('neutral', 2)],
    'advance_guard': [('toward', 4), ('guard', 4)],
    'retreat_counter': [('away', 3), ('uppercut', 1), ('neutral', 4)],
    'jump_kick': [('jump', 2), ('kick', 4), ('neutral', 2)],
    'jump_sweep': [('jump', 2), ('kick', 3), ('sweep', 3)],
    'guard_sweep': [('guard', 4), ('sweep', 2), ('neutral', 2)],
    'walk_sweep': [('toward', 4), ('sweep', 2), ('guard', 2)],
    'fireball_release': [('fireball', 1), ('neutral', 7)],
    'fireball_guard': [('fireball', 1), ('guard', 7)],
    'spin_release': [('spin', 1), ('neutral', 7)],
    'spin_kick': [('spin', 2), ('kick', 4), ('guard', 2)],
    'uppercut_release': [('uppercut', 1), ('neutral', 7)],
    'uppercut_guard': [('uppercut', 2), ('guard', 6)],
    'sweep_pressure': [('sweep', 2), ('guard', 2), ('toward', 2), ('sweep', 2)],
    'neutral_kick': [('neutral', 3), ('kick', 3), ('guard', 2)],
    'charge_boom': [('guard', 10), ('boom', 1), ('neutral', 1), ('guard', 8)],
    'charge_flash': [('guard', 10), ('flash', 1), ('neutral', 8)],
    'boom_flash': [('guard', 10), ('boom', 1), ('guard', 10), ('flash', 1), ('guard', 6)],
    'charge_boom_fast': [('guard', 10), ('boom', 1)],
    'boom_pursuit': [('guard', 10), ('boom', 1), ('toward', 8), ('kick', 1), ('guard', 4)],
    'guard_jump_kick': [('guard', 8), ('jump', 2), ('kick', 3), ('guard', 8)],
    'existing_zoning': [('existing_zoning', 8)],
    'existing_pressure': [('existing_pressure', 8)],
    'reactive_intercept': [('existing_reactive', 8)],
}


def motif_action(mode, env, actor, obs, names,controller_actions=None,source_action=None,previous=None):
    if controller_actions is not None and mode in controller_actions:
        return controller_actions[mode]
    if mode == 'source':
        return int(actor.predict(obs, True)[0]) if source_action is None else source_action
    if mode in ('existing_zoning','existing_pressure','existing_reactive'):
        from experiments.league_adaptive.characters import generic_opponent
        style='reactive_intercept' if mode=='existing_reactive' else mode.removeprefix('existing_')
        return generic_opponent(dict(style=style,period=24,charge_decisions=10),
            env.game.raw,env.player,env.episode_steps+1,names,previous=previous)
    forward='RIGHT' if env.game.raw[f'p{3-env.player}_x']>=env.game.raw[f'p{env.player}_x'] else 'LEFT'
    if mode in ('boom','flash'):
        return names[forward+'_Y_tap' if mode=='boom' else 'UP_B_tap']
    if mode in ('uppercut', 'spin'):
        face = 'R' if env.game.raw[f'p{3-env.player}_x'] >= env.game.raw[f'p{env.player}_x'] else 'L'
        name=f'SHORYUKEN_{face}_X' if mode == 'uppercut' else f'TATSUMAKI_{face}_C'
        if name not in names:
            back='LEFT' if forward=='RIGHT' else 'RIGHT'
            name=f'DP_{forward}_X' if mode=='uppercut' else f'QCF_{back}_C'
        return names[name]
    return recovery_action(mode, env.game.raw, env.player, names)


def perform_motif(name, env, actor, obs, raw, names, trace,controllers=None):
    plan=MOTIFS[name] if name in MOTIFS else [(name,8)]
    for mode, count in plan:
        for _ in range(count):
            before = readings(env)
            # Even forced training inputs traverse the real observation/GRU
            # history. Replace its predicted input with the input we execute.
            proposal=int(actor.predict(obs,True)[0])
            controller_actions={key:int(other.predict(obs,True)[0]) for key,other in (controllers or {}).items()}
            previous=trace[-1]['before'] if trace else None
            if previous is None and len(raw['positions'])>1:
                position=raw['positions'][-2]
                previous={f'p{env.player}_x':float(position[0]*1024),f'p{3-env.player}_x':float(position[2]*1024)}
            action = motif_action(mode, env, actor, obs, names,controller_actions,proposal,previous)
            if hasattr(actor.policy,'record_executed_actions'):
                actor.policy.record_executed_actions(action,obs['position'][None])
            for other in (controllers or {}).values():
                if hasattr(other.policy,'record_executed_actions'):
                    other.policy.record_executed_actions(action,obs['position'][None])
                elif other.get_state() is not None:
                    raise ValueError('A search controller cannot record the actual executed input history')
            decision = env.episode_steps
            obs, info = advance(env, obs, action, raw, True)
            trace.append(dict(decision=decision, action=action, mode=mode,
                opponent_action=info['opponent_action'], before=before, after=readings(env),
                damage_dealt=info['damage_dealt'], damage_taken=info['damage_taken']))
            if env.ended:
                return obs, info
    return obs, info


def readings(env):
    return {k: int(v) for k, v in env.game.raw.items()}


def diagnose(config, out):
    actor = load_actor(config['source'])
    settings=config['diagnosis']
    source_pool=json.loads((ROOT/settings.get('pool','configs/focus_pool.json')).read_text(encoding='utf-8'))
    if isinstance(source_pool,dict):source_pool=source_pool['pool']
    pool = [r for r in source_pool
            if r['id'] in config['diagnosis']['check_conditions']]
    env = FightEnv(actor.definition, pool)
    roots, rows, traces = [], [], []
    started = time.monotonic()
    try:
        cases=settings.get('conditions') or [dict(opponent=r['id'],player=2,offset=o) for r in pool for o in (0,12)]
        for case in cases:
            rival=next(r for r in pool if r['id']==case['opponent']);index=pool.index(rival)
            player,offset=int(case['player']),int(case['offset'])
            obs, _ = env.reset(options=dict(opponent=index, player=player, offset=offset))
            actor.reset()
            raw, trace = begin_episode(obs), []
            while not env.ended:
                decision = env.episode_steps
                if decision in settings['root_decisions']:
                    if len(roots)>=settings.get('maximum_roots',100000):raise RuntimeError('Declared root limit exceeded')
                    roots.append(dict(opponent=rival['id'], player=player, offset=offset, decision=decision,
                        saved=save_fight(env, actor), prefix=copy.deepcopy(raw)))
                before = readings(env)
                action = int(actor.predict(obs, True)[0])
                obs, info = advance(env, obs, action, raw)
                trace.append(dict(decision=decision, action=action,
                    action_name=actor.actions[action]['name'], opponent_action=info['opponent_action'],
                    before=before, after=readings(env), damage_dealt=info['damage_dealt'], damage_taken=info['damage_taken']))
                if env.native_frames >= settings['native_frames']:
                    raise RuntimeError('Diagnosis budget reached before completing declared conditions')
            row = dict(opponent=rival['id'], player=player, offset=offset, **terminal(info))
            rows.append(row)
            traces.append(dict(condition=row, trace=trace))
            if settings.get('verify_all',False):
                root=next(r for r in roots if all(r[k]==case[k] for k in ('opponent','player','offset')) and r['decision']==0)
                obs=restore_fight(env,actor,root['saved'])
                _,replayed=continue_episode(env,actor,obs,copy.deepcopy(root['prefix']))
                if terminal(replayed)!={k:row[k] for k in terminal(replayed)}:
                    raise RuntimeError('A selected condition does not replay exactly')
                row['unforced_replay_equal']=True
            print('DIAGNOSIS', json.dumps(row), flush=True)
        # An exact full continuation replay checks opponent counters, image
        # stacks and RNG restoration before interpreting any forced sequence.
        target = config['target']
        root = next(r for r in roots if all(r[k] == target[k] for k in ('opponent', 'player', 'offset')) and r['decision'] == 0)
        obs = restore_fight(env, actor, root['saved'])
        _, info = continue_episode(env, actor, obs, copy.deepcopy(root['prefix']))
        expected = next(r for r in rows if all(r[k] == target[k] for k in ('opponent', 'player', 'offset')))
        if terminal(info) != {k: expected[k] for k in terminal(info)}:
            raise RuntimeError('Unforced restored continuation differs')
        torch.save(dict(roots=roots, pool=pool, source=config['source']), out/'roots.pt')
        write(out/'traces.json', traces)
        result = dict(rows=rows, native_frames=env.native_frames, seconds=time.monotonic()-started,
            unforced_replay_equal=True, source=config['source'], decisions_are_native_frames=False,
            state_contents='Simulator, both image histories, opponent controller count/state and Python/NumPy/Torch RNGs.')
        write(out/'summary.json', result)
        return result
    finally:
        env.close()
        actor.close()


def search(config, out, diagnosis):
    """Full motif rollouts, then a bounded beam over complete exchanges.

    RAM only prioritizes training-time nodes. The saved learner input remains
    legal image stacks and own-first coordinates, with actual physical inputs.
    """
    bundle = torch.load(ROOT/diagnosis/'roots.pt', map_location='cpu', weights_only=False)
    actor = load_actor(config['source'])
    env = FightEnv(actor.definition, bundle['pool'])
    target, settings = config['target'], config['search']
    rival=next(r for r in bundle['pool'] if r['id']==target['opponent'])
    controllers={name:load_actor(path) for name,path in settings.get('neural_sources',{}).items()}
    if any(a.actions!=actor.actions for a in controllers.values()):
        raise ValueError('Search controllers require identical actual physical inputs')
    motifs=list(settings.get('motifs',list(MOTIFS)))
    if any(name not in MOTIFS and name not in controllers for name in motifs):raise ValueError('Unknown sequence source')
    proposals=list(settings.get('candidate_prefixes',[]))
    if settings.get('transferred_plan'):
        proposals.append(config['transferred_plans'][settings['transferred_plan']])
    for proposal in proposals:
        if any(name not in MOTIFS and name not in controllers for name in proposal['plan']+[proposal['tail']]):
            raise ValueError('Unknown transferred sequence source')
    roots = [r for r in bundle['roots'] if all(r[k] == target[k] for k in ('opponent', 'player', 'offset'))
             and r['decision'] in settings['roots']]
    if controllers:
        if any(r['decision'] for r in roots) and any(hasattr(a.policy,'get_inference_state') for a in controllers.values()):
            raise ValueError('A recurrent search teacher needs a standard-start root; use recover_skills for a real student prefix')
        augmented=[]
        for root in roots:
            restore_fight(env,actor,root['saved'])
            for other in controllers.values():other.reset()
            augmented.append(dict(root,saved=save_fight(env,actor,controllers)))
        roots=augmented
    names = {a['name']: i for i, a in enumerate(actor.actions)}
    episodes, rows, winners, failures = [], [], [], []
    directory = out/'experience'
    directory.mkdir()
    started = time.monotonic()
    budget = settings['native_frames']
    complete_reserve=int(settings.get('complete_native_frame_reserve',3000))
    if complete_reserve<=0:raise ValueError('Full-continuation reserve must be positive')

    def persist(raw, row, trace, success):
        # Canonical storage maps back to physical actions in the learner.
        # One-hot executed labels describe an achievable sequence, not Q.
        raw['supervised'] = [bool(success)]*len(raw['physical_actions'])
        save_episode(directory, episodes, raw, None, dict(group='pve' if rival['kind']=='cpu' else f"p{target['player']}",
            condition=dict(opponent=target['opponent'], opponent_kind=rival['kind'], player=target['player'],
                           offset=target['offset'], root=row['root'], **terminal(row)),
            origin='winning_sequence' if success else 'failed_sequence',
            dependence_group=[target['opponent'], target['player'], target['offset']],
            continuation=row['continuation'], search_plan=row['plan']), teacher=actor,executed_labels=True)
        write(out/f"trace_{len(episodes)-1:03d}.json", dict(condition=row, trace=trace))
        return episodes[-1]['file']

    def report(root, plan, info, raw, trace, continuation):
        row = dict(root=root['decision'], plan=plan, continuation=continuation,
            prefix_source=bundle['source'],**terminal(info))
        rows.append(row)
        success = info['result']=='win' and (rival['kind']!='cpu' or
            info['opponent_hp']<=0 and info['end_reason'] in ('signed_hp_ko','round_counter'))
        if success:
            row['experience_file'] = persist(raw, row, trace, True)
            winners.append(row)
        elif len(failures) < settings['maximum_saved_failures']:
            row['experience_file'] = persist(raw, row, trace, False)
            failures.append(row)
        write(out/'progress.json', dict(rows=rows, winners=winners, native_frames=env.native_frames,
            complete_branches=len(rows), seconds=time.monotonic()-started))
        print('SEQUENCE', json.dumps(row), flush=True)

    try:
        # First examine complete sequences rather than isolated six-frame
        # perturbations. The source motif is the exact matched baseline.
        root = min(roots,key=lambda r:r['decision'])
        for proposal in proposals:
            if env.native_frames+max(6000,complete_reserve)>budget:break
            obs=restore_fight(env,actor,root['saved'],controllers);raw=copy.deepcopy(root['prefix']);trace=[]
            for name in proposal['plan']:
                obs,info=perform_motif(name,env,actor,obs,raw,names,trace,controllers)
                if env.ended:break
            while not env.ended:
                obs,info=perform_motif(proposal['tail'],env,actor,obs,raw,names,trace,controllers)
            report(root,proposal['plan'],info,raw,trace,
                'Predeclared transferred exchange then repeated '+proposal['tail']+'; source policy '+config['source'])
        for name in motifs:
            if env.native_frames + complete_reserve > budget:
                break
            obs = restore_fight(env, actor, root['saved'],controllers)
            raw, trace = copy.deepcopy(root['prefix']), []
            while not env.ended:
                obs, info = perform_motif(name, env, actor, obs, raw, names, trace,controllers)
            report(root, [name], info, raw, trace, 'Repeated sampled motif throughout complete round; no source takeover except source motif')
        for root in ([] if winners else roots):
            if env.native_frames + 6000 > budget:
                break
            beam = [dict(saved=root['saved'], raw=copy.deepcopy(root['prefix']), plan=[], trace=[], priority=0.)]
            for depth in range(settings['maximum_segments']):
                candidates = []
                for parent in beam:
                    for name in motifs:
                        calls=sum(count for _,count in MOTIFS.get(name,[(name,8)]))
                        if env.native_frames+6*calls>budget:
                            break
                        if len(parent['raw']['physical_actions'])-root['decision']+calls>settings.get('maximum_prefix_decisions',100000):
                            continue
                        obs = restore_fight(env, actor, parent['saved'],controllers)
                        raw, trace = copy.deepcopy(parent['raw']), list(parent['trace'])
                        obs, info = perform_motif(name, env, actor, obs, raw, names, trace,controllers)
                        plan = parent['plan']+[name]
                        if env.ended:
                            report(root, plan, info, raw, trace, 'Recorded complete beam sequence, no source takeover')
                            continue
                        priority = float(max(0,info['self_hp'])-max(0,info['opponent_hp']))
                        # Distance/movement are explanatory readings, never
                        # rewards or an improvement criterion. Break ties by
                        # actual damage dealt, then candidate insertion order.
                        priority += .001*(176-max(0,info['opponent_hp']))
                        candidate = dict(saved=save_fight(env, actor,controllers), raw=raw, trace=trace, plan=plan, priority=priority)
                        candidates.append(candidate)
                        candidates.sort(key=lambda c: c['priority'], reverse=True)
                        del candidates[settings['beam_width']:]
                    if env.native_frames + 48 > budget:
                        break
                beam = candidates
                if depth+1 in settings.get('complete_rollout_depths',[]):
                    for node in beam[:settings.get('complete_rollout_leaders',1)]:
                        for tail in settings.get('complete_rollout_tails',['source']):
                            if env.native_frames+complete_reserve>budget:break
                            obs=restore_fight(env,actor,node['saved'],controllers)
                            raw,trace=copy.deepcopy(node['raw']),list(node['trace'])
                            while not env.ended:
                                obs,info=perform_motif(tail,env,actor,obs,raw,names,trace,controllers)
                            report(root,node['plan'],info,raw,trace,
                                'Saved exchange prefix then repeated '+tail+'; source policy '+config['source'])
                write(out/'beam.json', dict(root=root['decision'], depth=depth+1, native_frames=env.native_frames,
                    leaders=[dict(priority=c['priority'], plan=c['plan']) for c in beam]))
                if winners or not beam or env.native_frames + 48 > budget:
                    break
            if winners:
                break
        result = dict(status='winning_sequence_found' if winners else 'no_complete_winning_sequence_within_budget',
            rows=rows, winners=winners, native_frames=env.native_frames, seconds=time.monotonic()-started,
            source=config['source'], target=target, budget=budget,
            scope='Conditional training-time sequence evidence, not an independent learned-policy win or oracle. All branches from a condition share one dependence group.')
        write(out/'summary.json', result)
        return result
    finally:
        env.close()
        actor.close()
        for controller in controllers.values():controller.close()


def crossed_prefixes(config,out):
    """Separate opening-trajectory changes from subsequent control changes."""
    plan=config['opening_cross']
    actors={name:load_actor(path) for name,path in plan['actors'].items()}
    base_name=next(iter(actors));base=actors[base_name]
    if any(actor.actions!=base.actions for actor in actors.values()):
        raise ValueError('Crossed control requires the same physical action catalog')
    other={k:a for k,a in actors.items() if k!=base_name}
    pool=json.loads((ROOT/plan.get('pool','configs/focus_pool.json')).read_text(encoding='utf-8'))
    rival=next(r for r in pool if r['id']==plan['opponent'])
    recorded=None
    if plan.get('recorded_own_actions'):
        candidates=torch.load(ROOT/plan['recorded_own_actions'],map_location='cpu',weights_only=False)['episodes']
        recorded=next(e for e in candidates if e['opponent_config']==rival and e['player']==plan['player'] and e['offset']==plan['offset'])
    env=FightEnv(base.definition,[rival]);rows=[];episodes=[]
    experience=out/'experience';experience.mkdir()
    try:
        env.reset(options=dict(opponent=0,player=plan['player'],offset=plan['offset']))
        saved=save_fight(env,base,other)
        for prefix in actors:
            for continuation in plan.get('continuations',list(actors)):
                obs=restore_fight(env,base,saved,other);raw=begin_episode(obs)
                disagreements=[]
                for _ in range(plan['prefix_decisions']):
                    proposals={name:int(actor.predict(obs,True)[0]) for name,actor in actors.items()}
                    action=proposals[prefix]
                    # A continuation must have its own hidden representation
                    # of the actually executed prefix, including exploration
                    # or another controller's choices. Never copy a different
                    # network's hidden state or silently omit that history.
                    for name,actor in actors.items():
                        if name==prefix:continue
                        if hasattr(actor.policy,'record_executed_actions'):
                            actor.policy.record_executed_actions(action,obs['position'][None])
                        elif actor.get_state() is not None:
                            raise ValueError('Stateful continuation cannot record actual prefix actions')
                    if len(set(proposals.values()))>1:
                        disagreements.append(dict(decision=env.episode_steps,proposals=proposals,executed=action))
                    obs,info=advance(env,obs,action,raw,True)
                    if env.ended:break
                prefix_readings=readings(env)
                first_game_difference=None
                if not env.ended:
                    if continuation=='recorded_own_inputs':
                        if recorded is None:raise ValueError('Recorded continuation is missing its real episode')
                        names={a['name']:i for i,a in enumerate(base.actions)}
                        while not env.ended:
                            decision=env.episode_steps
                            if decision<=len(recorded['actions']) and first_game_difference is None:
                                if not np.array_equal(obs['image'],recorded['frames'][decision:decision+4]) or not np.array_equal(obs['position'],recorded['positions'][decision]):
                                    first_game_difference=decision
                            action=int(recorded['actions'][decision]) if decision<len(recorded['actions']) else names['NO_OP']
                            base.predict(obs,True);base.policy.record_executed_actions(action,obs['position'][None])
                            obs,info=advance(env,obs,action,raw,True)
                    else:_,info=continue_episode(env,actors[continuation],obs,raw,True)
                if env.native_frames>plan['native_frames']:raise RuntimeError('Crossed-prefix budget exceeded')
                row=dict(prefix=prefix,prefix_decisions=plan['prefix_decisions'],continuation=continuation,
                    prefix_readings=prefix_readings,prefix_disagreements=disagreements,
                    first_game_difference_from_recorded_path=first_game_difference,
                    **terminal(info));rows.append(row)
                if continuation in actors and prefix!=continuation and plan.get('save_experience',True):
                    raw['supervised']=[info['result']=='win']*len(raw['physical_actions'])
                    save_episode(experience,episodes,raw,None,dict(group=f"p{plan['player']}",
                        condition=dict(opponent=plan['opponent'],opponent_kind=rival['kind'],player=plan['player'],offset=plan['offset'],root=0,**terminal(info)),
                        origin='crossed_actual_prefix_and_frozen_continuation',
                        dependence_group=[plan['opponent'],plan['player'],plan['offset']],
                        prefix_actor=plan['actors'][prefix],continuation=plan['actors'][continuation]),teacher=base,executed_labels=True)
                print('CROSSED_PREFIX',json.dumps(row),flush=True)
        result=dict(rows=rows,native_frames=env.native_frames,configuration=plan,
            scope='Conditional full games after actual frozen-policy prefixes, not independent model wins or first-action oracle values.')
        write(out/'summary.json',result);return result
    finally:
        env.close()
        for actor in actors.values():actor.close()


def recover_skills(config,out):
    """Continuous supervision after actual failed student histories.

    Factual and restored continuations use the same existing native/episode
    implementation. A winning teacher at its own start is not assumed to be
    successful here; only complete verified suffixes become policy examples.
    """
    settings=config['student_recovery'];rows=[];episodes=[];seen=[];attempted=set();incomplete=[]
    if any(mode not in ('teacher',*MOTIFS) for mode in settings['continuations']):
        raise ValueError('Unknown complete recovery continuation')
    directory=out/'experience';directory.mkdir();frames=0;started=time.monotonic()
    keys=('opponent','player','offset')
    for case_number,case in enumerate(settings['cases']):
        if frames+settings.get('case_native_frame_reserve',12000)>settings['native_frames']:break
        student=load_actor(case['student']);teacher=load_actor(case['teacher'])
        action_map=None;timing_overrides=case.get('teacher_input_overrides',{})
        if student.actions!=teacher.actions:
            if (not settings.get('allow_exact_action_translation',False)
                    or hasattr(teacher.policy,'record_executed_actions')
                    or teacher.definition.get('inference_hold_steps',0)):
                raise ValueError('Different catalogs require an explicitly allowed stateless teacher and exact physical translation')
            # Match all six actual native key sets, never names or approximate
            # timing. An unsupported action is an error, not a fabricated
            # student label. Current recurrent teachers still require the
            # same catalog so their actual input history stays meaningful.
            signature=lambda commands:tuple(tuple(sorted(keys)) for keys in commands)
            targets={signature(commands):i for i,commands in enumerate(compiled(student.actions))}
            action_map={i:targets[signature(commands)] for i,commands in enumerate(compiled(teacher.actions))
                        if signature(commands) in targets}
            # A separately declared timing proposal is not an exact teacher
            # translation. Its changed live game must win before it supplies
            # actual executed-action supervision; never relabel it as equal.
            for old_name,new_name in timing_overrides.items():
                old=next(i for i,a in enumerate(teacher.actions) if a['name']==old_name)
                new=next(i for i,a in enumerate(student.actions) if a['name']==new_name)
                action_map[old]=new
        elif timing_overrides:
            raise ValueError('Timing proposals require explicitly different catalogs')
        env=FightEnv(student.definition,[case['rival']]);others={'teacher':teacher};roots=[]
        try:
            obs,_=env.reset(options=dict(opponent=0,player=case['player'],offset=case['offset']))
            student.reset();teacher.reset();raw=begin_episode(obs)
            while not env.ended:
                decision=env.episode_steps
                if decision in case.get('roots',settings['roots']):
                    roots.append(dict(decision=decision,saved=save_fight(env,student,others),prefix=copy.deepcopy(raw)))
                action=int(student.predict(obs,True)[0]);teacher.predict(obs,True)
                if hasattr(teacher.policy,'record_executed_actions'):
                    teacher.policy.record_executed_actions(action,obs['position'][None])
                elif teacher.get_state() is not None:
                    raise ValueError('Recovery teacher cannot record the actual student prefix')
                obs,info=advance(env,obs,action,raw)
            expected=case['expected']
            if terminal(info)!={k:expected[k] for k in terminal(info)}:raise RuntimeError('Student factual recovery condition differs')
            actual=terminal(info)
            group='pve' if case['rival']['kind']=='cpu' else f"p{case['player']}"
            save_episode(directory,episodes,raw,None,dict(group=group,
                condition=dict(opponent=case['rival']['id'],opponent_kind=case['rival']['kind'],
                    player=case['player'],offset=case['offset'],**actual),
                origin='actual_student_factual_not_policy_supervision',source_model=case['student'],
                policy_supervised=False,integration_task=case['task'],
                dependence_group=[case['rival']['id'],case['player'],case['offset']]),
                teacher=student,executed_labels=True)
            if roots:
                root=roots[0];obs=restore_fight(env,student,root['saved'],others)
                _,replay=continue_episode(env,student,obs,copy.deepcopy(root['prefix']))
                if terminal(replay)!=actual:raise RuntimeError('Unforced student replay differs')
            names={a['name']:i for i,a in enumerate(student.actions)}
            for root in roots:
                if len(seen)>=settings['maximum_roots']:break
                signature=(json.dumps(case['rival'],sort_keys=True),case['player'],case['offset'],
                    root['decision'],tuple(root['prefix']['physical_actions']))
                # Same condition and complete executed prefix determine the
                # same native scene. Reuse that evidence instead of counting
                # two deterministic student prefixes as diverse experience.
                if signature in seen:
                    rows.append(dict(case=case,root=root['decision'],status='reused_identical_executed_prefix'));continue
                seen.append(signature)
                for mode in settings['continuations']:
                    if frames+env.native_frames+settings.get('continuation_native_frame_reserve',3000)>settings['native_frames']:break
                    attempted.add((case_number,root['decision'],mode))
                    obs=restore_fight(env,student,root['saved'],others);branch=copy.deepcopy(root['prefix']);trace=[]
                    initial_readings=readings(env)
                    if mode=='teacher':
                        before=[initial_readings]
                        def observe(current,previous,info,action):
                            after=readings(current)
                            trace.append(dict(decision=current.episode_steps-1,action=action,
                                mode='teacher',before=before[0],after=after,
                                damage_dealt=info['damage_dealt'],damage_taken=info['damage_taken']))
                            before[0]=after
                        try:
                            _,result=continue_episode(env,teacher,obs,branch,True,action_map=action_map,observer=observe)
                        except UnsupportedPhysicalAction as error:
                            unknown=dict(case=case_number,opponent=case['rival']['id'],player=case['player'],offset=case['offset'],
                                root=root['decision'],continuation=case['teacher'],status='unknown_unsupported_teacher_input',
                                source_input=teacher.actions[error.action],decisions_executed=env.episode_steps,
                                simulated_native_frames=env.game.elapsed,input_timing_overrides=timing_overrides,
                                first_different_actual_input=next((i for i,(a,b) in enumerate(zip(raw['physical_actions'],branch['physical_actions'])) if a!=b),None),
                                complete_game=False,positive_supervision=False)
                            incomplete.append(unknown)
                            write(out/f'incomplete_{case_number}_{root["decision"]}.json',dict(condition=unknown,trace=trace))
                            write(out/'progress.json',dict(rows=rows,not_completed=incomplete,native_frames=frames+env.native_frames,restored_roots=len(seen)))
                            print('RECOVERY_NOT_COMPLETED',json.dumps(unknown),flush=True)
                            continue
                    else:
                        while not env.ended:
                            obs,result=perform_motif(mode,env,student,obs,branch,names,trace)
                    source_actions=raw['physical_actions'];executed=branch['physical_actions']
                    difference=next((i for i,(a,b) in enumerate(zip(source_actions,executed)) if a!=b),None)
                    success=result['result']=='win' and (case['rival']['kind']!='cpu' or
                        result['opponent_hp']<=0 and result['end_reason'] in ('signed_hp_ko','round_counter'))
                    # Retain actual student history, but never teach that
                    # uncontrolled prefix as if it had been a teacher action.
                    branch['supervised']=[False]*root['decision']+[bool(success)]*(len(executed)-root['decision'])
                    continuation=case['teacher'] if mode=='teacher' else 'repeat:'+mode
                    row=dict(student=case['student'],teacher=case['teacher'],task=case['task'],
                        condition={k:case[k] for k in keys if k in case},opponent=case['rival']['id'],
                        player=case['player'],offset=case['offset'],root=root['decision'],
                        continuation=continuation,baseline=actual,first_different_actual_input=difference,
                        unforced_replay_equal=True,exact_physical_translation=mode=='teacher' and action_map is not None and not timing_overrides,
                        input_timing_overrides=timing_overrides,**terminal(result))
                    save_episode(directory,episodes,branch,None,dict(group=group,
                        condition=dict(opponent=case['rival']['id'],opponent_kind=case['rival']['kind'],
                            player=case['player'],offset=case['offset'],root=root['decision'],**terminal(result)),
                        origin='verified_student_history_recovery' if success else 'failed_student_history_recovery',
                        prefix_source=case['student'],source_model=case['teacher'] if mode=='teacher' else 'executable:'+mode,
                        continuation=continuation,supervision_start=root['decision'],policy_supervised=success,
                        integration_task=case['task'],dependence_group=[case['rival']['id'],case['player'],case['offset']]),
                        teacher=student if action_map is not None else teacher,
                        executed_labels=action_map is not None or mode!='teacher' or teacher.definition.get('physical_actions',False))
                    row['experience_file']=episodes[-1]['file'];rows.append(row)
                    write(out/f"trace_{len(episodes)-1:03d}.json",dict(condition=row,
                        root_readings=initial_readings,trace=trace))
                    write(out/'progress.json',dict(rows=rows,native_frames=frames+env.native_frames,restored_roots=len(seen)))
                    print('STUDENT_RECOVERY',json.dumps(row),flush=True)
        finally:
            frames+=env.native_frames;env.close();student.close();teacher.close()
    pending=[dict(case=case_number,task=case['task'],root=root,continuation=mode,status='not_completed',
        reason='Budget, root not reached in factual game, or identical prefix already recorded; never a complete win/loss.')
        for case_number,case in enumerate(settings['cases']) for root in case.get('roots',settings['roots'])
        for mode in settings['continuations'] if (case_number,root,mode) not in attempted]
    result=dict(rows=rows,native_frames=frames,seconds=time.monotonic()-started,restored_roots=len(seen),not_completed=pending+incomplete,
        scope='Conditional full continuations after real student actions; no global action oracle, independent-win inflation or off-policy PPO advantage samples.')
    write(out/'summary.json',result);return result


def verify_sequences(config,out):
    """Independently replay selected complete input streams for supervision.

    Opponents respond again, and the frozen actor records every executed
    action/history. A successful script is never labeled an independent
    greedy student win. Shared evaluator recorders own the raw episode format.
    """
    from experiments.ryu_counterq.replay import TrajectoryRecorder
    settings=config['demonstration_learning'];actor_path=config['source'];actor=load_actor(actor_path)
    if not actor.definition.get('physical_actions'):raise ValueError('Use explicitly physical search trajectories')
    document=json.loads((ROOT/config['diagnosis']['pool']).read_text(encoding='utf-8'))
    pool=document['pool'] if isinstance(document,dict) else document
    env=FightEnv(actor.definition,pool,reward=settings['reward']);episodes=[];rows=[]
    try:
        selected=[]
        for directory in settings['search_directories']:
            directory=ROOT/directory;result=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            winner=next(iter(result['winners']),None)
            if winner is None:continue
            selected.append((directory,winner,result['target']))
        recovery_counts={}
        for directory in settings.get('recovery_directories',[]):
            directory=ROOT/directory;result=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            chosen_files=settings.get('selected_recovery_files',{}).get(directory.relative_to(ROOT).as_posix())
            used_roots=set()
            for row in result['rows']:
                if row.get('result')!='win':continue
                if chosen_files is not None and row['experience_file'] not in chosen_files:continue
                rival=next(r for r in pool if r['id']==row['opponent'])
                if rival['kind']=='cpu' and (row['opponent_hp']>0 or row['end_reason'] not in ('signed_hp_ko','round_counter')):continue
                key=(row['opponent'],row['player'],row['offset']);root=key+(row['root'],)
                if root in used_roots or recovery_counts.get(key,0)>=settings.get('maximum_recoveries_per_condition',2):continue
                used_roots.add(root);recovery_counts[key]=recovery_counts.get(key,0)+1
                target={k:row[k] for k in ('opponent','player','offset')}
                selected.append((directory,dict(row,plan=[row['continuation']]),target))
        if settings.get('selected_recovery_files'):
            expected={(directory,name) for directory,names in settings['selected_recovery_files'].items() for name in names}
            actual={(directory.relative_to(ROOT).as_posix(),row['experience_file']) for directory,row,_ in selected}
            if expected!=actual:raise ValueError('Declared compatible complete recovery recordings are missing')
        for directory,winner,target in selected:
            recording_path=winner.get('student',config['source'])
            if recording_path!=actor_path:
                actor.close();actor=load_actor(recording_path);actor_path=recording_path
                if actor.actions!=env.definition['actions']:raise ValueError('Recovery action catalogs differ')
            rival=next(r for r in pool if r['id']==target['opponent'])
            source=directory/'experience'/winner['experience_file']
            with np.load(source,allow_pickle=False) as data:
                labels=data['actions'].copy();frames=data['frames'].copy();positions=data['positions'].copy();mask=data['supervised'].copy()
            if not mask.any():raise ValueError('Selected demonstration has no verified supervised suffix')
            start=int(np.flatnonzero(mask)[0])
            if not mask[start:].all():raise ValueError('A supervised recovery must be one continuous successful suffix')
            p,o=target['player'],target['offset']
            obs,_=env.reset(options=dict(opponent=pool.index(rival),player=p,offset=o));actor.reset()
            captures=set(settings.get('capture_native_frames',[]))
            if captures:
                import cv2
                def capture(rgb,raw,k1,k2,n):
                    if n in captures:
                        cv2.imwrite(str(out/f"{rival['id']}_{n}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                env.game.on_frame=capture
            recorder=TrajectoryRecorder(pool,directory.relative_to(ROOT).as_posix()+'/'+winner['experience_file'])
            for i,action in enumerate(labels):
                if not np.array_equal(obs['image'],frames[i:i+4]) or not np.array_equal(obs['position'],positions[i]):
                    raise RuntimeError('Controlled demonstration legal history does not replay')
                actor.predict(obs,True);actor.policy.record_executed_actions(int(action),obs['position'][None])
                nxt,_,_,_,info=env.step(int(action));recorder.record_transition(actor,obs,nxt,info);obs=nxt
                if env.ended and i!=len(labels)-1:raise RuntimeError('Replayed controlled trajectory ended early')
            if not env.ended or terminal(info)!={k:winner[k] for k in terminal(info)}:
                raise RuntimeError('Controlled full-match result did not reproduce')
            episode=recorder.episodes[0]
            episode.update(controlled_replay_equal=True,independent_learned_control=False,
                supervision_only=True,data_phase='controlled_demonstration',search_plan=winner['plan'],
                continuation=winner['continuation'],supervision_start=start,recording_network=actor_path)
            episodes.append(episode);rows.append(dict(opponent=rival['id'],opponent_config=rival,player=p,offset=o,
                source=source.relative_to(ROOT).as_posix(),plan=winner['plan'],supervision_start=start,**terminal(info)))
            if env.native_frames>settings['verification_native_frame_cap']:raise RuntimeError('Demonstration verification budget exceeded')
        torch.save(dict(episodes=episodes),out/'verified_episodes.pt')
        value=dict(rows=rows,episodes=len(episodes),native_frames=env.native_frames,
            transitions=sum(len(e['actions']) for e in episodes),controlled_replay_equal=True,
            independent_learned_control=False,selection=('Predeclared exact complete winning recordings selected by joint legal-history compatibility; not health or future policy scores.'
                if settings.get('selected_recovery_files') else 'First complete KO/win per search condition, bounded distinct successful roots in original report order.'),
            scope='Auxiliary continuous action supervision only; no PPO advantages, fabricated source-greedy labels or optimal-Q targets.')
        write(out/'summary.json',value);return value
    finally:env.close();actor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['diagnose', 'search','cross','recover','verify'])
    parser.add_argument('--config', default='configs/tactical_discovery.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--diagnosis', default='artifacts/tactical_discovery/diagnosis')
    parser.add_argument('--experiment',help='Named, predeclared exploration override in the shared configuration.')
    args = parser.parse_args()
    setup()
    config = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if args.experiment:
        override=config['explorations'][args.experiment]
        for key,value in override.items():
            if key in ('search','diagnosis','demonstration_learning','student_recovery','opening_cross'):config[key].update(value)
            elif key in ('source','target'):config[key]=value
    out = (ROOT/args.out).resolve()
    if not out.is_relative_to((ROOT/'artifacts').resolve()):
        parser.error('Process data must remain under artifacts/')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'configuration.json',config)
    if args.stage == 'diagnose':
        diagnose(config, out)
    elif args.stage=='cross':
        crossed_prefixes(config,out)
    elif args.stage=='recover':
        recover_skills(config,out)
    elif args.stage=='verify':
        verify_sequences(config,out)
    else:
        search(config, out, args.diagnosis)


if __name__ == '__main__':
    main()
