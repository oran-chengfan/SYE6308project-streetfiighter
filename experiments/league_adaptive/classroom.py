"""Course-notebook orchestration using the existing Counter-Ryu implementation.

No second trainer, referee or action-selection implementation lives here.
Student cells provide observation, action, reward and model configuration.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.league_adaptive.runtime import ROOT, load_actor, source_module, write


def new_run(workspace, name):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = Path(workspace) / 'runs' / name / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_module(name, *arguments):
    subprocess.run([sys.executable, '-m', name, *map(str, arguments)], cwd=ROOT, check=True)


def prepare_student(directory, initial, output, *, scratch=False):
    """Apply the four actual editing files; make a fresh-phase initialization."""
    from experiments.ryu_counterq.policy import construct_sequence_q, restrict_sequence_actions
    directory, output = Path(directory), Path(output)
    texts = {name: (directory / f'student_{name}.py').read_text(encoding='utf-8')
             for name in ('observation', 'actions', 'reward', 'model')}
    modules = {name: source_module(text) for name, text in texts.items()}
    actor = load_actor(initial)
    try:
        original = copy.deepcopy(actor.definition)
        if original['kind'] != 'sequence_ppo':
            raise ValueError('This course adaptation uses the current sequence PPO model.')
        policy = actor.policy
        if len(original['actions']) > 64:
            indices = json.loads((ROOT / 'configs/ryu_targeted_delivery.json').read_text())['course_delivery']['action_indices']
            policy, original, _ = restrict_sequence_actions(
                dict(definition=original, state_dict=policy.state_dict()), indices)
        actions = copy.deepcopy(original['actions'][:15]) + copy.deepcopy(modules['actions'].CUSTOM_ACTIONS)
        if not 15 <= len(actions) <= 64:
            raise ValueError('Keep the 15 basic inputs and at most 64 total inputs.')
        from experiments.league_adaptive.runtime import compiled
        commands=compiled(actions)
        if any(len(command)!=6 for command in commands):
            raise ValueError('Each action must contain exactly six native frames.')
        if not scratch and actions != original['actions']:
            raise ValueError('Changed action meanings need scratch=True, not reused F weights.')
        definition = copy.deepcopy(original)
        definition.update(actions=actions, observation_source=texts['observation'],
                          observation_config=copy.deepcopy(modules['observation'].INPUT_CONFIG))
        definition = modules['model'].configure_definition(definition)
        if definition.get('kind') != 'sequence_ppo' or definition.get('character') != original.get('character'):
            raise ValueError('Keep the declared sequence PPO learner and character.')
        if scratch:
            for key in ('deployment_action_indices', 'internal_action_count', 'internal_action_mirror'):
                definition.pop(key, None)
        elif definition['observation_config'] != original['observation_config']:
            raise ValueError('Changed input shapes/configuration need scratch=True.')
        learner = construct_sequence_q(definition)
        if not scratch:
            try:
                learner.load_state_dict(policy.state_dict(), strict=True)
            except RuntimeError as error:
                raise ValueError('Student model architecture does not match these weights; use scratch=True.') from error
        output.mkdir(parents=True, exist_ok=False)
        for name, text in texts.items():
            (output / f'student_{name}.py').write_text(text, encoding='utf-8')
        path = output / 'initial.pt'
        torch.save(dict(definition=definition, state_dict=learner.state_dict(),
                        training=dict(source=str(initial), scratch=scratch,
                                      optimizer='fresh phase; not exact interrupted-run resume')), path)
        write(output / 'definition.json', definition)
        return path
    finally:
        actor.close()


def check_environment(agent, student_directory, opponent, ports, output):
    from experiments.league_adaptive.environment import TrainingFightEnv
    actor = load_actor(agent)
    reward_source = (Path(student_directory) / 'student_reward.py').read_text(encoding='utf-8')
    config = dict(reward='terminal_score', student_reward_source=reward_source,
                  training_group_weights=dict(pve=1, p1=1, p2=1))
    env = TrainingFightEnv(actor.definition, [opponent], config, 1110)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for port in ports:
            observation, _ = env.reset(options=dict(player=port, offset=0))
            actor.reset()
            raw = output / f'p{port}_raw.png'
            stack = output / f'p{port}_input.png'
            cv2.imwrite(str(raw), cv2.cvtColor(env.game.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(stack), np.concatenate(list(observation['image']), axis=1))
            rewards = []
            for _ in range(8):
                action = int(actor.predict(observation, True)[0])
                observation, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                if done or truncated:
                    break
            if not env.observation_space.contains(observation) or not np.isfinite(rewards).all():
                raise RuntimeError('Invalid model input or reward during course environment check.')
            stepped = output / f'p{port}_stepped_input.png'
            cv2.imwrite(str(stepped), np.concatenate(list(observation['image']), axis=1))
            rows.append(dict(player=port, image_shape=list(observation['image'].shape),
                             position=observation['position'].tolist(), input_count=int(actor.policy.action_space.n),
                             rewards=rewards, raw=str(raw), input=str(stack), stepped=str(stepped)))
        report = dict(status='ENVIRONMENT_CHECK_PASSED', agent=str(agent), cases=rows)
        write(output / 'summary.json', report)
        return report
    finally:
        env.close()
        actor.close()


def train_ppo(initial, student_directory, pool, settings, output, *, ports=(1, 2)):
    """Pass course controls to the existing recurrent on-policy trainer."""
    settings = copy.deepcopy(settings)
    for name in ('observation', 'actions', 'reward', 'model'):
        filename=f'student_{name}.py'
        if (Path(student_directory)/filename).read_text(encoding='utf-8') != (Path(initial).parent/filename).read_text(encoding='utf-8'):
            raise ValueError('Student files changed after preparation; rerun the environment-check section.')
    if settings['gamma'] != 1.0 or settings.get('device', 'cpu') != 'cpu':
        raise ValueError('The current verified sequence trainer uses gamma=1 and CPU; other values need an explicit implementation.')
    length = int(settings['sequence_length'])
    if int(settings['batch_size']) % length:
        raise ValueError('batch_size must contain whole recurrent sequences.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reward_source = (Path(student_directory) / 'student_reward.py').read_text(encoding='utf-8')
    shared = dict(pool=copy.deepcopy(pool), learner=dict(
        gamma=1., reward='terminal_score', student_reward_source=reward_source,
        learner_sides=list(ports), coverage_offsets=list(settings.get('coverage_offsets', [0, 12])),
        training_group_weights=copy.deepcopy(settings['training_group_weights'])),
        ppo=dict(n_envs=settings['n_envs'], n_steps=settings['n_steps'], sequence_length=length,
                 batch_sequences=settings['batch_size'] // length, n_epochs=settings['n_epochs'],
                 clip_range=settings['clip_range'], target_kl=settings['target_kl']))
    write(output / 'shared_config.json', shared)
    write(output / 'pool.json', pool)
    command = ['--initial', initial, '--pool', output / 'pool.json', '--config', output / 'shared_config.json',
               '--out', output / 'learner', '--steps', settings['total_timesteps'], '--seed', settings['seed'],
               '--lr', settings['learning_rate'], '--entropy', settings['ent_coef'], '--family-entropy', 0,
               '--gae-lambda', settings['gae_lambda'], '--checkpoint-every', settings['checkpoint_every'],
               '--max-seconds', settings['max_seconds'], '--capture-updates']
    run_module('experiments.ryu_adaptive_curriculum.train_direct_sequence', *command)
    report = json.loads((output / 'learner/result.json').read_text(encoding='utf-8'))
    last = output / 'learner' / f"checkpoint_{report['steps']}.pt"
    if not last.is_file() or report['optimizer_updates'] < 1:
        raise RuntimeError('The phase did not produce a trained checkpoint.')
    report.update(initial_agent=str(initial), last_agent=str(last), selection='last; no unmeasured best claim')
    write(output / 'summary.json', report)
    return last, report


def evaluate_cases(agent, cases, output, *, record_video=True, post_match_seconds=1.):
    from experiments.league_adaptive.evaluate import evaluate, exact_opponent_pool
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pool = exact_opponent_pool([case['opponent'] for case in cases])
    if record_video:
        if post_match_seconds < 1:
            raise ValueError('Keep at least one second after the scored terminal frame.')
        pool_path = output / 'pool.json'
        write(pool_path, pool)
        rows = []
        for i, case in enumerate(cases):
            folder = output / f'game_{i + 1:02d}'
            run_module('experiments.league_adaptive.record', '--agent', agent,
                       '--opponent-pool', pool_path, '--opponent-id', case['opponent']['id'],
                       '--player', case['player'], '--offset', case['offset'], '--out', folder,
                       '--label', 'Counter-Ryu', '--post-match-seconds', post_match_seconds)
            row = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            row.update(title=case.get('title', case['opponent']['id']), opponent_config=case['opponent'])
            rows.append(row)
    else:
        actor = load_actor(agent)
        conditions = [dict(opponent=c['opponent']['id'], player=c['player'], offset=c['offset']) for c in cases]
        try:
            report = evaluate(actor, pool, output / 'evaluation', cpu=False,
                              offsets=tuple(sorted({c['offset'] for c in cases})),
                              conditions=conditions, arena_call_sequence=True)
            rows = report['rows']
        finally:
            actor.close()
    result = dict(agent=str(agent), games=len(rows), wins=sum(r['result'] == 'win' for r in rows),
                  ko_wins=sum(r['result'] == 'win' and r['end_reason'] == 'signed_hp_ko' for r in rows),
                  by_port={str(p): dict(games=sum(r['player'] == p for r in rows),
                                      wins=sum(r['player'] == p and r['result'] == 'win' for r in rows)) for p in (1, 2)},
                  rows=rows, scope='Actual complete rounds; repeated known conditions are not new holdout evidence.')
    write(output / 'summary.json', result)
    return result


def show_result(report):
    from IPython.display import Image, Markdown, Video, display
    if 'cases' in report:
        print(report['status'])
        for case in report['cases']:
            print('P', case['player'], 'input', case['image_shape'], 'actions', case['input_count'])
            for name in ('raw', 'input', 'stepped'):
                display(Image(filename=case[name]))
    elif 'rows' in report:
        print({key: report[key] for key in ('games', 'wins', 'ko_wins', 'by_port')})
        for row in report['rows']:
            if row.get('video'):
                display(Markdown(f"**{row.get('title', 'Match')} · P{row['player']} · {row['result']} · HP {row['self_hp']} : {row['opponent_hp']}**"))
                display(Video(filename=row['video'], embed=True, width=640))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


def show_training_review(report):
    import matplotlib.pyplot as plt
    history = report.get('history', [])
    if not history:
        print('No training history in this session.')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for label in ('actor_loss', 'value_loss'):
        axes[0].plot([r['steps'] for r in history], [r['optimizer'].get(label, np.nan) for r in history], label=label)
    for port in ('CPU', 'P1', 'P2'):
        axes[1].plot([r['steps'] for r in history], [r['completed_rounds'].get(port + '_win', 0) for r in history], label=port)
    axes[0].set_title('Training losses'); axes[1].set_title('Cumulative sampled wins (not validation)')
    for axis in axes:
        axis.set_xlabel('Environment transitions'); axis.legend()
    fig.tight_layout()
    plt.show()


def export_model(agent, output):
    run_module('experiments.ryu_direct_league.portable_export', '--agent', agent, '--out', output,
               '--course-notebook', ROOT / 'notebooks/upstream/sye6308-sf2-classroom-2026.ipynb')
    return Path(output) / 'policy.zip'


def reproduce_integration(config_file, variant, iterations, output):
    """Explicit historical recipe; not a continuation of the notebook learner."""
    config = json.loads(Path(config_file).read_text(encoding='utf-8'))
    rule = copy.deepcopy(config['readout_feasibility'])
    rule.update(copy.deepcopy(rule['variants'][variant])); rule.pop('variants')
    needed = [rule['source'], rule['bank'], config['base_config']]
    needed += [x['path'] if isinstance(x, dict) else x for x in rule.get('independent_success_sources', [])]
    needed += rule.get('verified_episode_sources', [])
    missing = [p for p in needed if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError('Historical trajectory data is absent; use the Windows research environment:\n' + '\n'.join(missing))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write(output / 'action_indices.json', config['course_delivery']['action_indices'])
    rule.update(phase='integration', maximum_iterations=int(iterations), maximum_closure_calls=2 * int(iterations),
                evaluation_action_indices=(output / 'action_indices.json').relative_to(ROOT).as_posix())
    config['experiment'] = output.relative_to(ROOT / 'artifacts').as_posix()
    config['readout_feasibility']['variants'][variant] = rule
    write(output / 'config.json', config)
    run_module('experiments.ryu_counterq.capability', '--config', output / 'config.json',
               '--stage', 'readout-feasibility', '--experiment', variant)
    return output / 'integration/agent.pt'
