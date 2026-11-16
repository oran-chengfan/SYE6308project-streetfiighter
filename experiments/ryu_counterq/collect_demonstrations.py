"""Continuous legal demonstrations and same-state tactical verification.

One episode stores overlapping images once. Native CPU never supplies action
labels. Branches share explicit dependence groups and are not separate wins.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.league_adaptive.environment import FightEnv
from experiments.league_adaptive.runtime import ROOT, compiled, load_actor, save_fight, restore_fight, setup, write
from experiments.ryu_counterq.policy import initialize_sequence_q
from experiments.ryu_counterq.collect_recoveries import RECOVERY_PREFIXES, recovery_action
from experiments.ryu_direct_league.environment import win_quality_reward


class UnsupportedPhysicalAction(ValueError):
    """A frozen teacher requested an input absent from the declared map."""
    def __init__(self,action):
        self.action=int(action)
        super().__init__(f'Unsupported physical teacher action {self.action}')


def begin_episode(obs):
    return dict(frames=[f.copy() for f in obs['image']], positions=[obs['position'].copy()],
        physical_actions=[], rewards=[], dones=[], supervised=[])


def advance(env, obs, action, episode, supervised=False):
    before = dict(env.game.raw)
    nxt, _, terminated, truncated, info = env.step(action)
    if not np.array_equal(obs['image'][1:], nxt['image'][:-1]):
        raise RuntimeError('Demonstration image stack is not continuous')
    episode['frames'].append(nxt['image'][-1].copy())
    episode['positions'].append(nxt['position'].copy())
    episode['physical_actions'].append(int(action))
    episode['rewards'].append(win_quality_reward(before, info, env.player, terminated or truncated))
    episode['dones'].append(bool(terminated or truncated))
    episode['supervised'].append(bool(supervised))
    return nxt, info


def continue_episode(env, actor, obs, episode=None, supervised=False, action_map=None, observer=None):
    """Continue a restored fight; demonstration capture is optional."""
    info=env.last_info
    while not env.ended:
        action=int(actor.predict(obs,True)[0])
        if action_map is not None:
            if action not in action_map:raise UnsupportedPhysicalAction(action)
            action=action_map[action]
        previous=obs
        if episode is None:obs,_,_,_,info=env.step(action)
        else:obs,info=advance(env,obs,action,episode,supervised)
        if observer is not None:observer(env,previous,info,action)
    return obs, info


def terminal(info):
    return {k:info[k] for k in ('result','self_hp','opponent_hp','native_frames','end_reason')}


def save_episode(directory, episodes, raw, encoder, metadata, teacher=None,executed_labels=False):
    n = len(raw['physical_actions'])
    if not n or not raw['dones'][-1] or any(raw['dones'][:-1]):
        raise ValueError('A demonstration must be a complete continuous round')
    frames = np.stack(raw['frames']).astype(np.uint8)
    positions = np.stack(raw['positions']).astype(np.float32)
    images = np.stack([frames[i:i+4] for i in range(n+1)])
    maps, targets = [], []
    for start in range(0,n+1,128):
        observation = dict(image=images[start:start+128],position=positions[start:start+128])
        with torch.no_grad():
            if encoder is not None:
                tensor, _ = encoder.obs_to_tensor(observation)
                maps.append(encoder.encode_maps(tensor).cpu().numpy())
            if teacher is not None and not executed_labels:
                t, _ = teacher.policy.obs_to_tensor(observation)
                targets.append(teacher.policy.get_distribution(t).distribution.probs.cpu().numpy())
    if encoder is None:
        if teacher is None:raise ValueError('A legal action definition is required for contiguous observations')
        from experiments.league_adaptive.policy import mirror_indices
        mirror=(np.arange(len(teacher.actions)) if getattr(teacher,'definition',{}).get('physical_actions',False)
                else np.asarray(mirror_indices(teacher.actions)))
    else:mirror = encoder.physical_mirror.cpu().numpy()
    physical = np.asarray(raw['physical_actions'],np.int64)
    flip = positions[:-1,4]<0
    actions = np.where(flip,mirror[physical],physical)
    if targets:
        q = np.concatenate(targets)[:n]
        q = np.where(flip[:,None],q[:,mirror],q)
    else:
        q = np.eye(len(teacher.actions) if teacher is not None else 42,dtype=np.float32)[actions]
    rewards = np.asarray(raw['rewards'],np.float32)
    returns = np.cumsum(rewards[::-1],dtype=np.float64)[::-1].astype(np.float32)
    filename = f'episode_{len(episodes):04d}.npz'
    optional={} if not maps else dict(maps=np.concatenate(maps))
    np.savez_compressed(directory/filename,frames=frames,positions=positions,**optional,
        actions=actions.astype(np.int16),rewards=rewards,dones=np.asarray(raw['dones'],bool),
        targets=q.astype(np.float32 if encoder is None else np.float16),supervised=np.asarray(raw['supervised'],bool),returns=returns)
    episodes.append(dict(file=filename,steps=n,**metadata))
    write(directory/'manifest.json',dict(episodes=episodes,
        legal_history_storage='T+4 uint8 frames; observation t is frames[t:t+4].',
        features='Legal contiguous observations; features computed by learner.' if encoder is None else
                 'Fixed copied reference convolutional stem; replay and inference share float16 quantization.',
        scope='Full native trajectories; overlapping branches share a dependence group.'))


def collect_base(config,out,encoder):
    source = load_actor(config['source'])
    teacher=load_actor(config['label_source']) if config.get('label_source') else source
    pool = json.loads((ROOT/config['demonstration_pool']).read_text(encoding='utf-8'))
    present={int(r['stars']) for r in pool if r['kind']=='cpu'}
    if config.get('include_cpu',True):
        pool.extend(dict(id=f'CPU_Ryu_{s}',kind='cpu',stars=s,character='Ryu') for s in range(1,9) if s not in present)
    env = FightEnv(source.definition,pool)
    episodes, matches = [], []
    prior_frames=0; prior_seconds=0.
    if (out/'summary.json').is_file():
        prior=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        matches=prior['rows']; prior_frames=prior['native_frames']; prior_seconds=prior['seconds']
        episodes=json.loads((out/'manifest.json').read_text(encoding='utf-8'))['episodes']
    completed={(r['opponent'],r['player'],r['offset']) for r in matches}
    selected = (None if config.get('conditions') is None else
                {(r['opponent'],int(r['player']),int(r['offset'])) for r in config['conditions']})
    started = time.monotonic()
    try:
        if config.get('arena_call_sequence'):
            source.predict(dict(image=np.zeros((4,84,96),np.uint8),
                position=np.asarray([100,192,200,192,100,0],np.float32)/1024.),True)
        for index,rival in enumerate(pool):
            for player in ((1,) if rival['kind']=='cpu' else config.get('ports',(1,2))):
                for offset in ((0,) if rival['kind']=='cpu' else config.get('offsets',(0,12))):
                    if selected is not None and (rival['id'],player,offset) not in selected:continue
                    if (rival['id'],player,offset) in completed:continue
                    obs,_ = env.reset(options=dict(opponent=index,player=player,offset=offset))
                    if not config.get('arena_call_sequence'):source.reset()
                    ep = begin_episode(obs)
                    _,info = continue_episode(env,source,obs,ep,True)
                    row = dict(opponent=rival['id'],opponent_kind=rival['kind'],player=player,offset=offset,**terminal(info))
                    matches.append(row)
                    if info['result']=='win':
                        save_episode(out,episodes,ep,encoder,dict(group='pve' if rival['kind']=='cpu' else f'p{player}',
                            condition=row,origin='independent_actor_success_reference_labels' if teacher is not source else 'reference_success',
                            dependence_group=[rival['id'],player,offset]),teacher)
                    print('BASE_DEMO',json.dumps(row),flush=True)
        write(out/'summary.json',dict(rows=matches,successful_episodes=len(episodes),native_frames=prior_frames+env.native_frames,
            seconds=prior_seconds+time.monotonic()-started,acting_source=config['source'],
            label_source=config.get('label_source',config['source']),arena_call_sequence=bool(config.get('arena_call_sequence')),
            cpu_labels='Only controllable P1 labels; never CPU inputs.'))
    finally:
        env.close(); source.close()
        if teacher is not source:teacher.close()


def collect_recoveries(config,out,encoder):
    """First historical winning prefix per failed root, without a fresh search."""
    source = load_actor(config['source'])
    historical = json.loads((ROOT/'reports/migration_1022/counterfactual_recoveries.json').read_text(encoding='utf-8'))
    names = {'stress_heldout_fireball_timing':'heldout_fireball_timing','current_strong_response':'strong_response'}
    pool = [r for r in json.loads((ROOT/'configs/focus_pool.json').read_text(encoding='utf-8')) if r['id'] in names.values()]
    env = FightEnv(source.definition,pool)
    actions = {a['name']:i for i,a in enumerate(source.actions)}
    plans = dict(RECOVERY_PREFIXES)
    selected = []
    for root in historical['roots']:
        if root['opponent'] not in names or root['root'] not in (0,16,48) or root['branches'][0]['result']=='win':continue
        winner = next((b for b in root['branches'][1:] if b['result']=='win'),None)
        if winner:selected.append((root,winner))
    write(out/'selection.json',[dict(opponent=names[r['opponent']],player=r['player'],offset=r['offset'],root=r['root'],plan=w['plan']) for r,w in selected])
    episodes, rows = [], []
    started = time.monotonic()
    try:
        for root,expected in selected:
            opponent = names[root['opponent']]
            index = next(i for i,r in enumerate(pool) if r['id']==opponent)
            obs,_ = env.reset(options=dict(opponent=index,player=root['player'],offset=root['offset']))
            source.reset(); prefix = begin_episode(obs)
            for _ in range(root['root']):
                obs,_ = advance(env,obs,int(source.predict(obs,True)[0]),prefix)
                if env.ended:raise RuntimeError('Historical root is after terminal')
            saved = save_fight(env,source)
            _,baseline = continue_episode(env,source,obs,copy.deepcopy(prefix))
            if terminal(baseline)!=terminal(root['branches'][0]):
                raise RuntimeError('Historical baseline differs; stop tactical interpretation')
            obs = restore_fight(env,source,saved)
            _,replay = continue_episode(env,source,obs,copy.deepcopy(prefix))
            if terminal(replay)!=terminal(baseline):raise RuntimeError('Unforced replay differs')
            obs = restore_fight(env,source,saved)
            alternative = copy.deepcopy(prefix)
            for mode,repeats in plans[expected['plan']]:
                for _ in range(repeats):
                    obs,info = advance(env,obs,recovery_action(mode,env.game.raw,env.player,actions),alternative,True)
                    if env.ended:break
                if env.ended:break
            if not env.ended:_,info = continue_episode(env,source,obs,alternative)
            if terminal(info)!=terminal(expected):raise RuntimeError('Historical winning prefix differs')
            row = dict(opponent=opponent,player=root['player'],offset=root['offset'],root=root['root'],plan=expected['plan'],
                baseline=terminal(baseline),alternative=terminal(info),unforced_replay_equal=True)
            rows.append(row)
            save_episode(out,episodes,alternative,encoder,dict(group=f"p{root['player']}",condition=row,
                origin='verified_historical_prefix',dependence_group=[opponent,root['player'],root['offset']]))
            write(out/'progress.json',dict(rows=rows,native_frames=env.native_frames))
            print('VERIFIED_PREFIX',json.dumps(row),flush=True)
        write(out/'summary.json',dict(rows=rows,native_frames=env.native_frames,seconds=time.monotonic()-started,
            scope='Verified prefix plus frozen reference continuation; only forced-prefix labels supervised; overlapping roots dependent.'))
    finally:
        env.close(); source.close()


def collect_learned(config,out,encoder):
    """Verify one preselected eight-call learned motif per source/condition."""
    reference=load_actor(config['source'])
    pool=json.loads((ROOT/config['learned_demo_pool']).read_text(encoding='utf-8'))
    env=FightEnv(reference.definition,pool)
    names={a['name']:i for i,a in enumerate(reference.actions)}
    commands=compiled(reference.actions)
    episodes,rows=[],[]
    factual_frames=0; branch_frames=0; started=time.monotonic()
    settings=config['learned_demonstrations']
    write(out/'selection_rule.json',settings)
    try:
        for path in settings['sources']:
            teacher=load_actor(path)
            if teacher.definition['observation_config']!=reference.definition['observation_config']:
                raise ValueError('Demonstrator observation convention differs')
            mapping=[names[a['name']] for a in teacher.actions]
            if any(a!=commands[i] for a,i in zip(compiled(teacher.actions),mapping)):
                raise ValueError('Demonstrator physical input meaning differs')
            for index,rival in enumerate(pool):
                for player in ((1,) if rival['kind']=='cpu' else (1,2)):
                    for offset in ((0,) if rival['kind']=='cpu' else (0,12)):
                        obs,_=env.reset(options=dict(opponent=index,player=player,offset=offset))
                        teacher.reset(); raw=begin_episode(obs); roots={}; motion=[]
                        before_frames=env.native_frames-offset
                        while not env.ended:
                            decision=len(raw['physical_actions'])
                            if decision in settings['root_decisions']:
                                roots[decision]=(save_fight(env,teacher),copy.deepcopy(raw))
                            before=dict(env.game.raw)
                            action=mapping[int(teacher.predict(obs,True)[0])]
                            obs,info=advance(env,obs,action,raw)
                            after=env.game.raw; p=player; q=3-p
                            motion.append(dict(walk_pixels=abs(after[f'p{p}_x']-before[f'p{p}_x'])
                                if reference.actions[action]['name'] in ('LEFT','RIGHT') and min(before[f'p{p}_y'],after[f'p{p}_y'])>=184 else 0,
                                dealt=max(0,max(0,before[f'p{q}_hp'])-max(0,after[f'p{q}_hp'])),
                                taken=max(0,max(0,before[f'p{p}_hp'])-max(0,after[f'p{p}_hp']))))
                        factual_frames+=env.native_frames-before_frames
                        row=dict(source=path,opponent=rival['id'],player=player,offset=offset,factual=terminal(info),accepted=False)
                        selected=None
                        for root in settings['root_decisions']:
                            window=motion[root:root+settings['prefix_calls']]
                            if root not in roots or len(window)!=settings['prefix_calls']:continue
                            walk=sum(x['walk_pixels'] for x in window); dealt=sum(x['dealt'] for x in window); taken=sum(x['taken'] for x in window)
                            if taken==0 and (walk>=8 or dealt>=12):
                                selected=(root,walk,dealt); break
                        if selected is None:
                            row['reason']='No predeclared movement/pressure motif'; rows.append(row); continue
                        root,walk,dealt=selected
                        saved,prefix=roots[root]
                        branch_start=env.native_frames
                        obs=restore_fight(env,teacher,saved)
                        _,replay=continue_episode(env,teacher,obs,copy.deepcopy(prefix),action_map=mapping)
                        if terminal(replay)!=terminal(info):raise RuntimeError('Learned demonstrator unforced replay differs')
                        obs=restore_fight(env,teacher,saved); reference.reset()
                        _,baseline=continue_episode(env,reference,obs,copy.deepcopy(prefix))
                        obs=restore_fight(env,teacher,saved); reference.reset(); alternative=copy.deepcopy(prefix)
                        for action in raw['physical_actions'][root:root+settings['prefix_calls']]:
                            obs,branch_info=advance(env,obs,action,alternative,True)
                            if env.ended:break
                        if not env.ended:_,branch_info=continue_episode(env,reference,obs,alternative)
                        branch_frames+=env.native_frames-branch_start
                        improved=branch_info['result']=='win' and (baseline['result']!='win' or
                            branch_info['self_hp']>=baseline['self_hp']+settings['minimum_winning_hp_gain'])
                        row.update(root=root,walk_pixels=walk,dealt=dealt,baseline=terminal(baseline),alternative=terminal(branch_info),
                            accepted=bool(improved),unforced_replay_equal=True)
                        rows.append(row)
                        if improved:
                            save_episode(out,episodes,alternative,encoder,dict(group='pve' if rival['kind']=='cpu' else f'p{player}',
                                condition=row,origin='verified_learned_prefix',dependence_group=[rival['id'],player,offset]))
                        write(out/'progress.json',dict(rows=rows,native_frames=env.native_frames))
                        print('LEARNED_MOTIF',json.dumps(row),flush=True)
            teacher.close()
        write(out/'summary.json',dict(rows=rows,accepted=len(episodes),native_frames=env.native_frames,
            factual_native_frames=factual_frames,additional_branch_native_frames=branch_frames,seconds=time.monotonic()-started,
            scope='One fixed motif per demonstrator/condition; no retry after branch failure. Only the verified prefix is cloned. CPU control labels never used.'))
    finally:
        env.close(); reference.close()


def collect_quality(config,out,encoder):
    """Higher-quality controlled PvE trajectories, reward-based Q losses only."""
    reference=load_actor(config['source'])
    pool=[dict(id=f'CPU_Ryu_{s}',kind='cpu',stars=s,character='Ryu') for s in (7,8)]
    env=FightEnv(reference.definition,pool)
    names={a['name']:i for i,a in enumerate(reference.actions)}
    commands=compiled(reference.actions)
    episodes,rows=[],[]; started=time.monotonic()
    for path in config['learned_demonstrations']['sources']:
        teacher=load_actor(path)
        mapping=[names[a['name']] for a in teacher.actions]
        if teacher.definition['observation_source']!=reference.definition['observation_source'] or any(
            a!=commands[i] for a,i in zip(compiled(teacher.actions),mapping)):
            raise ValueError('Quality demonstrator interface differs')
        try:
            for index,rival in enumerate(pool):
                obs,_=env.reset(options=dict(opponent=index,player=1,offset=0))
                teacher.reset(); saved=save_fight(env,teacher); raw=begin_episode(obs)
                _,actual=continue_episode(env,teacher,obs,raw,action_map=mapping)
                obs=restore_fight(env,teacher,saved)
                _,replay=continue_episode(env,teacher,obs,begin_episode(obs),action_map=mapping)
                if terminal(actual)!=terminal(replay):raise RuntimeError('Quality trajectory does not replay')
                obs=restore_fight(env,teacher,saved); reference.reset()
                _,baseline=continue_episode(env,reference,obs,begin_episode(obs))
                ko=actual['result']=='win' and actual['opponent_hp']<=0 and actual['end_reason'] in ('signed_hp_ko','round_counter')
                accepted=ko and baseline['result']=='win' and actual['self_hp']>=baseline['self_hp']+12
                row=dict(source=path,stars=rival['stars'],baseline=terminal(baseline),demonstration=terminal(actual),
                    unforced_replay_equal=True,accepted=bool(accepted))
                rows.append(row)
                if accepted:
                    save_episode(out,episodes,raw,encoder,dict(group='pve',condition=row,
                        origin='verified_quality_trajectory_Q_reward_losses_only',dependence_group=[rival['id'],1,0]))
                print('QUALITY_DEMO',json.dumps(row),flush=True)
        finally:teacher.close()
    write(out/'summary.json',dict(rows=rows,native_frames=env.native_frames,seconds=time.monotonic()-started,
        accepted=len(episodes),scope='Complete controllable P1 actions, not native CPU labels. No behavior-cloning supervision; actual rewards/returns are conditional trajectory evidence.'))
    env.close(); reference.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/tactical_research.json')
    parser.add_argument('--kind',choices=('base','recoveries','learned','quality'),required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--observations-only',action='store_true',
                        help='Collect base legal histories without constructing the legacy Q encoder.')
    args=parser.parse_args(); setup()
    if args.observations_only and args.kind!='base':
        parser.error('Raw observation collection is supported by the base collector only')
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    out=Path(args.out)
    if out.exists() and not (args.kind=='base' and (out/'manifest.json').is_file() and (out/'summary.json').is_file()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    encoder=None
    if not args.observations_only:
        encoder,_=initialize_sequence_q(torch.load(config['source'],map_location='cpu',weights_only=False))
    if args.kind=='base':collect_base(config,out,encoder)
    elif args.kind=='recoveries':collect_recoveries(config,out,encoder)
    elif args.kind=='learned':collect_learned(config,out,encoder)
    else:collect_quality(config,out,encoder)


if __name__=='__main__':main()
