"""Complete local DDQN checkpoints at true fight boundaries.

Inference weights remain separate. Old inference/Adam files are initialization,
not resumable training checkpoints. No simulator mid-fight state is fabricated.
"""
import copy
import random
from pathlib import Path
import numpy as np
import torch


FORMAT='sequence_ddqn_boundary_state'


def progress_metadata(progress):
    result={k:copy.deepcopy(v) for k,v in progress.items() if k!='payoff'}
    result['sampler_conditions']=len(progress['payoff'])
    return result


def random_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state().clone())


def restore_random_state(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])


def contract(config,pool,character):
    io_keys={'save_training_state','autosave_native_frames','save_replay','finish_episode_at_budget'}
    result=dict(character=character,pool=copy.deepcopy(pool),rules=copy.deepcopy(config.get('rules',{})),
        learner={k:copy.deepcopy(v) for k,v in config['learner'].items() if k not in io_keys},
        network=copy.deepcopy(config['network']),actions=copy.deepcopy(config['actions']),
        counterfactual=copy.deepcopy(config.get('counterfactual',{})))
    if config.get('history_refresh') is not None:result['history_refresh']=copy.deepcopy(config['history_refresh'])
    return result


def restore_optimizer(optimizer,state):
    """Own the moments: loading on the same device may alias snapshot tensors."""
    optimizer.load_state_dict(copy.deepcopy(state))


def learner_components(policy,learning_rate,optimizer_state=None):
    target=copy.deepcopy(policy);target.reset_inference_state();target.optimizer=None
    target.set_training_mode(False)
    for parameter in target.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=learning_rate)
    if optimizer_state is not None:restore_optimizer(optimizer,optimizer_state)
    return target,optimizer


def transfer_adam_state(old_policy,new_policy,old_state,learning_rate):
    """Transfer unchanged named parameters; new parameters start without moments."""
    if len(old_state['param_groups'])!=1:raise ValueError('Expected the shared single-group Adam')
    old_names=[n for n,p in old_policy.named_parameters() if p.requires_grad]
    new_names=[n for n,p in new_policy.named_parameters() if p.requires_grad]
    old_ids=old_state['param_groups'][0]['params']
    if len(old_names)!=len(old_ids):raise ValueError('Original optimizer parameter order is unavailable')
    old_parameters=dict(old_policy.named_parameters());new_parameters=dict(new_policy.named_parameters())
    optimizer=torch.optim.Adam([p for p in new_policy.parameters() if p.requires_grad],lr=learning_rate)
    result=optimizer.state_dict();new_ids=result['param_groups'][0]['params'];old_by_name=dict(zip(old_names,old_ids))
    result['param_groups'][0].update({k:copy.deepcopy(v) for k,v in old_state['param_groups'][0].items() if k!='params'})
    for name,new_id in zip(new_names,new_ids):
        if name not in old_by_name:continue
        if new_parameters[name].shape!=old_parameters[name].shape:raise ValueError('Shared optimizer parameter changed shape')
        old_id=old_by_name[name]
        if old_id in old_state['state']:result['state'][new_id]=copy.deepcopy(old_state['state'][old_id])
    if not set(old_names)<=set(new_names):raise ValueError('Model conversion removed a trained parameter')
    restore_optimizer(optimizer,result)
    return optimizer,dict(shared_parameters=len(old_names),new_parameters=[n for n in new_names if n not in old_by_name])


def load_checkpoint(path,config,pool,character):
    state=torch.load(path,map_location='cpu',weights_only=False)
    required={'online','target','optimizer','replay','progress','rng','exploration','branch_rng','environment','actor_state'}
    if state.get('format')!=FORMAT or not required<=state.keys() or not state.get('at_fight_boundary'):
        raise ValueError('Missing full DDQN state. Old weights require --start initialization, not --resume.')
    if state['contract']!=contract(config,pool,character):
        raise ValueError('Full resume changed network, reward, optimizer, exploration or opponent contract')
    return state


def save_checkpoint(path,*,policy,target,optimizer,replay,definition,config,pool,progress,
                    explore,branch_rng,actor,env,purpose='continuation',history_seed_ids=()):
    if not env.ended:raise ValueError('A complete training snapshot requires a fight boundary')
    rng=random_state()
    state=dict(format=FORMAT,at_fight_boundary=True,purpose=purpose,
        definition=copy.deepcopy(definition),contract=contract(config,pool,definition['character']),
        online=policy.state_dict(),target=target.state_dict(),optimizer=optimizer.state_dict(),
        replay=replay.state_dict(),progress=copy.deepcopy(progress),rng=rng,
        exploration=dict(rng=copy.deepcopy(explore.rng.bit_generator.state),pending=list(explore.pending)),
        branch_rng=copy.deepcopy(branch_rng.bit_generator.state),actor_state=actor.get_state(),
        history_seed_ids=list(history_seed_ids),
        environment=dict(episode_index=env.episode_index,pool_scores=copy.deepcopy(env.pool_scores),
            rng=None if getattr(env,'_np_random',None) is None else copy.deepcopy(env.np_random.bit_generator.state),
            controllers=[dict(count=c.count,actor=None if c.actor is None else c.actor.get_state()) for c in env.controllers]))
    write_checkpoint(path,state)
    # Serialization and evidence output must not advance the learning RNG.
    restore_random_state(rng)
    return state['progress']


def write_checkpoint(path,state):
    """Atomic local serialization shared by boundary saves and explicit migration."""
    if state.get('format')!=FORMAT or not state.get('at_fight_boundary'):raise ValueError('Invalid boundary snapshot')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing');torch.save(state,temporary);temporary.replace(path)


def restore_components(state,*,policy,target,optimizer,replay,explore,branch_rng,actor,env):
    policy.load_state_dict(state['online']);target.load_state_dict(state['target'])
    restore_optimizer(optimizer,state['optimizer']);replay.load_state_dict(state['replay'])
    explore.rng.bit_generator.state=state['exploration']['rng'];explore.pending.clear()
    explore.pending.extend(state['exploration']['pending']);branch_rng.bit_generator.state=state['branch_rng']
    actor.set_state(state['actor_state']);env.episode_index=state['environment']['episode_index']
    env.pool_scores=copy.deepcopy(state['environment']['pool_scores'])
    if state['environment']['rng'] is not None:env.np_random.bit_generator.state=state['environment']['rng']
    if len(env.controllers)!=len(state['environment']['controllers']):raise ValueError('Controller pool changed')
    for controller,saved in zip(env.controllers,state['environment']['controllers']):
        controller.count=saved['count']
        if controller.actor is not None:controller.actor.set_state(saved['actor'])
    # Constructors load opponent models and may consume RNG. Restore last.
    restore_random_state(state['rng'])
    return copy.deepcopy(state['progress'])
