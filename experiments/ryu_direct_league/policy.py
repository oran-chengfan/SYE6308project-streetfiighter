"""Ordinary trainable PPO actor/critic; no frozen combat expert or tactical rules."""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from experiments.league_adaptive.runtime import spaces,load_actor,Actor
from experiments.league_adaptive.policy import mirror_indices


class DirectEncoder(BaseFeaturesExtractor):
    def __init__(self,observation_space,image_features=512,position_features=64,canonical=False):
        super().__init__(observation_space,image_features+position_features)
        self.canonical=bool(canonical)
        cnn=nn.Sequential(nn.Conv2d(4,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),
            nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad():width=cnn(torch.zeros(1,*observation_space['image'].shape)).shape[1]
        self.image_net=nn.Sequential(cnn,nn.Linear(width,image_features),nn.ReLU(),nn.LayerNorm(image_features))
        self.position_net=nn.Sequential(nn.Linear(6,position_features),nn.ReLU(),nn.LayerNorm(position_features))
    def transform(self,observations):
        image=observations['image'];position=observations['position'].clone()
        self.current_flip=position[:,4]<0
        if self.canonical:
            flip=self.current_flip
            image=torch.where(flip[:,None,None,None],image.flip(-1),image)
            position[:,0]=torch.where(flip,.5-position[:,0],position[:,0])
            position[:,2]=torch.where(flip,.5-position[:,2],position[:,2])
            position[:,4]=position[:,4].abs()
        return image,position
    def forward(self,observations):
        image,position=self.transform(observations)
        return torch.cat((self.image_net(image),self.position_net(position)),1)


class MotionEncoder(DirectEncoder):
    """Add learned frame-difference features; the initial action head ignores them."""
    def __init__(self,observation_space,canonical=False):
        super().__init__(observation_space,canonical=canonical)
        cnn=nn.Sequential(nn.Conv2d(7,16,8,4),nn.ReLU(),nn.Conv2d(16,32,4,2),nn.ReLU(),
            nn.Conv2d(32,32,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad():width=cnn(torch.zeros(1,7,*observation_space['image'].shape[-2:])).shape[1]
        self.motion_net=nn.Sequential(cnn,nn.Linear(width,128),nn.Tanh())
        self._features_dim+=128
    def forward(self,observations):
        image,position=self.transform(observations)
        motion=torch.cat((image,image[:,1:]-image[:,:-1]),1)
        return torch.cat((self.image_net(image),self.position_net(position),self.motion_net(motion)),1)


class DirectPolicy(MultiInputActorCriticPolicy):
    def __init__(self,*args,action_mirror=None,canonical=False,**kwargs):
        self.action_mirror_list=action_mirror;self.canonical=bool(canonical)
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.register_buffer('physical_mirror',torch.tensor(self.action_mirror_list,dtype=torch.long))
    def _get_action_dist_from_latent(self,latent_pi):
        logits=self.action_net(latent_pi)
        if self.canonical:
            flip=self.features_extractor.current_flip
            logits=torch.where(flip[:,None],logits[:,self.physical_mirror],logits)
        return self.action_dist.proba_distribution(action_logits=logits)
    def _get_constructor_parameters(self):
        values=super()._get_constructor_parameters()
        values.update(action_mirror=self.action_mirror_list,canonical=self.canonical)
        return values


def policy_kwargs(definition):
    canonical=definition.get('canonical',False)
    encoder=MotionEncoder if definition.get('encoder')=='motion_residual' else DirectEncoder
    return dict(features_extractor_class=encoder,features_extractor_kwargs=dict(canonical=canonical),
        net_arch=dict(pi=[],vf=[]),action_mirror=mirror_indices(definition['actions']),canonical=canonical)


def construct_direct(definition):
    import gymnasium as gym
    return DirectPolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,**policy_kwargs(definition))


def initialize(source,actions,canonical=False):
    original=load_actor(source)
    if type(original.policy).__name__!='MultiInputActorCriticPolicy':
        raise ValueError('Initialize from the plain learned actor, without a routed/frozen expert architecture')
    if original.actions!=actions[:len(original.actions)]:raise ValueError('Physical action prefix changed')
    definition={k:copy.deepcopy(original.definition[k]) for k in ('observation_source','observation_config')}
    definition.update(kind='direct_ppo',actions=copy.deepcopy(actions),canonical=bool(canonical))
    policy=construct_direct(definition);state=policy.state_dict();old=original.policy.state_dict()
    for key,value in old.items():
        if key in state and state[key].shape==value.shape:state[key]=value.clone()
    n=len(original.actions);state['action_net.weight'][:n]=old['action_net.weight'];state['action_net.bias'][:n]=old['action_net.bias']
    names={a['name']:i for i,a in enumerate(original.actions)}
    # New crouching buttons inherit the corresponding learned tap, with a
    # lower initial logit. This preserves the old greedy action without a mask.
    for i,action in enumerate(actions[n:],n):
        index=names[action['name'].rsplit('_',1)[-1]+'_tap']
        state['action_net.weight'][i]=old['action_net.weight'][index]
        state['action_net.bias'][i]=old['action_net.bias'][index]-1.
    policy.load_state_dict(state,strict=True)
    for parameter in policy.parameters():parameter.requires_grad_(True)
    return policy,definition


def save(path,policy,definition,training=None,optimizer=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(definition=definition,state_dict={k:v.detach().cpu() for k,v in policy.state_dict().items()},training=training or {})
    if optimizer is not None:payload['optimizer_state']=optimizer.state_dict()
    temporary=path.with_suffix('.writing');torch.save(payload,temporary);temporary.replace(path)
