"""Compact trainable responses on visible history, optionally anchored to a prior."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from experiments.league_adaptive.runtime import legacy_policy,spaces


def mirror_indices(actions):
    ids={a['name']:i for i,a in enumerate(actions)};result=[]
    for a in actions:
        name=a['name'].replace('RIGHT','TEMP').replace('LEFT','RIGHT').replace('TEMP','LEFT')
        name=name.replace('_R_','_TEMP_').replace('_L_','_R_').replace('_TEMP_','_L_')
        result.append(ids[name])
    return result


class VisibleHistoryEncoder(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,160)
        cnn=nn.Sequential(nn.Conv2d(7,16,8,4),nn.ReLU(),nn.Conv2d(16,32,4,2),nn.ReLU(),
            nn.Conv2d(32,32,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad():width=cnn(torch.zeros(1,7,*observation_space['image'].shape[-2:])).shape[1]
        self.visual=nn.Sequential(cnn,nn.Linear(width,128),nn.Tanh())
        self.coordinates=nn.Sequential(nn.Linear(6,32),nn.Tanh())
    def forward(self,obs):
        self.current_observations={k:v.detach() for k,v in obs.items()}
        image=obs['image'];pos=obs['position'].clone();flip=pos[:,4]<0
        image=torch.where(flip[:,None,None,None],image.flip(-1),image)
        pos[:,0]=torch.where(flip,.5-pos[:,0],pos[:,0]);pos[:,2]=torch.where(flip,.5-pos[:,2],pos[:,2]);pos[:,4]=pos[:,4].abs()
        history=torch.cat((image,image[:,1:]-image[:,:-1]),dim=1)
        return torch.cat((self.visual(history),self.coordinates(pos*8)),dim=1)


class ResponsePolicy(MultiInputActorCriticPolicy):
    def __init__(self,*args,action_mirror=None,reference_definition=None,prior_strength=.5,**kwargs):
        self.action_mirror_list=action_mirror;self.reference_definition=reference_definition;self.prior_strength=float(prior_strength)
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.register_buffer('physical_mirror',torch.tensor(self.action_mirror_list,dtype=torch.long))
        if self.reference_definition is not None:
            self.reference=legacy_policy(self.reference_definition,self.observation_space)
            for p in self.reference.parameters():p.requires_grad_(False)
            self.reference.optimizer=None
            nn.init.zeros_(self.action_net.weight);nn.init.zeros_(self.action_net.bias)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
    def retained_distribution(self,obs):
        # Exact observation bytes index a pure inference cache. This carries no
        # opponent belief, game counter or information between distinct inputs.
        if not hasattr(self,'_prior_cache'):self._prior_cache={}
        images=(obs['image'].detach()*255.).round().to(torch.uint8).cpu().numpy()
        positions=obs['position'].detach().cpu().numpy()
        keys=[image.tobytes()+pos.tobytes() for image,pos in zip(images,positions)]
        missing=[i for i,k in enumerate(keys) if k not in self._prior_cache]
        if missing:
            indices=torch.as_tensor(missing,device=obs['image'].device)
            self.reference.set_training_mode(False)
            with torch.no_grad():
                raw=dict(image=obs['image'][indices]*255.,position=obs['position'][indices])
                values=self.reference.get_distribution(raw).distribution.probs.detach().cpu()
            for i,value in zip(missing,values):self._prior_cache[keys[i]]=value.clone()
        values=torch.stack([self._prior_cache[k] for k in keys]).to(obs['image'].device)
        while len(self._prior_cache)>1536:self._prior_cache.pop(next(iter(self._prior_cache)))
        return values
    def _get_action_dist_from_latent(self,latent_pi):
        scores=self.action_net(latent_pi);obs=self.features_extractor.current_observations;flip=obs['position'][:,4]<0
        scores=torch.where(flip[:,None],scores[:,self.physical_mirror],scores)
        if self.reference_definition is not None:
            with torch.no_grad():prior=self.retained_distribution(obs)
            scores=scores+self.prior_strength*prior.clamp_min(1e-6).log()
        return self.action_dist.proba_distribution(action_logits=scores)
    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters()
        result.update(action_mirror=self.action_mirror_list,reference_definition=self.reference_definition,prior_strength=self.prior_strength)
        return result


def construct_policy(definition):
    if definition['kind']=='counterfactual_option_planner':
        from experiments.ryu_option_planner.policy import construct_planner
        return construct_planner(definition)
    if definition['kind']=='context_double_dqn':
        from experiments.ryu_counterq.policy import construct_counterq
        return construct_counterq(definition)
    if definition['kind']=='recurrent_direct_ppo':
        from experiments.ryu_adaptive_curriculum.recurrent_direct import construct_recurrent
        return construct_recurrent(definition)
    if definition['kind']=='memory_ppo':
        from experiments.ryu_adaptive_curriculum.memory_policy import construct_memory
        return construct_memory(definition)
    if definition['kind']=='legacy':return legacy_policy(definition,spaces(definition))
    if definition['kind']=='neural_mixture':
        from experiments.ryu_direct_league.neural_mixture import construct_mixture
        return construct_mixture(definition)
    if definition['kind']=='direct_ppo':
        from experiments.ryu_direct_league.policy import construct_direct
        return construct_direct(definition)
    if definition['kind']!='response':raise ValueError('Unsupported policy kind in the minimal handoff: '+definition['kind'])
    return ResponsePolicy(spaces(definition),__import__('gymnasium').spaces.Discrete(len(definition['actions'])),lambda _:1e-4,
        features_extractor_class=VisibleHistoryEncoder,net_arch=dict(pi=[128],vf=[128]),
        action_mirror=mirror_indices(definition['actions']),reference_definition=definition.get('reference_definition'),
        prior_strength=definition.get('prior_strength',.5))


def response_definition(base,reference=False,prior_strength=.5):
    definition={k:copy.deepcopy(base[k]) for k in ('actions','observation_source','observation_config')}
    definition.update(kind='response',prior_strength=float(prior_strength),reference_definition=copy.deepcopy(base) if reference else None)
    if reference and base['kind']!='legacy':raise ValueError('This study anchors the main actor to the frozen original candidate')
    return definition


def save_policy(path,policy,definition,training=None):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(definition=definition,state_dict={k:v.detach().cpu() for k,v in policy.state_dict().items()},training=training or {})
    temp=path.with_suffix('.writing');torch.save(payload,temp);temp.replace(path)
