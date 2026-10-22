"""Legal observation/action memory with an explicitly retained neural prior."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.ryu_direct_league.neural_mixture import NeuralMixturePolicy,mixture_kwargs
from experiments.league_adaptive.policy import mirror_indices
from experiments.league_adaptive.runtime import spaces
from experiments.ryu_adaptive_curriculum.history_protocol import continuity_reset


class MemoryPolicy(NeuralMixturePolicy):
    def __init__(self,*args,memory_hidden=128,memory_enabled=True,memory_smoothing=.1,memory_action_mirror=None,memory_learn_visual=False,**kwargs):
        self.memory_hidden=int(memory_hidden);self.memory_enabled=bool(memory_enabled)
        self.memory_smoothing=float(memory_smoothing);self.memory_action_mirror=memory_action_mirror
        self.memory_learn_visual=bool(memory_learn_visual)
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.register_buffer('memory_physical_mirror',torch.tensor(self.memory_action_mirror,dtype=torch.long))
        if self.memory_learn_visual:
            self.adaptive_visual=copy.deepcopy(self.features_extractor.visual)
            for parameter in self.adaptive_visual.parameters():parameter.requires_grad_(True)
        self.history_gru=nn.GRUCell(self.features_dim+self.action_space.n+(128 if self.memory_learn_visual else 0),self.memory_hidden)
        self.history_action=nn.Linear(self.memory_hidden,self.action_space.n)
        self.history_value=nn.Linear(self.memory_hidden,1)
        nn.init.zeros_(self.history_action.weight);nn.init.zeros_(self.history_action.bias)
        nn.init.zeros_(self.history_value.weight);nn.init.zeros_(self.history_value.bias)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters()
        result.update(memory_hidden=self.memory_hidden,memory_enabled=self.memory_enabled,
            memory_smoothing=self.memory_smoothing,memory_action_mirror=self.memory_action_mirror,memory_learn_visual=self.memory_learn_visual)
        return result
    def encode_base(self,observations):
        # The retained neural feature/prior networks are fixed during this
        # controlled experiment; sequence gradients train the new memory only.
        with torch.no_grad():
            features=self.extract_features(observations)
            actor_latent,critic_latent=self.mlp_extractor(features)
            prior=self._get_action_dist_from_latent(actor_latent).distribution.probs
            log_prior=((1-self.memory_smoothing)*prior+self.memory_smoothing/self.action_space.n).log()
            values=self.value_net(critic_latent).flatten()
            flip=observations['position'][:,4]<0
        return features.detach(),log_prior.detach(),values.detach(),flip.detach()
    def encode_adaptive(self,observations):
        if not self.memory_learn_visual:return None
        image=observations['image'].float()/255.;flip=observations['position'][:,4]<0
        image=torch.where(flip[:,None,None,None],image.flip(-1),image)
        return self.adaptive_visual(torch.cat((image,image[:,1:]-image[:,:-1]),dim=1))
    def memory_step(self,features,log_prior,base_values,hidden,previous_actions,episode_starts,flip,adaptive=None):
        hidden=hidden*(~episode_starts[:,None]).float()
        previous_actions=torch.where(episode_starts,torch.zeros_like(previous_actions),previous_actions)
        if not self.memory_enabled:hidden=torch.zeros_like(hidden)
        if self.memory_learn_visual:
            if adaptive is None:raise ValueError('Trainable visual features must be recomputed from legal images')
            features=torch.cat((features,adaptive),dim=1)
        inputs=torch.cat((features,F.one_hot(previous_actions,self.action_space.n).float()),dim=1)
        next_hidden=self.history_gru(inputs,hidden)
        delta=self.history_action(next_hidden)
        delta=torch.where(flip[:,None],delta[:,self.memory_physical_mirror],delta)
        distribution=torch.distributions.Categorical(logits=log_prior+delta)
        values=base_values+self.history_value(next_hidden).flatten()
        return distribution,values,next_hidden
    def canonical_actions(self,actions,flip):
        return torch.where(flip,self.memory_physical_mirror[actions],actions)
    def reset_inference_state(self):
        super().reset_inference_state()
        self._memory_hidden_state=None;self._memory_previous_actions=None;self._memory_images=None
        self.inference_history_resets=0
    def get_inference_state(self):
        if self._memory_hidden_state is None:return None
        return dict(hidden=self._memory_hidden_state.copy(),previous_actions=self._memory_previous_actions.copy(),
            images=self._memory_images.copy(),reset_count=self.inference_history_resets)
    def set_inference_state(self,state):
        if state is None:self.reset_inference_state();return
        self._memory_hidden_state=np.asarray(state['hidden']).copy()
        self._memory_previous_actions=np.asarray(state['previous_actions']).copy()
        self._memory_images=np.asarray(state['images']).copy();self.inference_history_resets=int(state['reset_count'])
    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        self.set_training_mode(False)
        if state is not None:self.set_inference_state(state)
        tensor,vectorized=self.obs_to_tensor(observation);images=tensor['image'].detach().cpu().numpy();batch=len(images)
        if self._memory_hidden_state is None or len(self._memory_hidden_state)!=batch:
            self._memory_hidden_state=np.zeros((batch,self.memory_hidden),np.float32)
            self._memory_previous_actions=np.zeros(batch,np.int64);self._memory_images=None
        starts=continuity_reset(self._memory_images,images)
        if episode_start is not None:starts|=np.asarray(episode_start,bool).reshape(-1)
        self.inference_history_resets+=int(starts.sum())
        with torch.no_grad():
            features,prior,base_values,flip=self.encode_base(tensor)
            distribution,values,hidden=self.memory_step(features,prior,base_values,
                torch.as_tensor(self._memory_hidden_state,device=self.device),
                torch.as_tensor(self._memory_previous_actions,device=self.device),
                torch.as_tensor(starts,device=self.device),flip,self.encode_adaptive(tensor))
            actions=distribution.probs.argmax(1) if deterministic else distribution.sample()
            self._memory_hidden_state=hidden.cpu().numpy().copy()
            self._memory_previous_actions=self.canonical_actions(actions,flip).cpu().numpy().copy()
            self._memory_images=images.copy()
            self.last_memory_probabilities=distribution.probs.detach();self.last_memory_values=values.detach()
            result=actions.cpu().numpy()
        return (result if vectorized else result[0]),None
    def observe_and_force(self,observation,action):
        """Update the legal belief once, recording the actual intervened input."""
        self.predict(observation,deterministic=True)
        tensor,_=self.obs_to_tensor(observation)
        physical=torch.full((len(tensor['image']),),int(action),dtype=torch.long,device=self.device)
        self._memory_previous_actions=self.canonical_actions(physical,tensor['position'][:,4]<0).cpu().numpy().copy()


def memory_kwargs(definition):
    kwargs=mixture_kwargs(definition)
    kwargs.update(memory_hidden=definition.get('memory_hidden',128),memory_enabled=definition.get('memory_enabled',True),
        memory_smoothing=definition.get('memory_smoothing',.1),memory_action_mirror=mirror_indices(definition['actions']),memory_learn_visual=definition.get('memory_learn_visual',False))
    return kwargs


def construct_memory(definition):
    import gymnasium as gym
    return MemoryPolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,**memory_kwargs(definition))


def initialize_memory(source,memory_enabled=True):
    definition=copy.deepcopy(source['definition'])
    if definition['kind']!='neural_mixture' or definition.get('residual_actions') or definition.get('action_logit_correction'):
        raise ValueError('Use the retained unmodified neural mixture as the explicit prior')
    definition.update(kind='memory_ppo',memory_enabled=bool(memory_enabled),memory_hidden=128,memory_smoothing=.1)
    policy=construct_memory(definition);state=policy.state_dict()
    for name,value in source['state_dict'].items():state[name]=value.clone()
    policy.load_state_dict(state,strict=True)
    return policy,definition
