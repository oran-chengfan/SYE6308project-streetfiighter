"""A fully trainable recurrent physical-action policy without retained routing.

The initial feedforward function is copied from one learned actor. Every CNN,
coordinate, recurrent, action and value parameter can then learn. There is no
teacher probability added to the logits and no action mask.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.ryu_direct_league.policy import DirectPolicy, policy_kwargs
from experiments.league_adaptive.policy import mirror_indices
from experiments.league_adaptive.runtime import spaces
from experiments.ryu_adaptive_curriculum.history_protocol import continuity_reset


def action_groups(actions):
    groups = []
    names = []
    for action in actions:
        name = action['name']
        if name in ('LEFT', 'RIGHT'): group = 'walking'
        elif name.startswith('UP'): group = 'jump'
        elif name.startswith('DOWN'): group = 'guard_crouch'
        elif name.startswith('SHORYUKEN'): group = 'uppercut'
        elif name.startswith('HADOUKEN'): group = 'projectile'
        elif name.startswith('TATSUMAKI'): group = 'spinning_kick'
        elif name.startswith('CROUCH_KICK'): group = 'crouching_kick'
        elif name == 'NO_OP': group = 'neutral'
        else: group = 'normal_button'
        if group not in names: names.append(group)
        groups.append(names.index(group))
    return names, groups


class HealthReader(nn.Module):
    """Read both absolute health bars from the four legal grayscale frames."""
    def __init__(self):
        super().__init__();self.linear=nn.Linear(4*11*48,2)
    def forward(self,images):
        hud=F.avg_pool2d(images[:,:,:22,:].float()/255.,2,2).flatten(1)
        return self.linear(hud).clamp(0,1)


class RecurrentDirectPolicy(DirectPolicy):
    def __init__(self, *args, memory_hidden=128, memory_enabled=True, family_indices=None, duration_choices=(1,),health_context=False,mask_health=False,action_temperature=1., **kwargs):
        self.memory_hidden = int(memory_hidden)
        self.memory_enabled = bool(memory_enabled)
        self.family_indices = family_indices
        self.duration_choices = tuple(int(x) for x in duration_choices)
        self.health_context=bool(health_context);self.mask_health=bool(mask_health);self.action_temperature=float(action_temperature)
        if not self.duration_choices or self.duration_choices[0]!=1 or min(self.duration_choices)<1:
            raise ValueError('Duration choices must begin with one six-frame call')
        super().__init__(*args, **kwargs)
        self.reset_inference_state()

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.history_gru = nn.GRUCell(self.features_dim + self.action_space.n + int(len(self.duration_choices)>1) + (11 if self.health_context else 0), self.memory_hidden)
        if self.health_context:self.health_reader=HealthReader()
        self.history_action = nn.Linear(self.memory_hidden, self.action_space.n)
        self.history_value = nn.Linear(self.memory_hidden, 1)
        nn.init.zeros_(self.history_action.weight); nn.init.zeros_(self.history_action.bias)
        nn.init.zeros_(self.history_value.weight); nn.init.zeros_(self.history_value.bias)
        if len(self.duration_choices)>1:
            self.history_duration = nn.Linear(self.memory_hidden,self.action_space.n*len(self.duration_choices))
            nn.init.zeros_(self.history_duration.weight); nn.init.zeros_(self.history_duration.bias)
        self.register_buffer('action_families', torch.tensor(self.family_indices, dtype=torch.long))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        values = super()._get_constructor_parameters()
        values.update(memory_hidden=self.memory_hidden, memory_enabled=self.memory_enabled, family_indices=self.family_indices,duration_choices=self.duration_choices,
                      health_context=self.health_context,mask_health=self.mask_health,action_temperature=self.action_temperature)
        return values

    def canonical_actions(self, actions, flip):
        return torch.where(flip, self.physical_mirror[actions], actions) if self.canonical else actions

    def sequence_step(self, features, hidden, previous_actions, starts, flip, remaining=None,health=None,raw_positions=None,controller_left=None,elapsed=None):
        hidden = hidden * (~starts[:, None]).float()
        previous_actions = torch.where(starts, torch.zeros_like(previous_actions), previous_actions)
        if not self.memory_enabled: hidden = torch.zeros_like(hidden)
        inputs = torch.cat((features, F.one_hot(previous_actions, self.action_space.n).float()), dim=1)
        if len(self.duration_choices)>1:
            if remaining is None: raise ValueError('The duration-aware critic requires actual pending input duration')
            remaining=torch.where(starts,torch.zeros_like(remaining),remaining)
            inputs=torch.cat((inputs,remaining[:,None].float()/max(self.duration_choices)),1)
        if self.health_context:
            if any(x is None for x in (health,raw_positions,controller_left,elapsed)):raise ValueError('Health context requires only the declared legal image-derived/history features')
            own=torch.where(controller_left,health[:,0],health[:,1]);other=torch.where(controller_left,health[:,1],health[:,0])
            hp=torch.stack((own,other,own-other),1)*4.
            if self.mask_health:hp=torch.zeros_like(hp)
            context=torch.cat((raw_positions*8.,hp,(controller_left.float()*2.-1.)[:,None],elapsed[:,None].float()/300.),1)
            inputs=torch.cat((inputs,context),1)
        hidden = self.history_gru(inputs, hidden)
        logits = self.action_net(features) + self.history_action(hidden)
        if self.canonical: logits = torch.where(flip[:, None], logits[:, self.physical_mirror], logits)
        distribution = torch.distributions.Categorical(logits=logits/self.action_temperature)
        value = self.value_net(features).flatten() + self.history_value(hidden).flatten()
        return distribution, value, hidden

    def duration_logits(self, hidden, flip):
        if len(self.duration_choices)==1: return hidden.new_zeros((len(hidden),self.action_space.n,1))
        logits=self.history_duration(hidden).reshape(len(hidden),self.action_space.n,len(self.duration_choices))/self.action_temperature
        if self.canonical: logits=torch.where(flip[:,None,None],logits[:,self.physical_mirror],logits)
        return logits

    def duration_distribution(self, hidden, actions, flip):
        logits=self.duration_logits(hidden,flip)
        return torch.distributions.Categorical(logits=logits[torch.arange(len(hidden),device=hidden.device),actions])

    def joint_entropy(self, distribution, hidden, flip):
        conditional=torch.distributions.Categorical(logits=self.duration_logits(hidden,flip)).entropy()
        return distribution.entropy()+(distribution.probs*conditional).sum(1)

    def read_health(self,observations):
        return self.health_reader(observations['image']) if self.health_context else None

    def family_entropy(self, distribution):
        probabilities = distribution.probs
        n = int(self.action_families.max()) + 1
        grouped = torch.zeros(len(probabilities), n, device=probabilities.device)
        grouped = grouped.scatter_add(1, self.action_families[None].expand(len(probabilities), -1), probabilities)
        return -(grouped * grouped.clamp_min(1e-12).log()).sum(1)

    def reset_inference_state(self):
        self._history_hidden = None
        self._history_actions = None
        self._history_images = None
        self._history_remaining = None
        self._history_committed = None
        self._history_controller_left=None;self._history_elapsed=None
        self.inference_history_resets = 0

    def get_inference_state(self):
        if self._history_hidden is None: return None
        return dict(hidden=self._history_hidden.copy(), actions=self._history_actions.copy(),
                    images=self._history_images.copy(), reset_count=self.inference_history_resets,
                    remaining=self._history_remaining.copy(),committed=self._history_committed.copy(),
                    controller_left=self._history_controller_left.copy(),elapsed=self._history_elapsed.copy())

    def set_inference_state(self, state):
        if state is None: self.reset_inference_state(); return
        self._history_hidden = state['hidden'].copy(); self._history_actions = state['actions'].copy()
        self._history_images = state['images'].copy(); self.inference_history_resets = int(state['reset_count'])
        self._history_remaining=state.get('remaining',np.zeros(len(self._history_hidden),np.int64)).copy()
        self._history_committed=state.get('committed',np.zeros(len(self._history_hidden),np.int64)).copy()
        self._history_controller_left=state.get('controller_left',np.ones(len(self._history_hidden),bool)).copy()
        self._history_elapsed=state.get('elapsed',np.zeros(len(self._history_hidden),np.int64)).copy()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.set_training_mode(False)
        if state is not None: self.set_inference_state(state)
        tensor, vectorized = self.obs_to_tensor(observation)
        images = tensor['image'].detach().cpu().numpy()
        batch = len(images)
        if self._history_hidden is None or len(self._history_hidden) != batch:
            self._history_hidden = np.zeros((batch, self.memory_hidden), np.float32)
            self._history_actions = np.zeros(batch, np.int64); self._history_images = None
            self._history_remaining=np.zeros(batch,np.int64); self._history_committed=np.zeros(batch,np.int64)
            self._history_controller_left=np.ones(batch,bool);self._history_elapsed=np.zeros(batch,np.int64)
        starts = continuity_reset(self._history_images, images)
        if episode_start is not None: starts |= np.asarray(episode_start, bool).reshape(-1)
        self.inference_history_resets += int(starts.sum())
        self._history_remaining[starts]=0
        # Only complete default starts are used. Initial self-first coordinates
        # identify which visible health bar belongs to us; retain this legal
        # observation-derived identity after the characters cross each other.
        self._history_controller_left[starts]=(tensor['position'][:,4].cpu().numpy()>=0)[starts]
        self._history_elapsed[starts]=0
        with torch.no_grad():
            features = self.extract_features(tensor)
            flip = tensor['position'][:, 4] < 0
            distribution, values, hidden = self.sequence_step(features,
                torch.as_tensor(self._history_hidden, device=self.device),
                torch.as_tensor(self._history_actions, device=self.device),
                torch.as_tensor(starts, device=self.device), flip,torch.as_tensor(self._history_remaining,device=self.device),
                self.read_health(tensor),tensor['position'],torch.as_tensor(self._history_controller_left,device=self.device),torch.as_tensor(self._history_elapsed,device=self.device))
            pending=torch.as_tensor(self._history_remaining,device=self.device)
            option_starts=pending==0
            proposed = distribution.probs.argmax(1) if deterministic else distribution.sample()
            actions=torch.where(option_starts,proposed,torch.as_tensor(self._history_committed,device=self.device))
            duration_distribution=self.duration_distribution(hidden,actions,flip)
            duration_indices=duration_distribution.probs.argmax(1) if deterministic else duration_distribution.sample()
            durations=torch.as_tensor(self.duration_choices,device=self.device)[duration_indices]
            self._history_remaining=torch.where(option_starts,durations-1,pending-1).cpu().numpy().copy()
            self._history_committed=actions.cpu().numpy().copy()
            self._history_hidden = hidden.cpu().numpy().copy()
            self._history_actions = self.canonical_actions(actions, flip).cpu().numpy().copy()
            self._history_images = images.copy()
            self._history_elapsed+=1
            self.last_memory_probabilities = torch.where(option_starts[:,None],distribution.probs,F.one_hot(actions,self.action_space.n).float()).detach()
            self.last_memory_values = values.detach()
            self.last_option_starts=option_starts.cpu().numpy().copy()
            self.last_duration_choices=durations.cpu().numpy().copy()
            result = actions.cpu().numpy()
        return (result if vectorized else result[0]), None

    def observe_and_force(self, observation, action):
        self.predict(observation, deterministic=True)
        tensor, _ = self.obs_to_tensor(observation)
        physical = torch.full((len(tensor['image']),), int(action), device=self.device, dtype=torch.long)
        self._history_actions = self.canonical_actions(physical, tensor['position'][:, 4] < 0).cpu().numpy().copy()

    def observe_and_force_option(self, observation, action, duration, deterministic=True):
        """Intervene on one complete action-duration choice at a free boundary."""
        if int(duration) not in self.duration_choices: raise ValueError('Undeclared duration')
        if self._history_remaining is not None and bool(np.any(self._history_remaining!=0)):
            raise ValueError('A complete-option intervention requires a free decision boundary')
        self.predict(observation,deterministic=deterministic)
        tensor,_=self.obs_to_tensor(observation)
        physical=torch.full((len(tensor['image']),),int(action),dtype=torch.long,device=self.device)
        self._history_actions=self.canonical_actions(physical,tensor['position'][:,4]<0).cpu().numpy().copy()
        self._history_remaining[:]=int(duration)-1
        self._history_committed[:]=int(action)


def recurrent_kwargs(definition):
    values = policy_kwargs(definition)
    values.update(memory_hidden=definition.get('memory_hidden', 128), memory_enabled=definition.get('memory_enabled', True),
                  family_indices=action_groups(definition['actions'])[1],duration_choices=definition.get('duration_choices',(1,)),
                  health_context=definition.get('health_context',False),mask_health=definition.get('mask_health',False),action_temperature=definition.get('action_temperature',1.))
    return values


def construct_recurrent(definition):
    import gymnasium as gym
    return RecurrentDirectPolicy(spaces(definition), gym.spaces.Discrete(len(definition['actions'])), lambda _:1e-4,
                                 **recurrent_kwargs(definition))


def initialize(source, memory=True):
    definition = copy.deepcopy(source['definition'])
    if definition['kind'] != 'direct_ppo': raise ValueError('Start from a single full-action actor')
    definition.update(kind='recurrent_direct_ppo', memory_hidden=128, memory_enabled=bool(memory))
    policy = construct_recurrent(definition)
    state = policy.state_dict()
    for name, value in source['state_dict'].items():
        if name not in state or state[name].shape != value.shape: raise ValueError('Initialization shape changed: '+name)
        state[name] = value.clone()
    policy.load_state_dict(state, strict=True)
    return policy, definition


def initialize_durations(source, durations=(1,2,4,8)):
    definition=copy.deepcopy(source['definition'])
    if definition['kind']!='recurrent_direct_ppo' or len(definition.get('duration_choices',(1,)))!=1:
        raise ValueError('Extend a one-call recurrent actor')
    definition['duration_choices']=list(durations)
    policy=construct_recurrent(definition); state=policy.state_dict()
    for name,value in source['state_dict'].items():
        if name=='history_gru.weight_ih':
            state[name][:,:value.shape[1]]=value.clone(); state[name][:,-1]=0
        else:
            if name not in state or state[name].shape!=value.shape: raise ValueError('Unexpected duration initialization shape: '+name)
            state[name]=value.clone()
    policy.load_state_dict(state,strict=True)
    return policy,definition


def initialize_health(source,reader,mask=False):
    definition=copy.deepcopy(source['definition'])
    if definition['kind']!='recurrent_direct_ppo' or definition.get('health_context'):raise ValueError('Extend a recurrent actor without the HUD branch')
    definition.update(health_context=True,mask_health=bool(mask))
    policy=construct_recurrent(definition);state=policy.state_dict()
    for name,value in source['state_dict'].items():
        if name=='history_gru.weight_ih':
            state[name][:,:value.shape[1]]=value.clone();state[name][:,value.shape[1]:]=0
        else:
            if name not in state or state[name].shape!=value.shape:raise ValueError('Unexpected health initialization shape: '+name)
            state[name]=value.clone()
    state['health_reader.linear.weight']=reader['weight'].clone();state['health_reader.linear.bias']=reader['bias'].clone()
    policy.load_state_dict(state,strict=True)
    return policy,definition
