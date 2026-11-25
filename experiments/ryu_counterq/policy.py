"""Legal-observation Q policies with reconstructible history.

New sequence learners re-encode legal replay images with trainable filters.
Historical copied-sensor weights retain their original frozen-stem default.
The older ContextQ policy is retained for historical weights; its health
reader is not part of the active sequence learner.
"""
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.ryu_direct_league.policy import DirectPolicy,policy_kwargs
from experiments.ryu_adaptive_curriculum.recurrent_direct import HealthReader
from experiments.ryu_adaptive_curriculum.history_protocol import continuity_reset
from experiments.league_adaptive.runtime import spaces,compiled
from experiments.league_adaptive.characters import advance_input_durations
from experiments.league_adaptive.policy import VisibleHistoryEncoder, mirror_indices
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class SequenceQPolicy(MultiInputActorCriticPolicy):
    """Single recurrent learner over every physical input, with no expert votes.

    New models train the visual stem and use physical previous actions;
    historical frozen/canonical models remain loadable. Q heads start
    independently of PPO values/logits.
    Recurrent resets use only continuity of the legal four-image observation.
    """
    def __init__(self, *args, action_mirror=None, memory_hidden=128,
                 trainable_visual=False, physical_actions=False, signed_relative_position=False,
                 input_history=None,physical_commands=None,facing_canonical=False,visual_normalization=None,action_readout=None,
                 deployment_action_indices=None,internal_action_count=None,internal_action_mirror=None, **kwargs):
        self.action_mirror_list = action_mirror
        self.memory_hidden = int(memory_hidden)
        self.trainable_visual=bool(trainable_visual)
        self.physical_actions=bool(physical_actions)
        self.signed_relative_position=bool(signed_relative_position)
        self.facing_canonical=bool(facing_canonical)
        if visual_normalization not in (None,'layer'):raise ValueError('Unknown visual normalization')
        self.visual_normalization=visual_normalization
        if action_readout not in (None,'relu'):raise ValueError('Unknown action readout')
        self.action_readout=action_readout
        self.deployment_action_indices=None if deployment_action_indices is None else list(deployment_action_indices)
        self.internal_action_count=internal_action_count
        self.internal_action_mirror=internal_action_mirror
        if self.facing_canonical and not self.physical_actions:raise ValueError('Facing normalization expects recorded physical inputs')
        self.input_history=copy.deepcopy(input_history);self.physical_commands=physical_commands
        if self.input_history is not None:
            if (not self.physical_actions or self.input_history['mode'] not in ('real','zero')
                    or not 0<int(self.input_history['cap_native_frames'])<=255
                    or self.input_history['normalization']<=0 or physical_commands is None):
                raise ValueError('Input durations require explicit physical commands and a real/zero configuration')
        super().__init__(*args, **kwargs)
        self.reset_inference_state()

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.visual_normalization=='layer':
            self.features_extractor.visual[2]=nn.Sequential(
                nn.LayerNorm(self.features_extractor.visual[1].out_features),nn.Tanh())
        self.register_buffer('physical_mirror', torch.tensor(self.action_mirror_list, dtype=torch.long))
        count=int(self.action_space.n if self.internal_action_count is None else self.internal_action_count)
        if self.deployment_action_indices is not None:
            indices=self.deployment_action_indices
            if len(indices)!=self.action_space.n or len(set(indices))!=len(indices) or min(indices)!=0 or max(indices)>=count:
                raise ValueError('Invalid external-to-internal deployment action catalog')
        elif count!=self.action_space.n:
            raise ValueError('An internal action width requires its explicit physical mapping')
        self.sequence_gru = nn.GRU(160 + count, self.memory_hidden, batch_first=True)
        self.action_net = nn.Linear(160 + self.memory_hidden, count)
        if self.action_readout=='relu':
            width=160+self.memory_hidden
            self.action_net=nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,count))
        self.value_net = nn.Linear(160 + self.memory_hidden, 1)
        output=self.action_net if self.action_readout is None else self.action_net[-1]
        nn.init.normal_(output.weight, std=.01)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(self.value_net.weight)
        nn.init.zeros_(self.value_net.bias)
        if self.input_history is not None:
            # Preserve every old matrix shape and initial inference. This is
            # an input projection, not another expert or action correction.
            self.input_duration_projection=nn.Linear(3,160,bias=False)
            nn.init.zeros_(self.input_duration_projection.weight)
        for parameter in self.features_extractor.visual[0].parameters():
            parameter.requires_grad_(self.trainable_visual)
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        values = super()._get_constructor_parameters()
        values.update(action_mirror=self.action_mirror_list, memory_hidden=self.memory_hidden,
                      trainable_visual=self.trainable_visual,physical_actions=self.physical_actions,
                      signed_relative_position=self.signed_relative_position,
                      input_history=self.input_history,physical_commands=self.physical_commands,facing_canonical=self.facing_canonical,
                      visual_normalization=self.visual_normalization,action_readout=self.action_readout,
                      deployment_action_indices=self.deployment_action_indices,internal_action_count=self.internal_action_count,
                      internal_action_mirror=self.internal_action_mirror)
        return values

    def encode_maps(self, observations):
        image = observations['image'].float()/255.
        flip = (observations['position'][:, 4] < 0) & (not self.physical_actions or self.facing_canonical)
        image = torch.where(flip[:, None, None, None], image.flip(-1), image)
        history = torch.cat((image, image[:, 1:]-image[:, :-1]), 1)
        if self.trainable_visual:return self.features_extractor.visual[0](history)
        with torch.no_grad():return self.features_extractor.visual[0](history).to(torch.float16)

    def observation_sequence_values(self, images, positions, previous, hidden=None, burn_in=0,input_durations=None,return_features=False):
        """Re-encode legal replay bytes with current filters, including burn-in.

        The recurrent input is the actual preceding physical action for new
        models. Historical canonical models retain their original semantics.
        """
        batch,length=images.shape[:2]
        maps=self.encode_maps(dict(image=images.flatten(0,1),position=positions.flatten(0,1)))
        return self.sequence_values(maps.reshape(batch,length,-1),positions,previous,hidden,burn_in,input_durations,return_features)

    def sequence_values(self, maps, positions, previous, hidden=None, burn_in=0,input_durations=None,return_features=False):
        batch, length, width = maps.shape
        visual = self.features_extractor.visual[2](
            self.features_extractor.visual[1](maps.float().reshape(-1, width)))
        raw_pos = positions.reshape(-1, 6)
        flip = (raw_pos[:, 4] < 0) & (not self.physical_actions or self.facing_canonical)
        # PPO passes positions as a view of the differentiable feature tensor.
        # Keep normalization out of place: abs backward needs the original dx.
        relative_x=raw_pos[:,4]
        if self.facing_canonical or not (self.physical_actions and self.signed_relative_position):relative_x=relative_x.abs()
        pos=torch.stack((torch.where(flip,.5-raw_pos[:,0],raw_pos[:,0]),raw_pos[:,1],
            torch.where(flip,.5-raw_pos[:,2],raw_pos[:,2]),raw_pos[:,3],relative_x,raw_pos[:,5]),1)
        features = torch.cat((visual, self.features_extractor.coordinates(pos*8)), 1).reshape(batch, length, 160)
        if self.input_history is not None:
            if input_durations is None or tuple(input_durations.shape)!=(batch,length,3):
                raise ValueError('Enabled input history requires real contiguous recorded durations')
            duration=input_durations.to(features.device).float()/self.input_history['normalization']
            if self.facing_canonical:
                duration=torch.where(flip.reshape(batch,length,1),duration[:,:,[1,0,2]],duration)
            if self.input_history['mode']=='zero':duration=torch.zeros_like(duration)
            features=features+self.input_duration_projection(duration)
        if self.facing_canonical:
            previous=torch.where(flip.reshape(batch,length),self.physical_mirror[previous.long()],previous)
        internal_previous=previous.long()
        if self.deployment_action_indices is not None:
            internal_previous=torch.as_tensor(self.deployment_action_indices,device=previous.device)[internal_previous]
        count=self.action_space.n if self.internal_action_count is None else int(self.internal_action_count)
        inputs = torch.cat((features, F.one_hot(internal_previous, count).float()), 2)
        if burn_in:
            with torch.no_grad():
                prefix, hidden = self.sequence_gru(inputs[:, :burn_in].detach(), hidden)
            tail, hidden = self.sequence_gru(inputs[:, burn_in:], hidden.detach())
            recurrent = torch.cat((prefix, tail), 1)
            features = torch.cat((features[:, :burn_in].detach(), features[:, burn_in:]), 1)
        else:
            recurrent, hidden = self.sequence_gru(inputs, hidden)
        combined = torch.cat((features, recurrent), 2)
        advantage = self.action_net(combined)
        q = self.value_net(combined) + advantage - advantage.mean(2, keepdim=True)
        if self.deployment_action_indices is not None:
            if self.facing_canonical:
                if self.internal_action_mirror is None:raise ValueError('Missing original canonical action map')
                q=torch.where(flip.reshape(batch,length,1),q[:,:,self.internal_action_mirror],q)
            self.last_internal_critic=q.mean(2)
            q=q[:,:,self.deployment_action_indices]
        elif self.facing_canonical:
            q=torch.where(flip.reshape(batch,length,1),q[:,:,self.physical_mirror],q)
        return (q, hidden, combined) if return_features else (q, hidden)

    def reset_inference_state(self):
        self._sequence_hidden = None
        self._sequence_previous = None
        self._sequence_images = None
        self._sequence_input_durations=None;self._input_durations_before=None
        self.inference_history_resets = 0

    def get_inference_state(self):
        if self._sequence_hidden is None:
            return None
        result=dict(hidden=self._sequence_hidden.detach().cpu().clone(),
                    previous=self._sequence_previous.copy(), images=self._sequence_images.copy(),
                    reset_count=self.inference_history_resets)
        if self.input_history is not None:
            result.update(input_durations=self._sequence_input_durations.copy(),
                input_durations_before=self._input_durations_before.copy())
        return result

    def set_inference_state(self, state):
        if state is None:
            self.reset_inference_state()
            return
        self._sequence_hidden = state['hidden'].clone().to(self.device)
        self._sequence_previous = state['previous'].copy()
        self._sequence_images = state['images'].copy()
        self.inference_history_resets = int(state['reset_count'])
        if self.input_history is not None:
            self._sequence_input_durations=state['input_durations'].copy()
            self._input_durations_before=state['input_durations_before'].copy()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.set_training_mode(False)
        if state is not None:
            self.set_inference_state(state)
        tensor, vectorized = self.obs_to_tensor(observation)
        images = tensor['image'].cpu().numpy()
        batch = len(images)
        starts = continuity_reset(self._sequence_images, images)
        if episode_start is not None:
            starts |= np.asarray(episode_start, bool).reshape(-1)
        if self._sequence_hidden is None or self._sequence_hidden.shape[1] != batch:
            self._sequence_hidden = torch.zeros(1, batch, self.memory_hidden, device=self.device)
            self._sequence_previous = np.zeros(batch, np.int64)
            if self.input_history is not None:self._sequence_input_durations=np.zeros((batch,3),np.uint8)
            starts[:] = True
        self._sequence_hidden[:, starts] = 0
        self._sequence_previous[starts] = 0
        duration_argument={}
        if self.input_history is not None:
            self._sequence_input_durations[starts]=0
            self._input_durations_before=self._sequence_input_durations.copy()
            self.last_input_durations_before=self._input_durations_before.copy()
            duration_argument['input_durations']=torch.as_tensor(self._input_durations_before,device=self.device)[:,None]
        self.inference_history_resets += int(starts.sum())
        self.last_hidden_before=self._sequence_hidden.detach().clone()
        self.last_previous_before=self._sequence_previous.copy()
        with torch.no_grad():
            maps = self.encode_maps(tensor)
            q, hidden = self.sequence_values(maps[:, None], tensor['position'][:, None],
                torch.as_tensor(self._sequence_previous, device=self.device)[:, None], self._sequence_hidden,**duration_argument)
            canonical_q = q[:, 0]
            flip = (tensor['position'][:, 4] < 0) & (not self.physical_actions)
            physical_q = torch.where(flip[:, None], canonical_q[:, self.physical_mirror], canonical_q)
            canonical = canonical_q.argmax(1)
            if not deterministic:
                canonical = torch.where(torch.rand(batch, device=self.device)<.05,
                    torch.randint(self.action_space.n, (batch,), device=self.device), canonical)
            actions = torch.where(flip, self.physical_mirror[canonical], canonical)
            self._sequence_previous = canonical.cpu().numpy().copy()
            self._sequence_hidden = hidden.detach()
            self._sequence_images = images.copy()
            if getattr(self,'probability_policy',False):self.last_policy_logits=physical_q.detach()
            else:self.last_context_q_values = physical_q.detach()
            self.last_memory_values = physical_q.max(1).values.detach()
            # The cloning-link softmax is an identity-test score. Acting uses
            # argmax (or explicit epsilon exploration), not this distribution.
            self.last_memory_probabilities = torch.softmax(physical_q/.2, 1).detach()
            result = actions.cpu().numpy()
            if self.input_history is not None:self._record_input_durations(result)
        return (result if vectorized else result[0]), None

    def observe_and_force(self, observation, action):
        self.predict(observation, deterministic=True)
        tensor, _ = self.obs_to_tensor(observation)
        self.record_executed_actions(action, tensor['position'].cpu().numpy())

    def record_executed_actions(self, action, positions):
        """Keep recurrent action history faithful after training exploration."""
        actual = np.broadcast_to(np.asarray(action, np.int64), (len(self._sequence_previous),))
        flip = (np.asarray(positions)[:, 4] < 0) & (not self.physical_actions)
        mirror = self.physical_mirror.cpu().numpy()
        self._sequence_previous = np.where(flip, mirror[actual], actual).copy()
        if self.input_history is not None:self._record_input_durations(actual)

    def _record_input_durations(self,actual):
        # Rebase on the pre-decision values. Exploration replaces the greedy
        # command; it must never count both proposed and executed inputs.
        self._sequence_input_durations=np.stack([advance_input_durations(before,
            self.physical_commands[int(action)],self.input_history['cap_native_frames'])
            for before,action in zip(self._input_durations_before,actual)])


class SequencePPOPolicy(SequenceQPolicy):
    """Same legal recurrent encoder and parameter capacity, PPO objectives.

    Only fresh, shared random parameters initialize this policy. Dueling
    scores become categorical logits; their action mean is the critic. The
    common baseline cancels in the categorical distribution. No learned Q
    estimate or replay return is treated as a PPO probability/value target.
    """
    probability_policy=True

    def __init__(self, *args, action_temperature=1., **kwargs):
        self.action_temperature=float(action_temperature)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), action_temperature=self.action_temperature)

    def extract_features(self,observations,features_extractor=None):
        return torch.cat((self.encode_maps(observations),observations['position']),1)

    def sequence_step(self,features,hidden,previous,starts,flip,input_durations=None):
        hidden=hidden*(~starts[:,None]).float()
        previous=torch.where(starts,torch.zeros_like(previous),previous)
        if input_durations is not None:
            input_durations=torch.where(starts[:,None],torch.zeros_like(input_durations),input_durations)
        score,next_hidden=self.sequence_values(features[:,:-6][:,None],
            features[:,-6:][:,None],previous[:,None],hidden[None],
            input_durations=None if input_durations is None else input_durations[:,None])
        score=score[:,0]
        physical_flip=flip & (not self.physical_actions)
        logits=torch.where(physical_flip[:,None],score[:,self.physical_mirror],score)
        critic=score.mean(1) if self.deployment_action_indices is None else self.last_internal_critic[:,0]
        return torch.distributions.Categorical(logits=logits/self.action_temperature),critic,next_hidden[0]

    def canonical_actions(self,actions,flip):
        return torch.where(flip,self.physical_mirror[actions],actions) if not self.physical_actions else actions

    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        _,_=super().predict(observation,state,episode_start,deterministic=True)
        distribution=torch.distributions.Categorical(logits=self.last_policy_logits/self.action_temperature)
        actions=distribution.probs.argmax(1) if deterministic else distribution.sample()
        tensor,vectorized=self.obs_to_tensor(observation)
        self.record_executed_actions(actions.cpu().numpy(),tensor['position'].cpu().numpy())
        self.last_memory_probabilities=distribution.probs.detach()
        self.last_memory_values=(self.last_policy_logits.mean(1) if self.deployment_action_indices is None
            else self.last_internal_critic[:,0]).detach()
        result=actions.cpu().numpy()
        return (result if vectorized else result[0]),None


def sequence_q_kwargs(definition):
    kwargs=dict(features_extractor_class=VisibleHistoryEncoder, net_arch=dict(pi=[], vf=[]),
        action_mirror=list(range(len(definition['actions']))) if definition.get('physical_actions') and not definition.get('facing_canonical') else mirror_indices(definition['actions']),
        memory_hidden=definition.get('memory_hidden', 128),
        trainable_visual=definition.get('trainable_visual',False),physical_actions=definition.get('physical_actions',False),
        signed_relative_position=definition.get('signed_relative_position',False),input_history=definition.get('input_history'),
        facing_canonical=definition.get('facing_canonical',False),visual_normalization=definition.get('visual_normalization'),
        action_readout=definition.get('action_readout'),
        deployment_action_indices=definition.get('deployment_action_indices'),internal_action_count=definition.get('internal_action_count'),
        internal_action_mirror=definition.get('internal_action_mirror'),
        physical_commands=compiled(definition['actions']) if definition.get('input_history') is not None else None)
    if definition.get('kind')=='sequence_ppo':
        kwargs['action_temperature']=float(definition.get('action_temperature',1.))
    return kwargs


def construct_sequence_q(definition):
    import gymnasium as gym
    cls=SequencePPOPolicy if definition['kind']=='sequence_ppo' else SequenceQPolicy
    return cls(spaces(definition), gym.spaces.Discrete(len(definition['actions'])),
        lambda _:1e-4, **sequence_q_kwargs(definition))


def restrict_sequence_actions(source,indices,maximum_actions=64):
    """Explicit deployment catalog reduction, retaining the full recurrent body.

    Internal logits and actual-previous-action channels keep their original
    widths and indices. Preferred physical inputs are unchanged whenever the
    original greedy input is retained. Probabilities are conditional on the
    retained catalog; unobserved excluded winners are NOT guaranteed absent.
    Original models and definitions remain untouched. This is not training.
    """
    old=source['definition'];indices=list(map(int,indices))
    if (old['kind'] not in ('sequence_ppo','sequence_double_q') or not old.get('physical_actions')
            or old.get('deployment_action_indices') is not None):
        raise ValueError('Restrict one original physical sequence policy only')
    if indices!=sorted(set(indices)) or indices[:15]!=list(range(15)) or not 15<=len(indices)<=maximum_actions:
        raise ValueError('Retain the course15basics and a sorted bounded unique catalog')
    if max(indices)>=len(old['actions']):raise ValueError('Unknown original physical action')
    mirrored=mirror_indices(old['actions'])
    if any(mirrored[i] not in indices for i in indices):raise ValueError('Catalog must include horizontal mirror partners')
    definition=copy.deepcopy(old)
    definition.update(actions=[copy.deepcopy(old['actions'][i]) for i in indices],
        deployment_action_indices=indices,internal_action_count=len(old['actions']),
        internal_action_mirror=source['state_dict']['physical_mirror'].tolist())
    policy=construct_sequence_q(definition);state=policy.state_dict()
    for name,value in source['state_dict'].items():
        if name=='physical_mirror':continue
        if state[name].shape!=value.shape:raise ValueError('Unexpected recurrent conversion shape: '+name)
        state[name]=value.clone()
    policy.load_state_dict(state,strict=True)
    policy.set_training_mode(False)
    return policy,definition,dict(original_actions=len(old['actions']),external_actions=len(indices),
        original_indices=indices,internal_inference_parameters_unchanged=True,optimizer_transferred=False,
        scope='Catalog reduction for the published course64-action limit. Deterministic behavior identity is conditional on retained original winners and must be tested on real histories. Not a universal lossless conversion.')


def initialize_action_readout(source):
    """One nonlinear readout initialized to the original linear function.

    ReLU(x)-ReLU(-x)=x. All shared visual/recurrent/value tensors are retained;
    old model definitions keep their linear head and inference semantics.
    """
    if source['definition'].get('action_readout') is not None:
        raise ValueError('Readout conversion expects the existing linear policy')
    definition=copy.deepcopy(source['definition']);definition['action_readout']='relu'
    policy=construct_sequence_q(definition);state=policy.state_dict()
    for name,value in source['state_dict'].items():
        if not name.startswith('action_net.'):state[name]=value.clone()
    width=160+policy.memory_hidden;identity=torch.eye(width)
    state['action_net.0.weight']=torch.cat((identity,-identity),0)
    state['action_net.0.bias']=torch.zeros(2*width)
    weight=source['state_dict']['action_net.weight']
    state['action_net.2.weight']=torch.cat((weight,-weight),1)
    state['action_net.2.bias']=source['state_dict']['action_net.bias'].clone()
    policy.load_state_dict(state,strict=True)
    return policy,definition,dict(identity_formula='ReLU(x)-ReLU(-x)=x',hidden_width=2*width,
        shared_inference_tensors_unchanged=True,optimizer_state_transferred=False,
        scope='Explicit single action-readout capacity experiment, not another expert or changed legal input/history.')


def initialize_visual_normalization(source):
    """Explicit research conversion; old weights retain their original meaning.

    The measured saturated projection motivates one LayerNorm before tanh.
    Shared tensors and Adam moments transfer by name; the two new affine
    parameters start at identity without moments. This changes initial
    behavior and requires a zero-update full-game comparison.
    """
    from experiments.ryu_counterq.training_state import transfer_adam_state
    if source['definition'].get('visual_normalization') is not None:
        raise ValueError('The source already defines visual normalization')
    old=construct_sequence_q(source['definition']);old.load_state_dict(source['state_dict'])
    definition=copy.deepcopy(source['definition']);definition['visual_normalization']='layer'
    policy=construct_sequence_q(definition)
    result=policy.load_state_dict(source['state_dict'],strict=False)
    expected=set(policy.state_dict())-set(source['state_dict'])
    if (set(result.missing_keys)!=expected or result.unexpected_keys or not expected
            or any(not key.endswith(('visual.2.0.weight','visual.2.0.bias')) for key in expected)):
        raise ValueError('Normalization conversion changed an unrelated tensor')
    if any(not torch.equal(policy.state_dict()[name],value) for name,value in source['state_dict'].items()):
        raise ValueError('A shared inference tensor changed during conversion')
    transfer=None
    if 'optimizer_state' in source:
        policy.optimizer,transfer=transfer_adam_state(old,policy,source['optimizer_state'],
            source['optimizer_state']['param_groups'][0]['lr'])
    return policy,definition,dict(shared_inference_tensors_unchanged=True,
        new_parameters=sorted(expected),optimizer=transfer,
        initialization_is_behavior_identical=False,
        scope='Explicit projection conditioning, not a new expert, recurrent module or hidden input.')


def initialize_sequence_q(source):
    definition = {k:copy.deepcopy(source['definition'][k]) for k in ('actions','observation_source','observation_config')}
    definition.update(kind='sequence_double_q', memory_hidden=128)
    policy = construct_sequence_q(definition)
    encoder = {k.removeprefix('features_extractor.'):v for k,v in source['state_dict'].items()
               if k.startswith('features_extractor.')}
    policy.features_extractor.load_state_dict(encoder, strict=True)
    return policy, definition


def random_sequence_q(character, network=None):
    """Common random DDQN initialization; PPO logits/values are never Q labels."""
    from experiments.league_adaptive.characters import CHARACTERS,physical_actions
    from experiments.league_adaptive.runtime import TARGET
    if character not in CHARACTERS:raise ValueError(character)
    source=torch.load(TARGET,map_location='cpu',weights_only=False)['definition']
    definition={k:copy.deepcopy(source[k]) for k in ('observation_source','observation_config')}
    definition.update(kind='sequence_double_q',character=character,actions=physical_actions(int((network or {}).get('tap_delay_frames',0))),
                      memory_hidden=128,trainable_visual=True,physical_actions=True)
    definition.update(network or {})
    return construct_sequence_q(definition),definition


def make_extras(position,health,controller_left,elapsed,pending):
    own=torch.where(controller_left,health[:,0],health[:,1]);other=torch.where(controller_left,health[:,1],health[:,0])
    return torch.cat((pending[:,None].float()/8.,position*8.,
                      torch.stack((own,other,own-other),1)*4.,
                      (controller_left.float()*2.-1.)[:,None],elapsed[:,None].float()/300.),1)


class ContextQPolicy(DirectPolicy):
    def __init__(self,*args,context_length=16,context_durations=(1,2,4,8),q_bound=1.5,history_enabled=True,context_offsets=None,**kwargs):
        self.context_length=int(context_length);self.context_durations=tuple(context_durations)
        self.context_offsets=None if context_offsets is None else tuple(int(v) for v in context_offsets)
        self.q_bound=float(q_bound);self.history_enabled=bool(history_enabled)
        super().__init__(*args,**kwargs);self.reset_inference_state()

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.health_reader=HealthReader()
        self.context_gru=nn.GRU(self.features_dim+self.action_space.n+12+int(self.context_offsets is not None),128,batch_first=True)
        self.context_advantage=nn.Linear(128,self.action_space.n*len(self.context_durations))
        self.context_value=nn.Linear(128,1)
        nn.init.zeros_(self.context_advantage.weight);nn.init.zeros_(self.context_advantage.bias)
        nn.init.zeros_(self.context_value.weight);nn.init.zeros_(self.context_value.bias)
        for p in self.features_extractor.image_net[0].parameters():p.requires_grad_(False)
        for p in self.health_reader.parameters():p.requires_grad_(False)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        values=super()._get_constructor_parameters()
        values.update(context_length=self.context_length,context_durations=self.context_durations,q_bound=self.q_bound,history_enabled=self.history_enabled,context_offsets=self.context_offsets)
        return values

    def encode_maps(self,observations):
        image=observations['image'].float()/255.;flip=observations['position'][:,4]<0
        if self.canonical:image=torch.where(flip[:,None,None,None],image.flip(-1),image)
        with torch.no_grad():
            # Identical quantization is used in RAM replay and inference.
            return self.features_extractor.image_net[0](image).to(torch.float16)

    def context_representation(self,maps,positions,extras,previous,lengths):
        batch,length,width=maps.shape
        image_net=self.features_extractor.image_net
        visual=image_net[3](image_net[2](image_net[1](maps.float().reshape(-1,width))))
        pos=positions.reshape(-1,6).clone();flip=pos[:,4]<0
        if self.canonical:
            pos[:,0]=torch.where(flip,.5-pos[:,0],pos[:,0]);pos[:,2]=torch.where(flip,.5-pos[:,2],pos[:,2]);pos[:,4]=pos[:,4].abs()
        coordinates=self.features_extractor.position_net(pos)
        features=torch.cat((visual,coordinates),1).reshape(batch,length,-1)
        inputs=torch.cat((features,F.one_hot(previous,self.action_space.n).float(),extras),2)
        if self.context_offsets is not None:
            gap=torch.zeros_like(extras[:,:,11]);gap[:,1:]=(extras[:,1:,11]-extras[:,:-1,11])*300./8.
            inputs=torch.cat((inputs,gap.clamp(0,12)[:,:,None]),2)
        packed=nn.utils.rnn.pack_padded_sequence(inputs,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.context_gru(packed);hidden=hidden[0]
        current=features[torch.arange(batch,device=features.device),lengths.to(features.device)-1]
        return current,hidden

    def context_values(self,maps,positions,extras,previous,lengths):
        current,hidden=self.context_representation(maps,positions,extras,previous,lengths)
        advantage=self.action_net(current).repeat_interleave(len(self.context_durations),dim=1)+self.context_advantage(hidden)
        value=self.value_net(current)+self.context_value(hidden)
        raw=value+advantage-advantage.mean(1,keepdim=True)
        return self.q_bound*torch.tanh(raw/self.q_bound)

    def histories_to_tensors(self,histories):
        if self.context_offsets is not None:
            histories=[[h[len(h)-1-age] for age in sorted(self.context_offsets,reverse=True) if age<len(h)] for h in histories]
        lengths=[len(h) for h in histories];length=max(lengths);batch=len(histories)
        width=histories[0][0]['map'].shape[0]
        maps=np.zeros((batch,length,width),np.float16);positions=np.zeros((batch,length,6),np.float32)
        extras=np.zeros((batch,length,12),np.float32);previous=np.zeros((batch,length),np.int64)
        for i,history in enumerate(histories):
            for t,row in enumerate(history):
                maps[i,t]=row['map'];positions[i,t]=row['position'];extras[i,t]=row['extras'];previous[i,t]=row['previous']
        return tuple(torch.as_tensor(x,device=self.device) for x in (maps,positions,extras,previous))+ (torch.as_tensor(lengths,dtype=torch.long),)

    def reset_inference_state(self):
        self._q_streams=None;self._q_images=None;self._history_remaining=None;self.inference_history_resets=0

    def get_inference_state(self):
        return None if self._q_streams is None else dict(streams=copy.deepcopy(self._q_streams),images=self._q_images.copy(),reset_count=self.inference_history_resets)

    def set_inference_state(self,state):
        if state is None:self.reset_inference_state();return
        self._q_streams=copy.deepcopy(state['streams']);self._q_images=state['images'].copy();self.inference_history_resets=int(state['reset_count'])
        self._history_remaining=np.asarray([s['pending'] for s in self._q_streams],np.int64)

    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        self.set_training_mode(False)
        if state is not None:self.set_inference_state(state)
        tensor,vectorized=self.obs_to_tensor(observation);images=tensor['image'].cpu().numpy();batch=len(images)
        if self._q_streams is None or len(self._q_streams)!=batch:self._q_streams=[None]*batch;self._q_images=None
        starts=continuity_reset(self._q_images,images)
        if episode_start is not None:starts|=np.asarray(episode_start,bool).reshape(-1)
        self.inference_history_resets+=int(starts.sum())
        positions=tensor['position'];flip=positions[:,4]<0
        for i in range(batch):
            if starts[i] or self._q_streams[i] is None:
                self._q_streams[i]=dict(history=deque(maxlen=self.context_length),role_left=bool(positions[i,4]>=0),elapsed=0,pending=0,committed=0,previous=0,chosen_duration=1)
        roles=torch.as_tensor([s['role_left'] for s in self._q_streams],device=self.device)
        elapsed=torch.as_tensor([s['elapsed'] for s in self._q_streams],device=self.device)
        pending=torch.as_tensor([s['pending'] for s in self._q_streams],device=self.device)
        with torch.no_grad():
            maps=self.encode_maps(tensor);health=self.health_reader(tensor['image'])
            extra=make_extras(positions,health,roles,elapsed,pending)
            for i,stream in enumerate(self._q_streams):
                stream['history'].append(dict(map=maps[i].cpu().numpy().copy(),position=positions[i].cpu().numpy().copy(),extras=extra[i].cpu().numpy().copy(),previous=stream['previous']))
            histories=[list(s['history'])[-(self.context_length if self.history_enabled else 1):] for s in self._q_streams]
            q=self.context_values(*self.histories_to_tensors(histories))
            options=q.argmax(1)
            if not deterministic:
                explore=torch.rand(batch,device=self.device)<.1
                options=torch.where(explore,torch.randint(q.shape[1],(batch,),device=self.device),options)
            actions=[];option_starts=[];durations=[]
            for i,stream in enumerate(self._q_streams):
                free=stream['pending']==0
                if free:
                    canonical=int(options[i])//len(self.context_durations)
                    physical=int(self.physical_mirror[canonical]) if bool(flip[i]) and self.canonical else canonical
                    duration=int(self.context_durations[int(options[i])%len(self.context_durations)])
                    stream['committed']=physical;stream['pending']=duration;stream['chosen_duration']=duration
                action=stream['committed'];stream['pending']-=1
                stream['previous']=int(self.physical_mirror[action]) if bool(flip[i]) and self.canonical else action
                stream['elapsed']+=1;actions.append(action);option_starts.append(free);durations.append(stream['chosen_duration'])
            actions_tensor=torch.as_tensor(actions,device=self.device)
            self.last_context_q_values=q.detach();self.last_memory_values=q.max(1).values.detach()
            self.last_memory_probabilities=F.one_hot(actions_tensor,self.action_space.n).float()
            self.last_option_starts=np.asarray(option_starts,bool);self.last_duration_choices=np.asarray(durations,np.int64)
            self._history_remaining=np.asarray([s['pending'] for s in self._q_streams],np.int64)
            self._q_images=images.copy();result=np.asarray(actions,np.int64)
        return (result if vectorized else result[0]),None

    def observe_and_force(self,observation,action):
        self.predict(observation,deterministic=True);tensor,_=self.obs_to_tensor(observation)
        for i,stream in enumerate(self._q_streams):
            flip=bool(tensor['position'][i,4]<0)
            stream['previous']=int(self.physical_mirror[action]) if flip and self.canonical else int(action)

    def observe_and_force_option(self,observation,action,duration,deterministic=True):
        if duration not in self.context_durations:raise ValueError('Undeclared duration')
        if self._history_remaining is not None and np.any(self._history_remaining):raise ValueError('Intervene at a free option boundary')
        self.observe_and_force(observation,action)
        for stream in self._q_streams:stream['committed']=int(action);stream['pending']=int(duration)-1;stream['chosen_duration']=int(duration)
        self._history_remaining=np.asarray([s['pending'] for s in self._q_streams],np.int64)


def counterq_kwargs(definition):
    values=policy_kwargs(definition)
    values.update(context_length=definition.get('context_length',16),context_durations=definition.get('context_durations',(1,2,4,8)),
                  q_bound=definition.get('q_bound',1.5),history_enabled=definition.get('history_enabled',True),context_offsets=definition.get('context_offsets'))
    return values


def construct_counterq(definition):
    import gymnasium as gym
    return ContextQPolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,**counterq_kwargs(definition))


def initialize(source,reader,context_length=16):
    definition={k:copy.deepcopy(source['definition'][k]) for k in ('actions','observation_source','observation_config','canonical')}
    definition.update(kind='context_double_dqn',context_length=int(context_length),context_durations=[1,2,4,8],q_bound=1.5)
    policy=construct_counterq(definition);state=policy.state_dict()
    for name,value in source['state_dict'].items():
        if name in state and state[name].shape==value.shape:state[name]=value.clone()
    state['action_net.weight']*=.05;state['action_net.bias']*=.05
    if 'history_gru.weight_ih' in source['state_dict']:
        old=source['state_dict']['history_gru.weight_ih'];state['context_gru.weight_ih_l0'][:,:old.shape[1]]=old.clone();state['context_gru.weight_ih_l0'][:,old.shape[1]:]=0
        for a,b in [('weight_hh_l0','weight_hh'),('bias_ih_l0','bias_ih'),('bias_hh_l0','bias_hh')]:state['context_gru.'+a]=source['state_dict']['history_gru.'+b].clone()
    state['health_reader.linear.weight']=reader['weight'].clone();state['health_reader.linear.bias']=reader['bias'].clone()
    policy.load_state_dict(state,strict=True)
    return policy,definition


def expand_multiscale(policy,definition):
    definition=copy.deepcopy(definition)
    definition.update(context_length=96,context_offsets=list(range(8))+list(range(11,40,4))+list(range(47,96,8)))
    expanded=construct_counterq(definition);state=expanded.state_dict();old=policy.state_dict()
    for key,value in old.items():
        if key=='context_gru.weight_ih_l0':
            state[key][:,:value.shape[1]]=value;state[key][:,value.shape[1]:]=0
        else:
            if state[key].shape!=value.shape:raise ValueError('Unexpected multiscale architecture change')
            state[key]=value.clone()
    expanded.load_state_dict(state,strict=True)
    return expanded,definition
