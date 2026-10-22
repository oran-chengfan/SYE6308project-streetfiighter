"""One legal-observation Q policy with reconstructible bounded history.

Convolution filters and the validated health reader are fixed perceptual
sensors. The spatial projection, coordinates, recurrent belief and every
action-value parameter learn. No frozen combat actor supplies action proposals.
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
from experiments.league_adaptive.runtime import spaces


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
