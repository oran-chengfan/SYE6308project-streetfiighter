"""A learned complete-option planner retaining the established PPO motor policy.

The ensemble predicts paired outcome differences of complete input plans
followed by the retained policy. Its disagreement is a conservative heuristic,
not a guarantee for unseen opponents. No rule chooses a plan by distance/style.
"""
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.ryu_counterq.policy import ContextQPolicy,counterq_kwargs,make_extras
from experiments.ryu_adaptive_curriculum.history_protocol import continuity_reset
from experiments.ryu_option_planner.options import PLANS,plan_input


class OptionPlannerPolicy(ContextQPolicy):
    def __init__(self,*args,baseline_definition=None,planner_ensemble=5,planner_beta=1.,planner_margin=.2,planner_action_names=None,planner_semantics=False,planner_control='paired',**kwargs):
        self.baseline_definition=copy.deepcopy(baseline_definition);self.planner_ensemble=int(planner_ensemble)
        self.planner_action_names=planner_action_names
        self.planner_semantics=bool(planner_semantics)
        self.planner_control=planner_control
        self.planner_beta=float(planner_beta);self.planner_margin=float(planner_margin)
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        from experiments.league_adaptive.policy import construct_policy
        self.incumbent=construct_policy(self.baseline_definition);self.incumbent.optimizer=None
        for p in self.incumbent.parameters():p.requires_grad_(False)
        for module in (self.action_net,self.value_net,self.context_advantage,self.context_value):
            for p in module.parameters():p.requires_grad_(False)
        self.planner_heads=nn.ModuleList([nn.Sequential(nn.Linear(self.features_dim+128+(90 if self.planner_semantics else 0),128),nn.Tanh(),nn.Linear(128,len(PLANS))) for _ in range(self.planner_ensemble)])
        for head in self.planner_heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        if self.planner_control=='ppo':
            width=self.features_dim+128+(90 if self.planner_semantics else 0)+2*len(PLANS)
            self.plan_actor=nn.Linear(width,len(PLANS));self.plan_critic=nn.Linear(width,1)
            for layer in (self.plan_actor,self.plan_critic):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            with torch.no_grad():self.plan_actor.bias[0]=5.5
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
        self.planner_names=None if self.planner_action_names is None else {name:i for i,name in enumerate(self.planner_action_names)}

    def _get_constructor_parameters(self):
        values=super()._get_constructor_parameters();values.update(baseline_definition=self.baseline_definition,planner_ensemble=self.planner_ensemble,
            planner_beta=self.planner_beta,planner_margin=self.planner_margin,planner_action_names=self.planner_action_names,planner_semantics=self.planner_semantics,planner_control=self.planner_control)
        return values

    def option_values(self,*context):
        return self.values_from_features(self.planner_features(*context))

    def planner_features(self,maps,positions,extras,previous,lengths):
        current,hidden=self.context_representation(maps,positions,extras,previous,lengths);features=torch.cat((current,hidden),1)
        if not self.planner_semantics:return features
        rows=torch.arange(len(lengths),device=features.device);last=lengths.to(features.device)-1
        pos=positions[rows,last];latest=extras[rows,last];age=(latest[:,11,None]-extras[:,:,11])*300.
        valid=torch.arange(positions.shape[1],device=features.device)[None]<lengths.to(features.device)[:,None]
        motion=[];health=[]
        for lag in (1.,4.,8.,16.):
            distance=(age-lag).abs().masked_fill(~valid,1e6);index=distance.argmin(1);span=age[rows,index].clamp_min(1.)
            motion.append((pos-positions[rows,index])*64./span[:,None]);health.append((latest[:,7:10]-extras[rows,index,7:10])*4.)
        histogram=(F.one_hot(previous,self.action_space.n).float()*valid[:,:,None]).sum(1)/lengths.to(features.device)[:,None]
        semantic=torch.cat((latest,torch.cat(motion,1),torch.cat(health,1),histogram),1)
        if semantic.shape[1]!=90:raise RuntimeError('Semantic feature width changed')
        return torch.cat((features,semantic),1)

    def values_from_features(self,features):
        raw=torch.stack([head(features) for head in self.planner_heads],1)
        return torch.cat((raw[:,:,:1].tanh(),2.*torch.tanh(raw[:,:,1:]/2.)),2)

    def decisions_from_features(self,features):
        predictions=self.values_from_features(features)
        belief=torch.cat((features,predictions.mean(1),torch.sqrt(predictions.var(1,unbiased=False)+1e-6)),1)
        return torch.distributions.Categorical(logits=self.plan_actor(belief)),self.plan_critic(belief).flatten()

    def decisions(self,*context):return self.decisions_from_features(self.planner_features(*context))

    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        self.set_training_mode(False);self.incumbent.set_training_mode(False)
        if state is not None:self.set_inference_state(state)
        tensor,vectorized=self.obs_to_tensor(observation);images=tensor['image'].cpu().numpy();batch=len(images)
        if self._q_streams is None or len(self._q_streams)!=batch:self._q_streams=[None]*batch;self._q_images=None
        starts=continuity_reset(self._q_images,images)
        if episode_start is not None:starts|=np.asarray(episode_start,bool).reshape(-1)
        self.inference_history_resets+=int(starts.sum());position=tensor['position'];flip=position[:,4]<0
        for i in range(batch):
            if starts[i] or self._q_streams[i] is None:
                self._q_streams[i]=dict(history=deque(maxlen=self.context_length),role_left=bool(position[i,4]>=0),elapsed=0,previous=0,pending=0,queue=[],selected=0,chosen_duration=1)
        roles=torch.as_tensor([s['role_left'] for s in self._q_streams],device=self.device)
        elapsed=torch.as_tensor([s['elapsed'] for s in self._q_streams],device=self.device)
        with torch.no_grad():
            maps=self.encode_maps(tensor);extra=make_extras(position,self.health_reader(tensor['image']),roles,elapsed,torch.zeros(batch,device=self.device))
            for i,s in enumerate(self._q_streams):
                s['history'].append(dict(map=maps[i].cpu().numpy().copy(),position=position[i].cpu().numpy().copy(),extras=extra[i].cpu().numpy().copy(),previous=s['previous']))
            histories=[list(s['history'])[-(self.context_length if self.history_enabled else 1):] for s in self._q_streams]
            features=self.planner_features(*self.histories_to_tensors(histories));values=self.values_from_features(features);advantage=values[:,:,1:]
            score=advantage.mean(1)-self.planner_beta*advantage.std(1,unbiased=False)-self.planner_margin
            if self.planner_control=='ppo':
                distribution,_=self.decisions_from_features(features)
                choices=(distribution.probs.argmax(1) if deterministic else distribution.sample()).cpu().numpy()
            else:
                best=score.argmax(1);choices=torch.where(score.max(1).values>0,best+1,torch.zeros_like(best)).cpu().numpy()
            free=[not s['queue'] for s in self._q_streams]
            baseline_indices=[i for i in range(batch) if free[i] and choices[i]==0]
            baseline_actions={}
            if baseline_indices:
                raw={key:value[baseline_indices].cpu().numpy() for key,value in tensor.items()}
                picked=self.incumbent.predict(raw,deterministic=True)[0]
                baseline_actions={i:int(action) for i,action in zip(baseline_indices,np.asarray(picked).reshape(-1))}
            if self.planner_names is None:raise RuntimeError('Physical action names were not initialized')
            actions=[];selected=[];durations=[]
            for i,s in enumerate(self._q_streams):
                if free[i]:
                    s['selected']=int(choices[i]);s['queue']=[mode for mode,count in PLANS[s['selected']][1] for _ in range(count)]
                    s['chosen_duration']=max(1,len(s['queue']))
                if s['queue']:
                    mode=s['queue'].pop(0)
                    action=int(mode) if isinstance(mode,(int,np.integer)) else plan_input(mode,float(position[i,4]),self.planner_names)
                else:action=baseline_actions[i]
                s['pending']=len(s['queue']);s['previous']=int(self.physical_mirror[action]) if bool(flip[i]) and self.canonical else action;s['elapsed']+=1
                actions.append(action);selected.append(s['selected']);durations.append(s['chosen_duration'])
            actions_tensor=torch.as_tensor(actions,device=self.device);self.last_memory_probabilities=F.one_hot(actions_tensor,self.action_space.n).float()
            means=values.mean(1);ids=torch.as_tensor(selected,device=self.device)
            self.last_memory_values=(means[:,0]+torch.where(ids>0,means.gather(1,ids[:,None]).flatten(),torch.zeros(batch,device=self.device))).clamp(-1,1).detach()
            self.last_planner_options=np.asarray(selected,np.int64);self.last_planner_scores=score.detach();self.last_option_starts=np.asarray(free,bool)
            self.last_duration_choices=np.asarray(durations,np.int64);self._history_remaining=np.asarray([s['pending'] for s in self._q_streams],np.int64)
            self._q_images=images.copy();actions=np.asarray(actions,np.int64)
        return (actions if vectorized else actions[0]),None

    def observe_and_force(self,observation,action):
        self.predict(observation,deterministic=True);tensor,_=self.obs_to_tensor(observation)
        for i,s in enumerate(self._q_streams):
            s['previous']=int(self.physical_mirror[action]) if bool(tensor['position'][i,4]<0) and self.canonical else int(action)

    def observe_and_force_option(self,observation,action,duration,deterministic=True):
        self.observe_and_force(observation,action)
        for s in self._q_streams:s['queue']=[int(action)]*(int(duration)-1);s['pending']=len(s['queue']);s['chosen_duration']=int(duration)
        self._history_remaining=np.asarray([s['pending'] for s in self._q_streams],np.int64)


def planner_kwargs(definition):
    values=counterq_kwargs(definition);values.update(baseline_definition=definition['baseline_definition'],planner_ensemble=definition.get('planner_ensemble',5),
        planner_beta=definition.get('planner_beta',1.),planner_margin=definition.get('planner_margin',.2),planner_action_names=[a['name'] for a in definition['actions']],planner_semantics=definition.get('planner_semantics',False),planner_control=definition.get('planner_control','paired'))
    return values


def construct_planner(definition):
    import gymnasium as gym
    from experiments.league_adaptive.runtime import spaces
    policy=OptionPlannerPolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:3e-4,**planner_kwargs(definition))
    policy.planner_names={a['name']:i for i,a in enumerate(definition['actions'])}
    return policy


def initialize(sensor,baseline):
    if sensor['definition']['actions']!=baseline['definition']['actions'] or sensor['definition']['observation_config']!=baseline['definition']['observation_config']:
        raise ValueError('Retained combat policy and planner observations/actions must match exactly')
    definition=copy.deepcopy(sensor['definition']);definition.update(kind='counterfactual_option_planner',baseline_definition=copy.deepcopy(baseline['definition']),
        context_length=96,context_offsets=list(range(8))+list(range(11,40,4))+list(range(47,96,8)),planner_ensemble=5,planner_beta=1.,planner_margin=.2,
        planner_plan_names=[label for label,_ in PLANS])
    policy=construct_planner(definition);state=policy.state_dict()
    for key,value in sensor['state_dict'].items():
        if key=='context_gru.weight_ih_l0':state[key][:,:value.shape[1]]=value;state[key][:,value.shape[1]:]=0
        elif key in state and state[key].shape==value.shape:state[key]=value.clone()
    for key,value in baseline['state_dict'].items():state['incumbent.'+key]=value.clone()
    policy.load_state_dict(state,strict=True)
    return policy,definition
