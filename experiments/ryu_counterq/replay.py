"""RAM-only transition graph; alternative histories share their actual prefix."""
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


class Replay:
    def __init__(self,nodes=150000,transitions=32768,width=3584,context=16,seed=2231,offsets=None):
        self.capacity=int(nodes);self.transition_capacity=int(transitions);self.context=int(context);self.rng=np.random.default_rng(seed)
        self.offsets=None if offsets is None else tuple(int(v) for v in offsets)
        self.maps=np.empty((nodes,width),np.float16);self.positions=np.empty((nodes,6),np.float32);self.extras=np.empty((nodes,12),np.float32)
        self.previous=np.empty(nodes,np.int16);self.parent=np.full(nodes,-1,np.int64);self.node_stamp=np.full(nodes,-1,np.int64);self.node_count=0
        self.starts=np.empty(transitions,np.int64);self.ends=np.empty(transitions,np.int64);self.actions=np.empty(transitions,np.int64)
        self.rewards=np.empty(transitions,np.float32);self.dones=np.empty(transitions,bool);self.durations=np.empty(transitions,np.int16)
        self.transition_stamp=np.full(transitions,-1,np.int64);self.transition_count=0;self.groups=defaultdict(lambda:deque(maxlen=4096))

    def add_node(self,parent,feature_map,position,extras,previous):
        number=self.node_count;index=number%self.capacity;self.node_count+=1
        self.maps[index]=feature_map;self.positions[index]=position;self.extras[index]=extras;self.previous[index]=previous
        self.parent[index]=parent;self.node_stamp[index]=number
        return number

    def add_transition(self,start,action,reward,end,done,duration,group):
        number=self.transition_count;index=number%self.transition_capacity;self.transition_count+=1
        self.starts[index]=start;self.ends[index]=end;self.actions[index]=action;self.rewards[index]=reward;self.dones[index]=done;self.durations[index]=duration
        self.transition_stamp[index]=number;self.groups[group].append(number)

    def history(self,number):
        values=[]
        while number>=0 and len(values)<self.context:
            index=number%self.capacity
            if self.node_stamp[index]!=number:return None
            values.append(number);number=int(self.parent[index])
        return values[::-1]

    def contexts(self,numbers,device='cpu'):
        histories=[self.history(int(n)) for n in numbers]
        if any(not h for h in histories):raise ValueError('A requested context is no longer in RAM replay')
        if self.offsets is not None:
            histories=[[h[len(h)-1-age] for age in sorted(self.offsets,reverse=True) if age<len(h)] for h in histories]
        lengths=np.asarray([len(h) for h in histories],np.int64);length=int(lengths.max());batch=len(histories)
        ids=np.zeros((batch,length),np.int64);mask=np.arange(length)[None]<lengths[:,None]
        for i,h in enumerate(histories):ids[i,:len(h)]=h
        maps=np.zeros((batch,length,self.maps.shape[1]),np.float16);positions=np.zeros((batch,length,6),np.float32)
        extras=np.zeros((batch,length,12),np.float32);previous=np.zeros((batch,length),np.int64)
        ix=ids[mask]%self.capacity
        maps[mask]=self.maps[ix];positions[mask]=self.positions[ix];extras[mask]=self.extras[ix];previous[mask]=self.previous[ix]
        return tuple(torch.as_tensor(x,device=device) for x in (maps,positions,extras,previous))+(torch.as_tensor(lengths),)

    def sample(self,batch=64):
        n=min(self.transition_count,self.transition_capacity)
        if n<batch:return None
        chosen=[];groups=list(self.groups);attempts=0
        while len(chosen)<batch and attempts<batch*40:
            attempts+=1
            if self.rng.random()<.5:
                group=groups[int(self.rng.integers(len(groups)))];values=self.groups[group]
                number=values[int(self.rng.integers(len(values)))];i=number%self.transition_capacity
                if self.transition_stamp[i]!=number:continue
            else:i=int(self.rng.integers(n))
            if self.history(int(self.starts[i])) is None or self.history(int(self.ends[i])) is None:continue
            chosen.append(i)
        if len(chosen)<batch:return None
        i=np.asarray(chosen)
        return dict(observation=self.contexts(self.starts[i]),next_observation=self.contexts(self.ends[i]),
                    action=torch.as_tensor(self.actions[i]),reward=torch.as_tensor(self.rewards[i]),done=torch.as_tensor(self.dones[i]),
                    duration=torch.as_tensor(self.durations[i],dtype=torch.float32))


class Demonstrations(Replay):
    """Permanent, group-balanced real trajectories, separate from the ring."""
    def __init__(self,path,context=16,seed=2231,objective='graded',offsets=None,prefix_priority=False):
        with np.load(path,allow_pickle=False) as source:
            nodes=len(source['maps']);transitions=len(source['transition_start'])
            super().__init__(nodes=nodes,transitions=transitions,width=source['maps'].shape[1],context=context,seed=seed,offsets=offsets)
            for key in ('maps','positions','extras','previous','parent'):setattr(self,key,source[key].copy())
            self.node_stamp=np.arange(nodes);self.node_count=nodes
            for dest,key in [('starts','start'),('ends','end'),('actions','action'),('rewards','reward'),('dones','done'),('durations','duration')]:
                setattr(self,dest,source['transition_'+key].copy())
            self.returns=source['transition_returns'].copy();self.wins=source['transition_win'].copy()
            self.labels=source['transition_group'].copy();self.transition_count=transitions
            self.transition_stamp=np.arange(transitions)
            self.demo_groups={str(g):np.flatnonzero(self.labels==g) for g in np.unique(self.labels)}
        self.prefix=np.zeros(transitions,bool);self.imitation=self.wins.copy();self.priority_groups=None
        self.supervision_group=np.zeros(transitions,np.int64)
        annotation=Path(path).parent/'prefix_annotations.npz'
        if annotation.is_file():
            with np.load(annotation,allow_pickle=False) as data:
                if not np.array_equal(data['start'],self.starts) or not np.array_equal(data['end'],self.ends):raise ValueError('Recovery prefix annotations and native transition IDs differ')
                self.prefix=data['intervention'].copy()
                if prefix_priority:
                    self.imitation=data['beneficial'].copy()
                    self.supervision_group[:]=1
                    self.priority_groups={g:ids[self.prefix[ids]] for g,ids in self.demo_groups.items() if self.prefix[ids].any()}
        if objective=='win':
            # Original native reward is terminal(+1/-1/0)+0.25*netHPdelta.
            # Recover its exact potential from actual parent/child edges.
            terminals=np.where(self.dones,np.where(self.rewards>.5,1.,np.where(self.rewards<-.5,-1.,0.)),0.)
            increments=np.full(nodes,np.nan,np.float64)
            for end,change in zip(self.ends,self.rewards-terminals):
                if np.isfinite(increments[end]) and not np.isclose(increments[end],change):raise ValueError('Conflicting native HP changes on a shared edge')
                increments[end]=change
            potential=np.zeros(nodes,np.float64)
            for node,parent in enumerate(self.parent):
                if parent<0:continue
                if parent>=node or not np.isfinite(increments[node]):raise ValueError('Incomplete native potential path')
                potential[node]=potential[parent]+increments[node]
            terminal=np.where(self.dones,np.where(self.wins,1.,-1.),0.)
            self.rewards=(terminal+np.where(self.dones,0.,potential[self.ends])-potential[self.starts]).astype(np.float32)
            self.returns=(np.where(self.wins,1.,-1.)-potential[self.starts]).astype(np.float32)
        # Collectors can append an episode forwards or backwards. Follow only
        # contiguous rows whose native end/start IDs match, never another fork.
        self.nstep_ends=self.ends.copy();self.nstep_rewards=self.rewards.astype(np.float32).copy()
        self.nstep_dones=self.dones.copy();self.nstep_durations=self.durations.astype(np.float32).copy()
        for i in range(transitions):
            cursor=i
            while self.nstep_durations[i]<32 and not self.nstep_dones[i]:
                successors=[j for j in (cursor+1,cursor-1) if 0<=j<transitions and self.starts[j]==self.ends[cursor]]
                if not successors:break
                if len(successors)!=1:raise ValueError('Ambiguous contiguous demonstration continuation')
                cursor=successors[0];self.nstep_ends[i]=self.ends[cursor];self.nstep_rewards[i]+=self.rewards[cursor]
                self.nstep_dones[i]=self.dones[cursor];self.nstep_durations[i]+=self.durations[cursor]

    def sample(self,batch=64):
        groups=list(self.demo_groups)
        chosen=[]
        for row in range(batch):
            grouped=self.priority_groups if self.priority_groups is not None and row<batch//2 else self.demo_groups
            names=list(grouped);chosen.append(int(self.rng.choice(grouped[names[int(self.rng.integers(len(names)))]])))
        i=np.asarray(chosen)
        return dict(observation=self.contexts(self.starts[i]),next_observation=self.contexts(self.ends[i]),
                    action=torch.as_tensor(self.actions[i]),reward=torch.as_tensor(self.rewards[i],dtype=torch.float32),
                    done=torch.as_tensor(self.dones[i]),duration=torch.as_tensor(self.durations[i],dtype=torch.float32),
                    returns=torch.as_tensor(self.returns[i],dtype=torch.float32),win=torch.as_tensor(self.wins[i]),
                    imitation=torch.as_tensor(self.imitation[i]),prefix=torch.as_tensor(self.prefix[i]),
                    supervision_group=torch.as_tensor(self.supervision_group[i]),
                    nstep_next_observation=self.contexts(self.nstep_ends[i]),nstep_reward=torch.as_tensor(self.nstep_rewards[i]),
                    nstep_done=torch.as_tensor(self.nstep_dones[i]),nstep_duration=torch.as_tensor(self.nstep_durations[i]))


def concatenate_contexts(parts):
    max_length=max(int(part[0].shape[1]) for part in parts);joined=[]
    for component in range(4):
        values=[]
        for part in parts:
            value=part[component];missing=max_length-value.shape[1]
            values.append(F.pad(value,(0,missing)) if component==3 else F.pad(value,(0,0,0,missing)))
        joined.append(torch.cat(values,0))
    return tuple(joined)+(torch.cat([part[4] for part in parts]),)


class DemonstrationMix:
    """Equal source mass, group balance within each native source."""
    def __init__(self,paths,context=16,seed=2231,objective='graded',offsets=None,prefix_priority=False):
        self.sources=[Demonstrations(path,context,seed+53*i,objective,offsets,prefix_priority) for i,path in enumerate(paths)]
        self.transition_count=sum(source.transition_count for source in self.sources)

    def sample(self,batch=64):
        counts=[batch//len(self.sources)+(i<batch%len(self.sources)) for i in range(len(self.sources))]
        parts=[source.sample(count) for source,count in zip(self.sources,counts) if count]
        return {key:concatenate_contexts([part[key] for part in parts]) if key in ('observation','next_observation','nstep_next_observation') else torch.cat([part[key] for part in parts]) for key in parts[0]}


def demonstration_objective(q,data,margin=.1,return_anchor=True):
    """Retain successful actions and achievable returns; defeats remain TD data.

    Observed teacher returns are a lower-bound training anchor, not an oracle
    or a target that says a losing teacher action is the best available action.
    """
    chosen=q.gather(1,data['action'][:,None]).flatten();success=data.get('imitation',data['win'])
    if not bool(success.any()):return q.sum()*0.,dict(margin=0.,return_lower_bound=0.,accuracy=None)
    costs=torch.full_like(q,float(margin));costs.scatter_(1,data['action'][:,None],0.)
    group=data.get('supervision_group',torch.zeros_like(data['action']))
    per_example=(q+costs).max(1).values-chosen
    masks=[success&(group==g) for g in torch.unique(group[success])]
    # Equal teaching-source mass prevents hundreds of continuation attacks
    # from washing out the rare whole-prefix interventions that changed a loss.
    imitation=torch.stack([per_example[mask].mean() for mask in masks]).mean()
    lower_bound=torch.stack([F.smooth_l1_loss(chosen[mask],torch.maximum(chosen[mask].detach(),data['returns'][mask])) for mask in masks]).mean() if return_anchor else q.sum()*0.
    return imitation+lower_bound,dict(margin=float(imitation.detach()),return_lower_bound=float(lower_bound.detach()),
                                    accuracy=float((q.argmax(1)[success]==data['action'][success]).float().mean()),
                                    prefix_samples=int(data['prefix'].sum()) if 'prefix' in data else 0,imitation_samples=int(success.sum()))


def double_q_target(online_next,target_next,rewards,dones,durations,gamma):
    selected=online_next.argmax(1)
    continuation=target_next.gather(1,selected[:,None]).flatten()
    return rewards+(~dones).float()*(float(gamma)**durations)*continuation
