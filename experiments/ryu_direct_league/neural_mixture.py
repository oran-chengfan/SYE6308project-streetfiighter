"""Learn a state-conditioned mixture of previously learned neural actors.

All proposals come from neural policies. No distance/air-state rule chooses a
proposal. The shared router and critic learn from actual online game rewards.
Frozen neural constituents deliberately retain their separately learned skills.
"""
import copy
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from experiments.league_adaptive.policy import VisibleHistoryEncoder,mirror_indices
from experiments.league_adaptive.runtime import spaces,compiled,load_actor


class NeuralMixturePolicy(MultiInputActorCriticPolicy):
    def __init__(self,*args,expert_definitions=None,physical_action_maps=None,inference_hold_steps=0,residual_actions=False,residual_action_mirror=None,action_logit_correction=False,correction_smoothing=.1,correction_temperature=1.,**kwargs):
        self.expert_definitions=expert_definitions;self.physical_action_maps=physical_action_maps
        self.residual_actions=bool(residual_actions);self.residual_action_mirror=residual_action_mirror
        self.action_logit_correction=bool(action_logit_correction);self.correction_smoothing=float(correction_smoothing);self.correction_temperature=float(correction_temperature)
        if self.action_logit_correction and (self.residual_actions or inference_hold_steps):raise ValueError('Use one action expansion mechanism')
        if not 0<self.correction_smoothing<1 or self.correction_temperature<=0:raise ValueError('Invalid full action prior')
        self.inference_hold_steps=int(inference_hold_steps);self.reset_inference_state()
        if self.residual_actions and self.inference_hold_steps:raise ValueError('Residual actions use the trained marginal policy without inference holding')
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        from experiments.league_adaptive.policy import construct_policy
        self.experts=nn.ModuleList([construct_policy(d) for d in self.expert_definitions])
        for i,expert in enumerate(self.experts):
            expert.optimizer=None
            for p in expert.parameters():p.requires_grad_(False)
            self.register_buffer('action_map_'+str(i),torch.tensor(self.physical_action_maps[i],dtype=torch.long))
        self.action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,len(self.experts))
        nn.init.zeros_(self.action_net.weight);nn.init.zeros_(self.action_net.bias)
        with torch.no_grad():self.action_net.bias[0]=3.
        if self.residual_actions:
            self.register_buffer('residual_physical_mirror',torch.tensor(self.residual_action_mirror,dtype=torch.long))
            self.residual_action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,self.action_space.n)
            self.residual_gate=nn.Linear(self.mlp_extractor.latent_dim_pi,1)
            nn.init.zeros_(self.residual_action_net.weight);nn.init.zeros_(self.residual_action_net.bias)
            nn.init.zeros_(self.residual_gate.weight)
            with torch.no_grad():self.residual_gate.bias.fill_(float(np.log(1/9)))
        if self.action_logit_correction:
            self.register_buffer('correction_physical_mirror',torch.tensor(self.residual_action_mirror,dtype=torch.long))
            self.correction_action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,self.action_space.n)
            nn.init.zeros_(self.correction_action_net.weight);nn.init.zeros_(self.correction_action_net.bias)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
        self._proposal_cache={}
    def proposals(self,obs):
        image=(obs['image'].detach()*255.).round().to(torch.uint8);position=obs['position'].detach()
        images=image.cpu().numpy();positions=position.cpu().numpy();keys=[a.tobytes()+b.tobytes() for a,b in zip(images,positions)]
        missing=[i for i,k in enumerate(keys) if k not in self._proposal_cache]
        if missing:
            ix=torch.as_tensor(missing,device=self.device);raw=dict(image=image[ix],position=position[ix]);values=[]
            with torch.no_grad():
                for j,expert in enumerate(self.experts):
                    expert.set_training_mode(False);a=expert.get_distribution(raw).distribution.probs.argmax(1)
                    values.append(getattr(self,'action_map_'+str(j))[a])
            values=torch.stack(values,1).cpu()
            for i,value in zip(missing,values):self._proposal_cache[keys[i]]=value.clone()
        result=torch.stack([self._proposal_cache[k] for k in keys]).to(self.device)
        limit=4096 if self.training else 512
        while len(self._proposal_cache)>limit:self._proposal_cache.pop(next(iter(self._proposal_cache)))
        return result
    def _get_action_dist_from_latent(self,latent_pi):
        weights=self.action_net(latent_pi).softmax(1);obs=self.features_extractor.current_observations
        proposals=self.proposals(obs);probabilities=torch.zeros(len(weights),self.action_space.n,device=weights.device)
        self.last_route_probabilities=weights.detach();self.last_neural_proposals=proposals.detach()
        probabilities=probabilities.scatter_add(1,proposals,weights)
        if self.residual_actions:
            logits=self.residual_action_net(latent_pi);flip=obs['position'][:,4]<0
            logits=torch.where(flip[:,None],logits[:,self.residual_physical_mirror],logits)
            residual=logits.softmax(1);gate=self.residual_gate(latent_pi).sigmoid()
            probabilities=(1-gate)*probabilities+gate*residual
            self.last_residual_probability=gate.detach()
        if self.action_logit_correction:
            # Every action has positive initial exploration mass. Unbounded
            # logit changes allow any action to become the greedy choice.
            prior=(1-self.correction_smoothing)*probabilities+self.correction_smoothing/self.action_space.n
            delta=self.correction_action_net(latent_pi);flip=obs['position'][:,4]<0
            delta=torch.where(flip[:,None],delta[:,self.correction_physical_mirror],delta)
            logits=prior.log()/self.correction_temperature+delta
            self.last_action_logit_correction=delta.detach()
            return self.action_dist.proba_distribution(action_logits=logits)
        return self.action_dist.proba_distribution(action_logits=probabilities.clamp_min(1e-12).log())
    def route_distribution(self,observations):
        features=self.extract_features(observations);latent=self.mlp_extractor.forward_actor(features)
        return torch.distributions.Categorical(logits=self.action_net(latent))
    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters();result.update(expert_definitions=self.expert_definitions,physical_action_maps=self.physical_action_maps,inference_hold_steps=self.inference_hold_steps,residual_actions=self.residual_actions,residual_action_mirror=self.residual_action_mirror,action_logit_correction=self.action_logit_correction,correction_smoothing=self.correction_smoothing,correction_temperature=self.correction_temperature);return result
    def reset_inference_state(self):self._held_neural_route=None;self._route_remaining=None
    def get_inference_state(self):
        return None if self._held_neural_route is None else dict(held=self._held_neural_route.copy(),remaining=self._route_remaining.copy())
    def set_inference_state(self,state):
        if state is None:self.reset_inference_state()
        else:self._held_neural_route=np.asarray(state['held']).copy();self._route_remaining=np.asarray(state['remaining']).copy()
    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        if not self.inference_hold_steps:return super().predict(observation,state=state,episode_start=episode_start,deterministic=deterministic)
        self.set_training_mode(False);tensor,vectorized=self.obs_to_tensor(observation);batch=len(tensor['image'])
        if self._held_neural_route is None or len(self._held_neural_route)!=batch:
            self._held_neural_route=np.zeros(batch,np.int64);self._route_remaining=np.zeros(batch,np.int64)
        image=tensor['image']
        if image.shape[1]!=4:raise ValueError('Observation-derived reset requires the declared four grayscale histories')
        # Side.reset duplicates its current legal image four times. Thus every
        # new match reconsiders its route even when the Arena retains the model
        # object. Visually static histories also permit harmless replanning.
        duplicate=(image[:,1:]==image[:,:1]).flatten(1).all(1).cpu().numpy();pending=(self._route_remaining<=0)|duplicate
        if episode_start is not None:pending|=np.asarray(episode_start,dtype=bool).reshape(-1)
        with torch.no_grad():
            route=self.route_distribution(tensor);proposals=self.proposals(self.features_extractor.current_observations)
            choices=(route.probs.argmax(1) if deterministic else route.sample()).cpu().numpy()
            self._held_neural_route[pending]=choices[pending];self._route_remaining[pending]=self.inference_hold_steps
            held=torch.as_tensor(self._held_neural_route,device=self.device);action=proposals[torch.arange(batch,device=self.device),held].cpu().numpy()
            self.last_route_probabilities=route.probs.detach();self.last_neural_proposals=proposals.detach();self.last_committed_neural_route=self._held_neural_route.copy()
        self._route_remaining-=1
        return (action if vectorized else action[0]),None


def mixture_kwargs(definition):
    return dict(features_extractor_class=VisibleHistoryEncoder,net_arch=dict(pi=[128],vf=[128]),expert_definitions=definition['expert_definitions'],physical_action_maps=definition['physical_action_maps'],inference_hold_steps=definition.get('inference_hold_steps',0),residual_actions=definition.get('residual_actions',False),residual_action_mirror=mirror_indices(definition['actions']) if definition.get('residual_actions') or definition.get('action_logit_correction') else None,action_logit_correction=definition.get('action_logit_correction',False),correction_smoothing=definition.get('correction_smoothing',.1),correction_temperature=definition.get('correction_temperature',1.))


def construct_mixture(definition):
    import gymnasium as gym
    return NeuralMixturePolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,**mixture_kwargs(definition))


def initialize(paths):
    actors=[load_actor(path) for path in paths];base=actors[0].definition
    if any(x.definition['observation_config']!=base['observation_config'] for x in actors):raise ValueError('Neural actors must share the legal observation preprocessing')
    physical={tuple(tuple(frame) for frame in c):i for i,c in enumerate(compiled(base['actions']))};maps=[]
    for actor in actors:
        if actor.definition.get('reference_definition') is not None:raise ValueError('Use neural constituents without frozen composite reference routing')
        maps.append([physical[tuple(tuple(frame) for frame in c)] for c in compiled(actor.actions)])
    definition={k:copy.deepcopy(base[k]) for k in ('actions','observation_source','observation_config')}
    definition.update(kind='neural_mixture',expert_definitions=[copy.deepcopy(x.definition) for x in actors],physical_action_maps=maps,expert_sources=list(paths),constituents='Previously learned neural actors; retained weights. Router and critic train online. No hand-authored combat proposals.')
    policy=construct_mixture(definition)
    for target,actor in zip(policy.experts,actors):target.load_state_dict(actor.policy.state_dict(),strict=True);actor.close()
    for actor in actors:
        if type(actor.policy.features_extractor) is VisibleHistoryEncoder:
            policy.features_extractor.load_state_dict(actor.policy.features_extractor.state_dict(),strict=True)
            policy.mlp_extractor.load_state_dict(actor.policy.mlp_extractor.state_dict(),strict=True)
            break
    nn.init.zeros_(policy.value_net.weight);nn.init.zeros_(policy.value_net.bias)
    return policy,definition


if __name__=='__main__':
    import argparse
    from experiments.league_adaptive.runtime import setup
    from experiments.ryu_direct_league.policy import save
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();setup();policy,definition=initialize(a.actors);save(a.out,policy,definition,dict(initialization='Initial greedy physical action equals the first learned actor: its mixture mass exceeds all remaining masses combined. Frozen constituents are explicit.'))
