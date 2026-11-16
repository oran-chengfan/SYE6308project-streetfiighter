"""Learn a state-conditioned mixture of previously learned neural actors.

All proposals come from neural policies. No distance/air-state rule chooses a
proposal. The shared router and critic learn from actual online game rewards.
Frozen neural constituents deliberately retain their separately learned skills.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from experiments.league_adaptive.policy import VisibleHistoryEncoder,mirror_indices
from experiments.league_adaptive.runtime import ROOT,spaces,compiled,load_actor,source_module


class NeuralMixturePolicy(MultiInputActorCriticPolicy):
    def __init__(self,*args,expert_definitions=None,physical_action_maps=None,inference_hold_steps=0,residual_actions=False,residual_action_mirror=None,action_logit_correction=False,correction_smoothing=.1,correction_temperature=1.,route_logit_bias=None,correction_only=False,correction_spatial_features=False,full_distribution_experts=False,trainable_experts=None,**kwargs):
        self.expert_definitions=expert_definitions;self.physical_action_maps=physical_action_maps
        self.full_distribution_experts=bool(full_distribution_experts)
        self.trainable_expert_indices=tuple(trainable_experts or ())
        if self.trainable_expert_indices and not self.full_distribution_experts:
            raise ValueError('Trainable skills require fresh full distributions, not cached greedy proposals')
        if self.full_distribution_experts and (residual_actions or action_logit_correction or inference_hold_steps):
            raise ValueError('Full skill distributions use one learned marginal without extra action correction or holding')
        self._route_bias_values=copy.deepcopy(route_logit_bias)
        self.residual_actions=bool(residual_actions);self.residual_action_mirror=residual_action_mirror
        self.action_logit_correction=bool(action_logit_correction);self.correction_smoothing=float(correction_smoothing);self.correction_temperature=float(correction_temperature)
        self.correction_only=bool(correction_only)
        self.correction_spatial_features=bool(correction_spatial_features)
        if self.correction_only and not self.action_logit_correction:raise ValueError('Correction-only learning requires the full physical-action head')
        if self.correction_spatial_features and not self.correction_only:raise ValueError('Spatial correction requires the immutable inherited actor')
        if self.action_logit_correction and (self.residual_actions or inference_hold_steps):raise ValueError('Use one action expansion mechanism')
        if not 0<=self.correction_smoothing<1 or self.correction_temperature<=0:raise ValueError('Invalid full action prior')
        self.inference_hold_steps=int(inference_hold_steps);self.reset_inference_state()
        if self.residual_actions and self.inference_hold_steps:raise ValueError('Residual actions use the trained marginal policy without inference holding')
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        from experiments.league_adaptive.policy import construct_policy
        self.experts=nn.ModuleList([construct_policy(d) for d in self.expert_definitions])
        if len(set(self.trainable_expert_indices))!=len(self.trainable_expert_indices) or any(
                i not in range(len(self.experts)) for i in self.trainable_expert_indices):
            raise ValueError('Invalid trainable skill indices')
        for i,expert in enumerate(self.experts):
            expert.optimizer=None
            if i not in self.trainable_expert_indices:
                for p in expert.parameters():p.requires_grad_(False)
            else:
                # The outer critic owns value learning. Keep the selected
                # actor's own trainable perception/body/action expression and
                # its constructor's frozen internal constituents intact.
                for module in (expert.value_net,expert.mlp_extractor.value_net):
                    for p in module.parameters():p.requires_grad_(False)
            self.register_buffer('action_map_'+str(i),torch.tensor(self.physical_action_maps[i],dtype=torch.long))
        self.action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,len(self.experts))
        if self._route_bias_values is not None:
            if len(self._route_bias_values)!=len(self.experts) or not np.isfinite(self._route_bias_values).all():
                raise ValueError('One finite route bias per expert is required')
            self.register_buffer('route_logit_bias',torch.tensor(self._route_bias_values,dtype=torch.float32))
        nn.init.zeros_(self.action_net.weight);nn.init.zeros_(self.action_net.bias)
        with torch.no_grad():self.action_net.bias[0]=3.
        if self.full_distribution_experts:
            self.register_buffer('physical_mirror',torch.tensor(self.residual_action_mirror,dtype=torch.long))
        if self.residual_actions:
            self.register_buffer('residual_physical_mirror',torch.tensor(self.residual_action_mirror,dtype=torch.long))
            self.residual_action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,self.action_space.n)
            self.residual_gate=nn.Linear(self.mlp_extractor.latent_dim_pi,1)
            nn.init.zeros_(self.residual_action_net.weight);nn.init.zeros_(self.residual_action_net.bias)
            nn.init.zeros_(self.residual_gate.weight)
            with torch.no_grad():self.residual_gate.bias.fill_(float(np.log(1/9)))
        if self.action_logit_correction:
            self.register_buffer('correction_physical_mirror',torch.tensor(self.residual_action_mirror,dtype=torch.long))
            width=self.mlp_extractor.latent_dim_pi
            if self.correction_spatial_features:
                self.features_extractor.expose_spatial_context=True
                width+=self.features_extractor.visual[1].in_features+self.features_extractor.coordinates[0].out_features
            self.correction_action_net=nn.Linear(width,self.action_space.n)
            nn.init.zeros_(self.correction_action_net.weight);nn.init.zeros_(self.correction_action_net.bias)
        if self.correction_only:
            for name,parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith(('correction_action_net.','value_net.','mlp_extractor.value_net.')))
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
    def route_logits(self,latent):
        scores=self.action_net(latent)
        return scores if self._route_bias_values is None else scores+self.route_logit_bias
    def _get_action_dist_from_latent(self,latent_pi):
        weights=self.route_logits(latent_pi).softmax(1);obs=self.features_extractor.current_observations
        if self.full_distribution_experts:
            raw=dict(image=(obs['image']*255.).round().to(torch.uint8),position=obs['position'])
            distributions=[]
            for j,expert in enumerate(self.experts):
                learning=j in self.trainable_expert_indices
                expert.set_training_mode(self.training and learning)
                with torch.set_grad_enabled(torch.is_grad_enabled() and learning):
                    source=expert.get_distribution(raw).distribution.probs
                mapping=getattr(self,'action_map_'+str(j)).expand(len(source),-1)
                physical=torch.zeros(len(source),self.action_space.n,device=source.device).scatter_add(1,mapping,source)
                distributions.append(physical)
            physical=torch.stack(distributions,1)
            self.last_route_probabilities=weights.detach()
            self.last_neural_proposals=physical.argmax(2).detach()
            # The identical expression acts, explores, learns and exports.
            # It can select actions that are neither skill's greedy proposal.
            marginal=(weights[:,:,None]*physical).sum(1)
            self.action_dist.distribution=torch.distributions.Categorical(probs=marginal)
            return self.action_dist
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
            flip=obs['position'][:,4]<0
            correction_features=latent_pi
            if self.correction_spatial_features:
                spatial=self.features_extractor.current_spatial_context
                spatial=spatial*(latent_pi.shape[1]/spatial.shape[1])**.5
                correction_features=torch.cat((latent_pi,spatial),dim=1)
            self.last_correction_context=(correction_features.detach(),probabilities.detach(),flip.detach())
            logits=self.corrected_logits(correction_features,probabilities,flip)
            return self.action_dist.proba_distribution(action_logits=logits)
        return self.action_dist.proba_distribution(action_logits=probabilities.clamp_min(1e-12).log())
    def corrected_logits(self,features,base_probabilities,flip):
        """The same physical-action expression for acting and frozen-context learning."""
        prior=(1-self.correction_smoothing)*base_probabilities+self.correction_smoothing/self.action_space.n
        # Keep the retained128-column calculation exact at zero expansion.
        width=self.mlp_extractor.latent_dim_pi
        head=self.correction_action_net
        delta=F.linear(features[:,:width],head.weight[:,:width].contiguous(),head.bias)
        if self.correction_spatial_features:
            delta=delta+F.linear(features[:,width:],head.weight[:,width:].contiguous())
        delta=torch.where(flip[:,None],delta[:,self.correction_physical_mirror],delta)
        self.last_action_logit_correction=delta.detach()
        return prior.clamp_min(1e-12).log()/self.correction_temperature+delta
    def route_distribution(self,observations):
        features=self.extract_features(observations)
        if not self.share_features_extractor:features=features[0]
        latent=self.mlp_extractor.forward_actor(features)
        return torch.distributions.Categorical(logits=self.route_logits(latent))
    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters();result.update(expert_definitions=self.expert_definitions,physical_action_maps=self.physical_action_maps,inference_hold_steps=self.inference_hold_steps,residual_actions=self.residual_actions,residual_action_mirror=self.residual_action_mirror,action_logit_correction=self.action_logit_correction,correction_smoothing=self.correction_smoothing,correction_temperature=self.correction_temperature,route_logit_bias=None if self._route_bias_values is None else self.route_logit_bias.detach().cpu().tolist(),correction_only=self.correction_only,correction_spatial_features=self.correction_spatial_features,full_distribution_experts=self.full_distribution_experts,trainable_experts=list(self.trainable_expert_indices));return result
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
    expanded = definition.get('residual_actions') or definition.get('action_logit_correction') or definition.get('full_distribution_experts')
    return dict(
        features_extractor_class=VisibleHistoryEncoder,
        net_arch=dict(pi=[128],vf=[128]),
        share_features_extractor=definition.get('share_features_extractor',True),
        expert_definitions=definition['expert_definitions'],
        physical_action_maps=definition['physical_action_maps'],
        inference_hold_steps=definition.get('inference_hold_steps',0),
        residual_actions=definition.get('residual_actions',False),
        residual_action_mirror=mirror_indices(definition['actions']) if expanded else None,
        action_logit_correction=definition.get('action_logit_correction',False),
        correction_smoothing=definition.get('correction_smoothing',.1),
        correction_temperature=definition.get('correction_temperature',1.),
        correction_only=definition.get('correction_only',False),
        correction_spatial_features=definition.get('correction_spatial_features',False),
        full_distribution_experts=definition.get('full_distribution_experts',False),
        trainable_experts=definition.get('trainable_experts',[]),
        route_logit_bias=definition.get('route_logit_bias'))


def construct_mixture(definition):
    import gymnasium as gym
    return NeuralMixturePolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,**mixture_kwargs(definition))


def physical_action_map(actions,base_actions):
    key=lambda command:tuple(tuple(sorted(frame)) for frame in command)
    physical={key(command):i for i,command in enumerate(compiled(base_actions))}
    return [physical[key(command)] for command in compiled(actions)]


def initialize_skill_branches(source):
    """Keep the complete source skill and learn another with private features.

    Both start with the source's entire physical distribution. No teacher
    weight, port label, skill identity or game metadata chooses an action.
    """
    actor=load_actor(source);base=actor.definition
    try:
        if base['kind']!='neural_mixture' or base.get('full_distribution_experts') or base.get('inference_hold_steps'):
            raise ValueError('Create one stateless complete-skill comparison from the retained mixture')
        learned=copy.deepcopy(base);learned['correction_only']=False
        definition={k:copy.deepcopy(base[k]) for k in ('actions','observation_source','observation_config')}
        identity=list(range(len(base['actions'])))
        definition.update(kind='neural_mixture',expert_definitions=[copy.deepcopy(base),learned],
            physical_action_maps=[identity,identity],expert_sources=[str(source),str(source)],
            full_distribution_experts=True,trainable_experts=[1],
            constituents='Complete immutable source path plus a source-initialized trainable visual tactical path; legal-observation selector, full physical probability marginal.')
        policy=construct_mixture(definition)
        for skill in policy.experts:skill.load_state_dict(actor.policy.state_dict(),strict=True)
        for name in ('features_extractor','mlp_extractor','value_net'):
            getattr(policy,name).load_state_dict(getattr(actor.policy,name).state_dict(),strict=True)
        nn.init.zeros_(policy.action_net.weight);nn.init.zeros_(policy.action_net.bias)
        return policy,definition
    finally:actor.close()


def extend_experts(source,paths,initial_bias=-20.):
    """Add learned responses while retaining every original inference weight.

    A new score starts at the mean retained score. Its explicit bias therefore
    bounds initial mass independently of the arbitrary shared logit offset.
    The bias is a training admission schedule, fixed throughout each rollout.
    """
    original=load_actor(source);definition=copy.deepcopy(original.definition)
    if (definition['kind']!='neural_mixture' or definition.get('retained_expert_count') or definition.get('route_logit_bias') is not None
            or any(definition.get(k) for k in ('residual_actions','action_logit_correction','inference_hold_steps'))):
        raise ValueError('Extend the retained plain mixture once, without another expansion mechanism')
    if not 1<=len(paths)<=2:raise ValueError('Use one or two validated complementary response experts')
    count=len(definition['expert_definitions']);actors=[load_actor(path) for path in paths]
    reference_observation=source_module(definition['observation_source'])
    for actor in actors:
        if actor.definition['kind']!='response' or actor.definition['observation_config']!=definition['observation_config']:
            raise ValueError('New response experts must share the legal input configuration')
        candidate_observation=source_module(actor.definition['observation_source'])
        for path in (ROOT/'assets/starts').glob('*.npz'):
            with np.load(path,allow_pickle=False) as data:rgb=data['rgb']
            if not np.array_equal(reference_observation.preprocess(rgb),candidate_observation.preprocess(rgb)):
                raise ValueError('Response preprocessing differs from the retained policy')
        definition['expert_definitions'].append(copy.deepcopy(actor.definition))
        definition['physical_action_maps'].append(physical_action_map(actor.actions,definition['actions']))
    definition['expert_sources']=list(definition['expert_sources'])+list(paths)
    definition.update(retained_expert_count=count,route_logit_bias=[0.]*count+[float(initial_bias)]*len(paths))
    policy=construct_mixture(definition);state=policy.state_dict();old=original.policy.state_dict()
    for key,value in old.items():
        if key not in ('action_net.weight','action_net.bias'):state[key]=value.clone()
    for key in ('action_net.weight','action_net.bias'):
        state[key][:count]=old[key]
        state[key][count:]=old[key].mean(dim=0)
    policy.load_state_dict(state,strict=True)
    for index,actor in enumerate(actors,count):
        policy.experts[index].load_state_dict(actor.policy.state_dict(),strict=True);actor.close()
    original.close()
    return policy,definition


def initialize(paths):
    actors=[load_actor(path) for path in paths];base=actors[0].definition
    if any(x.definition['observation_config']!=base['observation_config'] for x in actors):raise ValueError('Neural actors must share the legal observation preprocessing')
    maps=[]
    for actor in actors:
        if actor.definition.get('reference_definition') is not None:raise ValueError('Use neural constituents without frozen composite reference routing')
        maps.append(physical_action_map(actor.actions,base['actions']))
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
