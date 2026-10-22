"""Current-policy complete-option counterfactuals with matching own memory.

Candidate interventions change an entire constant physical input and its
duration. Stochastic continuations share random streams on the six-frame clock;
greedy continuations are separately labelled. No global oracle is assumed.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.league_adaptive.runtime import Actor,load_actor,setup,write
from experiments.ryu_adaptive_curriculum.recurrent_direct import construct_recurrent
from experiments.ryu_direct_league.environment import RyuEnv


def result(info):
    return {k:info[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')}


def utility(info):
    signed=int(info['result']=='win')-int(info['result']=='loss')
    return signed+.25*(max(0,info['self_hp'])-max(0,info['opponent_hp']))/176.


class OptionValues:
    def __init__(self,definition,pool,out,seed=2231):
        if len(definition.get('duration_choices',(1,)))<=1: raise ValueError('Counterfactual options require explicit learned durations')
        self.definition=copy.deepcopy(definition);self.pool=copy.deepcopy(pool);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.seed=int(seed);self.index=0;self.additional_native_frames=0
        with torch.random.fork_rng(devices=[]):
            policy=construct_recurrent(definition)
        for p in policy.parameters():p.requires_grad_(False)
        policy.optimizer=None;self.actor=Actor(policy,definition)
        self.env=RyuEnv(definition,pool,0,gamma=1.)
        self.ids={a['name']:i for i,a in enumerate(definition['actions'])}

    def restore(self,root):
        observation=self.env.restore(root['saved'])
        self.actor.set_state(copy.deepcopy(root['actor_state']))
        return observation

    def finish(self,observation,deterministic):
        while True:
            observation,_,t,tr,info=self.env.step(int(self.actor.predict(observation,deterministic)[0]))
            if t or tr:return info

    def factual(self,opponent,player,offset):
        self.actor.reset();observation,_=self.env.reset(options=dict(opponent=opponent,player=player,offset=offset))
        roots=[];seen=0;reservoir=np.random.default_rng(self.seed+self.index*337)
        for step in range(1300):
            free=self.actor.policy._history_remaining is None or int(self.actor.policy._history_remaining[0])==0
            candidate=None
            if free and 12<=step<=160:
                seen+=1
                selected=len(roots)<4 or int(reservoir.integers(seen))<4
                if selected:
                    candidate=dict(saved=self.env.save(),actor_state=copy.deepcopy(self.actor.get_state()),
                                   observation={k:v.copy() for k,v in observation.items()},decision=step,
                                   random_state=torch.get_rng_state().clone(),opponent=copy.deepcopy(self.pool[opponent]))
            action=int(self.actor.predict(observation,False)[0]);duration=int(self.actor.policy.last_duration_choices[0])
            if candidate is not None:
                candidate.update(factual_action=action,factual_duration=duration)
                if len(roots)<4:roots.append(candidate)
                else:roots[int(reservoir.integers(4))]=candidate
            observation,_,t,tr,info=self.env.step(action)
            if t or tr:
                for root in roots:root['factual_info']=result(info)
                return roots,info
        raise RuntimeError('Factual game did not terminate')

    def pairs(self,root):
        p=root['saved']['player'];raw=root['saved']['game']['readings'];dx=raw[f'p{3-p}_x']-raw[f'p{p}_x']
        face='R' if dx>=0 else 'L';back='LEFT' if dx>=0 else 'RIGHT'
        pairs=[(root['factual_action'],root['factual_duration'])]
        obs=self.restore(root);a=int(self.actor.predict(obs,True)[0]);d=int(self.actor.policy.last_duration_choices[0]);pairs.append((a,d))
        for name in ('LEFT','RIGHT','DOWN_'+back):
            pairs.extend((self.ids[name],d) for d in (1,4,8))
        for name in (f'SHORYUKEN_{face}_X',f'HADOUKEN_{face}_Y','CROUCH_KICK_C','C_tap'):
            pairs.extend((self.ids[name],d) for d in (1,8))
        return list(dict.fromkeys(pairs))

    def branch(self,root,action,duration,mode,replicate=0):
        observation=self.restore(root)
        if replicate==0:torch.set_rng_state(root['random_state'])
        else:torch.manual_seed(self.seed+self.index*991+replicate*1709)
        self.actor.policy.observe_and_force_option(observation,action,duration,deterministic=mode=='greedy')
        observation,_,t,tr,info=self.env.step(action)
        if not (t or tr):info=self.finish(observation,mode=='greedy')
        return result(info)

    def query(self,policy,steps,*,replicates=2,roots_per_query=1):
        self.actor.policy.load_state_dict(policy.state_dict(),strict=True);self.actor.policy.set_training_mode(False)
        before=self.env.native_frames;started=time.monotonic();queries=[]
        with torch.random.fork_rng(devices=[]):
            for _ in range(roots_per_query):
                condition=self.index%(len(self.pool)*4);rival=condition//4;port=1+(condition%4)//2;offset=12*(condition%2)
                if self.pool[rival]['kind']=='cpu':port,offset=1,0
                torch.manual_seed(self.seed+self.index*433)
                roots,factual_info=self.factual(rival,port,offset)
                if not roots:
                    self.index+=1;continue
                root=roots[self.index%len(roots)];pairs=self.pairs(root);rows=[]
                for action,duration in pairs:
                    stochastic=[self.branch(root,action,duration,'stochastic',r) for r in range(replicates)]
                    greedy=self.branch(root,action,duration,'greedy')
                    rows.append(dict(action=action,name=self.definition['actions'][action]['name'],duration=duration,
                                     stochastic=stochastic,greedy=greedy,
                                     stochastic_utility=float(np.mean([utility(r) for r in stochastic])),greedy_utility=utility(greedy)))
                if rows[0]['stochastic'][0]!=root['factual_info']:raise RuntimeError('Whole-option factual baseline did not reproduce its full native game')
                # A separate unforced replay checks both the root RNG and the
                # ordinary policy call, independently of the intervention API.
                observation=self.restore(root);torch.set_rng_state(root['random_state'])
                replay=self.finish(observation,False)
                if result(replay)!=root['factual_info']:raise RuntimeError('Unforced factual replay changed')
                directory=self.out/f'query_{self.index:04d}'
                directory.mkdir();torch.save(root,directory/'root.pt')
                report=dict(steps=steps,opponent=self.pool[rival],player=port,offset=offset,root_decision=root['decision'],
                            factual=result(factual_info),pairs=rows,
                            scope='Constant physical input for the declared number of six-frame calls; actual opponent acts throughout. Identical own/controller histories and matched random streams. Greedy and stochastic continuations remain distinct conditional outcomes.')
                write(directory/'values.json',report)
                queries.append(dict(root=root,rows=rows))
                print('OPTION_VALUES',steps,self.pool[rival]['id'],port,offset,root['decision'],
                      [(r['name'],r['duration'],round(r['stochastic_utility'],3),round(r['greedy_utility'],3)) for r in rows],flush=True)
                self.index+=1
        self.additional_native_frames+=self.env.native_frames-before
        write(self.out/'live.json',dict(steps=steps,queries=self.index,additional_native_frames=self.additional_native_frames,
                                       seconds=time.monotonic()-started,roots=len(queries)))
        return queries

    def close(self):self.env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--agent',required=True);parser.add_argument('--pool',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--queries',type=int,default=2)
    parser.add_argument('--replicates',type=int,default=2);args=parser.parse_args();setup()
    actor=load_actor(args.agent);pool=json.loads(Path(args.pool).read_text())
    query=OptionValues(actor.definition,pool,args.out)
    try:
        rows=[]
        for _ in range(args.queries):
            values=query.query(actor.policy,0,replicates=args.replicates)
            for value in values:
                rows.append(dict(factual=value['root']['factual_info'],pairs=value['rows']))
        write(Path(args.out)/'summary.json',dict(agent=args.agent,queries=rows,additional_native_frames=query.additional_native_frames))
    finally:query.close()


if __name__=='__main__':main()
