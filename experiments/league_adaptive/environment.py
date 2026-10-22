"""Two controller ports, a frozen opponent pool and faithful native CPU rounds."""
from __future__ import annotations
import copy
from collections import Counter,deque
import gymnasium as gym
import numpy as np
from experiments.league_adaptive.runtime import ROOT,NativeGame,Side,compiled,load_actor,spaces

from experiments.league_adaptive.tactics import choose as tactic


def family(name):
    if name.startswith('SHORYUKEN'):return 'uppercut_input'
    if name.startswith('HADOUKEN'):return 'projectile_input'
    if name.startswith('TATSUMAKI'):return 'spinning_kick_input'
    if name.startswith('CROUCH_KICK'):return 'crouching_kick_input'
    if name in ('LEFT','RIGHT'):return 'walking'
    if name.startswith('UP'):return 'jump_input'
    if name.startswith('DOWN'):return 'crouch_or_guard_input'
    if name=='NO_OP':return 'neutral'
    return 'normal_button_input'


def scheduled_style(config,decision_number):
    """Observer-only label for a controlled switching opponent's next input."""
    if config.get('kind')=='tactical' and config.get('style')=='switching':
        return config['styles'][((int(decision_number)-1)//int(config['switch_every']))%len(config['styles'])]
    return None


class Controller:
    def __init__(self,config,definition):
        self.config=dict(config);self.count=0;self.actor=None
        if config['kind']=='agent':self.actor=load_actor(config['path']);self.definition=self.actor.definition
        else:self.definition=definition
        self.actions={a['name']:i for i,a in enumerate(self.definition['actions'])};self.commands=compiled(self.definition['actions'])
    def reset(self):
        self.count=0
        if self.actor is not None:self.actor.reset()
    def choose(self,side,raw):
        self.count+=1
        if self.actor is not None:return int(self.actor.predict(side.observe(raw),True)[0])
        c=self.config;style=c['style'];p=side.player;dx=raw[f'p{3-p}_x']-raw[f'p{p}_x'];distance=abs(dx)
        face='R' if dx>=0 else 'L';forward,back=('RIGHT','LEFT') if dx>=0 else ('LEFT','RIGHT')
        phase=(self.count+c.get('phase',0))%c.get('period',16)
        if c['kind']=='tactical':return tactic(style,raw,p,self.count,self.actions,c)
        if style=='fixed_right':name='SHORYUKEN_R_Z'
        elif style=='fixed_left':name='SHORYUKEN_L_Z'
        elif style=='dp':name=f'SHORYUKEN_{face}_Z'
        elif style=='fireball':
            name=f'HADOUKEN_{face}_Y' if distance>70 else f'SHORYUKEN_{face}_X'
            if phase in (0,1):name='NO_OP'
        elif style=='rush':
            name=forward if distance>42 else 'Z_tap' if phase%2 else 'C_tap'
            if distance>100 and phase in (0,1,2):name='UP_'+forward
        elif style=='mixed':
            name=f'HADOUKEN_{face}_Z' if distance>100 and phase<8 else forward if distance>100 else f'SHORYUKEN_{face}_Y' if phase<6 else f'TATSUMAKI_{face}_B' if phase<11 else 'DOWN_'+back
        elif style=='defensive':name=f'SHORYUKEN_{face}_Z' if distance<55 and phase<7 else 'DOWN_'+back
        else:raise ValueError(style)
        return self.actions[name]


class FightEnv(gym.Env):
    metadata={}
    def __init__(self,definition,pool,rank=0,balanced=False,capture=False,reward='win_probability'):
        self.definition=definition;self.pool=copy.deepcopy(pool);self.rank=rank;self.balanced=balanced;self.capture=capture
        self.reward_kind=reward;self.controllers=[Controller(c,definition) for c in pool]
        self.game=NativeGame();self.commands=compiled(definition['actions'])
        self.observation_space=spaces(definition);self.action_space=gym.spaces.Discrete(len(self.commands))
        self.episode_index=0;self.native_frames=0;self.ended=True;self.sides={};self.pool_scores={}
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);o=options or {}
        weights=np.asarray([c.get('weight',1.) for c in self.pool],float);weights/=weights.sum()
        index=int(o.get('opponent',self.np_random.choice(len(self.pool),p=weights)))
        self.current=self.pool[index];self.controller=self.controllers[index];self.controller.reset()
        self.player=int(o.get('player',1+self.rank%2 if self.balanced else 1+(self.episode_index+self.rank)%2))
        cpu=self.current['kind']=='cpu'
        if cpu:self.player=1
        self.offset=int(o.get('offset',0 if cpu else 12*(self.episode_index%2)))
        self.game.open('pve' if cpu else 'pvp',int(self.current.get('stars',1)),self.current.get('character','Ryu'))
        for _ in range(self.offset):self.game.step_keys()
        self.native_frames+=self.offset
        self.sides={self.player:Side(self.definition,self.player),3-self.player:Side(self.controller.definition,3-self.player)}
        for side in self.sides.values():side.reset(self.game.rgb)
        self.episode_index+=1;self.episode_steps=0;self.episode_return=0.;self.ended=False;self.last_info={};self.recent=deque(maxlen=8)
        return self.sides[self.player].observe(self.game.raw),{}
    def step(self,action,*,opponent_action=None):
        if self.ended:raise RuntimeError('Reset before stepping')
        if self.capture and self.episode_steps%8==0:self.recent.append(self.save())
        before=dict(self.game.raw);cpu=self.current['kind']=='cpu';index=int(action)
        if not 0<=index<len(self.commands):raise ValueError('Invalid learner action')
        if cpu:
            if opponent_action is not None:raise ValueError('Cannot override native CPU')
            other=-1;enemy=((),)*6
        else:
            if opponent_action is None:other=self.controller.choose(self.sides[3-self.player],before)
            else:self.controller.count+=1;other=int(opponent_action)
            enemy=self.controller.commands[other]
        pair=(self.commands[index],enemy) if self.player==1 else (enemy,self.commands[index])
        winner=reason=None;executed=0
        for k1,k2 in zip(*pair):
            raw,winner,reason=self.game.step_keys(k1,k2);executed+=1
            if reason:break
        self.native_frames+=executed
        for side in self.sides.values():side.update(self.game.rgb)
        p=self.player;q=3-p
        result='cutoff' if reason=='frame_cap' else 'draw' if winner=='draw' else 'win' if winner==f'p{p}' else 'loss' if winner else 'ongoing'
        dealt=max(0,max(before[f'p{q}_hp'],0)-max(raw[f'p{q}_hp'],0));taken=max(0,max(before[f'p{p}_hp'],0)-max(raw[f'p{p}_hp'],0))
        phi0=.5*(max(before[f'p{p}_hp'],0)-max(before[f'p{q}_hp'],0))/176
        phi1=.5*(max(raw[f'p{p}_hp'],0)-max(raw[f'p{q}_hp'],0))/176 if reason is None else 0.
        reward=float(result=='win')+phi1-phi0
        if self.reward_kind=='terminal_score':reward=float(result=='win')-float(result=='loss')+phi1-phi0
        if self.reward_kind=='damage':reward=(dealt-taken)/176+float(result=='win')-float(result=='loss')
        self.episode_steps+=1;self.episode_return+=reward;self.ended=reason is not None
        info=dict(result=result,end_reason=reason,self_hp=raw[f'p{p}_hp'],opponent_hp=raw[f'p{q}_hp'],damage_dealt=dealt,damage_taken=taken,step_reward=reward,
            native_frames=self.game.elapsed,elapsed_native_frames=executed,opponent=self.current['id'],opponent_kind=self.current['kind'],player=p,offset=self.offset,action=index,opponent_action=other)
        if self.ended:info['episode']=dict(r=self.episode_return,l=self.episode_steps,t=self.game.elapsed/self.game.fps)
        self.last_info=info
        return self.sides[p].observe(raw),reward,bool(reason and reason!='frame_cap'),reason=='frame_cap',info
    def save(self):
        return dict(game=self.game.snapshot(),controller=self.controllers.index(self.controller),count=self.controller.count,player=self.player,
            controller_actor_state=None if self.controller.actor is None else self.controller.actor.get_state(),
            frames={p:[f.copy() for f in s.frames] for p,s in self.sides.items()},
            bookkeeping={n:copy.deepcopy(getattr(self,n)) for n in ('episode_steps','episode_return','episode_index','ended','last_info','offset')},
            rng=copy.deepcopy(self.np_random.bit_generator.state))
    def restore(self,saved):
        self.game.restore(saved['game']);self.player=saved['player'];index=saved['controller']
        self.controller=self.controllers[index];self.current=self.pool[index];self.controller.count=saved['count']
        if self.controller.actor is not None:self.controller.actor.set_state(saved.get('controller_actor_state'))
        self.sides={self.player:Side(self.definition,self.player),3-self.player:Side(self.controller.definition,3-self.player)}
        for p,frames in saved['frames'].items():self.sides[p].frames=deque([f.copy() for f in frames],maxlen=4)
        for name,value in saved['bookkeeping'].items():setattr(self,name,copy.deepcopy(value))
        self.np_random.bit_generator.state=copy.deepcopy(saved['rng'])
        return self.sides[self.player].observe(self.game.raw)
    def set_pool_weights(self,weights):
        if len(weights)!=len(self.pool) or min(weights)<=0:raise ValueError('Positive pool weights required')
        for c,w in zip(self.pool,weights):c['weight']=float(w)
    def close(self):self.game.close()
