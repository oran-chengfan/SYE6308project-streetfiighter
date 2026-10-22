"""Ryu-only native matches; ordinary rewards contain no move preference."""
import copy
from collections import deque
import numpy as np
from experiments.league_adaptive.environment import FightEnv,Controller


class RyuEnv(FightEnv):
    def __init__(self,definition,pool,rank,gamma=.99):
        if any(p.get('character','Ryu')!='Ryu' for p in pool):raise ValueError('This study is Ryu versus Ryu only')
        self.gamma=float(gamma)
        self.cf_capture=False;self.cf_samples=[];self.cf_completed=deque(maxlen=8);self.cf_seen=0
        self.randomize_behavior=False
        self.track_scenes=False
        self.backplay_roots=None;self.backplay_active=False;self.backplay_level=0;self.backplay_results=deque(maxlen=16);self.backplay_count=0
        super().__init__(definition,pool,rank,balanced=True,reward='terminal_score')
    def enable_behavior_randomization(self,seed):self.randomize_behavior=True;self.behavior_rng=np.random.default_rng(seed)
    def enable_backplay(self,path,seed):
        import torch
        self.backplay_roots=torch.load(path,map_location='cpu',weights_only=False)['roots'];self.backplay_rng=np.random.default_rng(seed)
    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        self.backplay_active=False
        if self.backplay_roots is not None and self.current['kind']!='cpu' and self.backplay_rng.random()<.5:
            thresholds=(.95,.875,.75,.625,.5,.375,.25,.125,0.)
            candidates=[x for x in self.backplay_roots if x['player']==self.player and x['fraction']>=thresholds[self.backplay_level]]
            chosen=candidates[int(self.backplay_rng.integers(len(candidates)))];saved=copy.copy(chosen['saved'])
            saved['controller']=next(i for i,x in enumerate(self.pool) if x['id']==chosen['opponent']['id'])
            episode_index=self.episode_index;sampling_rng=copy.deepcopy(self.np_random.bit_generator.state)
            observation=self.restore(saved);self.episode_index=episode_index;self.np_random.bit_generator.state=sampling_rng
            self.episode_steps=0;self.episode_return=0.;self.ended=False;self.last_info={}
            self.backplay_active=True;self.backplay_fraction=chosen['fraction']
        self.controller.config=copy.deepcopy(self.current)
        if self.randomize_behavior and self.current.get('randomize_behavior',True) and self.current['kind'] in ('rule','tactical'):
            c=self.controller.config;rng=self.behavior_rng
            c['period']=int(rng.integers(8,41));c['phase']=int(rng.integers(c['period']))
            c['close']=int(rng.integers(44,77));c['far']=int(rng.integers(90,151))
            if c['style']=='switching':
                c['switch_every']=int(rng.integers(40,121));rng.shuffle(c['styles'])
        return observation,info
    def save(self):
        result=super().save();result['controller_configuration']=copy.deepcopy(self.controller.config);return result
    def restore(self,saved):
        observation=super().restore(saved)
        self.controller.config=copy.deepcopy(saved.get('controller_configuration',self.current))
        return observation
    def enable_cf_capture(self,seed):self.cf_capture=True;self.cf_rng=np.random.default_rng(seed)
    def _capture_cf(self):
        if self.episode_steps==0:self.cf_samples=[];self.cf_seen=0
        self.cf_seen+=1
        # Preserve coverage of early decisions as well as the rest of a round.
        # This changes training-state sampling, never the deployed action rule.
        if self.cf_seen<=40:
            if len(self.cf_samples)<2:self.cf_samples.append(self.save())
            else:
                index=int(self.cf_rng.integers(self.cf_seen))
                if index<2:self.cf_samples[index]=self.save()
        elif len(self.cf_samples)<4:self.cf_samples.append(self.save())
        else:
            index=int(self.cf_rng.integers(self.cf_seen-40))
            if index<2:self.cf_samples[index+2]=self.save()
    def pop_cf_roots(self,count=1):
        values=[]
        for _ in range(min(count,len(self.cf_completed))):
            index=next((i for i,x in enumerate(self.cf_completed) if x['round_result']=='loss'),len(self.cf_completed)-1)
            value=self.cf_completed[index];del self.cf_completed[index];values.append(value)
        return values
    def step(self,action,*,opponent_action=None):
        if self.cf_capture:self._capture_cf()
        before=dict(self.game.raw);p=self.player;q=3-p
        observation,_,terminated,truncated,info=super().step(action,opponent_action=opponent_action)
        if self.track_scenes:
            distance=abs(before[f'p{q}_x']-before[f'p{p}_x']);lead=before[f'p{p}_hp']-before[f'p{q}_hp']
            timer=before['timer_bcd'];seconds=10*(timer>>4)+(timer&15)
            info['scene']='|'.join(('close' if distance<70 else 'middle' if distance<130 else 'far',
                'air' if before[f'p{q}_y']<184 else 'ground',
                'ahead' if lead>24 else 'behind' if lead < -24 else 'even',
                'late' if seconds<=20 else 'early' if seconds>=70 else 'middle_time',
                'enemy_right' if before[f'p{q}_x']>=before[f'p{p}_x'] else 'enemy_left'))
        prior=.5*(max(0,before[f'p{p}_hp'])-max(0,before[f'p{q}_hp']))/176.
        after=0. if terminated or truncated else .5*(max(0,info['self_hp'])-max(0,info['opponent_hp']))/176.
        reward=float(info['result']=='win')-float(info['result']=='loss')+self.gamma*after-prior
        # Replace bookkeeping inherited from the research environment.
        self.episode_return+=reward-info['step_reward'];info['step_reward']=reward
        if 'episode' in info:info['episode']['r']=self.episode_return
        if 'episode' in info and self.backplay_active:
            self.backplay_results.append(float(info['result']=='win'));self.backplay_count+=1
            if self.backplay_count%8==0 and len(self.backplay_results)>=8:
                rate=float(np.mean(self.backplay_results))
                if rate>=.75 and self.backplay_level<8:self.backplay_level+=1;self.backplay_results.clear()
                elif rate<.25 and self.backplay_level>0:self.backplay_level-=1;self.backplay_results.clear()
            info['backplay_training_start']=dict(fraction=self.backplay_fraction,level=self.backplay_level)
        if 'episode' in info and self.randomize_behavior and self.current['kind'] in ('rule','tactical'):
            info['training_rival_parameters']={k:v for k,v in self.controller.config.items() if k in ('style','period','phase','close','far','styles','switch_every')}
        if self.cf_capture and 'episode' in info and self.cf_samples:
            root=self.cf_samples[int(self.cf_rng.integers(len(self.cf_samples)))]
            self.cf_completed.append(dict(saved=root,opponent=copy.deepcopy(self.current),round_result=info['result']))
        self.last_info=info
        return observation,reward,terminated,truncated,info
    def add_opponent(self,entry):
        if entry.get('character','Ryu')!='Ryu':raise ValueError('Unexpected character')
        if any(p['id']==entry['id'] for p in self.pool):return
        self.pool.append(copy.deepcopy(entry));self.controllers.append(Controller(entry,self.definition))


class Worker:
    def __init__(self,definition,pool,rank,gamma,capture_seed=None,behavior_seed=None,backplay_path=None,track_scenes=False):self.definition,self.pool,self.rank,self.gamma,self.capture_seed,self.behavior_seed,self.backplay_path,self.track_scenes=definition,pool,rank,gamma,capture_seed,behavior_seed,backplay_path,track_scenes
    def __call__(self):
        from experiments.league_adaptive.runtime import setup
        setup();env=RyuEnv(self.definition,self.pool,self.rank,self.gamma)
        env.track_scenes=self.track_scenes
        if self.capture_seed is not None:env.enable_cf_capture(self.capture_seed)
        if self.behavior_seed is not None:env.enable_behavior_randomization(self.behavior_seed)
        if self.backplay_path is not None:env.enable_backplay(self.backplay_path,self.rank+41000)
        return env
