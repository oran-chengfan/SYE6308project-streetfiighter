"""Actual counterfactual native transitions; no simulated game is counted as real."""
from experiments.league_adaptive.runtime import setup
from experiments.ryu_adaptive_curriculum.train_direct_sequence import CoverageEnv
from experiments.ryu_direct_league.environment import RyuEnv


class BranchingEnv(CoverageEnv):
    def __init__(self,definition,pool,rank,seed,objective='graded'):
        super().__init__(definition,pool,rank,seed,reward_kind='damage',damage_weight=.25)
        self.branch_native_frames=0;self.q_objective=objective

    def q_reward(self,before,info,done):
        if self.q_objective=='graded':
            return .25*(info['damage_dealt']-info['damage_taken'])/176.+float(info['result']=='win')-float(info['result']=='loss')
        p=self.player;q=3-p;prior=.25*(max(0,before[f'p{p}_hp'])-max(0,before[f'p{q}_hp']))/176.
        after=0. if done else .25*(max(0,info['self_hp'])-max(0,info['opponent_hp']))/176.
        terminal=(1. if info['result']=='win' else -1.) if done else 0.
        return terminal+after-prior

    def step(self,action,**kwargs):
        before=dict(self.game.raw);observation,old,t,tr,info=super().step(action,**kwargs)
        reward=self.q_reward(before,info,bool(t or tr));self.episode_return+=reward-old;info['step_reward']=reward
        if 'episode' in info:info['episode']['r']=self.episode_return
        return observation,reward,t,tr,info

    def counterfactual_plans(self,plans):
        root=self.save();frames_before=self.native_frames;result=[]
        try:
            for plan in plans:
                self.restore(root);segments=[];ended=False
                for action,duration in plan:
                    ticks=[]
                    for _ in range(int(duration)):
                        # Bypass CoverageEnv's real completed-game scheduler.
                        raw_before=dict(self.game.raw);observation,_,t,tr,info=RyuEnv.step(self,int(action))
                        reward=self.q_reward(raw_before,info,bool(t or tr))
                        ticks.append(dict(observation={k:v.copy() for k,v in observation.items()},reward=reward,done=bool(t or tr),
                                          result=info['result'],self_hp=info['self_hp'],opponent_hp=info['opponent_hp']))
                        if t or tr:ended=True;break
                    segments.append(dict(action=int(action),duration=int(duration),ticks=ticks))
                    if ended:break
                result.append(segments)
        finally:
            self.restore(root);self.branch_native_frames+=self.native_frames-frames_before
        return dict(opponent=self.current['id'],player=self.player,offset=self.offset,plans=result)


class Worker:
    def __init__(self,definition,pool,rank,seed,objective='graded'):self.definition,self.pool,self.rank,self.seed,self.objective=definition,pool,rank,seed,objective
    def __call__(self):setup();return BranchingEnv(self.definition,self.pool,self.rank,self.seed,self.objective)
