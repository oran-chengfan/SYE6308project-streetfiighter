from experiments.ryu_counterq.environment import BranchingEnv
from experiments.league_adaptive.runtime import setup


class OptionEnv(BranchingEnv):
    def step(self,action,**kwargs):
        if int(action)<0:
            # Pause this independent simulator, including its opponent. No
            # native frame or actor history is advanced while other workers
            # finish options from the same frozen collecting policy.
            return self.sides[self.player].observe(self.game.raw),0.,False,False,dict(paused=True)
        observation,reward,t,tr,info=super().step(action,**kwargs);info['paused']=False
        return observation,reward,t,tr,info


class Worker:
    def __init__(self,definition,pool,rank,seed):self.definition,self.pool,self.rank,self.seed=definition,pool,rank,seed
    def __call__(self):setup();return OptionEnv(self.definition,self.pool,self.rank,self.seed,objective='win')
