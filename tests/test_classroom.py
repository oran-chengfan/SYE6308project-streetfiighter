"""Course editing controls must affect learning without changing the referee."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.league_adaptive.environment import FightEnv, TrainingFightEnv, sample_training_condition


class ClassroomControlsTests(unittest.TestCase):
    def test_requested_port_and_disabled_cpu_group(self):
        pool=[dict(id='cpu',kind='cpu'),dict(id='agent',kind='agent')]
        config=dict(learner_sides=[2],coverage_offsets=[0,12],
                    training_group_weights=dict(pve=0,p1=0,p2=1))
        rng=np.random.default_rng(1110)
        choices=[sample_training_condition(pool,config,rng) for _ in range(100)]
        self.assertTrue(all(index==1 and port==2 for index,port,offset in choices))
        self.assertEqual({offset for index,port,offset in choices},{0,12})

    def test_invalid_port_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            sample_training_condition([dict(id='a',kind='agent')],
                                      dict(learner_sides=[3]),np.random.default_rng(1))

    def test_student_reward_preserves_terminal_outcome_and_updates_episode_return(self):
        env=TrainingFightEnv.__new__(TrainingFightEnv)
        env.student_reward=lambda t: 2*t['damage_dealt']-t['damage_taken']
        env.episode_return=11.
        info=dict(result='win',end_reason='signed_hp_ko',player=2,
                  damage_dealt=5,damage_taken=2,step_reward=1.,episode=dict(r=11.,l=10))
        with patch.object(FightEnv,'step',return_value=({},1.,True,False,info)):
            observation,reward,done,truncated,result=env.step(0)
        self.assertEqual(reward,8.)
        self.assertEqual(env.episode_return,18.)
        self.assertEqual(result['episode']['r'],18.)
        self.assertEqual((result['result'],result['end_reason'],result['player']),('win','signed_hp_ko',2))
        self.assertTrue(done)
        self.assertFalse(truncated)

    def test_nonfinite_student_reward_fails_before_learning(self):
        env=TrainingFightEnv.__new__(TrainingFightEnv)
        env.student_reward=lambda t: float('nan')
        env.episode_return=0.
        with patch.object(FightEnv,'step',return_value=({},0.,False,False,{})):
            with self.assertRaises(ValueError):env.step(0)


if __name__=='__main__':unittest.main()
