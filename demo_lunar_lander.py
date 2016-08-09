from controller import algo
import gym.envs.box2d.lunar_lander as ll
import numpy as np

class DemoLunarLander(algo.Algorithm):
    def __init__(self):
        algo.Algorithm.__init__(self)
    def _learn_iteration(self, buf, dry_run):
        pass
    def _save(self, fn):
        pass
    def _load(self, fn):
        pass
    def _reset(self):
        pass
    def control(self, s, action_space):
        self.continuous = True
        a  = ll.heuristic(self, s)
        #rand = 0.02
        #a += np.random.uniform( low=-rand, high=+rand, size=(4,) ) 
        return a
