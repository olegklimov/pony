from controller import algo
import gym.envs.box2d.bipedal_walker as w
import numpy as np

class DemoBipedalWalker(algo.Algorithm):
    def __init__(self):
        algo.Algorithm.__init__(self, BATCH=100)
    def _learn_iteration(self, buf, dry_run):
        pass
    def _save(self, fn):
        pass
    def _load(self, fn):
        pass
    def control(self, s, action_space):
        a  = w.heuristic(self, s)
        rand = 0.02
        a += np.random.uniform( low=-rand, high=+rand, size=(4,) ) 
        return a
