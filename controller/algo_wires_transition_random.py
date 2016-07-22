import value_WIRES as wires
import algo
import transition_model
import xp

import numpy as np
from threading import Lock

class WiresTransitionRandom(algo.Algorithm):
    def __init__(self):
        algo.Algorithm.__init__(self, BATCH=100)
        self.wires = wires.ValueWIRES()
        self.trans = transition_model.Transition()

    def _learn_iteration(self, buf, dry_run):
        self.wires.learn_iteration(buf, dry_run)
        self.trans.learn_iteration(buf, dry_run)

    def _save(self, fn):
        self.trans.model.save_weights(fn + "_trans.h5", overwrite=True)
        self.wires.V_stable.model.save_weights(fn + "_wires.h5", overwrite=True)

    def _load(self, fn):
        self.trans.model.load_weights(fn + "_trans.h5")
        self.wires.V_stable.model.load_weights(fn + "_wires.h5")
        self.wires.V_online.model.load_weights(fn + "_wires.h5")

    def control(self, s, action_space):
        RAND = 50
        input = np.zeros(shape=(RAND, xp.STATE_DIM + xp.ACTION_DIM))
        for c in range(10):
            if c==0:
                for i in range(RAND):
                    input[i, xp.STATE_DIM:] = action_space.sample()
            else:
                k = 0.1 / c
                for i in range(RAND):
                    input[i, xp.STATE_DIM:] = best_action + k*action_space.sample()
            v1sn = self.trans.predict(input)
            v1v  = self.wires.evaluate(v1sn)
            #v1v = np.zeros(shape=(xp.ACTION_DIM,))
            v1i = np.argmax(v1v)
            best_action = input[v1i, xp.STATE_DIM:]
            #print "best_action", best_action
        print "val%02i %0.2f" % (c, max(v1v))
        return best_action
