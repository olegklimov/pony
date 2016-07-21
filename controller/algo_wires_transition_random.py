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
        v1   = np.zeros(shape=(RAND, xp.STATE_DIM + xp.ACTION_DIM))
        for i in range(RAND):
            v1[i, :xp.STATE_DIM] = s
            v1[i, xp.STATE_DIM:] = action_space.sample()
        print "v1", v1
        v1sn = self.trans.predict(v1)
        print "v1sn", v1sn
        v1v  = self.wires.evaluate(v1sn)
        print "v1v", v1v
        v1i  = np.argmax(v1v)
        print "v1i", v1i
        return v1[v1i, xp.STATE_DIM:]
