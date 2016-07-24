import value_WIRES as wires
import algo
import advantage
import xp

import numpy as np
from threading import Lock

class WiresAdvantageRandom(algo.Algorithm):
    def __init__(self):
        algo.Algorithm.__init__(self, BATCH=100)
        self.wires = wires.ValueWIRES()
        self.adv = advantage.Advantage()

    def _learn_iteration(self, buf, dry_run):
        self.wires.learn_iteration(buf, dry_run)
        self.adv.learn_iteration(buf, dry_run)

    def _save(self, fn):
        self.adv.model.save_weights(fn + "_adv.h5", overwrite=True)
        self.wires.V_stable.model.save_weights(fn + "_wires.h5", overwrite=True)

    def _load(self, fn):
        self.adv.model.load_weights(fn + "_adv.h5")
        self.wires.V_stable.model.load_weights(fn + "_wires.h5")
        self.wires.V_online.model.load_weights(fn + "_wires.h5")

    def _control(self, s, action_space):
        v1  = [action_space.sample() for x in range(50)]
        v1e = self.adv.estimate(s, v1)
        v1i = np.argmax(v1e)
        print "CONTROL1 %0.3f .. %0.3f  (%0.3f)" % (np.min(v1e), np.max(v1e), v1e[v1i])

        v2  = [(action_space.sample()*0.2 + v1[v1i]) for x in range(50)]
        v2e = self.adv.estimate(s, v2)
        v2i = np.argmax(v2e)
        print "CONTROL2 %0.3f .. %0.3f  (%0.3f)" % (np.min(v2e), np.max(v2e), v2e[v2i])

        v3  = [(action_space.sample()*0.1 + v2[v2i]) for x in range(50)]
        v3e = self.adv.estimate(s, v3)
        v3i = np.argmax(v3e)
        print "CONTROL3 %0.3f .. %0.3f  (%0.3f)" % (np.min(v3e), np.max(v3e), v3e[v3i])

        return v3[v3i]

