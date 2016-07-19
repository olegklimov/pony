import value_WIRES as wires
import advantage
import xp

import numpy as np
from threading import Lock

BATCH = 200

class WiresAdvantageRandom:
    def __init__(self):
        self.wires = wires.ValueWIRES()
        self.adv = advantage.Advantage()
        self.pause = False
        self.quit = False
        self.dry_run = True
        self.save_load_mutex = Lock()

    def learn_thread_func(self):
        while not self.quit:
            while self.pause and not self.quit and not self.dry_run:
                import time
                time.sleep(0.1)
            with self.save_load_mutex:
                buf = xp.batch(BATCH)
                self.wires.learn_iteration(buf, self.dry_run)
                self.adv.learn_iteration(buf, self.dry_run)
                if xp.epoch > 2: self.dry_run = False

    def save(self, fn):
        with self.save_load_mutex:
            print("SAVE %s" % fn)
            self.adv.model.save_weights(fn + "_adv.h5", overwrite=True)
            self.wires.V_stable.model.save_weights(fn + "_wires.h5", overwrite=True)

    def load(self, fn):
        with self.save_load_mutex:
            print("LOAD %s" % fn)
            self.adv.model.load_weights(fn + "_adv.h5")
            self.wires.V_stable.model.load_weights(fn + "_wires.h5")
            self.wires.V_online.model.load_weights(fn + "_wires.h5")
            self.dry_run = True
            xp.epoch = 0.0
            xp.epoch_sample_counter = 0

    def control(self, s, action_space):
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

