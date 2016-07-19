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
                if xp.epoch > 1: self.dry_run = False

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

    def control(self, s):
        V = 50
        v1 = np.random.uniform( low=-1.0, high=1.0, shape=(V,) )
        return a

