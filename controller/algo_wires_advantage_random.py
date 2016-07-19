import value_WIRES as wires
import advantage
import xp

import numpy as np

BATCH = 200

class WiresAdvantageRandom:
    def __init__(self):
        self.wires = wires.ValueWIRES()
        self.adv = advantage.Advantage()
        self.pause = False
        self.quit = False

    def learn_thread_func(self):
        while not self.quit:
            while self.pause and not self.quit:
                import time
                time.sleep(0.1)
            buf = xp.batch(BATCH)
            self.wires.learn_iteration(buf)
            self.adv.learn_iteration(buf)

    def save(self):
        pass

    def load(self):
        pass

    def control(self, s):
        V = 50
        v1 = np.random.uniform( low=-1.0, high=1.0, shape=(V,) )
        return a

