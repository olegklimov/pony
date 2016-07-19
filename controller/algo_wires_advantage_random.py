import value_WIRES as wires
import advantage
import xp

import numpy as np

BATCH = 200

class WiresAdvantageRandom:
    def __init__(self):
        self.wires = wires.ValueWIRES()
        self.adv = advantage.Advantage()

    def learn_thread_func(self):
        while 1:
            buf = xp.batch(BATCH)
            self.wires.learn_iteration(buf)
            self.adv.learn_iteration(buf)

    def control(self, s):
        return a
