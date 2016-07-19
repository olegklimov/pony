import numpy as np
import keras
import keras.models
import value_WIRES as wires
import xp

BATCH = 200

class WiresAdvantageRandom:
    def __init__(self):
        self.wires = wires.ValueWIRES()

    def learn_thread_func(self):
        while 1:
            buf = xp.batch(BATCH)
            self.wires.learn_iteration(buf)

    def control(self, s):
        return a
