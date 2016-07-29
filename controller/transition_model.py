import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Merge, merge, Input
from keras.regularizers import l2
from keras.models import Model
from threading import Lock

import xp

def clamp_minus_one_plus_one(x):
    return K.minimum( +1, K.maximum(x, -1) )  # as opposed to min/max, minimum/maximum is element-wise operations

class Transition:
    def __init__(self):
        self.model_mutex = Lock()

        clamp = Lambda(clamp_minus_one_plus_one)
        inp_s = Input( shape=(xp.STATE_DIM,) )
        inp_a = Input( shape=(xp.ACTION_DIM,) )
        d1 = Dense(512, activation='relu', W_regularizer=l2(0.001))
        d2 = Dense(512, activation='relu', W_regularizer=l2(0.001))
        out = Dense(xp.STATE_DIM)
        out_tensor = out(d2(d1( merge( [inp_s,clamp(inp_a)], mode='concat' ) )))

        self.model = Model( input=[inp_s, inp_a], output=out_tensor )
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        self.model.compile(loss='mae', optimizer=Adam(lr=0.0005, beta_2=0.9999))

    def learn_iteration(self, buf, dry_run):
        BATCH = len(buf)
        inp_s  = np.zeros( (BATCH, xp.STATE_DIM) )
        inp_a  = np.zeros( (BATCH, xp.ACTION_DIM) )
        target = np.zeros( (BATCH, xp.STATE_DIM) )

        for i,x in enumerate(buf):
            inp_s[i] = x.s 
            inp_a[i] = x.a
            target[i] = x.sn - x.s

        with self.model_mutex:
            test = self.model.predict([inp_s, inp_a])
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                xp.export_viz.state_trans[x.viz_n] = test[i] + x.s

        if dry_run:
            with self.model_mutex:
                loss = self.model.test_on_batch([inp_s, inp_a], target)
            #print("transition (test) %0.5f" % loss)
        else:
            with self.model_mutex:
                loss = self.model.train_on_batch([inp_s, inp_a], target)
            #print("transition %0.5f" % loss)
        return loss

    def predict(self, inp_s, inp_a):
        with self.model_mutex:
            return self.model.predict([inp_s, inp_a]) + inp_s
