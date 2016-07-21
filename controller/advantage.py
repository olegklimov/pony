import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense

import xp

class Advantage:
    def __init__(self):
        self.model = keras.models.Sequential()
        from keras.regularizers import l2
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01), batch_input_shape=(None, xp.STATE_DIM+xp.ACTION_DIM)))
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01)))
        self.model.add(Dense(1))
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005, beta_2=0.9999))

    def learn_iteration(self, buf, dry_run):
        BATCH = len(buf)
        input  = np.zeros( (BATCH, xp.STATE_DIM + xp.ACTION_DIM) )
        target = np.zeros( (BATCH, 1) )

        for i,x in enumerate(buf):
            input[i][:xp.STATE_DIM] = x.s
            input[i][xp.STATE_DIM:] = x.a
            target[i,0] = x.nv - x.v

        if dry_run:
            loss = self.model.test_on_batch(input, target)
            #print("advantage (test) %0.5f" % loss)
        else:
            loss = self.model.train_on_batch(input, target)
            print("advantage %0.5f" % loss)

    def estimate(self, s, these_actions):
        A = len(these_actions)
        input  = np.zeros( (A, xp.STATE_DIM + xp.ACTION_DIM) )
        for i in range(A):
            input[i][:xp.STATE_DIM] = s
            input[i][xp.STATE_DIM:] = these_actions[i]
        r = self.model.predict_on_batch(input)
        return r

