import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense
from threading import Lock

import xp

class Advantage:
    def __init__(self):
        self.model = keras.models.Sequential()
        from keras.regularizers import l2
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01), batch_input_shape=(None, xp.STATE_DIM+xp.ACTION_DIM)))
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01)))
        self.model.add(Dense(1))
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        #self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005, beta_2=0.9999))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))
        self.model_mutex = Lock()

    def learn_iteration(self, buf, dry_run):
        BATCH = len(buf)
        input  = np.zeros( (BATCH, xp.STATE_DIM + xp.ACTION_DIM) )
        target = np.zeros( (BATCH, 1) )

        for i,x in enumerate(buf):
            input[i][:xp.STATE_DIM] = x.s
            input[i][xp.STATE_DIM:] = x.a
            target[i,0] = x.nv - x.v

        with self.model_mutex:
            results = self.model.predict_on_batch(input)
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                xp.export_viz.Vpolicy[x.viz_n] = x.v + 10*results[i]
                #print x.nv, x.v, x.nv - x.v, "predicted", results[i]

        if dry_run:
            with self.model_mutex:
                loss = self.model.test_on_batch(input, target)
            #print("advantage (test) %0.5f" % loss)
        else:
            with self.model_mutex:
                loss = self.model.train_on_batch(input, target)
            print("advantage %0.5f" % loss)

    def estimate(self, s, these_actions):
        A = len(these_actions)
        input  = np.zeros( (A, xp.STATE_DIM + xp.ACTION_DIM) )
        for i in range(A):
            input[i][:xp.STATE_DIM] = s
            input[i][xp.STATE_DIM:] = these_actions[i]
        with self.model_mutex:
            r = self.model.predict_on_batch(input)
        return r

