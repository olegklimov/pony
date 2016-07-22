import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense
from threading import Lock

import xp

class Transition:
    def __init__(self):
        self.model = keras.models.Sequential()
        from keras.regularizers import l2
        self.model.add(Dense(512, activation='relu', W_regularizer=l2(0.0001), batch_input_shape=(None, xp.STATE_DIM+xp.ACTION_DIM)))
        self.model.add(Dense(512, activation='relu', W_regularizer=l2(0.0001)))
        self.model.add(Dense(xp.STATE_DIM))
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        self.model.compile(loss='mae', optimizer=Adam(lr=0.0005, beta_2=0.9999))
        self.model_mutex = Lock()

    def learn_iteration(self, buf, dry_run):
        BATCH = len(buf)
        input  = np.zeros( (BATCH, xp.STATE_DIM + xp.ACTION_DIM) )
        target = np.zeros( (BATCH, xp.STATE_DIM) )

        for i,x in enumerate(buf):
            input[i][:xp.STATE_DIM] = x.s 
            input[i][xp.STATE_DIM:] = x.a
            target[i] = x.sn - x.s

        with self.model_mutex:
            test = self.model.predict(input)
        for i,x in enumerate(buf):
            xp.export_viz.state_trans[x.viz_n] = test[i] + x.s

        if dry_run:
            with self.model_mutex:
                loss = self.model.test_on_batch(input, target)
            #print("transition (test) %0.5f" % loss)
        else:
            with self.model_mutex:
                loss = self.model.train_on_batch(input, target)
            print("transition %0.5f" % loss)

    def predict(self, s_and_a):
        with self.model_mutex:
            return self.model.predict(s_and_a) + s_and_a[:,:xp.STATE_DIM]
