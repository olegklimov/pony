import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense

import xp
# Uses:
# xp.STATE_DIM
# xp.ACTION_DIM

class Advantage:
    def __init__(self):
        self.model = keras.models.Sequential()
        from keras.regularizers import l2
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01), batch_input_shape=(None, xp.STATE_DIM+xp.ACTION_DIM)))
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01)))
        self.model.add(Dense(1, W_regularizer=l2(0.01)))
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005, beta_2=0.9999))

    def learn_iteration(self, buf):
        BATCH = len(buf)
        input  = np.zeros( (BATCH, xp.STATE_DIM + xp.ACTION_DIM) )
        target = np.zeros( (BATCH, 1) )

        for i,x in enumerate(buf):
            input[i][:xp.STATE_DIM] = x.s
            input[i][xp.STATE_DIM:] = x.a

        for i,x in enumerate(buf):
            target[i,0] = x.nv - x.v

        loss = self.model.train_on_batch(input, target)
        print "advantage %0.5f" % loss
