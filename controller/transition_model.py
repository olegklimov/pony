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

CRASH_OR_WIN_THRESHOLD = 50

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
        almost_out = d2(d1( merge( [inp_s,clamp(inp_a)], mode='concat' ) ))
        out_s = Dense(xp.STATE_DIM, W_regularizer=l2(0.001))
        out_r = Dense(1, W_regularizer=l2(0.001))
        out_tensor_s = out_s(almost_out)
        out_tensor_r = out_r(almost_out)

        self.model = Model( input=[inp_s, inp_a], output=[out_tensor_s, out_tensor_r] )
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0001, beta_2=0.999, epsilon=1e-5))
        #self.model.compile(loss='mse', optimizer=Adamax())
        #self.model.compile(loss='mae', optimizer=Adam(lr=0.00005, beta_2=0.999))

    def learn_iteration(self, buf, dry_run):
        BATCH = len(buf)
        inp_s  = np.zeros( (BATCH, xp.STATE_DIM) )
        inp_a  = np.zeros( (BATCH, xp.ACTION_DIM) )
        target_s = np.zeros( (BATCH, xp.STATE_DIM) )
        target_r = np.zeros( (BATCH, 1) )
        sample_weight = np.ones( (BATCH,) )

        good, bad = 0, 0
        for i,x in enumerate(buf):
            inp_s[i] = x.s 
            inp_a[i] = x.a
            target_s[i] = x.sn - x.s
            target_r[i] = 0 #x.r
            if np.abs(x.r) > CRASH_OR_WIN_THRESHOLD:
                # Don't try to predict wins and crashes: it's a physical model,
                # it only can approximate potential field r = V(s') - V(s)
                sample_weight[i] = 0.0
            if np.linalg.norm( target_s[i] ) > 1.0:
                # Don't try to approximate when contact with the ground changes, it is outliers.
                sample_weight[i] = 0.0
                bad += 1
            else:
                good += 1
        #print( "%i/%i" % (bad, (good+bad)) )

        with self.model_mutex:
            test_s, test_r = self.model.predict([inp_s, inp_a])
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                xp.export_viz.ttest[x.viz_n] = test_s[i] + x.s

        with self.model_mutex:
            if dry_run:
                loss = self.model.test_on_batch([inp_s, inp_a], [target_s, target_r], sample_weight=[sample_weight,sample_weight])
            else:
                loss = self.model.train_on_batch([inp_s, inp_a], [target_s, target_r], sample_weight=[sample_weight,sample_weight])
            #print("transition dry_run=%i %0.5f" % (dry_run, loss))
        #print self.model.metrics_names  # [loss, state_loss, reward_loss]
        return float(loss[1]), float(loss[2])

    def predict(self, inp_s, inp_a):
        with self.model_mutex:
            ret_s, ret_r = self.model.predict([inp_s, inp_a])
        return ret_s + inp_s, ret_r
