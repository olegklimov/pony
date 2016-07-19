#!/usr/bin/env python

import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense
import xp

BATCH = 200
GAMMA = 0.995
TAU   = 0.01

class VNetwork:
    def __init__(self):
        self.model = keras.models.Sequential()
        from keras.regularizers import l2
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01), batch_input_shape=(None,xp.STATE_DIM)))
        self.model.add(Dense(256, activation='relu', W_regularizer=l2(0.01)))
        self.model.add(Dense(1, W_regularizer=l2(0.01)))
        def one_sided_l2(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=-1)
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        sgd = SGD(lr=0.0001, decay=0, momentum=0.0, nesterov=False)
        self.model.compile(loss=one_sided_l2, optimizer=Adam(lr=0.0005, beta_2=0.9999)) #clipvalue=0.1

V_online = None
V_stable = None

def minibatch_from_replay_buffer():
    while 1:
        buf = xp.batch(BATCH)
        input  = np.zeros( (BATCH, xp.STATE_DIM) )
        target = np.zeros( (BATCH, 1) )

        # collect current values
        for i,x in enumerate(buf):
            input[i] = x.sn
        next_v = V_stable.model.predict(input)

        for i,x in enumerate(buf):
            input[i] = x.s
        stable_v = V_stable.model.predict(input)
        online_v = V_online.model.predict(input)

        for i,x in enumerate(buf):
            x.nv = next_v[i][0]
            x.v  = stable_v[i][0]
            x.ov = online_v[i][0]

        # WIRES
        N = len(xp.replay)
        v = 0
        for i in range(N-1,-1,-1):
            x = xp.replay[i]
            if x.terminal:
                v = 0
            if not x.terminal or x.r > 0:
                v  = max(v, x.nv)
            v *= GAMMA
            v += x.r
            x.target = v

        # save
        for i,x in enumerate(buf):
            target[i,0] = x.target
            xp.export_viz.state1[x.viz_n] = x.s
            xp.export_viz.state2[x.viz_n] = x.sn
            xp.export_viz.Vstable1[x.viz_n] = x.v
            xp.export_viz.Vstable2[x.viz_n] = x.nv
            xp.export_viz.Vonline1[x.viz_n] = x.ov
            xp.export_viz.Vtarget[x.viz_n]  = x.target
            xp.export_viz.step[x.viz_n] = x.step

        yield (input, target)

quit_flag = False

def slowly_transfer_weights_to_stable_network():
    ws_online = V_online.model.get_weights()
    ws_stable = V_stable.model.get_weights()
    for arr_online, arr_stable in zip(ws_online, ws_stable):
        arr_stable *= (1-TAU)
        arr_stable += TAU*arr_online
    V_stable.model.set_weights(ws_stable)

class BatchEndCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        if quit_flag: raise SystemExit("quit")
        print "train %0.5f" % logs.get('loss')
        slowly_transfer_weights_to_stable_network()

def learn_thread_func():
    V_online.model.fit_generator(
        minibatch_from_replay_buffer(),
        samples_per_epoch=1, nb_epoch=100000, verbose=0, max_q_size=1,
        callbacks=[BatchEndCallback()] # Inside callback set new y_ddqn target every minibatch
        );

