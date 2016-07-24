import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense
from keras.engine import Layer
from threading import Lock

import xp
# Uses:
# xp.replay
# xp.STATE_DIM
# xp.export_viz

CRASH_THRESHOLD = -50   # pin value function if crashed, ignore value regression

class Bell(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Bell, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.exp(-x*x)
        #return -K.abs(x)

    def get_config(self):
        config = {}
        base_config = super(Bell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class VNetwork:
    def __init__(self):
        self.model = keras.models.Sequential()
        from keras.regularizers import l2
        self.model.add(Dense(320, activation='relu', W_regularizer=l2(0.01), batch_input_shape=(None,xp.STATE_DIM)))
        self.model.add(Dense(320, activation='relu', W_regularizer=l2(0.01), batch_input_shape=(None,xp.STATE_DIM)))
        self.model.add(Dense(320, W_regularizer=l2(0.001)))  # hard_sigmoid
        self.model.add(Bell())
        self.model.add(Dense(1))
        def one_sided_l2(y_true, y_pred):
            return K.mean( K.square(y_true - y_pred), axis=-1)
            #return K.mean( (K.sign(y_true) + 1) * K.square(y_true - y_pred), axis=-1)
            #return \
            #    K.mean( (K.sign(y_true) + 1) * K.square(y_true - y_pred), axis=-1) + \
            #    K.mean( (K.sign(y_true) - 1) * (-1) * K.max(10+y_pred, 0), axis=-1)
        from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
        self.model.compile(loss=one_sided_l2, optimizer=Adam(lr=0.0005, beta_2=0.9999)) #clipvalue=0.1

class ValueWIRES:
    def __init__(self, GAMMA = 0.995, TAU = 0.01):
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.V_online = VNetwork()
        self.V_stable = VNetwork()
        self.stable_mutex = Lock()

    def _slowly_transfer_weights_to_stable_network(self):
        ws_online = self.V_online.model.get_weights()
        with self.stable_mutex:
            ws_stable = self.V_stable.model.get_weights()
        for arr_online, arr_stable in zip(ws_online, ws_stable):
            arr_stable *= (1-self.TAU)
            arr_stable += self.TAU*arr_online
        with self.stable_mutex:
            self.V_stable.model.set_weights(ws_stable)

    def learn_iteration(self, buf, dry_run):
        BATCH   = len(buf)
        TUNNEL  = 3*BATCH
        #FARAWAY = BATCH
        input  = np.zeros( (BATCH+TUNNEL, xp.STATE_DIM) )
        target = np.zeros( (BATCH+TUNNEL, 1) )

        # collect current values
        for i,x in enumerate(buf):
            input[i] = x.sn
        with self.stable_mutex:
            next_v = self.V_stable.model.predict(input[:BATCH])

        for i,x in enumerate(buf):
            input[i] = x.s
        with self.stable_mutex:
            stable_v = self.V_stable.model.predict(input[:BATCH])
        online_v = self.V_online.model.predict(input[:BATCH])

        for i,x in enumerate(buf):
            x.nv = next_v[i][0]
            x.v  = stable_v[i][0]
            x.ov = online_v[i][0]

        # WIRES
        N = len(xp.replay)
        v = 0
        episode = 0
        for i in range(N-1,-1,-1):
            x = xp.replay[i]
            if x.terminal:
                v = 0
                episode += 1
            if not (x.terminal and x.r < CRASH_THRESHOLD):
                v = max(v, x.nv)   # not crash
            v *= self.GAMMA
            v += x.r
            x.target = v           # -100 on crash
            x.episode = episode

        # save
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                target[i,0] = x.target
                xp.export_viz.state1[x.viz_n] = x.s
                xp.export_viz.state2[x.viz_n] = x.sn
                xp.export_viz.Vstable1[x.viz_n] = x.v
                xp.export_viz.Vstable2[x.viz_n] = x.nv
                xp.export_viz.Vonline1[x.viz_n] = x.ov
                xp.export_viz.Vtarget[x.viz_n]  = x.target
                xp.export_viz.step[x.viz_n] = x.step
                xp.export_viz.episode[x.viz_n] = x.episode

        MEANINGFUL_AXIS = 14
        TUNNEL_WIDTH = 0.1
        cursor = BATCH 
        while cursor < BATCH+TUNNEL:
            for i,x in enumerate(buf):
                if x.v < 0: continue
                tmp = np.random.uniform( low=-TUNNEL_WIDTH, high=+TUNNEL_WIDTH, size=(MEANINGFUL_AXIS,) )
                tmp.resize( (xp.STATE_DIM,) )
                input[cursor] = x.s + tmp
                viz_n = cursor-BATCH + N
                xp.export_viz.step[viz_n]    = x.step
                xp.export_viz.episode[viz_n] = x.episode
                cursor += 1
                if not (cursor < BATCH+TUNNEL): break
        neighbourhood = self.V_online.model.predict(input[BATCH:])
        cursor = BATCH
        with xp.replay_mutex:
            while cursor < BATCH+TUNNEL:
                t = neighbourhood[cursor-BATCH] * self.GAMMA
                target[cursor,0] = t
                viz_n = cursor-BATCH + N
                xp.export_viz.state1[viz_n] = input[cursor]
                xp.export_viz.state2[viz_n] = input[cursor]
                xp.export_viz.state_trans[viz_n] = input[cursor]
                xp.export_viz.state_policy[viz_n] = input[cursor]
                xp.export_viz.Vstable1[viz_n] = neighbourhood[cursor-BATCH]
                xp.export_viz.Vstable2[viz_n] = t
                xp.export_viz.Vonline1[viz_n] = 0
                xp.export_viz.Vtarget[viz_n]  = 0
                xp.export_viz.Vpolicy[viz_n] = t
                cursor += 1
            xp.export_viz.N[0] = N + TUNNEL

        #for i in range(BATCH,BATCH+FARAWAY):
        #    hypersphere = np.random.uniform(low=-1, high=+1, size=xp.STATE_DIM)
        #    n = np.linalg.norm(hypersphere, ord=1)
        #    hypersphere *= 15.0/n
        #    input[i] = hypersphere
        #axis = np.random.randint( low=0, high=MEANINGFUL_AXIS )
        #tmp = input[i,axis]
        #if np.random.randint(low=0,high=2)==1: tmp = +3.0
        #else: tmp = -3.0
        #input[i,axis] = tmp
        #target[i,0] = -200

        if 0:
            with self.stable_mutex:
                test = self.V_stable.model.predict(input[BATCH:])
            for i in range(BATCH,BATCH+FARAWAY):
                n = buf[i-BATCH].viz_n
                xp.export_viz.state1[n] = input[i]
                xp.export_viz.state2[n] = input[i]
                h = test[i-BATCH]
                xp.export_viz.Vstable1[n] = h
                xp.export_viz.Vstable2[n] = h + 1
                xp.export_viz.Vonline1[n] = h
                xp.export_viz.Vtarget[n]  = target[i,0]

        if dry_run:
            loss = self.V_online.model.test_on_batch(input, target)
            #print("WIRES (test) %0.5f" % loss)
        else:
            loss = self.V_online.model.train_on_batch(input, target)
            #print("WIRES %0.5f" % loss)
            self._slowly_transfer_weights_to_stable_network()

    def evaluate(self, sn):
        with self.stable_mutex:
            return self.V_stable.model.predict(sn)

