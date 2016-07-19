import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.layers.core import Dense

import xp
# Uses:
# xp.replay
# xp.STATE_DIM

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
        self.model.compile(loss=one_sided_l2, optimizer=Adam(lr=0.0005, beta_2=0.9999)) #clipvalue=0.1

class ValueWIRES:
    def __init__(self, GAMMA = 0.995, TAU = 0.01):
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.V_online = VNetwork()
        self.V_stable = VNetwork()

    def _slowly_transfer_weights_to_stable_network(self):
        ws_online = self.V_online.model.get_weights()
        ws_stable = self.V_stable.model.get_weights()
        for arr_online, arr_stable in zip(ws_online, ws_stable):
            arr_stable *= (1-self.TAU)
            arr_stable += self.TAU*arr_online
        self.V_stable.model.set_weights(ws_stable)

    def learn_iteration(self, buf, dry_run):
        BATCH = len(buf)
        input  = np.zeros( (BATCH, xp.STATE_DIM) )
        target = np.zeros( (BATCH, 1) )

        # collect current values
        for i,x in enumerate(buf):
            input[i] = x.sn
        next_v = self.V_stable.model.predict(input)

        for i,x in enumerate(buf):
            input[i] = x.s
        stable_v = self.V_stable.model.predict(input)
        online_v = self.V_online.model.predict(input)

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
            if not x.terminal or x.r > 0:
                v  = max(v, x.nv)
            v *= self.GAMMA
            v += x.r
            x.target = v
            x.episode = episode

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
            xp.export_viz.episode[x.viz_n] = x.episode

        if dry_run:
            loss = self.V_online.model.test_on_batch(input, target)
            print("WIRES (test) %0.5f" % loss)
        else:
            loss = self.V_online.model.train_on_batch(input, target)
            print("WIRES %0.5f" % loss)
            self._slowly_transfer_weights_to_stable_network()
