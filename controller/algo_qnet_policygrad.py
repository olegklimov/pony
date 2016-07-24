import algo
import xp
import transition_model

import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.regularizers import l2
from keras.layers.core import Dense
from keras.layers import merge, Input
from keras.models import Model
from keras.engine import Layer
from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
from threading import Lock

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

CRASH_THRESHOLD = -50   # pin value function if crashed, ignore value regression

class QNetPolicygrad(algo.Algorithm):
    def __init__(self, GAMMA=0.995, TAU=0.01, BATCH=100):
        algo.Algorithm.__init__(self, BATCH)
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.stable_mutex = Lock()

        #self.transition = transition_model.Transition()

        stable_inp_s = Input( shape=(xp.STATE_DIM,) )  # batch (None, ...) added automatically
        stable_inp_a = Input( shape=(xp.ACTION_DIM,) )
        stable_inp = merge( [stable_inp_s, stable_inp_a], mode='concat' )
        stable_d1 = Dense(320, activation='relu', W_regularizer=l2(0.01))
        stable_d2 = Dense(240, activation='relu', W_regularizer=l2(0.01))
        stable_d3 = Dense(120, W_regularizer=l2(0.01))
        stable_bell = Bell()
        stable_qout = Dense(1)

        online_inp_s = Input( shape=(xp.STATE_DIM,) )
        online_inp_a = Input( shape=(xp.ACTION_DIM,) )
        online_inp = merge( [online_inp_s, online_inp_a], mode='concat' )
        online_d1 = Dense(320, activation='relu', W_regularizer=l2(0.01))
        online_d2 = Dense(240, activation='relu', W_regularizer=l2(0.01))
        online_d3 = Dense(120, W_regularizer=l2(0.01))
        online_bell = Bell()
        online_qout = Dense(1)

        stable_out_tensor = stable_qout(stable_bell( stable_d3(stable_d2(stable_d1(stable_inp))) ))
        online_out_tensor = online_qout(online_bell( online_d3(online_d2(online_d1(online_inp))) ))

        self.Q_online = Model( input=[online_inp_s,online_inp_a], output=online_out_tensor )
        self.Q_stable = Model( input=[stable_inp_s,stable_inp_a], output=stable_out_tensor )

        self.Q_online.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))
        self.Q_stable.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))

        policy_d1 = Dense(320, activation='relu', W_regularizer=l2(0.01))
        policy_d2 = Dense(240, activation='relu', W_regularizer=l2(0.01))
        policy_d3 = Dense(120, W_regularizer=l2(0.01))
        policy_bell = Bell()
        policy_out = Dense(xp.ACTION_DIM)

        policy_out_tensor = policy_out(policy_bell( policy_d3(policy_d2(policy_d1(stable_inp_s))) ))
        policy_value_of_s = self.Q_stable( [stable_inp_s,policy_out_tensor] )

        self.policy_action = Model( input=[stable_inp_s], output=policy_out_tensor )
        self.policy_value  = Model( input=[stable_inp_s], output=policy_value_of_s )

        def only_up(y_true, y_pred):
            return K.mean( -y_pred, axis=-1 )
        self.policy_action.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))
        self.policy_value.compile( loss=only_up, optimizer=Adam(lr=0.0005, beta_2=0.9999))

    def _learn_iteration(self, buf, dry_run):
        #self.transition.learn_iteration(buf, dry_run)

        BATCH = len(buf)
        assert(self.BATCH==BATCH)
        TUNNEL  = 3*BATCH

        batch_s = np.zeros( (BATCH+TUNNEL, xp.STATE_DIM) )
        batch_a = np.zeros( (BATCH+TUNNEL, xp.ACTION_DIM) )
        batch_t = np.zeros( (BATCH+TUNNEL, 1) )

        for i,x in enumerate(buf):
            batch_s[i] = x.sn
        with self.stable_mutex:
            nv = self.policy_value.predict(batch_s[:BATCH])
        for i,x in enumerate(buf):
            x.nv = nv[i][0]

        # WIRES
        N = len(xp.replay)
        v = 0
        episode = 0
        for i in range(N-1,-1,-1):
            x = xp.replay[i]
            if x.terminal:
                v = 0
                episode += 1
            x.episode = episode
            v = v*self.GAMMA + x.r
            x.wires_v = v
            if not (x.terminal and x.r < CRASH_THRESHOLD):    # not crash 
                x.target_v = max(v, x.nv*self.GAMMA + x.r)
                v = max(x.target_v, x.v)                      # x.v is stable v
            else:
                x.target_v = v                                # -100 on crash

        for i,x in enumerate(buf):
            batch_s[i] = x.s
            batch_a[i] = x.a
        with self.stable_mutex:
            stable_v = self.Q_stable.predict( [batch_s[:BATCH], batch_a[:BATCH]] )
        for i,x in enumerate(buf):
            x.v = stable_v[i][0]

        # Viz
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                batch_t[i,0] = x.target_v
                xp.export_viz.state1[x.viz_n] = x.s
                xp.export_viz.state2[x.viz_n] = x.sn
                xp.export_viz.Vstable1[x.viz_n] = x.v
                xp.export_viz.Vstable2[x.viz_n] = x.nv
                xp.export_viz.Vonline1[x.viz_n] = 0
                xp.export_viz.Vtarget[x.viz_n]  = x.target_v
                xp.export_viz.step[x.viz_n] = x.step
                xp.export_viz.episode[x.viz_n] = x.episode
                # TODO
                xp.export_viz.state_policy[x.viz_n] = x.s
                xp.export_viz.Vpolicy[x.viz_n] = 0

        # Generate tunnel cloud
        MEANINGFUL_AXIS = 14
        TUNNEL_RADIUS_STATE = 0.1
        TUNNEL_RADIUS_ACTION = 0.1
        cursor = BATCH 
        while cursor < BATCH+TUNNEL:
            for i,x in enumerate(buf):
                if x.v < 0: continue
                tmp = np.random.uniform( low=-TUNNEL_RADIUS_STATE, high=+TUNNEL_RADIUS_STATE, size=(MEANINGFUL_AXIS,) )
                tmp.resize( (xp.STATE_DIM,) )
                batch_s[cursor] = x.s + tmp
                #tmp = np.random.uniform( low=-TUNNEL_RADIUS_ACTION, high=+TUNNEL_RADIUS_ACTION, size=(xp.ACTION_DIM,) )
                #batch_a[cursor] = x.a + tmp
                tmp = np.random.uniform( low=-1.1, high=+1.1, size=(xp.ACTION_DIM,) )
                batch_a[cursor] = tmp
                viz_n = cursor-BATCH + N
                xp.export_viz.step[viz_n]    = x.step
                xp.export_viz.episode[viz_n] = x.episode
                cursor += 1
                if not (cursor < BATCH+TUNNEL): break
            if cursor == BATCH: # no v>0, leave zeros
                break

        online_cloud = self.Q_online.predict( [batch_s[BATCH:], batch_a[BATCH:]] )
        cursor = BATCH
        with xp.replay_mutex:
            while cursor < BATCH+TUNNEL:
                t = online_cloud[cursor-BATCH] * self.GAMMA
                batch_t[cursor,0] = t
                viz_n = cursor-BATCH + N
                xp.export_viz.state1[viz_n] = batch_s[cursor]
                xp.export_viz.state2[viz_n] = batch_s[cursor]
                xp.export_viz.state_trans[viz_n] = batch_s[cursor]
                xp.export_viz.state_policy[viz_n] = batch_s[cursor]
                xp.export_viz.Vstable1[viz_n] = online_cloud[cursor-BATCH]
                xp.export_viz.Vstable2[viz_n] = t
                xp.export_viz.Vonline1[viz_n] = 0
                xp.export_viz.Vtarget[viz_n]  = 0
                xp.export_viz.Vpolicy[viz_n] = t
                cursor += 1
            xp.export_viz.N[0] = N + TUNNEL

        if dry_run:
            loss = self.Q_online.test_on_batch( [batch_s, batch_a], batch_t )
            #print("WIRES (test) %0.5f" % loss)
        else:
            loss = self.Q_online.train_on_batch( [batch_s, batch_a], batch_t )
            #print("WIRES %0.5f" % loss)
            test1 = self.Q_online.test_on_batch( [batch_s, batch_a], batch_t )
            loss = self.policy_value.train_on_batch(batch_s, batch_t)
            test2 = self.Q_online.test_on_batch( [batch_s, batch_a], batch_t )
            #print("POLICY %0.5f" % loss)
            self._slowly_transfer_weights_to_stable_network()

    def _save(self, fn):
        self.transition.model.save_weights(fn + "_transition.h5", overwrite=True)

    def _load(self, fn):
        self.transition.model.load_weights(fn + "_transition.h5")

    def _control(self, s, action_space):
        v1  = [action_space.sample() for x in range(50)]
        return v1[0]

    def _slowly_transfer_weights_to_stable_network(self):
        ws_online = self.Q_online.get_weights()
        with self.stable_mutex:
            ws_stable = self.Q_stable.get_weights()
        for arr_online, arr_stable in zip(ws_online, ws_stable):
            arr_stable *= (1-self.TAU)
            arr_stable += self.TAU*arr_online
        with self.stable_mutex:
            self.Q_stable.set_weights(ws_stable)
