import algo
import xp
import transition_model

import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.regularizers import l2, activity_l1l2, activity_l1, activity_l2
from keras.layers.core import Dense, Lambda
from keras.layers import Merge, merge, Input
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

def clamp_minus_one_plus_one(x):
    return K.minimum( +1, K.maximum(x, -1) )  # as opposed to min/max, minimum/maximum is element-wise operations

def parabola_of_x(x):
    return  (-0.01)*K.sum(K.square(x), axis=-1, keepdims=True)

def parabola_of_x_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = 1
    return tuple(shape)

class QNetPolicygrad(algo.Algorithm):
    def __init__(self, GAMMA=0.995, TAU=0.01, BATCH=100):
        algo.Algorithm.__init__(self, BATCH)
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.stable_mutex = Lock()

        nrm = Lambda(parabola_of_x, output_shape=parabola_of_x_shape)
        rst = Lambda(clamp_minus_one_plus_one)

        stable_inp_s = Input( shape=(xp.STATE_DIM,) )  # batch (None, ...) added automatically
        stable_inp_a = Input( shape=(xp.ACTION_DIM,) )
        stable_inp = merge( [stable_inp_s, rst(stable_inp_a)], mode='concat' )
        stable_d1 = Dense(320, activation='relu', W_regularizer=l2(0.001))
        stable_d2 = Dense(320, activation='relu', W_regularizer=l2(0.001))
        stable_d3 = Dense(320, W_regularizer=l2(0.001))
        stable_bell = Bell()
        stable_qout = Dense(1, bias=False)

        online_inp_s = Input( shape=(xp.STATE_DIM,) )
        online_inp_a = Input( shape=(xp.ACTION_DIM,) )
        online_inp = merge( [online_inp_s, rst(online_inp_a)], mode='concat' )
        online_d1 = Dense(320, activation='relu', W_regularizer=l2(0.001))
        online_d2 = Dense(320, activation='relu', W_regularizer=l2(0.001))
        online_d3 = Dense(320, W_regularizer=l2(0.001))
        online_bell = Bell()
        online_qout = Dense(1, bias=False)

        stable_out_tensor = merge( [
            stable_qout(stable_bell( stable_d3(stable_d2(stable_d1(stable_inp))) )),
            nrm(stable_inp)
            ], mode='sum' )
        online_out_tensor = merge( [
            online_qout(online_bell( online_d3(online_d2(online_d1(online_inp))) )),
            nrm(online_inp)
            ], mode='sum' )

        self.Q_online = Model( input=[online_inp_s,online_inp_a], output=online_out_tensor )
        self.Q_stable = Model( input=[stable_inp_s,stable_inp_a], output=stable_out_tensor )

        self.Q_online.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))
        self.Q_stable.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))

        stable_d1.trainable = False
        stable_d2.trainable = False
        stable_d3.trainable = False
        stable_qout.trainable = False

        # activity_regularizer=activity_l2(0.001))
        def only_up(y_true, y_pred):
            return K.mean( -y_pred, axis=-1 )
        def policy_net(inp_s):
            d1 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            d2 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            d3 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            out = Dense(xp.ACTION_DIM)
            out_tensor = out( d3(d2(d1(inp_s))) )
            value_of_s = self.Q_stable( [inp_s,out_tensor] )
            action = Model( input=[inp_s], output=out_tensor )
            action.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))  # really optimal values here
            value  = Model( input=[inp_s], output=value_of_s )
            value.compile(loss=only_up, optimizer=Adam(lr=0.0005, beta_2=0.9999))
            return action, value
        input_s = Input( shape=(xp.STATE_DIM,) )
        self.stable_policy_action, self.stable_policy_value = policy_net(input_s)
        self.online_policy_action, self.online_policy_value = policy_net(input_s)

        self.countdown = 0
        self.demo_policy_tolearn = 2000

    def _learn_demo_policy(self, buf, dry_run):
        batch_s = np.zeros( (self.BATCH, xp.STATE_DIM) )
        batch_a = np.zeros( (self.BATCH, xp.ACTION_DIM) )
        for i,x in enumerate(buf):
            batch_s[i] = x.s
            batch_a[i] = x.a
        loss = self.online_policy_action.train_on_batch( batch_s, batch_a )
        if self.demo_policy_tolearn==0:
            w = self.online_policy_action.get_weights()
            self.stable_policy_action.set_weights(w)
            print "COPY"
        else:
            print loss

    def _learn_iteration(self, buf, dry_run):
        if self.demo_policy_tolearn > 0 and not dry_run:
            self.demo_policy_tolearn -= 1
            self._learn_demo_policy(buf, dry_run)
            return
        #self.transition.learn_iteration(buf, dry_run)

        BATCH = len(buf)
        assert(self.BATCH==BATCH)

        batch_s = np.zeros( (BATCH, xp.STATE_DIM) )
        batch_a = np.zeros( (BATCH, xp.ACTION_DIM) )
        batch_t = np.zeros( (BATCH, 1) )

        for i,x in enumerate(buf):
            batch_s[i] = x.sn
        with self.stable_mutex:
            nv = self.stable_policy_value.predict(batch_s)
            #nv2_a = self.stable_policy_action.predict(batch_s)
            #nv2_v = self.Q_stable.predict( [batch_s, nv2_a] )
        for i,x in enumerate(buf):
            x.nv = nv[i][0]
            #print "action", nv2_a[i]
            #print "value1", nv[i][0]
            #print "value2", nv2_v[i][0]

        for i,x in enumerate(buf):
            batch_s[i] = x.s
            batch_a[i] = x.a
        with self.stable_mutex:
            stable_v = self.Q_stable.predict( [batch_s, batch_a] )
        for i,x in enumerate(buf):
            x.v = stable_v[i][0]

        # WIRES
        if self.countdown==0:
            N = len(xp.replay)
            self.N = N
            v = 0
            episode = 0
            #for i,x in enumerate(buf):
            for i in range(N-1,-1,-1):
                x = xp.replay[i]
                if x.terminal:
                    v = 0
                    episode += 1
                x.episode = episode
                v = v*self.GAMMA + x.r
                x.wires_v = v
            self.countdown = 20
        else:
            self.countdown -= 1

        # Viz
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                if not (x.terminal and x.r < CRASH_THRESHOLD):    # not crash
                    #x.target_v = x.nv*self.GAMMA + x.r
                    x.target_v = max(x.wires_v, x.nv*self.GAMMA + x.r)
                    #v = max(x.target_v, x.v)                     # x.v is stable v
                    #x.target_v = v
                else:
                    x.target_v = x.r
                if x.jpeg is not None and x.jpeg.find("27170") != -1:
                    print x.__dict__
                    print "r", x.r, "x.target_v", x.target_v
                #x.target_v = x.nv*self.GAMMA + x.r
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
                xp.export_viz.Vpolicy[x.viz_n] = x.target_v

        xp.export_viz.N[0] = self.N

        if dry_run:
            loss = self.Q_online.test_on_batch( [batch_s, batch_a], batch_t )
            #print("WIRES (test) %0.5f" % loss)
        else:
            wires_loss = self.Q_online.train_on_batch( [batch_s, batch_a], batch_t )
            #test1 = self.Q_stable.test_on_batch( [batch_s, batch_a], batch_t )
            policy_loss = self.online_policy_value.test_on_batch(batch_s, batch_t)
            #test2 = self.Q_stable.test_on_batch( [batch_s, batch_a], batch_t )
            #print("test1", test1, "test2", test2)
            print("WIRES %0.2f POLICY %0.5f" % (wires_loss, policy_loss))
            self._slowly_transfer_weights_to_stable_network(self.Q_stable, self.Q_online, self.TAU)
            self._slowly_transfer_weights_to_stable_network(self.stable_policy_value, self.online_policy_value, 0.0001)

    def _save(self, fn):
        self.transition.model.save_weights(fn + "_transition.h5", overwrite=True)

    def _load(self, fn):
        self.transition.model.load_weights(fn + "_transition.h5")

    def _control(self, s, action_space, flag):
        a = self.stable_policy_action.predict(s.reshape(1,xp.STATE_DIM))[0]
        import gym.envs.box2d.bipedal_walker as w
        ah  = w.heuristic(self, s)
        print "a", a
        print "h", ah
        if flag: return a
        return ah

    def _slowly_transfer_weights_to_stable_network(self, stable, online, TAU):
        ws_online = online.get_weights()
        with self.stable_mutex:
            ws_stable = stable.get_weights()
        for arr_online, arr_stable in zip(ws_online, ws_stable):
            arr_stable *= (1-TAU)
            arr_stable += TAU*arr_online
        with self.stable_mutex:
            stable.set_weights(ws_stable)
