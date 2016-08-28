import algo
import xp
import transition_model

import numpy as np
import keras
import keras.models
from keras import backend as K
from keras.regularizers import l2, activity_l1l2, activity_l1, activity_l2
from keras.layers.core import Dense, Lambda, Activation, ActivityRegularization
from keras.layers import Merge, merge, Input
from keras.models import Model
from keras.engine import Layer
from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
from keras.constraints import nonneg
from threading import Lock

#class Bell(Layer):
#    def __init__(self, **kwargs):
#        self.supports_masking = True
#        super(Bell, self).__init__(**kwargs)
#    def call(self, x, mask=None):
#        return K.exp(-x*x)
#        #return -K.abs(x)
#    def get_config(self):
#        config = {}
#        base_config = super(Bell, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))

CRASH_OR_WIN_THRESHOLD = 50   # pin value function if last state is win or crash, ignore bootstrap

def clamp_minus_one_plus_one(x):
    return K.minimum( +1, K.maximum(x, -1) )  # as opposed to min/max, minimum/maximum is element-wise operations

def gaussian_of_x(x):
    return K.sum( K.exp(-x*x), axis=-1, keepdims=True )
    #return K.sum(K.square(x), axis=-1, keepdims=True)

def gaussian_of_x_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = 1  # 320
    return tuple(shape)

class QNetPolicygrad(algo.Algorithm):
    def __init__(self, GAMMA=0.995, TAU=0.01, BATCH=100):
        algo.Algorithm.__init__(self, BATCH)
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.stable_mutex = Lock()
        self.online_mutex = Lock()

        clamp = Lambda(clamp_minus_one_plus_one)

        input_s = Input( shape=(xp.STATE_DIM,) )
        def qmodel():
            inp_s = Input( shape=(xp.STATE_DIM,),  name="inp_s")  # batch (None, ...) added automatically
            inp_a = Input( shape=(xp.ACTION_DIM,), name="inp_a")

            a1 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            a2 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            a3 = Dense(320, activation='relu', W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001))
            a_out = Dense(1, W_regularizer=l2(0.001), W_constraint=nonneg())

            #a_out = Dense(1, bias=False, W_constraint=nonneg())#, activity_regularizer=activity_l2(0.001))
            v1 = Dense(320, activation='relu', W_regularizer=l2(0.00001))
            v2 = Dense(320, activation='relu', W_regularizer=l2(0.00001))
            v3 = Dense(320, activation='relu', W_regularizer=l2(0.00001))
            v_out = Dense(1, W_regularizer=l2(0.00001))
            #, activity_regularizer=activity_l2(0.001)
            #a_bell = Bell()
            #wtf = ActivityRegularization(l2=0.01)

            if 1:
                gaussian = Lambda(gaussian_of_x, output_shape=gaussian_of_x_shape)
                parabolized_action = merge( [
                    a_out( a2(a1( merge([inp_s, inp_a], mode='concat') )) ),
                    #gaussian( merge([inp_s, inp_a], mode='concat') )
                    gaussian(inp_a)
                    ], mode='mul')
                out_tensor = merge( [
                    parabolized_action,
                    v_out( v2(v1(inp_s)) )
                    ], mode='sum' )
            else:
                out_tensor = a_out( a2(a1( merge([inp_s, inp_a], mode='concat') )) )
            #parabolized_value = merge( [
            #    v3(v2(v1(inp_s))),
            #    gaussian(inp_s)
            #    ], mode='mul')

            #out_tensor = a_out( a2(a1( merge([inp_s, inp_a], mode='concat') )) )

            #if self.mode in ['sum', 'mul', 'ave', 'max']:  'concat'
            #elif self.mode in ['dot', 'cos']:

            Qmod = Model( input=[inp_s,inp_a], output=out_tensor )
            Qmod.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))
            #return [v1,v2,v3,v_out, a1,a2,a3,a_out], Qmod
            return [v1,v2,v3,v_out, a1,a2,a3,a_out], Qmod

        stable_trainable, self.Q_stable = qmodel()
        online_trainable, self.Q_online = qmodel()

        for layer in stable_trainable:
            layer.trainable = False  # model already compiled (that's where this flag used), this assignment avoids learning by policy below

        def only_up(y_true, y_pred):
            return K.mean( -y_pred, axis=-1 )
        def close_to_previous_policy(act_previous, a_act_predicted):
            return 100*K.mean(K.square(a_act_predicted - act_previous), axis=-1)

        def policy_net(inp_s):
            d1 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            d2 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            d3 = Dense(320, activation='relu', W_regularizer=l2(0.001))
            out = Dense(xp.ACTION_DIM)
            out_action = clamp(out( d3(d2(d1(inp_s))) ))
            value_of_s = self.Q_stable( [inp_s,out_action] )
            action = Model( input=[inp_s], output=out_action )
            action.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_2=0.9999))  # really optimal values here
            value  = Model( input=[inp_s], output=[out_action,value_of_s] )
            value.compile(loss=[close_to_previous_policy,only_up], optimizer=Adam(lr=0.0005, beta_2=0.9999))
            return action, value
        self.stable_policy_action, self.stable_policy_value = policy_net(input_s)
        self.online_policy_action, self.online_policy_value = policy_net(input_s)

        self.trans = transition_model.Transition()

        self.countdown = 0
        self.demo_policy_tolearn = 0  #2000
        self.new_xp_counter = 5
        self.use_random_policy = True

    def _learn_demo_policy_supervised(self, buf, dry_run):
        batch_s = np.zeros( (self.BATCH, xp.STATE_DIM) )
        batch_a = np.zeros( (self.BATCH, xp.ACTION_DIM) )
        for i,x in enumerate(buf):
            batch_s[i] = x.s
            batch_a[i] = x.a
        with self.stable_mutex:
            loss = self.online_policy_action.train_on_batch( batch_s, batch_a )
            if self.demo_policy_tolearn==0:
                w = self.online_policy_action.get_weights()
                self.stable_policy_action.set_weights(w)
                print("copy to stable policy")
            elif self.demo_policy_tolearn % 100 == 0:
                print("%05i supervised demo learn %0.4f" % (self.demo_policy_tolearn, loss))

    def _learn_iteration(self, buf, dry_run):
        if self.demo_policy_tolearn > 0 and not dry_run:
            self.demo_policy_tolearn -= 1
            self._learn_demo_policy_supervised(buf, dry_run)
            return

        if self.use_random_policy:
            N = len(xp.replay)
            self.N = N
            if N > 1000:
                self.use_random_policy = False
                print("have %i random samples, start learning" % N)
            else:
                self.demo_policy_tolearn = 0  # random action taken, supervised demo learning not applicable
                return

        BATCH = len(buf)
        assert(self.BATCH==BATCH)

        batch_s = np.zeros( (BATCH, xp.STATE_DIM) )
        batch_a = np.zeros( (BATCH, xp.ACTION_DIM) )
        batch_t = np.zeros( (BATCH, 1) )

        for i,x in enumerate(buf):
            batch_s[i] = x.sn
        with self.stable_mutex:
            _, nv = self.stable_policy_value.predict(batch_s)
            _, pv = self.online_policy_value.predict(batch_s)
            pa = self.online_policy_action.predict(batch_s)
            ps = self.trans.predict(batch_s, pa)
            #nv2_a = self.stable_policy_action.predict(batch_s)
            #nv2_v = self.Q_stable.predict( [batch_s, nv2_a] )
        for i,x in enumerate(buf):
            x.nv = nv[i][0]
            x.pv = pv[i][0]
            x.pa = pa[i]
            x.ps = ps[i]

        # stable v
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
            self.global_maximum_reward = 0
            total_reward = 0
            v = 0
            episode = 0
            for i in range(N-1,-1,-1):
                x = xp.replay[i]
                if x.terminal:
                    v = 0
                    self.global_maximum_reward = max(self.global_maximum_reward, total_reward)
                    total_reward = 0
                    episode += 1
                x.episode = episode
                v = v*self.GAMMA + x.r
                x.wires_v = v
                if x.r>0: total_reward += x.r
            print "self.global_maximum_reward %s" % self.global_maximum_reward
            self.countdown = 20
        else:
            self.countdown -= 1

        # Viz
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                if not (x.terminal and np.abs(x.r) > CRASH_OR_WIN_THRESHOLD):    # not crash
                    bellman = x.nv*self.GAMMA + x.r
                    #bellman = min( bellman, x.stable_nv*(1/self.GAMMA) )
                    t = max(x.wires_v, bellman)
                    x.target_v = min(self.global_maximum_reward*1.05, t)
                    #x.target_v = v
                else:
                    x.target_v = x.r
                #x.target_v = x.nv*self.GAMMA + x.r
                batch_t[i,0] = x.target_v
                xp.export_viz.state1[x.viz_n] = x.s
                xp.export_viz.state2[x.viz_n] = x.sn
                xp.export_viz.Vstable1[x.viz_n] = x.v
                xp.export_viz.Vstable2[x.viz_n] = x.target_v
                xp.export_viz.Vonline1[x.viz_n] = 0
                xp.export_viz.Vtarget[x.viz_n]  = 0
                xp.export_viz.step[x.viz_n] = x.step
                xp.export_viz.episode[x.viz_n] = x.episode
                xp.export_viz.state_policy[x.viz_n] = x.ps
                xp.export_viz.Vpolicy[x.viz_n] = x.pv

        xp.export_viz.N[0] = self.N

        trans_loss = self.trans.learn_iteration(buf, dry_run)
        if dry_run:
            with self.online_mutex:
                loss = self.Q_online.test_on_batch( [batch_s, batch_a], batch_t )
            #print("WIRES (test) %0.5f" % loss)
        else:
            with self.online_mutex:
                wires_loss = self.Q_online.train_on_batch( [batch_s, batch_a], batch_t )
            #test1 = self.Q_stable.test_on_batch( [batch_s, batch_a], batch_t )
            with self.stable_mutex:
                stable_policy_a = self.stable_policy_action.predict(batch_s)
                policy_loss = self.online_policy_value.train_on_batch(batch_s, [stable_policy_a,batch_t])  # batch_t target not used, see only_up()
            #test2 = self.Q_stable.test_on_batch( [batch_s, batch_a], batch_t )
            #print("test1", test1, "test2", test2)
            #print("WIRES %0.4f POLICY %0.4f TRANS %0.4f" % (wires_loss, policy_loss, trans_loss))
            with self.online_mutex:
                self._slowly_transfer_weights_to_stable_network(self.Q_stable, self.Q_online, self.TAU)
            #self._slowly_transfer_weights_to_stable_network(self.stable_policy_value, self.online_policy_value, 0.0001)

    def _save(self, fn):
        self.Q_stable.save_weights(fn + "_qnet.h5", overwrite=True)
        self.stable_policy_action.save_weights(fn + "_policy.h5", overwrite=True)
        self.trans.model.save_weights(fn + "_trans.h5", overwrite=True)

    def _load(self, fn):
        self.Q_stable.load_weights(fn + "_qnet.h5")
        self.Q_online.load_weights(fn + "_qnet.h5")
        self.stable_policy_action.load_weights(fn + "_policy.h5")
        self.trans.model.load_weights(fn + "_trans.h5")
        self.demo_policy_tolearn = 0

    def _reset(self, new_experience):
        #self.heuristic_timeout = np.random.randint(low=0, high=100)
        if new_experience:
            if self.new_xp_counter > 0:
                self.new_xp_counter -= 1
                return
            #print "POLICY TO STABLE"
            with self.online_mutex:
                ws_online = self.online_policy_action.get_weights()
            with self.stable_mutex:
                self.stable_policy_action.set_weights(ws_online)

    def _control(self, s, action_space):
        if self.use_random_policy:
            return action_space.sample()
        with self.stable_mutex:
            a = self.online_policy_action.predict(s.reshape(1,xp.STATE_DIM))[0]
            #batch_s = np.zeros( (1, xp.STATE_DIM) )
            #batch_t = np.zeros( (1, 1) )
            #batch_s[0] = s
            #t1 = self.online_policy_value.predict(batch_s)
            #print "BEFORE", a
            #for c in range(50):
            #    policy_loss = self.online_policy_value.train_on_batch(batch_s, batch_t)
            #    print ("control policy_loss %0.5f" % policy_loss)
            #a = self.online_policy_action.predict(s.reshape(1,xp.STATE_DIM))[0]
            #t2 = self.online_policy_value.predict(batch_s)
            #print "VALUE %0.5f -> %0.5f" % (t1,t2)
        if 0:
            import gym.envs.box2d.bipedal_walker as w
            if self.heuristic_timeout > 0:
                self.heuristic_timeout -= 1
                if self.heuristic_timeout==0:
                    print("heuristic over")
                ah  = w.heuristic(self, s)
                return ah
        if 0:
            import gym.envs.box2d.lunar_lander as ll
            self.continuous = True
            a = ll.heuristic(self, s)
        return a

    def advantage_visualize(self, s, a, action_space):
        PIXELS = xp.ACTION_PIXELS
        batch_s = np.zeros( (PIXELS*xp.ACTION_DIM, xp.STATE_DIM) )
        batch_a = np.zeros( (PIXELS*xp.ACTION_DIM, xp.ACTION_DIM) )
        for i in range(xp.ACTION_DIM):
            pk = (action_space.high[i]-action_space.low[i]) / PIXELS
            for p in range(PIXELS):
                batch_s[i*PIXELS + p] = s
                batch_a[i*PIXELS + p] = a
                batch_a[i*PIXELS + p, i] = action_space.low[i] + p*pk
        with self.online_mutex:
            online_v = self.Q_online.predict( [batch_s, batch_a] )
        with self.stable_mutex:
            ast = self.stable_policy_action.predict(s.reshape(1,xp.STATE_DIM))[0]
            stable_v = self.Q_stable.predict( [batch_s, batch_a] )
        xp.export_viz.action_online[:] = a
        xp.export_viz.action_stable[:] = ast
        xp.export_viz.agraph_online[:] = online_v
        xp.export_viz.agraph_stable[:] = stable_v

    def _slowly_transfer_weights_to_stable_network(self, stable, online, TAU):
        ws_online = online.get_weights()
        with self.stable_mutex:
            ws_stable = stable.get_weights()
        for arr_online, arr_stable in zip(ws_online, ws_stable):
            arr_stable *= (1-TAU)
            arr_stable += TAU*arr_online
        with self.stable_mutex:
            stable.set_weights(ws_stable)

    def useful_to_think_more(self):
        return not self.use_random_policy
