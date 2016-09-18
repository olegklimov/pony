import algo
import xp
import transition_model

import numpy as np, time
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
from sklearn.neighbors import BallTree

CRASH_OR_WIN_THRESHOLD = 50   # pin value function if last state is win or crash, ignore bootstrap
ACTION_DISPERSE = 0.3

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
    def __init__(self, dir, experiment_name, GAMMA=0.99, TAU=0.01, BATCH=100):
        algo.Algorithm.__init__(self, dir, experiment_name, BATCH)
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.stable_mutex = Lock()
        self.online_mutex = Lock()

        clamp = Lambda(clamp_minus_one_plus_one)

        input_s = Input( shape=(xp.STATE_DIM,) )
        def qmodel():
            inp_s = Input( shape=(xp.STATE_DIM,),  name="inp_s")  # batch (None, ...) added automatically
            inp_a = Input( shape=(xp.ACTION_DIM,), name="inp_a")

            a1 = Dense(640, activation='relu', W_regularizer=l2(0.01))
            a2 = Dense(640, activation='relu', W_regularizer=l2(0.01))
            #a3 = Dense(320, activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01))
            a_out = Dense(1, W_regularizer=l2(0.01)) #W_constraint=nonneg())

            v1 = Dense(640, activation='relu', W_regularizer=l2(0.01))
            v2 = Dense(640, activation='relu', W_regularizer=l2(0.01))
            v_out = Dense(1, W_regularizer=l2(0.01))

            gaussian = Lambda(gaussian_of_x, output_shape=gaussian_of_x_shape)
            parabolized_action = merge( [
                a_out( a2(a1( merge([inp_s, inp_a], mode='concat') )) ),
                gaussian(inp_a)
                ], mode='mul')
            out_tensor = a_out( a2(a1( merge([inp_s, inp_a], mode='concat') )) )
            #out_tensor = merge( [
            #    parabolized_action,
            #    v_out( v2(v1(inp_s)) )
            #    ], mode='sum' )

            Qmod = Model( input=[inp_s,inp_a], output=out_tensor )
            Qmod.compile(loss='mse', optimizer=Adam(lr=0.0001, beta_2=0.9999))
            return [v1,v2,v_out, a1,a2,a_out], Qmod

        stable_trainable, self.Q_stable = qmodel()
        online_trainable, self.Q_online = qmodel()

        for layer in stable_trainable:
            layer.trainable = False  # model already compiled (that's where this flag used), this assignment avoids learning Q layers by learning policy

        self.stable_policy_action, self.stable_policy_value = self.policy_net(input_s)
        self.online_policy_action, self.online_policy_value = self.policy_net(input_s)

        self.trans = transition_model.Transition()

        self.countdown = 0
        self.demo_policy_tolearn = 0  #2000

    def policy_net(self, inp_s):
        clamp = Lambda(clamp_minus_one_plus_one)
        def only_up(y_true, y_pred):
            return K.mean( -y_pred, axis=-1 )
        def close_to_previous_policy(act_previous, a_act_predicted):
            return 0.000*K.mean(K.square(a_act_predicted - act_previous), axis=-1)
        d1 = Dense(320, activation='relu', W_regularizer=l2(0.01))
        d2 = Dense(320, activation='relu', W_regularizer=l2(0.01))
        #d3 = Dense(320, activation='relu', W_regularizer=l2(0.001))
        out = Dense(xp.ACTION_DIM, W_regularizer=l2(0.01))
        out_action = clamp(out( d2(d1(inp_s)) ))
        value_of_s = self.Q_stable( [inp_s,out_action] )    # Q_stable here! (only self. in function)
        action = Model( input=[inp_s], output=out_action )
        action.compile(loss='mse', optimizer=Adam(lr=0.0001, beta_2=0.9999))  # really optimal values here
        value  = Model( input=[inp_s], output=[out_action,value_of_s] )
        value.compile(loss=[close_to_previous_policy,only_up], optimizer=Adam(lr=0.0005, beta_2=0.9999))
        return action, value

    def export_policy(self):
        input_s = Input( shape=(xp.STATE_DIM,) )
        pi, _ = self.policy_net(input_s)
        with self.stable_mutex:
            w = self.online_policy_action.get_weights()
            pi.set_weights(w)
        def return_action(s):
            return pi.predict(s.reshape(1,xp.STATE_DIM))[0]
        return return_action

    def _learn_demo_policy_supervised(self, buf, dry_run):
        s = np.zeros( (self.BATCH, xp.STATE_DIM) )
        batch_a = np.zeros( (self.BATCH, xp.ACTION_DIM) )
        for i,x in enumerate(buf):
            s[i] = x.s
            batch_a[i] = x.a
        with self.stable_mutex:
            loss = self.online_policy_action.train_on_batch( s, batch_a )
            if self.demo_policy_tolearn==0:
                w = self.online_policy_action.get_weights()
                self.stable_policy_action.set_weights(w)
                print("copy to stable policy")
            elif self.demo_policy_tolearn % 100 == 0:
                print("%05i supervised demo learn %0.4f" % (self.demo_policy_tolearn, loss))

    def _test_still_need_random_policy(self):
        if self.use_random_policy:
            N = len(xp.replay)
            self.N = N
            if N > 1000:
                self.use_random_policy = False
                print("have %i random samples, start learning" % N)
            else:
                self.demo_policy_tolearn = 0  # random action taken, supervised demo learning not applicable

    def _learn_iteration(self, buf, dry_run):
        t0 = time.time()
        if self.demo_policy_tolearn > 0 and not dry_run:
            self.demo_policy_tolearn -= 1
            self._learn_demo_policy_supervised(buf, dry_run)
            return
        if self.use_random_policy: return

        B = len(buf)
        assert(self.BATCH==B)

        ## go ##

        STATE_DIM = xp.STATE_DIM
        s  = np.zeros( (2*B, STATE_DIM) )
        a  = np.zeros( (2*B, xp.ACTION_DIM) )
        sn = np.zeros( (2*B, STATE_DIM) )
        vt = np.zeros( (2*B, 1) )
        sample_weight = np.ones( (2*B,) )

        for i,x in enumerate(buf):
            s[i]   = x.s
            a[i]   = x.a
            sn[i]  = x.sn
            s[B+i] = x.s
        with self.stable_mutex:
            v = self.Q_stable.predict( [s[:B],a[:B]] )
            an, vn  = self.online_policy_value.predict(sn[:B]) # action at sn, uses Q_stable inside
            apolicy = self.online_policy_action.predict(s[B:]) # action at s
            apolicy_noisy = np.clip(apolicy, -1, +1) + np.random.uniform(low=-ACTION_DISPERSE, high=ACTION_DISPERSE, size=(B,xp.ACTION_DIM))
            apolicy_noisy = np.clip(apolicy_noisy, -1, +1)
            vpolicy = self.Q_stable.predict( [s[B:], apolicy_noisy] )
            phys_sn, phys_r  = self.trans.predict(s[B:], apolicy_noisy)  # predicted state and reward from transition model
            phys_an, phys_vn = self.online_policy_value.predict(phys_sn) # action at phys_sn, uses Q_stable inside
        t1 = time.time()

        N = len(xp.replay)
        self.N = N

        # target and viz
        X = np.zeros( (1,STATE_DIM+xp.ACTION_DIM,)  )
        bellman_term = 0.0
        bellman_term_n = 0.01
        bellman_phys = 0.0
        bellman_phys_n = 0.01
        bellman_qlrn = 0.0
        bellman_qlrn_n = 0.01
        with xp.replay_mutex:
            for i,x in enumerate(buf):
                # Can we trust vn = Q(sn, an) ?
                X[0][:STATE_DIM] = x.sn
                X[0][STATE_DIM:] = an[i]
                #count = self.neighbours.query_radius(X, r=0.1, count_only=True)[0]
                #count = 0
                #physics = np.random.randint(count+0, count+10) < 5
                #physics = True
                stuck   = np.linalg.norm(x.s - x.sn) < 0.01
                #print "count=%i, physics=%s, stuck=%i" % (count, physics, stuck)
                if x.terminal and np.abs(x.r) > CRASH_OR_WIN_THRESHOLD:
                    # crash or win
                    a[B+i]    = apolicy_noisy[i]
                    vt[i,0]   = x.r
                    vt[B+i,0] = x.r
                    stuck = False
                    sample_weight[i]   = 1.0
                    sample_weight[B+i] = 1.0
                    bellman = abs(vt[i,0] - vpolicy[i,0])
                    bellman_term += bellman
                    bellman_term_n += 1
                else:
                    # Q-learning
                    vt[i,0] = x.r + self.GAMMA*vn[i,0]
                    bellman = abs(vt[i,0] - v[i,0])
                    bellman_qlrn += bellman
                    bellman_qlrn_n += 1
                    sample_weight[i]   = 1.0
                    # Physics model
                    a[B+i]    = apolicy_noisy[i]
                    vt[B+i,0] = phys_r[i,0] + self.GAMMA*phys_vn[i,0]
                    bellman         = abs(vt[B+i,0] - vpolicy[i,0])
                    bellman_phys   += bellman
                    bellman_phys_n += 1
                    sample_weight[B+i] = 1.0

                f = 0
                #if physics: f |= 1
                if stuck:   f |= 2
                if stuck:   sample_weight[i] = 0.0

                xp.export_viz.s[x.viz_n]  = x.s

                xp.export_viz.vs[x.viz_n] = v[i,0]
                xp.export_viz.sn[x.viz_n] = x.sn
                xp.export_viz.vn[x.viz_n] = vn[i,0]
                xp.export_viz.vn_targ[x.viz_n] = vt[i,0]

                xp.export_viz.phys_vs[x.viz_n] = vpolicy[i]
                xp.export_viz.phys_sn[x.viz_n] = phys_sn[i]
                xp.export_viz.phys_vn[x.viz_n] = phys_vn[i,0]
                xp.export_viz.phys_vn_targ[x.viz_n] = vt[B+i,0]

                xp.export_viz.flags[x.viz_n] = f
                xp.export_viz.step[x.viz_n] = x.step
                xp.export_viz.episode[x.viz_n] = x.episode
                x.bellman_weights[x.viz_n] = bellman
        t2 = time.time()

        xp.export_viz.N[0] = self.N

        transition_loss, reward_loss = self.trans.learn_iteration(buf, dry_run)
        if dry_run:
            with self.online_mutex:
                loss = self.Q_online.test_on_batch( [s, a], vt, sample_weight=sample_weight  )
            policy_loss = 0.0
        else:
            with self.online_mutex:
                loss = self.Q_online.train_on_batch( [s, a], vt, sample_weight=sample_weight )
            #test1 = self.Q_stable.test_on_batch( [s, a], vt )
            with self.stable_mutex:
                policy_loss = self.online_policy_value.train_on_batch(s[B:], [apolicy,vt[B:]])  # vt target not used, see only_up(); apolicy used in close_to_previous_policy; also not apolicy_noisy.
                policy_loss = float(policy_loss[0])   # [loss, close_to_previous_policy, only_up], print self.online_policy_value.metrics_names
            #test2 = self.Q_stable.test_on_batch( [s, a], vt )
            #print("test1", test1, "test2", test2)  # test2 must be equal to test1, otherwise we're learning not only policy, but Q too.
            #print("WIRES %0.4f POLICY %0.4f TRANS %0.4f" % (loss, policy_loss, trans_loss))
            with self.online_mutex:
                self._slowly_transfer_weights_to_stable_network(self.Q_stable, self.Q_online, self.TAU)
        assert isinstance(bellman_qlrn + bellman_phys + bellman_term, float)
        assert isinstance(transition_loss, float), type(transition_loss)
        assert isinstance(policy_loss, float), type(policy_loss)
        t3 = time.time()
        print "time eval=%0.2fms target=%0.2fms learn=%0.2fms total=%0.2fms" % (1000*(t1-t0), 1000*(t2-t1), 1000*(t3-t2), 1000*(t3-t0))

        return [transition_loss, reward_loss, bellman_term/bellman_term_n, bellman_phys/bellman_phys_n, bellman_qlrn/bellman_qlrn_n]

    nameof_losses = ["transition", "reward", "bellman_term", "bellman_phys", "bellman_qlrn"]
    nameof_runind = ["score", "runtime"]

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

    def load_something_useful_on_start(self, fn):
        try: self.trans.model.load_weights(fn + "_trans.h5")
        except Exception, e: print("cannot load transition model weights: %s" % e)

    def _reset(self, new_experience):
        if new_experience: # have xp.replay_mutex locked if true
            if not self.pause:
                #print "POLICY TO STABLE"
                with self.online_mutex:
                    ws_online = self.online_policy_action.get_weights()
                with self.stable_mutex:
                    self.stable_policy_action.set_weights(ws_online)
            self._neighbours_reset()

    def _neighbours_reset(self):
        "xp.replay_mutex must be locked"
        count = len(xp.replay)
        if count > 0:
            #print "BALL TREE %i POINTS" % count
            STATE_DIM = xp.STATE_DIM
            ACTION_DIM = xp.ACTION_DIM
            X = np.zeros( (count, STATE_DIM+ACTION_DIM)  )
            for i in range(count):
                X[i][:STATE_DIM] = xp.replay[i].s
                X[i][STATE_DIM:] = xp.replay[i].a
            #print "--"
            self.neighbours = BallTree(X)
            #print "/BALL TREE"

    def _control(self, s, action_space):
        if self.use_random_policy:
            return action_space.sample()

        with self.stable_mutex:
            a = self.online_policy_action.predict(s.reshape(1,xp.STATE_DIM))[0]

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

        if np.random.randint(0,5) == 0:
            t = np.argmax( online_v )
            i = t // PIXELS
            p = t  % PIXELS
            # np.max( online_v ) == online_v[i*PIXELS + p]
            a = batch_a[i*PIXELS + p]
        else:
            a  = np.clip(a, -1, +1)
            a += np.random.normal(scale=ACTION_DISPERSE, size=(xp.ACTION_DIM,))
            a  = np.clip(a, -1, +1)

        return a

    def _slowly_transfer_weights_to_stable_network(self, stable, online, TAU):
        ws_online = online.get_weights()
        with self.stable_mutex:
            ws_stable = stable.get_weights()
        for arr_online, arr_stable in zip(ws_online, ws_stable):
            arr_stable *= (1-TAU)
            arr_stable += TAU*arr_online
        with self.stable_mutex:
            stable.set_weights(ws_stable)

