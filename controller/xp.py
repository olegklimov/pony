import numpy as np

STATE_DIM   = 0  # Dimensionality of s, determined from environment
ACTION_DIM  = 0  # Dimensionality of a, actions represented as single-hot vector, i.e. [0,0,0,0,1,0,0] for 4-th action

from threading import Lock
replay_mutex = Lock()
replay = []
replay_shuffled = []
replay_important = []
epoch = 0.0
epoch_sample_counter = 0

def init_from_env(env):
    global STATE_DIM, ACTION_DIM
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]

class XPoint:
    'experience point'
    def __init__(self, s, a, r, sn, step, terminal=False, jpeg=None):
        self.s = np.asarray(s)
        assert self.s.shape==(STATE_DIM,)
        self.a = np.asarray(a)
        assert self.a.shape==(ACTION_DIM,)
        self.r = r
        self.sn = np.asarray(sn)
        assert self.sn.shape==(STATE_DIM,)
        self.y_ddqn = 0.0
        self.y_maxq = 0.0
        self.step = int(step)
        self.sampled_counter = 0
        self.important = True
        assert isinstance(terminal, bool)
        self.terminal = terminal
        self.viz_n = 0
        self.jpeg = jpeg

        self.v = 0
        self.nv = 0

    def to_jsonable(self):
        j = { "s": self.s.tolist(), "sn": self.sn.tolist(), "a": self.a.tolist(), "r": self.r, "step": self.step }
        if self.jpeg is not None: j["jpeg"] = self.jpeg
        if self.terminal: j["terminal"] = True
        return j

import json

def load_lowlevel(fn):
    with open(fn) as f:
        j = json.loads(f.read());
    global STATE_DIM, ACTION_DIM
    if STATE_DIM==0:
        STATE_DIM  = len(j[0]["s"])
        ACTION_DIM = len(j[0]["a"])
        print("Load xp autodetected STATE_DIM={}, ACTION_DIM={}".format(STATE_DIM,ACTION_DIM))
    return [XPoint(**dict) for dict in j]

def save_lowlevel(fn, buf):
    with open(fn, "wb") as f:
        print >> f, json.dumps( [x.to_jsonable() for x in buf], indent=4, separators=(',', ': '))

def load(fn):
    global replay
    with replay_mutex:
        replay = load_lowlevel(fn)

def save(fn):
    with replay_mutex:
        save_lowlevel(fn, replay)

class ExportViz:
    def reopen(self, dir, N, STATE_DIM, mode):
        self.N        = np.memmap(dir+"/N", mode=mode, shape=(1), dtype=np.int32)
        self.state1   = np.memmap(dir+"/state1",  mode=mode, shape=(N,STATE_DIM), dtype=np.float32)
        self.state2   = np.memmap(dir+"/state2",  mode=mode, shape=(N,STATE_DIM), dtype=np.float32)
        self.state_trans  = np.memmap(dir+"/state_transition", mode=mode, shape=(N,STATE_DIM), dtype=np.float32)
        self.state_policy = np.memmap(dir+"/state_policy", mode=mode, shape=(N,STATE_DIM), dtype=np.float32)
        self.Vstable1 = np.memmap(dir+"/Vstable1", mode=mode, shape=(N,), dtype=np.float32)
        self.Vstable2 = np.memmap(dir+"/Vstable2", mode=mode, shape=(N,), dtype=np.float32)
        self.Vonline1 = np.memmap(dir+"/Vonline1", mode=mode, shape=(N,), dtype=np.float32)
        self.Vtarget  = np.memmap(dir+"/Vtarget", mode=mode, shape=(N,), dtype=np.float32)
        self.Vpolicy  = np.memmap(dir+"/Vpolicy", mode=mode, shape=(N,), dtype=np.float32)
        self.step     = np.memmap(dir+"/step",    mode=mode, shape=(N,), dtype=np.int32)
        self.episode  = np.memmap(dir+"/episode", mode=mode, shape=(N,), dtype=np.int32)
        self.jpeg     = np.memmap(dir+"/jpegmap", mode=mode, shape=(N*16,), dtype=np.int8)

export_viz = None

def export_viz_open(dir, mode="w+"):
    'with replay_mutex'
    N = len(replay)
    #print "EXPORT VIZ N=%i" % N
    global export_viz
    export_viz = None
    v = ExportViz()
    v.reopen(dir, N, STATE_DIM, mode)
    export_viz = v
    v.N[0] = N
    for x in replay:
        if x.v: continue
        export_viz.state1[x.viz_n] = x.s
        export_viz.state2[x.viz_n] = x.sn
        export_viz.state_trans[x.viz_n]  = x.s
        export_viz.state_policy[x.viz_n] = x.sn
        export_viz.Vtarget[x.viz_n]  = x.r
        export_viz.Vonline1[x.viz_n] = x.r
        export_viz.Vstable1[x.viz_n] = x.r
        export_viz.Vstable2[x.viz_n] = x.r
        export_viz.Vpolicy[x.viz_n]  = x.r
        export_viz.step[x.viz_n] = x.step
        if x.jpeg:
            import os
            j = x.jpeg[len(dir)+1:]
            for c in range(len(j)):
                assert c < 15
                export_viz.jpeg[x.viz_n*16 + c] = ord(j[c])
        else:
            export_viz.jpeg[x.viz_n*16 + 0] = 0

def shuffle():
    'with replay_mutex'
    global replay_shuffled
    import random
    N = len(replay)
    for n in range(N):
        replay[n].viz_n = n
    replay_shuffled = replay[:]
    random.shuffle(replay_shuffled)

def batch(BATCH_SIZE):
    with replay_mutex:
        half_replay = len(replay_shuffled) // 2
        buf = []
        while len(buf) < BATCH_SIZE:
            x = replay_shuffled.pop(0)
            buf.append(x)
            #if not x.important and len(buf) < BATCH_SIZE//3: # half will be important samples
                #print x.sampled_counter
            #    continue
            x.sampled_counter += 1
            #if x.important:
            #    replay_shuffled[half_replay:half_replay] = [x]
                #print len(replay_shuffled)
                #replay_shuffled.append(x)
                #replay_shuffled.insert(half_replay, x)
            #else:
            replay_shuffled.append(x)
        #print [x.sampled_counter for x in buf], len(replay_shuffled)
        assert( len(buf)==BATCH_SIZE )
        global epoch_sample_counter, epoch
        epoch_sample_counter += BATCH_SIZE
        epoch = float(epoch_sample_counter) / len(replay)
    return buf
