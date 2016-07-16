import numpy as np

STATE_DIM   = 0  # Dimensionality of s, determined from environment
ACTION_DIM  = 0  # Dimensionality of a, actions represented as single-hot vector, i.e. [0,0,0,0,1,0,0] for 4-th action

from threading import Lock
replay_mutex = Lock()
replay = []
replay_shuffled = []
replay_important = []

def init_from_env(env):
    global STATE_DIM, ACTION_DIM
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]

class XPoint:
    'experience point'
    def __init__(self, s, a, r, sn=None):
        self.s = np.asarray(s)
        assert self.s.shape==(STATE_DIM,)
        self.a = np.asarray(a)
        assert self.a.shape==(ACTION_DIM,)
        self.r = r
        if sn is None: # None means terminal point, no next state
            self.sn = None
        else:
            self.sn = np.asarray(sn)
            assert self.sn.shape==(STATE_DIM,)
        self.y_ddqn = 0.0
        self.y_maxq = 0.0
        self.sampled_counter = 0
        self.important = True  # New experience is important by defauls
        self.terminal = False
        self.viz_n = 0
        self.jpeg = None

    def to_jsonable(self):
        j = { "s": self.s.tolist(), "a": self.a.tolist(), "r": self.r }
        if self.sn is not None: j["sn"] = self.sn.tolist()
        if self.jpeg is not None: j["jpeg"] = self.jpeg
        return j

import json

def load_lowlevel(fn):
    with open(fn) as f:
        j = json.loads(f.read());
    return [XPoint(**dict) for dict in j]

def save_lowlevel(fn, buf):
    with open(fn, "wb") as f:
        print >> f, json.dumps( [x.to_jsonable() for x in buf], indent=4, separators=(',', ': '))

def load(fn):
    global replay
    replay = load_lowlevel(fn)

def save(fn):
    save_lowlevel(fn, replay)

class ExportViz:
    def reopen(self, N, STATE_DIM):
        self.state1 = np.memmap(".vizdata/state1", mode="w+", shape=(N,STATE_DIM), dtype=np.float32)
        self.state2 = np.memmap(".vizdata/state2", mode="w+", shape=(N,STATE_DIM), dtype=np.float32)
        self.V      = np.memmap(".vizdata/V", mode="w+", shape=(N,), dtype=np.float32)

export_viz = None

def shuffle():
    global replay_shuffled, export_viz
    import random    
    del export_viz
    export_viz = ExportViz()
    with replay_mutex:
        N = len(replay)
        print "RESHUFFLE, EXPORT VIZ N=%i" % N
        export_viz.reopen(N, STATE_DIM)
        for n in range(N):
            replay[n].viz_n = n        
        replay_shuffled = replay[:]
        random.shuffle(replay_shuffled)

def batch(BATCH_SIZE):
    half_replay = len(replay_shuffled) // 2
    with replay_mutex:
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
    return buf

