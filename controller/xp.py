import numpy as np, time

STATE_DIM   = 0  # Dimensionality of s, determined from environment
ACTION_DIM  = 0  # Dimensionality of a, actions represented as single-hot vector, i.e. [0,0,0,0,1,0,0] for 4-th action
ACTION_PIXELS = 100

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
    print "a"

class XPoint:
    'experience point'
    def __init__(self, s, a, r, sn, step, terminal=False, jpeg=None):
        self.s = np.asarray(s)
        assert self.s.shape==(STATE_DIM,)
        self.a = np.asarray(a)
        assert self.a.shape==(ACTION_DIM,)
        self.r = float(r)
        self.sn = np.asarray(sn)
        assert self.sn.shape==(STATE_DIM,)
        self.wires_v = 0.0
        self.episode = 1
        self.step = int(step)
        self.sampled_counter = 0
        self.important = True
        assert isinstance(terminal, bool)
        self.terminal = terminal
        self.viz_n = 0
        self.jpeg = jpeg

        self.v = 0
        self.vn = 0
        self.bellman = 1

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
        assert STATE_DIM>0
        assert ACTION_DIM>0
        self.N  = np.memmap(dir+"/mmap_N", mode=mode, shape=(1), dtype=np.int32)
        more_N = N + 1000
        self.s  = np.memmap(dir+"/mmap_s",  mode=mode, shape=(more_N,STATE_DIM), dtype=np.float32)
        self.v  = np.memmap(dir+"/mmap_v",  mode=mode, shape=(more_N,), dtype=np.float32)
        self.sn = np.memmap(dir+"/mmap_sn", mode=mode, shape=(more_N,STATE_DIM), dtype=np.float32)
        self.vn = np.memmap(dir+"/mmap_vn", mode=mode, shape=(more_N,), dtype=np.float32)
        self.sp = np.memmap(dir+"/mmap_sp", mode=mode, shape=(more_N,STATE_DIM), dtype=np.float32)
        self.vp = np.memmap(dir+"/mmap_vp", mode=mode, shape=(more_N,), dtype=np.float32)
        self.st = np.memmap(dir+"/mmap_st", mode=mode, shape=(more_N,STATE_DIM), dtype=np.float32)
        self.vt = np.memmap(dir+"/mmap_vt", mode=mode, shape=(more_N,), dtype=np.float32)
        self.ttest    = np.memmap(dir+"/mmap_ttest",    mode=mode, shape=(more_N,STATE_DIM), dtype=np.float32)
        self.step     = np.memmap(dir+"/mmap_step",     mode=mode, shape=(more_N,), dtype=np.int32)
        self.episode  = np.memmap(dir+"/mmap_episode",  mode=mode, shape=(more_N,), dtype=np.int32)
        self.jpeg     = np.memmap(dir+"/mmap_jpegmap",  mode=mode, shape=(more_N*16,), dtype=np.int8)
        self.flags    = np.memmap(dir+"/mmap_flags",    mode=mode, shape=(more_N,), dtype=np.int32)
        self.action_online = np.memmap(dir+"/mmap_action_online", mode=mode, shape=(ACTION_DIM,), dtype=np.float32)
        self.action_stable = np.memmap(dir+"/mmap_action_stable", mode=mode, shape=(ACTION_DIM,), dtype=np.float32)
        self.agraph_online = np.memmap(dir+"/mmap_agraph_online", mode=mode, shape=(ACTION_PIXELS*ACTION_DIM,1), dtype=np.float32)
        self.agraph_stable = np.memmap(dir+"/mmap_agraph_stable", mode=mode, shape=(ACTION_PIXELS*ACTION_DIM,1), dtype=np.float32)

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
        export_viz.s[x.viz_n]  = x.s
        export_viz.v[x.viz_n]  = x.r
        export_viz.sn[x.viz_n] = x.sn
        export_viz.vn[x.viz_n] = x.r
        export_viz.sp[x.viz_n] = x.sn
        export_viz.vp[x.viz_n] = x.r
        export_viz.st[x.viz_n] = x.sn
        export_viz.vt[x.viz_n] = x.r
        export_viz.step[x.viz_n] = x.step
        if x.jpeg:
            import os
            j = x.jpeg[x.jpeg.find('/')+1:]
            assert len(j) <= 15, "'%s' too long" % j
            for c in range(len(j)):
                export_viz.jpeg[x.viz_n*16 + c] = ord(j[c])
            export_viz.jpeg[x.viz_n*16 + len(j)] = 0
        else:
            export_viz.jpeg[x.viz_n*16 + 0] = 0

def shuffle():
    'with replay_mutex'
    global replay_shuffled, replay_idx
    import random
    N = len(replay)
    bellman_weights = np.ones( shape=(N,) )
    for n in range(N):
        replay[n].viz_n = n
        replay[n].bellman_weights = bellman_weights
        replay[n].bellman_weights[n] = replay[n].bellman
    replay_shuffled = replay[:]
    random.shuffle(replay_shuffled)
    replay_idx = np.arange(N, dtype=np.int32)

def batch(BATCH_SIZE):
    with replay_mutex:
        t0 = time.time()
        N = len(replay_shuffled)
        if N==0: return []
        half_replay = N // 2
        buf = []
        go_round_size = BATCH_SIZE * 3 // 4

        while len(buf) < go_round_size:
            x = replay_shuffled.pop(0)
            buf.append(x)
            x.sampled_counter += 1
            replay_shuffled.append(x)

        t00 = time.time()
        bellman_proportional_size = BATCH_SIZE - go_round_size
        bellman_sum = np.sum(replay[0].bellman_weights)
        probabilities = replay[0].bellman_weights / bellman_sum
        t1 = time.time()
        choice = np.random.choice( replay_idx, size=bellman_proportional_size, replace=False, p=probabilities )
        t2 = time.time()
        buf += [replay[x] for x in choice]
        t3 = time.time()
        #print "time choice=%0.2fms []=%0.2fms part2=%0.2fms total=%0.2fms" % (1000*(t2-t1), 1000*(t3-t2), 1000*(t3-t00), 1000*(t3-t0))

        assert( len(buf)==BATCH_SIZE )
        global epoch_sample_counter, epoch
        epoch_sample_counter += BATCH_SIZE
        epoch += float(go_round_size) / len(replay)

    return buf
