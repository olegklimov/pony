import numpy as np
import json, os

class Progress:
    def __init__(self, dir, task_name, desc, losses, runind):
        self.task_name = task_name
        self.dir = dir
        self.N = 0
        self.T = 0
        self.allocated_n = 0
        self.allocated_t = 0
        self.json_fn = dir + "/" + task_name + ".json"
        self.mapped_log_fn = dir+"/"+task_name+".log"
        self.mapped_tst_fn = dir+"/"+task_name+".tst"
        self.mapped_NT_fn   = dir+"/"+task_name+".NT"
        self.mapped_NT = None
        self.mapped_log = None
        self.mapped_tst = None
        self.desc = desc
        self.losses = losses
        self.runind = runind
        print("Progress: %s" % self.json_fn)

    def _ensure_mapped(self):
        if self.mapped_NT is None:
            # lazy save
            self.reopen(100, 100)
            self.mapped_NT = np.memmap(self.mapped_NT_fn, mode="w+", shape=(2,), dtype=np.int32)
            self.mapped_NT[0] = 0
            with open(self.json_fn, "wb") as f:
                print >> f, json.dumps( {
                    "desc": self.desc,
                    "color": "#%02x%02x%02x" % (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)),
                    "mmapped_log": self.mapped_log_fn,
                    "losses": self.losses,
                    "mmapped_tst": self.mapped_tst_fn,
                    "runind": self.runind,
                    "mmapped_NT": self.mapped_NT_fn
                    },
                    indent=4, separators=(',', ': '))

    def push_data_point(self, iter, epoch, ts, lr, loss1, loss2=0, loss3=0, loss4=0, loss5=0, loss6=0):
        self._ensure_mapped()
        N = self.N
        self.mapped_log[N][0] = iter
        self.mapped_log[N][1] = epoch
        self.mapped_log[N][2] = ts
        self.mapped_log[N][3] = lr
        self.mapped_log[N][4] = loss1
        self.mapped_log[N][5] = loss2
        self.mapped_log[N][6] = loss3
        self.mapped_log[N][7] = loss4
        self.mapped_log[N][8] = loss5
        self.mapped_log[N][9] = loss6
        self.N += 1
        if self.N >= self.allocated_n:
            self.reopen(self.N + 100, self.allocated_t)
        self.mapped_NT[0] = self.N

    def push_testrun_point(self, iter, epoch, ts, score, runtime):
        self._ensure_mapped()
        T = self.T
        self.mapped_tst[T][0] = iter
        self.mapped_tst[T][1] = epoch
        self.mapped_tst[T][2] = ts
        self.mapped_tst[T][4] = score
        self.mapped_tst[T][5] = runtime
        self.T += 1
        if self.T >= self.allocated_t:
            self.reopen(self.allocated_n, self.T + 100)
        self.mapped_NT[1] = self.T

    def reopen(self, more_n, more_t):
        self.allocated_n = more_n
        self.allocated_t = more_t
        if os.path.exists(self.mapped_log_fn):
            mode = "r+"
        else:
            mode = "w+"
        del self.mapped_log
        self.mapped_log = np.memmap(self.mapped_log_fn, mode=mode, shape=(more_n,10), dtype=np.float32)
        self.mapped_tst = np.memmap(self.mapped_tst_fn, mode=mode, shape=(more_t,10), dtype=np.float32)
