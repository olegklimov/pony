import numpy as np
import json, os

class Progress:
    def __init__(self, dir, task_name, desc, losses):
        self.task_name = task_name
        self.dir = dir
        self.N = 0
        self.allocated = 0
        self.json_fn = dir + "/" + task_name + ".json"
        self.mapped_log_fn = dir+"/"+task_name+".log"
        self.mapped_N_fn   = dir+"/"+task_name+".N"
        self.mapped_N = None
        self.mapped_log = None
        self.desc = desc
        self.losses = losses
        print("Progress: %s" % self.json_fn)

    def push_data_point(self, iter, epoch, ts, lr, loss1, loss2=0, loss3=0, loss4=0, loss5=0, loss6=0):
        if self.mapped_N is None:
            # lazy save
            self.reopen(100)
            self.mapped_N = np.memmap(self.mapped_N_fn, mode="w+", shape=(1,), dtype=np.int32)
            self.mapped_N[0] = 0
            with open(self.json_fn, "wb") as f:
                print >> f, json.dumps( {
                    "desc": self.desc,
                    "color": "#%x%x%x" % (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)),
                    "losses": self.losses,
                    "mmapped_log": self.mapped_log_fn,
                    "mmapped_N": self.mapped_N_fn
                    },
                    indent=4, separators=(',', ': '))
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
        if self.N >= self.allocated:
            self.reopen(self.N + 100)
        self.mapped_N[0] = self.N

    def reopen(self, more):
        self.allocated = more
        if os.path.exists(self.mapped_log_fn):
            mode = "r+"
        else:
            mode = "w+"
        del self.mapped_log
        self.mapped_log = np.memmap(self.mapped_log_fn, mode=mode, shape=(more,10), dtype=np.float32)
