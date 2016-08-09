import xp
from threading import Lock

class Algorithm:
    def __init__(self, BATCH=100):
        self.BATCH = BATCH
        self.pause = False
        self.quit = False
        self.dry_run = True
        self.save_load_mutex = Lock()

    def learn_thread_func(self):
        while not self.quit:
            while self.pause and not self.quit and not self.dry_run:
                import time
                time.sleep(0.1)
            with self.save_load_mutex:
                buf = xp.batch(self.BATCH)
                self._learn_iteration(buf, self.dry_run)
                if xp.epoch > 2: self.dry_run = False

    def save(self, fn):
        with self.save_load_mutex:
            print("SAVE %s" % fn)
            self._save(fn)

    def load(self, fn):
        with self.save_load_mutex:
            print("LOAD %s" % fn)
            self._load(fn)
            self.dry_run = True
            xp.epoch = 0.0
            xp.epoch_sample_counter = 0

    def reset(self):
        self._reset()

    def control(self, s, action_space):
        return self._control(s, action_space)

    def useful_to_think_more(self):
        return False
        
