import xp
from threading import Lock

class Algorithm:
    def __init__(self, BATCH=100):
        self.BATCH = BATCH
        self.pause = False
        self.quit = False
        self.dry_run = True
        self.use_random_policy = True
        self.save_load_mutex = Lock()

    def learn_thread_func(self):
        while 1:
            while (self.use_random_policy or self.pause) and not self.quit and not self.dry_run:
                import time
                time.sleep(0.1)
            if self.quit:
                break
            with self.save_load_mutex:
                self.run_single_learn_iteration(self.dry_run)
                if xp.epoch > 1: self.dry_run = False

    def run_single_learn_iteration(self, dry_run):
        buf = xp.batch(self.BATCH)
        self._learn_iteration(buf, dry_run)

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

    def reset(self, new_experience):
        self._reset(new_experience)

    def control(self, s, action_space):
        return self._control(s, action_space)

    def useful_to_think_more(self):
        return not self.use_random_policy

    def load_something_useful_on_start(self, fn):
        pass
