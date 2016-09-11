import xp
import progress
from threading import Lock
import time

class Algorithm:
    def __init__(self, dir, experiment_name, BATCH=100):
        self.BATCH = BATCH
        self.experiment_name = experiment_name
        self.pause = False
        self.quit = False
        self.dry_run = True
        self.use_random_policy = True
        self.save_load_mutex = Lock()
        self.iter_counter = 0

        import subprocess
        diff = subprocess.Popen(["git", "diff"], stdout=subprocess.PIPE).communicate()[0]
        self.progress = progress.Progress(dir, experiment_name, diff)
        self.progress_last_epoch = 0
        self.time_start = time.time()

    def learn_thread_func(self):
        while 1:
            while (self.use_random_policy or self.pause) and not self.quit and not self.dry_run:
                time.sleep(0.1)
            if self.quit:
                break
            with self.save_load_mutex:
                self.run_single_learn_iteration(self.dry_run)
                if xp.epoch > 1: self.dry_run = False

    def run_single_learn_iteration(self, dry_run):
        buf = xp.batch(self.BATCH)
        losses_array = self._learn_iteration(buf, dry_run)
        if not self.dry_run:
            self.iter_counter += 1

        epoch_int = int(xp.epoch)
        if epoch_int != self.progress_last_epoch:
            self.progress_last_epoch = epoch_int
            self.progress.push_data_point(self.iter_counter, xp.epoch, time.time() - self.time_start, 0.01, *losses_array)

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
