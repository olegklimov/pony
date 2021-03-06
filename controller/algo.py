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
        rev  = subprocess.Popen(["git", "log", "-1"], stdout=subprocess.PIPE).communicate()[0]
        diff = subprocess.Popen(["git", "diff"], stdout=subprocess.PIPE).communicate()[0]
        if self.nameof_losses:
            self.progress = progress.Progress(dir, experiment_name, rev + "\n" + diff, self.nameof_losses, self.nameof_runind)
            self.progress_last_epoch = 0
            self.time_start = time.time()
        else:
            self.progress = None

    nameof_runind = []
    nameof_losses = []

    def _test_still_need_random_policy(self):
        pass

    def learn_thread_func(self):
        self._test_still_need_random_policy()
        while 1:
            while (self.use_random_policy or self.pause) and not self.quit and not self.dry_run:
                self._test_still_need_random_policy()
                time.sleep(0.1)
            if self.quit:
                break
            with self.save_load_mutex:
                self.run_single_learn_iteration(self.dry_run)
                if xp.epoch > 1: self.dry_run = False

    def run_single_learn_iteration(self, dry_run):
        t1 = time.time()
        buf = xp.batch(self.BATCH)
        t2 = time.time()
        losses_array = self._learn_iteration(buf, dry_run)
        if not self.dry_run:
            self.iter_counter += 1
        if losses_array is None: return
        t3 = time.time()
        #print "%0.2fms = %0.2fms batch + %0.2fms iteration" % ( 1000*(t3-t1), 1000*(t2-t1), 1000*(t3-t2) )

        epoch_int = int(xp.epoch)
        if self.progress and epoch_int != self.progress_last_epoch or True:
            self.progress_last_epoch = epoch_int
            self.progress.push_data_point(self.iter_counter, xp.epoch, time.time() - self.time_start, 0.01, *losses_array)

    def push_testrun_point(self, score, runtime):
        if self.progress:
            self.progress.push_testrun_point(self.iter_counter, xp.epoch, time.time() - self.time_start, score, runtime)

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

    def do_action(self):
        "test something by pressing F5"
        pass


