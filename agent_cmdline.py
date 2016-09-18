import os, sys, shutil
os.environ['THEANO_FLAGS'] = "device=gpu"
import theano, random, time, argparse
theano.config.floatX = 'float32'
#theano.config.compute_test_value = 'warn'
theano.config.exception_verbosity = 'high'
import numpy as np
import scipy.misc
import gym
from gym.wrappers import SkipWrapper

import controller.xp as xp

# python -B agent_cmdline.py BipedalWalker-v2 --loadxp demos.json --viz-only
# python -B agent_cmdline.py BipedalWalker-v2 --loadxp demos.json --learn
# python -B agent_cmdline.py BipedalWalker-v2 --loadxp demos.json --control-from-iteration 1337
#
# --savexp try1.json
# --algo wires

parser = argparse.ArgumentParser(description="Reinforcement learning from demonstrations, control after certain learn iteration.")
parser.add_argument("env", metavar="ENV", nargs=1, help="gym environment to work with, also defines environment directory .ENV/")
parser.add_argument("--loadxp", nargs='+', help="load experience from environment directory")
parser.add_argument("--savexp", nargs=1, help="file and jpeg dir name prefix to save experience (J, Ctrl+S)")
parser.add_argument("--viz-only", help="export visualisation arrays and quit (default)", action="store_true")
parser.add_argument("--learn", nargs=1, help="learn and quit", default=["QP"])  #, action="store_true")
parser.add_argument("--frameskip", nargs=1, help="frame skipping", type=int, default=1)
parser.add_argument("--control-from-iteration", help="learn and quit", type=int, default=-1)
args = parser.parse_args()

env_type = args.env[0]
dir = "."+env_type
prefix = "_"
if args.savexp:
    prefix = args.savexp[0]
    experiment_name = args.savexp[0]
print("Environment dir: {}".format(dir))
print("Prefix: {}".format(prefix))
print("Experiment: {}".format(experiment_name))

dir_jpeg = dir + "/" + prefix
try: shutil.rmtree(dir_jpeg)
except: pass
os.makedirs(dir_jpeg)

if args.loadxp:
    for x in args.loadxp:
        fn = dir+"/"+x
        pack = xp.load_lowlevel(fn)
        print("Loaded {} ({} samples)".format(fn, len(pack)))
        xp.replay.extend(pack)
    xp.shuffle()
    print("Total {} samples".format(len(xp.replay)))
print

def make_env():
    env = gym.make(env_type)
    if args.frameskip > 1:
        wrap = SkipWrapper(args.frameskip)
        env = wrap(env)
    return env

args.frameskip = args.frameskip[0]
env = make_env()

if xp.STATE_DIM==0:
    xp.init_from_env(env)

try: xp.export_viz_open(dir_jpeg, "r+")
except: xp.export_viz_open(dir_jpeg, "w+")

if args.viz_only:
    sys.exit(0)

print("control-from-iteration: {}".format(args.control_from_iteration))
print("learn={}".format(args.learn[0]))

if args.learn[0]=="WAR":
    import controller.algo_wires_advantage_random as war
    alg = war.WiresAdvantageRandom()
elif args.learn[0]=="WTR":
    import controller.algo_wires_transition_random as wtr
    alg = wtr.WiresTransitionRandom()
elif args.learn[0]=="DBW":
    import demo_bipedal_walker
    alg = demo_bipedal_walker.DemoBipedalWalker()
elif args.learn[0]=="DLL":
    import demo_lunar_lander
    alg = demo_lunar_lander.DemoLunarLander()
elif args.learn[0]=="QP":
    import controller.algo_qnet_policygrad as qp
    alg = qp.QNetPolicygrad(dir + "/progress", experiment_name)
else:
    print("unknown algorithm %s" % args.learn[0])
    sys.exit(0)

human_sets_pause = 1
alg.pause = True
human_wants_quit = False
human_wants_restart = False
human_records_xp = True
new_xp = []

import pyglet
from pyglet.window import key as kk

def key_press(key, mod):
    global human_wants_restart, human_sets_pause, human_records_xp
    if key==32:
        human_sets_pause = (human_sets_pause+1) % 2
        print("pause %i (%s)" % (human_sets_pause, ["GO!","STOP","AUTO"][human_sets_pause]))
    elif key==0xff0d: human_wants_restart = True
    elif key==kk.F1:
        alg.pause = not alg.pause
        print("alg.pause=%i" % (alg.pause))
    elif key==kk.F2:
        alg.save(dir + "/_weights")
    elif key==kk.F3:
        alg.load(dir + "/_weights")
        with xp.replay_mutex:
            xp.export_viz_open(dir_jpeg, "r+")
    elif key==ord("j"):
        human_records_xp = not human_records_xp
        print("record=%i prefix=%s" % (human_records_xp, prefix))
    elif key==ord("s") and (mod & kk.MOD_CTRL):
        if human_records_xp:
            fn = dir+"/"+prefix+".json"
            print("SAVE XP {}".format(fn))
            xp.save_lowlevel(fn, new_xp)
        else:
            print("NOT RECORDING")
    else:
        print("key pressed {}".format(key))

def key_release(key, mod):
    pass
def close():
    global human_wants_quit
    human_wants_quit = True

sn = env.reset()
alg.reset(True)
env.render()
if "env" in env.__dict__: env.viewer = env.env.viewer
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
env.viewer.window.on_close = close
env.viewer.window.set_caption(prefix)
global_step_counter = 0

# before starting a thread, check if everything works in main thread.
alg.load_something_useful_on_start(dir + "/_weights")
alg.run_single_learn_iteration(False)
alg.control(sn, env.action_space)

from threading import Thread
learn_thread = Thread(target=alg.learn_thread_func)
learn_thread.daemon = True
learn_thread.start()

def testrun_rollout():
    "instead of sleeping, run a test, post score to progress log"
    env2 = make_env()
    score = 0
    ts = 0
    sn = env2.reset()
    while 1:
        s = sn
        a = alg.control(s, env2.action_space)
        sn, r, done, info = env2.step(a)
        if ts > env2.spec.timestep_limit / args.frameskip:
            done = True
        ts += 1
        score += r
        if done: break
    print "testrun: %0.2f in %i steps" % (score, ts)
    alg.push_testrun_point(score, ts)

def rollout():
    global human_wants_quit, human_wants_restart, human_sets_pause, sn, global_step_counter
    human_wants_restart = False
    track = []
    ts = 0
    r = 0
    score = 0
    while 1:
        s = sn

        while human_sets_pause and not human_wants_quit and not human_wants_restart:
            env.viewer.window.dispatch_events()
            time.sleep(0.2)
            a = alg.control(s, env.action_space)
        if human_wants_quit: break

        a = alg.control(s, env.action_space)

        sn, r, done, info = env.step(a)
        if ts > env.spec.timestep_limit / args.frameskip:
            done = True
            print("time limit hit")
        ts += 1
        score += r
        if human_wants_restart:
            done = True
            r = -100.0
            print("human -100")
        pt = xp.XPoint(s, a, r, sn, ts, done)

        env.render()

        if human_records_xp and (global_step_counter % 5 == 0 or done):
            rgb = env.render("rgb_array")
            jpeg_name = dir_jpeg + "/{:05}".format(global_step_counter)
            scipy.misc.imsave(jpeg_name, rgb, format="jpeg")
            pt.jpeg = jpeg_name

        track.append(pt)

        global_step_counter += 1
        if done: break

    print("total reward %0.2f, last reward %0.2f on step %i" % (score, r, ts))

    if track and human_records_xp and ts>2:
        new_xp.extend(track)
        with xp.replay_mutex:
            xp.replay.extend(track)
            xp.shuffle()
            xp.export_viz_open(dir_jpeg, "r+")
            print("now replay buffer have %i samples" % len(xp.replay))
            alg.reset(True)
        wait_t1 = time.time()
        while 1:
            env.viewer.window.dispatch_events()
            if not alg.useful_to_think_more(): break
            if human_wants_quit: break
            if human_wants_restart: break
            testrun_rollout()
            time.sleep(4)
            if time.time() > wait_t1 + 20: break
    else:
            alg.reset(False)
    sn = env.reset()

episode_n = 0
while not human_wants_quit:
    episode_n += 1
    print("episode %i" % episode_n)
    rollout()

alg.quit = True
learn_thread.join()
#pyglet.app.run()
