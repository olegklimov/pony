import os, sys
os.environ['THEANO_FLAGS'] = "device=gpu"
import theano, random, time, argparse
theano.config.floatX = 'float32'
import numpy as np
import gym

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
parser.add_argument("--viz-only", help="export visualisation arrays and quit (default)", action="store_true")
parser.add_argument("--learn", nargs=1, help="learn and quit")  #, action="store_true")
parser.add_argument("--control-from-iteration", help="learn and quit", type=int, default=-1)
args = parser.parse_args()

env_type = args.env[0]
dir = "."+env_type
print("Environment dir: {}".format(dir))
assert os.path.exists(dir)

if args.loadxp:
    for x in args.loadxp:
        fn = dir+"/"+x
        pack = xp.load_lowlevel(fn)
        print("Loaded {} ({} samples)".format(dir, len(pack)))
        xp.replay.extend(pack)
    xp.shuffle()
    print("Total {} samples".format(len(xp.replay)))
print

xp.export_viz_open(dir, "w+")
#del xp.export_viz
#xp.export_viz_open(dir, "r+")

if args.viz_only:
    sys.exit(0)

print("control-from-iteration: {}".format(args.control_from_iteration))
print("args.learn={}".format(args.learn[0]))

env = gym.make(env_type)

if args.learn[0]=="WAR":
    from threading import Thread
    import pyglet
    import controller.algo_wires_advantage_random as war
    alg = war.WiresAdvantageRandom()
    learn_thread = Thread(target=alg.learn_thread_func)
    learn_thread.daemon = True
    learn_thread.start()
else:
    print("unknown algorithm %s" % args.learn[0])
    sys.exit(0)
 
human_sets_pause = True
alg.pause = True
human_wants_quit = False
human_wants_restart = False

from pyglet.window import key as kk
def key_press(key, mod):
    global human_wants_restart, human_sets_pause
    if key==32: human_sets_pause = not human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==kk.F1: alg.pause = not alg.pause
    if key==kk.F2:
        alg.save(dir + "/_weights")
    if key==kk.F3:
        alg.load(dir + "/_weights")
        xp.export_viz_open(dir, "r+")

def key_release(key, mod):
    pass
def close():
    global human_wants_quit, human_wants_restart, human_sets_pause
    human_wants_quit = True

s = env.reset()
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
env.viewer.window.on_close = close

def rollout():
    global human_wants_quit, human_wants_restart, human_sets_pause, s
    human_wants_restart = False
    while not human_wants_quit:
        a = alg.control(s, env.action_space)
        s, r, done, info = env.step(a)
        env.render()
        if done: break
        while human_sets_pause and not human_wants_quit and not human_wants_restart:
            env.viewer.window.dispatch_events()
            import time
            time.sleep(0.2)
        if human_wants_restart: break
    s = env.reset()

while not human_wants_quit:
    rollout()

alg.quit = True
learn_thread.join()
#pyglet.app.run()
