import os, sys
os.environ['THEANO_FLAGS'] = "device=gpu"
import theano, random, time, argparse
theano.config.floatX = 'float32'
import numpy as np
import gym

import controller.value_iteration_with_gravity as gravity
import controller.xp as xp

# python -B agent_cmdline.py BipedalWalker-v2 --loadxp demos.json --viz-only
# python -B agent_cmdline.py BipedalWalker-v2 --loadxp demos.json --learn
# python -B agent_cmdline.py BipedalWalker-v2 --loadxp demos.json --control-from-iteration 1337
#
# --savexp try1.json
# --algo wires

parser = argparse.ArgumentParser(description="RL from demonstrations (optional), control environment after certain learn iteration.")
parser.add_argument("env", metavar="ENV", nargs=1, help="gym environment to work with, also defines environment directory .ENV/")
parser.add_argument("--loadxp", nargs='+', help="load experience from environment directory")
parser.add_argument("--viz-only", help="export visualisation arrays and quit", action="store_true")
parser.add_argument("--learn", help="learn and quit", action="store_true")
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

xp.export_viz_open(dir)

if args.viz_only:
    for x in xp.replay:
        xp.export_viz.state1[x.viz_n] = x.s   
        xp.export_viz.state2[x.viz_n] = x.sn
        xp.export_viz.Vtarget[x.viz_n]  = x.r
        xp.export_viz.Vonline1[x.viz_n] = x.r
        xp.export_viz.Vstable1[x.viz_n] = x.r
        xp.export_viz.Vstable2[x.viz_n] = x.r
        xp.export_viz.step[x.viz_n] = x.step
    del xp.export_viz
    xp.export_viz_open(dir, "r+")
    sys.exit(0)

elif args.learn or args.control_from_iteration>=0:
    print("control-from-iteration: {}".format(args.control_from_iteration))
    #env = gym.make( env_type )
    #print env
    from threading import Thread
    import pyglet
    gravity.V_online = gravity.VNetwork()
    gravity.V_stable = gravity.VNetwork()
    learn_thread = Thread(target=gravity.learn_thread_func)
    learn_thread.daemon = True
    learn_thread.start()
    pyglet.app.run()

else:
    parser.print_help()
