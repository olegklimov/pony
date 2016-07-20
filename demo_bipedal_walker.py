import os, shutil
import gym.envs.box2d.bipedal_walker as w
from controller import xp
import numpy as np
import scipy.misc

dir = ".BipedalWalker-v2"
dir_jpeg = dir + "/d"

try: shutil.rmtree(dir_jpeg)
except: pass
os.makedirs(dir_jpeg)

env = w.BipedalWalker()
s = env.reset()
xp.init_from_env(env)

step = 0
for e in range(40):
    s = env.reset()
    ts = 0
    episode = []
    s_moving_average = 0
    while True:
        a  = w.heuristic(env, s)
        rand = e/40.0*0.1
        a += np.random.uniform( low=-rand, high=+rand, size=(4,) ) 

        sn, r, done, info = env.step(a)

        s_moving_average = 0.9*s_moving_average + 0.1*s
        stuck = np.linalg.norm( s_moving_average - s) < 0.01
        if stuck: done = True

        pt = xp.XPoint(s, a, r, sn, ts, done)
        s = sn
        step += 1
        ts += 1
        if step % 5 == 0 or done:
            rgb = env.render("rgb_array")
            jpeg_name = dir_jpeg + "/{:05}.jpg".format(step)
            scipy.misc.imsave(jpeg_name, rgb)
            pt.jpeg = jpeg_name
        else:
            env.render()
        episode.append(pt)
        if done: break
    xp.replay.extend(episode)

xp.save(dir + "/demos.json")
