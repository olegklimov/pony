import os
import gym.envs.box2d.bipedal_walker as w
from controller import xp
import numpy as np
import scipy.misc


dir = ".BipedalWalker-v2"

try: os.mkdir(dir)
except: pass

env = w.BipedalWalker()
s = env.reset()
xp.init_from_env(env)

step = 0
for e in range(20):
    s = env.reset()
    episode = []
    while True:
        a = w.heuristic(env, s)
        sn, r, done, info = env.step(a)
        pt = xp.XPoint(s, a, r, sn if not done else None)
        s = sn
        step += 1
        if step % 20 == 0 or done:
            rgb = env.render("rgb_array")
            jpeg_name = dir + "/z{:05}.jpg".format(step)
            scipy.misc.imsave(jpeg_name, rgb)
            pt.jpeg = jpeg_name
        else:
            env.render()
        episode.append(pt)
        if done: break
    xp.replay.extend(episode)

xp.save(dir + "/demos.json")
