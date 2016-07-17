import os
import controller.xp as xp
import numpy as np

dir = ".Cliff-v0"
try: os.mkdir(dir)
except: pass

def synthetic_cliff():
    for c in range(15):
        alpha = 2*np.pi / 15 * c
        s = None
        for t in range(18):
            sn = np.array( [100 * np.sin(alpha) * t/6, 100 * np.cos(alpha) * t/6] )
            a  = np.array( [0] )
            r  = 0
            if t==17:
                r  = -100 if c!=10 else +100
            if s is not None:
                xp.replay.append( xp.XPoint(s,a,r,sn,t,t==17) )
            s = sn
    for c in range(3000):
        alpha = np.random.uniform(0, 2*np.pi) 
        beta  = np.random.uniform(0, 2*np.pi)
        rad   = np.random.uniform(50, 100)
        side_s  = np.array( [rad*np.sin(alpha), rad*np.cos(alpha)] )
        side_sn = np.array( [side_s[0] + 10*np.sin(beta), side_s[1] + 10*np.cos(beta)] )
        a = np.array( [0] )
        r = 0
        xp.replay.append( xp.XPoint(side_s,a,r,side_sn,0,False) )
#    for c in range(15):
#        alpha      = 2*np.pi / 15 * c
#        alpha_next = 2*np.pi / 15 * (c+1)
#        side_s  = np.array( [100 * np.sin(alpha     ) * 5/6, 100 * np.cos(alpha     ) * 5/6] )
#        side_sn = np.array( [100 * np.sin(alpha_next) * 5/6, 100 * np.cos(alpha_next) * 5/6] )
#        a = np.array( [1] )
#        r = 0
#        xp.replay.append( xp.XPoint(side_s,a,r,side_sn,c,False) )
#        xp.replay.append( xp.XPoint(side_sn,a,r,side_s,c,False) )


xp.STATE_DIM = 2
xp.ACTION_DIM = 1
synthetic_cliff()
print "generated cliff {} samples in replay buffer".format(len(xp.replay))
xp.save(dir + "/cliff.json")
