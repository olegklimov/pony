import numpy as np

def synthetic_cliff():
    for c in range(15):
        alpha      = 2*np.pi / 15 * c
        s = None
        for t in range(18):
            sn = np.array( [100 * math.sin(alpha) * t/6, 100 * math.cos(alpha) * t/6] )
            a  = np.array( [0] )
            r  = 0
            if t==17:
                r  = -100 if c!=10 else +100
            if s is not None:
                xp.replay.append( xp.XPoint(s,a,r,sn) )
                xp.replay[-1].terminal = t==17
            s = sn
    for c in range(15):
        alpha      = 2*np.pi / 15 * c
        alpha_next = 2*np.pi / 15 * (c+1)
        side_s  = np.array( [100 * math.sin(alpha     ) * 5/6, 100 * math.cos(alpha     ) * 5/6] )
        side_sn = np.array( [100 * math.sin(alpha_next) * 5/6, 100 * math.cos(alpha_next) * 5/6] )
        a = np.array( [0] )
        r = 0
        xp.replay.append( xp.XPoint(side_s,a,r,side_sn) )
        xp.replay[-1].terminal = False
        xp.replay.append( xp.XPoint(side_sn,a,r,side_s) )
        xp.replay[-1].terminal = False
    for c in range(300):
        alpha = np.random.uniform(2*np.pi)
        beta  = np.random.uniform(2*np.pi)
        side_s  = np.array( [100*math.sin(alpha), 100*math.cos(alpha)] )
        side_sn = np.array( [side_s[0] + 5*math.sin(beta), side_s[1] + 5*math.cos(beta)] )
        a = np.array( [0] )
        r = 0
        xp.replay.append( xp.XPoint(side_s,a,r,side_sn) )
        xp.replay[-1].terminal = False
        xp.replay.append( xp.XPoint(side_sn,a,r,side_s) )
        xp.replay[-1].terminal = False

