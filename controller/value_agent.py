import os
os.environ['THEANO_FLAGS'] = "device=gpu"
import theano, random, time
theano.config.floatX = 'float32'
import numpy as np
import value_iteration_with_gravity as gravity
import xp

if __name__=="__main__":
    from threading import Thread
    learn_thread = Thread(target=gravity.learn_thread_func)
    def do_anything(seconds):
        if seconds==2:
            learn_thread.daemon = True
            learn_thread.start()

    import xp
    xp.STATE_DIM = 3
    xp.ACTION_DIM = 3
    xp.ACTIONS = np.eye(xp.ACTION_DIM)
    gravity.V_online = gravity.VNetwork()
    gravity.V_stable = gravity.VNetwork()

    # synthetic data

    #synthetic_cliff()
    #xp.save("xp")
    xp.load("xp")
    for x in xp.replay:
        if x.sn is None:
            x.terminal = True
            x.sn = x.s
        x.important = False
    xp.shuffle()

    for x in xp.replay:
        x.target = x.r
        x.v = x.r
        x.ov = x.r
        x.nv = x.r


    ### debug vis ###

    from PyQt4 import QtGui, QtCore
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    class MyDynamicMplCanvas(FigureCanvas):
        def __init__(self):
            fig = Figure(figsize=(5,4), dpi=100)
            from mpl_toolkits.mplot3d import Axes3D
            FigureCanvas.__init__(self, fig)
            FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
            FigureCanvas.updateGeometry(self)
            self.axes1 = fig.add_subplot(111, projection='3d')
            timer = QtCore.QTimer(self)
            timer.timeout.connect(self.update_figure)
            timer.start(50)
            self.seconds = 0
            self.update_figure();

        def update_figure(self):
            with xp.replay_mutex:
                buf = xp.replay[:]
            X = []
            Y = []
            O = []
            V = []
            T = []
            self.axes1.hold(False)
            for x in buf:
                X.append( x.s[0] )
                Y.append( x.s[1] )
                O.append( x.ov )
                V.append( x.v )
                T.append( x.target )
                if x.terminal:
                    self.axes1.plot(np.array(X), np.array(Y), zs=np.array(V))
                    #self.axes1.plot(np.array(X), np.array(Y), zs=np.array(O))
                    #self.axes1.scatter(np.array(X), np.array(Y), zs=np.array(T))
                    self.axes1.hold(True)
                    X = []
                    Y = []
                    O = []
                    V = []
                    T = []
            #if len(X) > 0:
            #  self.axes1.plot(np.array(X), np.array(Y), zs=np.array(V))
            #    self.axes1.plot(np.array(X), np.array(Y), zs=np.array(O))

            X1 = [x.s[0]         for x in buf if x.important]
            Y1 = [x.s[1]         for x in buf if x.important]
            V1 = [x.target       for x in buf if x.important]
            X2 = [x.s[0]-x.sn[0] for x in buf if x.important]
            Y2 = [x.s[1]-x.sn[1] for x in buf if x.important]
            V2 = [x.target-x.nv  for x in buf if x.important]
            #self.axes1.quiver(X1,Y1,V1, X2,Y2,V2, length=10)

            try:
                self.draw()
            except KeyboardInterrupt:
                qApp.quit()
            do_anything(self.seconds)
            self.seconds += 1

    qApp = QtGui.QApplication([])
    f = MyDynamicMplCanvas()
    f.showMaximized()
    qApp.exec_();

