#include <QtCore/QTimer>
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include <QtOpenGL/QtOpenGL>
#include <GL/glu.h>
#include <boost/shared_ptr.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

using boost::shared_ptr;
using boost::iostreams::mapped_file_source;

float line_vertex[] = {
+1,+1,0, -1,+1,0,
-1,+1,0, -1,-1,0,
-1,-1,0, +1,-1,0,
+1,-1,0, +1,+1,0,
-2,0,0, +2,0,0,
0,-2,0, 0,+2,0,

+1,+1,-1, +1,+1,+1,
-1,+1,-1, -1,+1,+1,
-1,-1,-1, -1,-1,+1,
+1,-1,-1, +1,-1,+1,
};

struct Quiver {
	int N;
	int STATEDIM;
	mapped_file_source file_state1;
	mapped_file_source file_state2;
	mapped_file_source file_V;
	//mapped_file_source file_D;
	std::vector<float> vertex; // x y z
	std::vector<float> vcolor; // r g b

	void open()
	{
		file_state1.open(".vizdata/state1");
		file_state2.open(".vizdata/state2");
		file_V.open(".vizdata/V");
		//file_D.open(".vizdata/D");
	}

	void reprocess()
	{
		int new_N = file_V.size() / sizeof(float);
		STATEDIM = file_state1.size() / file_V.size();
		if (new_N!=N) printf("N=%i STATEDIM=%i\n", N, STATEDIM);
		N = new_N;
		float* s1 = (float*) file_state1.data();
		float* s2 = (float*) file_state2.data();
		float* V  = (float*) file_V.data();
		//float* D  = (float*) file_D.data();
		vertex.resize(6*N);
		vcolor.resize(3*N);
		for (int c=0; c<N; c++) {
			vertex[6*c+0] = s1[STATEDIM*c+0];
			vertex[6*c+1] = s1[STATEDIM*c+1];
			vertex[6*c+2] = V[c];
			vertex[6*c+3] = s2[STATEDIM*c+0];
			vertex[6*c+4] = s2[STATEDIM*c+1];
			vertex[6*c+5] = V[c];
			vcolor[3*c+0] = 0.0f;
			vcolor[3*c+1] = 1.0f;
			vcolor[3*c+2] = 1.0f;
		}
	}

	void draw()
	{
		glVertexPointer(3, GL_FLOAT, 0, vertex.data());
		glColorPointer(3, GL_FLOAT, 0, vcolor.data());
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glDrawArrays(GL_LINES, 0, N);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
};

class Viz: public QGLWidget {
	Q_OBJECT
public:
	QTimer* timer;
	
	Viz(QWidget *parent):
		QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
	{
		timer = new QTimer(this);
		QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));
		timer->start(1000/30);
	}

	shared_ptr<Quiver> q;

	void initializeGL()
	{
		qglClearColor(Qt::black);
		//qglClearColor(QColor(0xFF0000));
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

		//glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
		//glShadeModel(GL_SMOOTH);
		//glEnable(GL_LIGHTING);
		//glEnable(GL_LIGHT0);
		//glEnable(GL_MULTISAMPLE);

		//glEnable(GL_LINE_SMOOTH);
		//glHint(GL_LINE_SMOOTH_HINT,  GL_NICEST);

		//static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
		//glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	}

	void paintGL()
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//glOrtho(-5.5, +5.5, -5.5, +5.5, -5.0, 15.0);
		float near = 0.1;
		float far  = 5.0;
		float typical = 1.0;
		float r = typical/far*near;
		glFrustum(-r, +r, -r, +r, 0.1, 55.0);
		glMatrixMode(GL_MODELVIEW);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();
		glTranslatef(0.0, 0.0, -10.0);

		//0.0f, 0.0f, 0.0f,
		//1.0f, 1.0f, 1.0f
		//glTranslatef(0, 0, wheel);
		glRotatef(yrot, 1.0, 0.0, 0.0);
		glRotatef(xrot, 0.0, 0.0, 1.0);
		glScalef(wheel,wheel,wheel);
		glTranslatef(-user_x, -user_y, -user_z);

		glColor3f(1.0f, 1.0f, 1.0f);
		glVertexPointer(3, GL_FLOAT, 0, line_vertex);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_LINES, 0, sizeof(line_vertex)/sizeof(float)/3);
		glDisableClientState(GL_VERTEX_ARRAY);

		if (q) q->draw();
	}

	void resizeGL(int w, int h)
	{
		int side = std::max(w, h);
		glViewport((w - side) / 2, (h - side) / 2, side, side);
	}

	float user_x = 0;
	float user_y = 0;
	float user_z = 0;
	float xrot = 0;
	float yrot = -40;
	float zrot = 0;
	float wheel = 1;

	void keyPressEvent(QKeyEvent* kev)
	{
		double fx = cos(-xrot / 180 * 3.1415926);
		double fy = sin(-xrot / 180 * 3.1415926);
		double sx = -fy;
		double sy =  fx;
		if (kev->key()==Qt::Key_A || kev->key()==Qt::Key_D) {
			double sign = kev->key()==Qt::Key_A ?  -1 : +1;
			user_x += sign * 0.02 * fx;
			user_y += sign * 0.02 * fy;
		}
		if (kev->key()==Qt::Key_W || kev->key()==Qt::Key_S) {
			double sign = kev->key()==Qt::Key_S ?  -1 : +1;
			user_x += sign * 0.02 * sx;
			user_y += sign * 0.02 * sy;
		}
		if (kev->key()==Qt::Key_PageDown || kev->key()==Qt::Key_PageUp) {
			double sign = kev->key()==Qt::Key_PageDown ?  -1 : +1;
			user_z += sign * 0.02;
		}
		updateGL();
	}

	void wheelEvent(QWheelEvent* wev)
	{
		wheel += 0.001*wev->delta();
		updateGL();
	}

	void mousePressEvent(QMouseEvent* mev)
	{
		updateGL();
		if (mev->button()==Qt::LeftButton) {
			prev_mx = mev->x();
			prev_my = mev->y();
			drag = true;
		}
	}

	void mouseReleaseEvent(QMouseEvent* mev)
	{
		if (mev->button()==Qt::LeftButton)
			drag = false;
	}


	bool drag = false;
	double prev_mx;
	double prev_my;

	void mouseMoveEvent(QMouseEvent* mev)
	{
		if (drag) {
			xrot += 0.02*(mev->x() - prev_mx);
			yrot += 0.02*(mev->y() - prev_my);
			prev_mx = mev->x();
			prev_my = mev->y();
			updateGL();
		}
	}
	
public slots:
	void timeout()
	{
		if (q) {
			q->reprocess();
		}
		xrot += 0.01;
		updateGL();
	}
};

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	Viz window(0);
	try {
		shared_ptr<Quiver> q;
		q.reset(new Quiver);
		q->open();
		q->reprocess();
		window.q = q;

	} catch (const std::exception& e) {
		fprintf(stderr, "ERROR: %s\n", e.what());
	}

	window.showMaximized();
	return app.exec();
}

#include "../.generated/viz.moc"

