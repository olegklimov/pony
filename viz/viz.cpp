#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include <QtOpenGL/QGLWidget>
#include <QtOpenGL/QtOpenGL>
#include <GL/glu.h>
#include <boost/shared_ptr.hpp>
//#include <GL/glew.h>

using boost::shared_ptr;

float line_vertex[13000*3];
std::vector<float> s3;

struct Quiver {
	std::vector<float> state1; // x y z
	std::vector<float> state2; // x y z
	std::vector<float> color;  // r g b
};

class Viz: public QGLWidget {
public:
	Viz(QWidget *parent):
	QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
	{
		for (int c=0; c<13000/2; ++c) {
			float x = -1.0 + 2.0 * random() / RAND_MAX;
			float y = -1.0 + 2.0 * random() / RAND_MAX;
			line_vertex[6*c + 0] = x;
			line_vertex[6*c + 1] = y;
			line_vertex[6*c + 2] = 0.1 * sin(3*x) * cos(3*y);
			x += 0.1*(-1.0 + 2.0 * random() / RAND_MAX);
			y += 0.1*(-1.0 + 2.0 * random() / RAND_MAX);
			line_vertex[6*c + 3] = x;
			line_vertex[6*c + 4] = y;
			line_vertex[6*c + 5] = 0.1 * sin(3*x) * cos(3*y);
		}
	}

	void draw_quiver(const shared_ptr<Quiver>& q)
	{

	}

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
		glFrustum(-r, +r, -r, +r, 0.1, 15.0);
		printf(" + wheel=%0.2f\n",  + wheel);
		glMatrixMode(GL_MODELVIEW);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();
		glTranslatef(0.0, 0.0, -10.0);

		//0.0f, 0.0f, 0.0f,
		//1.0f, 1.0f, 1.0f
		glTranslatef(0, 0, wheel);
		glRotatef(yrot, 1.0, 0.0, 0.0);
		glRotatef(xrot, 0.0, 0.0, 1.0);
		glTranslatef(-user_x, -user_y, 0);

		glColor3f(0.0f, 1.0f, 1.0f);
		glVertexPointer(3, GL_FLOAT, 0, line_vertex);
		glEnableClientState(GL_VERTEX_ARRAY);
		//glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glDrawArrays(GL_LINES, 0, 13000);
		glDisableClientState(GL_VERTEX_ARRAY);

		//draw_axes(3.0);
	}

	void resizeGL(int w, int h)
	{
		side = std::max(w, h);
		printf("side=%i\n", side);
		glViewport((w - side) / 2, (h - side) / 2, side, side);
	}

	int side = 0;
	float user_x = 0;
	float user_y = 0;
	float xrot = 0;
	float yrot = -45;
	float zrot = 0;
	float wheel = 0;

	void keyPressEvent(QKeyEvent* kev)
	{
		double fx = cos(-xrot / 180 * 3.1415926);
		double fy = sin(-xrot / 180 * 3.1415926);
		double sx = -fy;
		double sy =  fx;
		if (kev->key()==Qt::Key_A || kev->key()==Qt::Key_D) {
			double sign = kev->key()==Qt::Key_A ?  +1 : -1;
			user_x += sign * 0.02 * fx;
			user_y += sign * 0.02 * fy;
		}
		if (kev->key()==Qt::Key_W || kev->key()==Qt::Key_S) {
			double sign = kev->key()==Qt::Key_S ?  +1 : -1;
			user_x += sign * 0.02 * sx;
			user_y += sign * 0.02 * sy;
		}
		updateGL();
	}

	void wheelEvent(QWheelEvent* wev)
	{
		wheel += 0.01*wev->delta();
		updateGL();
	}

	void mousePressEvent(QMouseEvent* mev)
	{
		updateGL();
		//lastPos = mev->pos();
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
};

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	Viz window(0);
	window.showMaximized();
	return app.exec();
}

