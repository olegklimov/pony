#include <QtCore/QTimer>
#include <QtGui/QApplication>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QSpinBox>
#include <QtGui/QRadioButton>

#include <QtOpenGL/QGLWidget>
#include <QtOpenGL/QtOpenGL>

#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <GL/glu.h>

#include "../t-sne/tsne.h"

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
	int N = 0;
	int STATEDIM;
	mapped_file_source file_state1;
	mapped_file_source file_state2;
	mapped_file_source file_Vonline1;
	mapped_file_source file_Vstable1;
	mapped_file_source file_Vstable2;
	mapped_file_source file_Vtarget;
	mapped_file_source file_step;
	mapped_file_source file_episode;
	mapped_file_source file_jpeg;
	
	//mapped_file_source file_D;
	std::vector<float> vertex; // x y z
	std::vector<float> vcolor; // r g b
	std::vector<int> rendered2index;
	std::string jpeg_dir;

	void open(const std::string& dir)
	{
		tsne_stop();
		file_state1.open(dir + "/state1");
		file_state2.open(dir + "/state2");
		file_Vonline1.open(dir + "/Vonline1");
		file_Vstable1.open(dir + "/Vstable1");
		file_Vstable2.open(dir + "/Vstable2");
		file_Vtarget.open(dir + "/Vtarget");
		file_step.open(dir + "/step");
		file_episode.open(dir + "/episode");
		file_jpeg.open(dir + "/jpegmap");
		jpeg_dir = dir;
	}

	int episode_of(int n)
	{
		if (n==-1) return -1;
		assert(n<N);
		const int* episode = (const int*) file_episode.data();
		return episode[n];
	}
	
	int episode_filter = -1;

	std::string jpeg_of(int n) 
	{
		assert(n<N);
		const char* jpeg = (const char*) file_jpeg.data();
		return jpeg_dir + "/" + (jpeg + 16*n); // ".BipedalWalker-v2/z00020.jpg"
	}

	void reprocess(
		const shared_ptr<Quiver>& myself,
		float xy_range,
		float z_range,
		bool tsne,
		int axis1, int axis2 // if tsne false
		)
	{
		int new_N = file_Vonline1.size() / sizeof(float);
		STATEDIM = file_state1.size() / file_Vonline1.size();
		if (new_N!=N) printf("N=%i STATEDIM=%i\n", new_N, STATEDIM);
		N = new_N;
		if (N==0) return;
		float* s1 = (float*) file_state1.data();
		float* s2 = (float*) file_state2.data();
		float* Vonline1 = (float*) file_Vonline1.data();
		float* Vstable1 = (float*) file_Vstable1.data();
		float* Vstable2 = (float*) file_Vstable2.data();
		float* Vtarget  = (float*) file_Vtarget.data();
		int* step       = (int*) file_step.data();
		float step1 = 1.0f / *std::max_element(step, step+N);
		vertex.resize(2*6*N);
		vcolor.resize(2*6*N);
		float z_range1 = 1.0 / z_range;
		float xy_range1 = 1.0 / xy_range;
		if (tsne && (int)tsne_x1x2.size() != 4*N) {
			tsne_start(myself);
			assert((int)tsne_x1x2.size() == 4*N);
		}
		int part2 = N*6;
		int cursor = 0;
		rendered2index.clear();
		for (int c=0; c<N; c++) {
			bool filtered = episode_filter != -1 && episode_of(c) != episode_filter;
			if (filtered) continue;
			rendered2index.push_back(c);
			if (!tsne) {
				vertex[6*cursor+0] = xy_range1*s1[STATEDIM*c+axis1];
				vertex[6*cursor+1] = xy_range1*s1[STATEDIM*c+axis2];
				vertex[6*cursor+3] = xy_range1*s2[STATEDIM*c+axis1];
				vertex[6*cursor+4] = xy_range1*s2[STATEDIM*c+axis2];
			} else {
				vertex[6*cursor+0] = xy_range1*tsne_x1x2[2*c+0];
				vertex[6*cursor+1] = xy_range1*tsne_x1x2[2*c+1];
				vertex[6*cursor+3] = xy_range1*tsne_x1x2[2*c+0 + N];
				vertex[6*cursor+4] = xy_range1*tsne_x1x2[2*c+1 + N];
			}
			vertex[6*cursor+2] = Vstable1[c] * z_range1;
			vertex[6*cursor+5] = Vstable2[c] * z_range1;

			vcolor[6*cursor+0] = 0.3f;
			vcolor[6*cursor+1] = 0.3f;
			vcolor[6*cursor+2] = 0.3f + step[c]*step1*0.7f;
			vcolor[6*cursor+3] = 0;
			vcolor[6*cursor+4] = 0.2f;
			vcolor[6*cursor+5] = 0.5*step[c]*step1;

			vertex[6*cursor+0+part2] = vertex[6*cursor+0];
			vertex[6*cursor+1+part2] = vertex[6*cursor+1];
			vertex[6*cursor+2+part2] = Vonline1[c] * z_range1;
			vertex[6*cursor+3+part2] = vertex[6*cursor+0];
			vertex[6*cursor+4+part2] = vertex[6*cursor+1];
			vertex[6*cursor+5+part2] = Vtarget[c] * z_range1;
			if (Vtarget[c] > Vonline1[c]) {
				vcolor[6*cursor+0+part2] = 0.0f;
				vcolor[6*cursor+1+part2] = 1.0f;
				vcolor[6*cursor+2+part2] = 0.0f;
				vcolor[6*cursor+3+part2] = 0.0f;
				vcolor[6*cursor+4+part2] = 1.0f;
				vcolor[6*cursor+5+part2] = 0.0f;
			} else {
				vcolor[6*cursor+0+part2] = 1.0f;
				vcolor[6*cursor+1+part2] = 0.0f;
				vcolor[6*cursor+2+part2] = 0.0f;
				vcolor[6*cursor+3+part2] = 1.0f;
				vcolor[6*cursor+4+part2] = 0.0f;
				vcolor[6*cursor+5+part2] = 0.0f;
			}
			cursor++;
		}
		render_N = cursor;
		assert((int)rendered2index.size()==render_N);
	}

	int render_N = 0;
	int tsne_iteration = 0;
	std::vector<double> tsne_x1x2;
	double tsne_error = 0;
	bool tsne_shuddown = false;
	bool tsne_done = false;
	boost::thread* tsne_thread = 0;

	void tsne_stop()
	{
		if (!tsne_thread) return;
		tsne_shuddown = true;	
		tsne_thread->join();
		delete tsne_thread;
	}

	void tsne_start(const shared_ptr<Quiver>& myself)
	{
		tsne_stop();
		tsne_x1x2.assign(4*N, 0.0);
		tsne_done = false;
		tsne_shuddown = false;
		tsne_thread = new boost::thread( [myself] ()  { myself->tsne_thread_func(); } );
	}

	void tsne_thread_func()
	{
		std::vector<double> v;
		std::vector<double> costs;
		v.resize(2*N*STATEDIM);
		float* s1 = (float*) file_state1.data();
		float* s2 = (float*) file_state2.data();
		for (int n=0; n<N*STATEDIM; n++) {
			v[n             ] = s1[n];
			v[n + N*STATEDIM] = s2[n];
		}
		costs.resize(2*N);
		double perplexity = 30;
		double theta = 0.5;
		int rand_seed = 5;
		srand((unsigned int) rand_seed);
		TSNE tsne;
		tsne.run(v.data(), 2*N, STATEDIM, tsne_x1x2.data(), 2, perplexity, theta, &tsne_iteration, &tsne_error, &tsne_shuddown);
		tsne_done = true;
	}

	void draw(int highlight_n)
	{
		if (vertex.empty()) return;
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(3, GL_FLOAT, 0, vertex.data());
		glColorPointer(3, GL_FLOAT, 0, vcolor.data());
		int part2 = N*2;
		glDrawArrays(GL_LINES, 0,     2*render_N);
		glDrawArrays(GL_LINES, part2, 2*render_N);

		glVertexPointer(3, GL_FLOAT, 6*4, vertex.data());
		glColorPointer(3, GL_FLOAT, 6*4, vcolor.data());
		glPointSize(3.0f);
		glDrawArrays(GL_POINTS, 0, render_N);

		if (highlight_n != -1) {
			glPointSize(9.0f);
			glDrawArrays(GL_POINTS, highlight_n, 1);
		}

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
};

class Viz: public QGLWidget {
	Q_OBJECT
public:
	Viz():
		QGLWidget(QGLFormat(QGL::SampleBuffers), 0)
	{
		setFocusPolicy(Qt::StrongFocus);
		setMouseTracking(true);
	}

	shared_ptr<Quiver> q;

	void initializeGL()
	{
		qglClearColor(Qt::black);
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glEnable(GL_DEPTH_TEST);
	}

	void paintGL()
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		float near = 0.1;
		float far  = 5.0;
		float typical = 1.0;
		float r = typical/far*near;
		glFrustum(-r, +r, -r, +r, 0.1, 55.0);

		glMatrixMode(GL_MODELVIEW);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();
		glTranslatef(0.0, 0.0, -10.0);
		glRotatef(yrot, 1.0, 0.0, 0.0);
		glRotatef(xrot, 0.0, 0.0, 1.0);
		glScalef(wheel,wheel,wheel);
		glTranslatef(-user_x, -user_y, -user_z);

		glColor3f(0.3f, 0.3f, 0.3f);
		glVertexPointer(3, GL_FLOAT, 0, line_vertex);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_LINES, 0, sizeof(line_vertex)/sizeof(float)/3);
		glDisableClientState(GL_VERTEX_ARRAY);

		closest_indx = -1;
		closest_rend = -1;
		double closest_dist = 60;
		
		if (q) {
			const char* jpeg = (const char*) q->file_jpeg.data();
			int mid_h = rect().height();
			std::array<GLdouble, 16> projection;
			std::array<GLdouble, 16> modelview;
			std::array<GLdouble, 3>  screen_coords;
			glGetDoublev(GL_PROJECTION_MATRIX, projection.data());
			glGetDoublev(GL_MODELVIEW_MATRIX, modelview.data());
			std::array<GLint,4> viewport;
			glGetIntegerv(GL_VIEWPORT, viewport.data());
			float* vertex = q->vertex.data();
			for (int c=0; c<q->render_N; c++) {
				int idx = q->rendered2index[c];
				if (jpeg[16*idx]==0) continue;
				gluProject(
					vertex[6*c+0], vertex[6*c+1], vertex[6*c+2],
					modelview.data(),
					projection.data(),
					viewport.data(),
					screen_coords.data(), screen_coords.data() + 1, screen_coords.data() + 2);
				int x = screen_coords[0];
				int y = mid_h - screen_coords[1];
				double dist = fabs(x - mouse_prev_x) + fabs(y - mouse_prev_y);
				if (dist < closest_dist) {
					closest_dist = dist;
					closest_rend = c;
					closest_indx = idx;
				}
			}
		}

		if (q) q->draw(closest_rend);

		if (closest_indx!=-1) {
			glLoadIdentity();
			//glPointSize(26.0f);
			//glColor3f(1.0, 1.0, 1.0);
			//glBegin(GL_POINTS);
			float x = (mouse_prev_x + 10 - rect().width()/2) / side*r*2;
			float y = (rect().height()/2 - 10 - mouse_prev_y) / side*r*2;
			//glVertex3f(x, y,-0.10);
			//glEnd();

			QImage test(q->jpeg_of(closest_indx).c_str());
			glRasterPos3f( x, y, -0.10 );
			glPixelZoom(1.0, -1.0);
			glDrawPixels(test.width(), test.height(), 
				GL_BGRA, GL_UNSIGNED_BYTE, 
				test.bits() );
		}
	}

	void resizeGL(int w, int h)
	{
		side = std::max(w, h);
		glViewport((w - side) / 2, (h - side) / 2, side, side);
	}

	int closest_indx = -1;
	int closest_rend = -1;
	int side;
	float user_x = 0;
	float user_y = 0;
	float user_z = 0;
	float xrot = 0;
	float yrot = -40;
	float zrot = 0;
	float wheel = 1;
	float z_range = 1.0;

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
		wheel *= (1 + 0.001*wev->delta());
		updateGL();
	}

	void mousePressEvent(QMouseEvent* mev)
	{
		updateGL();
		if (mev->button()==Qt::LeftButton) {
			mouse_init_x = mouse_prev_x = mev->x();
			mouse_init_y = mouse_prev_y = mev->y();
			drag = true;
			mouse_screenshot = false;
		}
	}

	void mouseReleaseEvent(QMouseEvent* mev)
	{
		if (mev->button()==Qt::LeftButton) {
			drag = false;
			double traveled = fabs(mouse_init_x - mev->x()) + fabs(mouse_init_y - mev->y());
			if (traveled < 5 && q) {
				q->episode_filter = q->episode_of(closest_indx);
			}
		}
	}

	bool mouse_screenshot = false;
	bool drag = false;
	double mouse_prev_x = 0;
	double mouse_prev_y = 0;
	double mouse_init_x = 0;
	double mouse_init_y = 0;

	void mouseMoveEvent(QMouseEvent* mev)
	{
		if (drag) {
			xrot += 0.02*(mev->x() - mouse_prev_x);
			yrot += 0.02*(mev->y() - mouse_prev_y);
			mouse_prev_x = mev->x();
			mouse_prev_y = mev->y();
			updateGL();
			mouse_screenshot = false;
		} else {
			mouse_prev_x = mev->x();
			mouse_prev_y = mev->y();
			mouse_screenshot = true;
		}
	}
};

class VizWindow: public QWidget {
	Q_OBJECT
public:
	Viz* viz_widget;
	QGridLayout* grid;
	
	QDoubleSpinBox* z_range;
	QDoubleSpinBox* xy_range;
	
	QSettings* ini;
	QTimer* timer;
	
	QRadioButton* radio_tsne;
	QRadioButton* radio_axis;
	QSpinBox* axis1;
	QSpinBox* axis2;
	
	VizWindow(const std::string& dir):
		QWidget(0)
	{
		timer = new QTimer(this);
		QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));
		timer->start(1000/30);
		
		ini = new QSettings( (dir + "/viz.ini").c_str(), QSettings::IniFormat, this);
		grid = new QGridLayout();
		setLayout(grid);
		
		int row = 0;
		grid->addWidget(new QLabel("Z range:"), row, 0);
		z_range = new QDoubleSpinBox();
		z_range->setRange(0.1, 1000);
		z_range->setSingleStep(0.1);
		z_range->setValue(ini->value("z_range").toDouble());
		grid->addWidget(z_range, row, 1);
		row++;

		grid->addWidget(new QLabel("XY range:"), row, 0);
		xy_range = new QDoubleSpinBox();
		xy_range->setRange(0.1, 1000);
		xy_range->setSingleStep(0.1);
		xy_range->setValue(ini->value("xy_range").toDouble());
		grid->addWidget(xy_range, row, 1);
		row++;

		{
			QGroupBox* box = new QGroupBox(tr("XY"));
			radio_tsne = new QRadioButton("t-SNE");
			radio_axis = new QRadioButton("Axis");
			radio_tsne->setChecked(ini->value("xy_tsne").toBool());
			radio_axis->setChecked(ini->value("xy_axis").toBool());
			grid->addWidget(box, row, 0, 1, 2);
			QVBoxLayout* vbox = new QVBoxLayout;
			box->setLayout(vbox);
			vbox->addWidget(radio_tsne);
			vbox->addWidget(radio_axis);
			axis1 = new QSpinBox();
			axis2 = new QSpinBox();
			axis1->setValue(ini->value("xy_axis_n1").toInt());
			axis2->setValue(ini->value("xy_axis_n2").toInt());
			vbox->addWidget(axis1);
			vbox->addWidget(axis2);
			row++;
		}

		grid->setRowStretch(row, 1);
		grid->setColumnStretch(2, 1);

		viz_widget = new Viz();
		grid->addWidget(viz_widget, 0, 2, row+1, 1);

		timeout();
	}

	~VizWindow()
	{
		ini->setValue("xy_range", xy_range->value());
		ini->setValue("z_range", z_range->value());
		ini->setValue("xy_tsne", radio_tsne->isChecked());
		ini->setValue("xy_axis", radio_axis->isChecked());
		ini->setValue("xy_axis_n1", axis1->value());
		ini->setValue("xy_axis_n2", axis2->value());
	}

public slots:
	void timeout()
	{
		if (viz_widget->q) {
			viz_widget->q->reprocess(
				viz_widget->q,
				xy_range->value(),
				z_range->value(),
				radio_tsne->isChecked(), axis1->value(), axis2->value()
				);
		}
		viz_widget->xrot += 0.01;
		viz_widget->updateGL();
	}
};

int main(int argc, char *argv[])
{
	if (argc<2) {
		fprintf(stderr, "Usage:\n%s DIR\n", argv[0]);
		return 1;
	}
	QApplication app(argc, argv);
	std::string dir = argv[1];
	VizWindow window(dir);
	try {
		shared_ptr<Quiver> q;
		q.reset(new Quiver);
		q->open(dir);
		window.viz_widget->q = q;

	} catch (const std::exception& e) {
		fprintf(stderr, "ERROR: %s\n", e.what());
		return 1;
	}

	window.showMaximized();
	return app.exec();
}

#include "../.generated/viz.moc"

