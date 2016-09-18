#include "viz-qnet3d.h"
#include <QtGui/QKeyEvent>

#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <GL/glu.h>

using boost::shared_ptr;
using boost::iostreams::mapped_file_source;

static float line_vertex[] = {
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
	mapped_file_source file_N;
	mapped_file_source file_s;
	mapped_file_source file_v;
	mapped_file_source file_sn;
	mapped_file_source file_vn;
	mapped_file_source file_st;
	mapped_file_source file_vt;
	mapped_file_source file_sp;
	mapped_file_source file_vp;
	mapped_file_source file_ttest;
	mapped_file_source file_step;
	mapped_file_source file_episode;
	mapped_file_source file_jpeg;
	mapped_file_source file_flags;
	mapped_file_source file_action_online;
	mapped_file_source file_action_stable;
	mapped_file_source file_agraph_online;
	mapped_file_source file_agraph_stable;

	std::vector<float> vertex; // x y z
	std::vector<float> vcolor; // r g b
	std::vector<int> rendered2index;
	std::string dir;
	std::string env_dir;

	void open(const std::string& env_dir_, const std::string& dir_)
	{
		dir = dir_;
		env_dir = env_dir_;
		file_N.open(dir + "/mmap_N");
		file_s .open(dir + "/mmap_s");
		file_v .open(dir + "/mmap_v");
		file_sn.open(dir + "/mmap_sn");
		file_vn.open(dir + "/mmap_vn");
		file_st.open(dir + "/mmap_st");
		file_vt.open(dir + "/mmap_vt");
		file_sp.open(dir + "/mmap_sp");
		file_vp.open(dir + "/mmap_vp");
		file_ttest.open(dir + "/mmap_ttest");
		file_step.open(dir + "/mmap_step");
		file_episode.open(dir + "/mmap_episode");
		file_jpeg.open(dir + "/mmap_jpegmap");
		file_flags.open(dir + "/mmap_flags");
		file_action_online.open(dir + "/mmap_action_online");
		file_action_stable.open(dir + "/mmap_action_stable");
		file_agraph_online.open(dir + "/mmap_agraph_online");
		file_agraph_stable.open(dir + "/mmap_agraph_stable");
	}

	void close()
	{
		file_N.close();
		file_s.close();
		file_v.close();
		file_sn.close();
		file_vn.close();
		file_st.close();
		file_vt.close();
		file_sp.close();
		file_vp.close();
		file_ttest.close();
		file_step.close();
		file_episode.close();
		file_jpeg.close();
		file_flags.close();
		file_action_online.close();
		file_action_stable.close();
		file_agraph_online.close();
		file_agraph_stable.close();
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
		return env_dir + "/" + (jpeg + 16*n); // ".BipedalWalker-v2/z00020.jpg"
	}

	float val_from_axis(float* s, float xy_range1, int axis, int step, float step_f, float step_k, float V, float V_range1, bool mode_transition)
	{
		if (axis>=0 && axis<STATEDIM) return s[axis]*xy_range1;
		if (axis==-1) return -1 + (step-step_f)*step_k;
		if (axis==-2) return mode_transition ? 0.0 : V*V_range1;
		return 0;
	}

	void print_about(int ind)
	{
		int i = ind;
		float* Vstable1 = (float*) file_v.data();
		float* Vstable2 = (float*) file_vn.data();
		const int* episode = (const int*) file_episode.data();
		float* Vtarget  = (float*) file_vt.data();
		const char* jpeg = (const char*) file_jpeg.data();
		int this_episode = episode[i];
		while (this_episode==episode[i]) {
			printf("P[%06i] x.v=%0.5f t=%0.5f x.sn=%0.5f jpeg=%s\n", i, Vstable1[i], Vtarget[i], Vstable2[i], jpeg + 16*i);
			++i;			
		}
	}

	void fill_color(float v, float* save_here)
	{
		QColor t;
		v = std::max(-1.0f, v);
		v = std::min(+1.0f, v);
		t.setHslF(0.5-0.5*v, 1, 0.5);
		qreal r,g,b;
		t.getRgbF(&r, &g, &b);
		save_here[0] = r;
		save_here[1] = g;
		save_here[2] = b;
	}

	void reprocess(
		float xy_range, float z_range,
		int axis1,
		int axis2,
		int axis3,
		int axis4,
		int timefilter_t1, int timefilter_t2,
		// s_vn
		bool mode_transition,
		bool mode_policy,
		bool mode_target
		)
	{
		if (!file_N.is_open()) return;
		int new_N = ((int*)file_N.data())[0];
		if (new_N!=N) {
			STATEDIM = file_s.size() / file_v.size();
			N = new_N;
			render_N = 0;
			printf("N=%i STATEDIM=%i\n", N, STATEDIM);
			close();
			open(env_dir, dir);
		}
		if (N==0) return;
		float* s1 = (float*) file_s.data();
		float* s2 = (float*) file_sn.data();
		float* v1 = (float*) file_v.data();
		float* v2 = (float*) file_vn.data();
		int* step = (int*)   file_step.data();
		int* flags = (int*)  file_flags.data();
		if (mode_transition) {
			s2 = (float*) file_ttest.data();
			v2 = v1;
		} else if (mode_target) {
			s2 = (float*) file_st.data();
			v2 = (float*) file_vt.data();
		} else if (mode_policy) {
			s2 = (float*) file_sp.data();
			v2 = (float*) file_vp.data();
		}
		float step_f;
		float step_k;
		if (timefilter_t1 >= 0) {
			step_f = timefilter_t1;
			step_k = 2.0f / std::max(timefilter_t2-timefilter_t1, 1);
		} else {
			step_f = 0;
			step_k = 2.0f / *std::max_element(step, step+N);
		}
		vertex.resize(2*6*N);
		vcolor.resize(2*6*N);
		float z_range1 = 1.0 / z_range;
		float xy_range1 = 1.0 / xy_range;
		//int part2 = N*6;
		int cursor = 0;
		rendered2index.clear();
		for (int c=0; c<N; c++) {
			bool filtered = episode_filter != -1 && episode_of(c) != episode_filter;
			if (timefilter_t1 >= 0) filtered |= (step[c] < timefilter_t1) || (step[c] > timefilter_t2);
			if (filtered) continue;
			rendered2index.push_back(c);

			vertex[6*cursor+0] = val_from_axis( s1+STATEDIM*c, xy_range1, axis1, step[c], step_f, step_k, v1[c], z_range1, mode_transition );
			vertex[6*cursor+1] = val_from_axis( s1+STATEDIM*c, xy_range1, axis2, step[c], step_f, step_k, v1[c], z_range1, mode_transition );
			vertex[6*cursor+2] = val_from_axis( s1+STATEDIM*c, xy_range1, axis3, step[c], step_f, step_k, v1[c], z_range1, mode_transition );
			float color1       = val_from_axis( s1+STATEDIM*c, xy_range1, axis4, step[c], step_f, step_k, v1[c], z_range1, mode_transition );
			fill_color(color1, vcolor.data()+6*cursor);

			vertex[6*cursor+3] = val_from_axis( s2+STATEDIM*c, xy_range1, axis1, step[c]+1, step_f, step_k, v2[c], z_range1, mode_transition );
			vertex[6*cursor+4] = val_from_axis( s2+STATEDIM*c, xy_range1, axis2, step[c]+1, step_f, step_k, v2[c], z_range1, mode_transition );
			vertex[6*cursor+5] = val_from_axis( s2+STATEDIM*c, xy_range1, axis3, step[c]+1, step_f, step_k, v2[c], z_range1, mode_transition );
			float color2       = val_from_axis( s2+STATEDIM*c, xy_range1, axis4, step[c]+1, step_f, step_k, v2[c], z_range1, mode_transition );
			fill_color(color2, vcolor.data()+6*cursor+3);

			if (flags[c] & 1) {
				vcolor[6*cursor+3] = 0.1;
				vcolor[6*cursor+4] = 0.1;
				vcolor[6*cursor+5] = 0.1;
			}
			if (flags[c] & 2) {
				vcolor[6*cursor+0] = 1.0;
				vcolor[6*cursor+1] = 1.0;
				vcolor[6*cursor+2] = 1.0;
				vcolor[6*cursor+3] = 1.0;
				vcolor[6*cursor+4] = 1.0;
				vcolor[6*cursor+5] = 1.0;
			}

			cursor++;
		}

		render_N = cursor;
		assert((int)rendered2index.size()==render_N);
	}

	int render_N = 0;
	
	void draw(int highlight_n)
	{
		if (vertex.empty()) return;
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(3, GL_FLOAT, 0, vertex.data());
		glColorPointer(3, GL_FLOAT, 0, vcolor.data());
		int part2 = N*2;
		glLineWidth(1.0);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

		glDrawArrays(GL_LINES, 0,     2*render_N);

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

	void actions_reprocess(float w, float h, float z_range1)
	{
		if (!file_N.is_open()) return;
		int new_action_dim = file_action_online.size() / sizeof(float);
		if (new_action_dim!=ACTION_DIM) {
			ACTION_DIM = new_action_dim;
			ACTION_PIXELS = file_agraph_online.size() / file_action_online.size();
			printf("ACTION_DIM=%i, PIXELS=%i\n", ACTION_DIM, ACTION_PIXELS);
		}

		int SPACING = 10;
		float window_h = std::min(h / ACTION_DIM, 300.0f);
		w -= SPACING;
		float window_w = window_h * 4/3;

		agraph_border.resize(6*3*ACTION_DIM);
		//agraph.resize(ACTION_DIM);
		agraph_online.resize(3*ACTION_DIM*ACTION_PIXELS);
		agraph_stable.resize(3*ACTION_DIM*ACTION_PIXELS);
		agraph_color.resize(3*ACTION_DIM*ACTION_PIXELS);
		agraph_action_online.resize(3*2*ACTION_DIM);
		agraph_action_stable.resize(3*2*ACTION_DIM);
		float* action_online = (float*)file_action_online.data();
		float* action_stable = (float*)file_action_stable.data();
		float* agraph_online_p = (float*)file_agraph_online.data();
		float* agraph_stable_p = (float*)file_agraph_stable.data();
		for (int c=0; c<ACTION_DIM; ++c) {
			float y = (window_h+SPACING)*c + SPACING;
			Graph g; // = agraph[c];
			g.dy = y + window_h/2;
			g.dx = w - window_w/2;
			g.kx = window_w/2;
			g.ky = -window_h/2;

			agraph_border[6*3*c +  0 + 0] = -1.0*g.kx + g.dx;
			agraph_border[6*3*c +  0 + 1] = +1.0*g.ky + g.dy;
			agraph_border[6*3*c +  0 + 2] = -0.1;
			agraph_border[6*3*c +  3 + 0] = -1.0*g.kx + g.dx;
			agraph_border[6*3*c +  3 + 1] = -1.0*g.ky + g.dy;
			agraph_border[6*3*c +  3 + 2] = -0.1;

			agraph_border[6*3*c +  6 + 0] = -1.0*g.kx + g.dx;
			agraph_border[6*3*c +  6 + 1] =  0.0*g.ky + g.dy;
			agraph_border[6*3*c +  6 + 2] = -0.1;
			agraph_border[6*3*c +  9 + 0] = +1.0*g.kx + g.dx;
			agraph_border[6*3*c +  9 + 1] =  0.0*g.ky + g.dy;
			agraph_border[6*3*c +  9 + 2] = -0.1;

			agraph_border[6*3*c + 12 + 0] = +1.0*g.kx + g.dx;
			agraph_border[6*3*c + 12 + 1] = -1.0*g.ky + g.dy;
			agraph_border[6*3*c + 12 + 2] = -0.1;
			agraph_border[6*3*c + 15 + 0] = +1.0*g.kx + g.dx;
			agraph_border[6*3*c + 15 + 1] = +1.0*g.ky + g.dy;
			agraph_border[6*3*c + 15 + 2] = -0.1;

			for (int p=0; p<ACTION_PIXELS; ++p) {
				float x = -1 + 2.0*p/ACTION_PIXELS;
				float ys = z_range1*agraph_stable_p[ACTION_PIXELS*c + p];
				if (ys> 1.0) ys =  1.0;
				if (ys<-1.0) ys = -1.0;
				float yo = z_range1*agraph_online_p[ACTION_PIXELS*c + p];
				if (yo> 1.0) yo =  1.0;
				if (yo<-1.0) yo = -1.0;
				agraph_online[ACTION_PIXELS*3*c + 3*p + 0] =  x*g.kx + g.dx;
				agraph_online[ACTION_PIXELS*3*c + 3*p + 1] = yo*g.ky + g.dy;
				agraph_online[ACTION_PIXELS*3*c + 3*p + 2] = -0.1;
				agraph_stable[ACTION_PIXELS*3*c + 3*p + 0] =  x*g.kx + g.dx;
				agraph_stable[ACTION_PIXELS*3*c + 3*p + 1] = ys*g.ky + g.dy;
				agraph_stable[ACTION_PIXELS*3*c + 3*p + 2] = -0.1;
				fill_color(ys, agraph_color.data() + ACTION_PIXELS*3*c + 3*p);
			}

			float x = action_online[c];
			agraph_action_online[2*3*c + 0 + 0] = x*g.kx + g.dx;
			agraph_action_online[2*3*c + 0 + 1] = +1.0*g.ky + g.dy;
			agraph_action_online[2*3*c + 0 + 2] = -0.1;
			agraph_action_online[2*3*c + 3 + 0] = x*g.kx + g.dx;
			agraph_action_online[2*3*c + 3 + 1] = -1.0*g.ky + g.dy;
			agraph_action_online[2*3*c + 3 + 2] = -0.1;
			x = action_stable[c];
			agraph_action_stable[2*3*c + 0 + 0] = x*g.kx + g.dx;
			agraph_action_stable[2*3*c + 0 + 1] = +1.0*g.ky + g.dy;
			agraph_action_stable[2*3*c + 0 + 2] = -0.1;
			agraph_action_stable[2*3*c + 3 + 0] = x*g.kx + g.dx;
			agraph_action_stable[2*3*c + 3 + 1] = -1.0*g.ky + g.dy;
			agraph_action_stable[2*3*c + 3 + 2] = -0.1;
		}
	}

	void actions_draw()
	{
		if (ACTION_DIM==0) return;

		glDisable(GL_DEPTH_TEST);
		glEnableClientState(GL_VERTEX_ARRAY);
		glColor3f(0.3f, 0.3f, 0.3f);
		glVertexPointer(3, GL_FLOAT, 0, agraph_border.data());
		glDrawArrays(GL_LINES, 0, 6*ACTION_DIM);

		glEnableClientState(GL_COLOR_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, agraph_online.data());
		glColorPointer(3, GL_FLOAT, 0, agraph_color.data());
		for (int c=0; c<ACTION_DIM; ++c)
			glDrawArrays(GL_LINE_STRIP, c*ACTION_PIXELS, ACTION_PIXELS);
		glDisableClientState(GL_COLOR_ARRAY);
		glColor3f(0.5f, 0.5f, 0.5f);
		glVertexPointer(3, GL_FLOAT, 0, agraph_stable.data());
		for (int c=0; c<ACTION_DIM; ++c)
			glDrawArrays(GL_LINE_STRIP, c*ACTION_PIXELS, ACTION_PIXELS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glVertexPointer(3, GL_FLOAT, 0, agraph_action_online.data());
		glDrawArrays(GL_LINES, 0, 2*ACTION_DIM);
		glColor3f(0.9f, 0.0f, 0.9f);
		glVertexPointer(3, GL_FLOAT, 0, agraph_action_stable.data());
		glDrawArrays(GL_LINES, 0, 2*ACTION_DIM);
		glDisableClientState(GL_VERTEX_ARRAY);
		glEnable(GL_DEPTH_TEST);
	}

	struct Graph {
		float dx, dy, kx, ky;
	};

	int ACTION_DIM = 0;
	int ACTION_PIXELS = 0;

	//std::vector<Graph> agraph;
	std::vector<float> agraph_border; // x y z
	std::vector<float> agraph_online; // x y z
	std::vector<float> agraph_color;  // r g b for online
	std::vector<float> agraph_stable; // x y z
	std::vector<float> agraph_action_online; // x y z two points for each action
	std::vector<float> agraph_action_stable; // x y z two points for each action
};

Viz::Viz():
	QGLWidget(QGLFormat(QGL::SampleBuffers), 0)
{
	setFocusPolicy(Qt::StrongFocus);
	setMouseTracking(true);
}

void Viz::initializeGL()
{
	qglClearColor(Qt::black);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_DEPTH_TEST);
}

void Viz::paintGL()
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

	if (q && q->file_N.is_open()) {
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

		std::string jpeg = q->jpeg_of(closest_indx);
		if (jpeg_reported != jpeg) {
			jpeg_reported = jpeg;
			printf("jpeg '%s'\n", jpeg.c_str());
		}

		QImage test(jpeg.c_str());
		if (test.isNull()) {
			fprintf(stderr, "cannot load jpeg '%s'\n", jpeg.c_str());
		} else {
			glRasterPos3f( x, y, -0.10 );
			glPixelZoom(1.0, -1.0);
			glDrawPixels(test.width(), test.height(),
				GL_BGRA, GL_UNSIGNED_BYTE,
				test.bits() );
		}
	}

	if (q) {
		glLoadIdentity();
		glScalef(1.0/side*r*2, -1.0/side*r*2, 1.0);
		glTranslatef(-rect().width()/2, -rect().height()/2, 0);
		q->actions_draw();
	}
}

void Viz::resizeGL(int w, int h)
{
	side = std::max(w, h);
	glViewport((w - side) / 2, (h - side) / 2, side, side);
}

void Viz::keyPressEvent(QKeyEvent* kev)
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

void Viz::wheelEvent(QWheelEvent* wev)
{
	wheel *= (1 + 0.001*wev->delta());
	updateGL();
}

void Viz::mousePressEvent(QMouseEvent* mev)
{
	updateGL();
	if (mev->button()==Qt::LeftButton) {
		mouse_init_x = mouse_prev_x = mev->x();
		mouse_init_y = mouse_prev_y = mev->y();
		drag = true;
		mouse_screenshot = false;
	}
}

void Viz::mouseReleaseEvent(QMouseEvent* mev)
{
	if (mev->button()==Qt::LeftButton) {
		drag = false;
		double traveled = fabs(mouse_init_x - mev->x()) + fabs(mouse_init_y - mev->y());
		if (traveled < 5 && q) {
			q->episode_filter = q->episode_of(closest_indx);
			if (closest_indx!=-1)
				q->print_about(closest_indx);
		}
	}
}

void Viz::mouseMoveEvent(QMouseEvent* mev)
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

void Viz::reopen(const std::string& env_dir, const std::string& dir)
{
	if (!q) {
		q.reset(new Quiver);
	}
	try {
		q->close();
		q->open(env_dir, dir);
	} catch (const std::exception& e) {
		fprintf(stderr, "Viz::reopen(): %s\n", e.what());
	}
}

void Viz::reprocess(
	float xy_range, float z_range,
	int axis1,
	int axis2,
	int axis3,
	int axis4,
	int timefilter_t1, int timefilter_t2,
	// s_vn
	bool mode_transition,
	bool mode_policy,
	bool mode_target)
{
	q->reprocess(
		xy_range, z_range,
		axis1, axis2, axis3, axis4,
		timefilter_t1, timefilter_t2,
		mode_transition,
		mode_policy,
		mode_target
		);
}

void Viz::actions_reprocess(double z_range)
{
	q->actions_reprocess(
		rect().width(), rect().height(),
		z_range);
}
