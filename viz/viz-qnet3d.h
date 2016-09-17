#include <QtOpenGL/QGLWidget>
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;

class Quiver;

class Viz: public QGLWidget {
public:
	Viz();
	void initializeGL();
	void paintGL();
	void resizeGL(int w, int h);
	void keyPressEvent(QKeyEvent* kev);
	void wheelEvent(QWheelEvent* wev);
	void mousePressEvent(QMouseEvent* mev);
	void mouseReleaseEvent(QMouseEvent* mev);
	void mouseMoveEvent(QMouseEvent* mev);

	void reprocess(
		float xy_range, float z_range,
		int axis1, int axis2, int axis3, int axis4,
		int timefilter_t1, int timefilter_t2,
		bool mode_transition, bool mode_policy, bool mode_target);
	void reopen(const std::string& env_dir, const std::string& dir);
	void actions_reprocess(
		double z_range);
	void timeout();

	shared_ptr<Quiver> q;

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

	std::string jpeg_reported;

	bool mouse_screenshot = false;
	bool drag = false;
	double mouse_prev_x = 0;
	double mouse_prev_y = 0;
	double mouse_init_x = 0;
	double mouse_init_y = 0;
};
