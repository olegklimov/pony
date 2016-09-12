#include "viz-qnet3d.h"
#include "viz-progress.h"
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

using boost::shared_ptr;

class VizWindow: public QWidget {
	Q_OBJECT
public:
	Viz* viz_widget;

	QDoubleSpinBox* z_range;
	QDoubleSpinBox* xy_range;

	QSettings* ini;
	QTimer* timer;
	QTimer* timer_long;

	QTabWidget* tabbed;

	QWidget* page_3d;
	QSpinBox* axis[4];
	QRadioButton* mode_s_vn;
	QRadioButton* mode_target;
	QRadioButton* mode_transition;
	QRadioButton* mode_policy;
	QCheckBox* timefilter_checkbox;
	QSpinBox* timefilter_t1;
	QSpinBox* timefilter_t2;

	QWidget* page_progress;
	//boost::shared_ptr<Progress> progress;

	VizWindow(const std::string& dir):
		QWidget(0),
		dir(dir)
	{
		timer = new QTimer(this);
		QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));
		timer->start(1000/10);

		timer_long = new QTimer(this);
		QObject::connect(timer_long, SIGNAL(timeout()), this, SLOT(timeout_long()));
		timer_long->start(1000);

		ini = new QSettings( (dir + "/viz.ini").c_str(), QSettings::IniFormat, this);

		tabbed = new QTabWidget();
		tabbed->setTabPosition(QTabWidget::West);

		create_3d_page();
		create_progress_page();
		tabbed->addTab(page_3d, "&QNet");
		tabbed->addTab(page_progress, "&Progress");

		QVBoxLayout* vbox = new QVBoxLayout();
		setLayout(vbox);
		vbox->addWidget(tabbed);
	}

	void create_progress_page()
	{
		page_progress = progress_widget_create(dir + "/progress");
		QGridLayout* grid = new QGridLayout();
		page_progress->setLayout(grid);
	}

	void create_3d_page()
	{
		page_3d = new QWidget();
		QGridLayout* grid = new QGridLayout();
		page_3d->setLayout(grid);

		int row = 0;
		grid->addWidget(new QLabel("Z range:"), row, 0);
		z_range = new QDoubleSpinBox();
		z_range->setRange(0.1, 1000000);
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
			QVBoxLayout* vbox = new QVBoxLayout;
			for (int c=0; c<4; c++) {
				axis[c] = new QSpinBox();
				axis[c]->setMinimum(-2);
				vbox->addWidget(axis[c]);
			}
			axis[0]->setValue(ini->value("xy_axis_n1").toInt());
			axis[1]->setValue(ini->value("xy_axis_n2").toInt());
			axis[2]->setValue(ini->value("xy_axis_n3").toInt());
			axis[3]->setValue(ini->value("xy_axis_n4").toInt());
			grid->addLayout(vbox, row, 0, 1, 2);
			row++;
		}
		
		{
			QGroupBox* box = new QGroupBox(tr("Mode"));
			mode_s_vn = new QRadioButton("s / vn");
			mode_target = new QRadioButton("target / target");
			mode_transition = new QRadioButton("transition / flat");
			mode_policy = new QRadioButton("policy");
			mode_s_vn->setChecked(ini->value("mode_s_vn").toBool());
			mode_target->setChecked(ini->value("mode_target").toBool());
			mode_transition->setChecked(ini->value("mode_transition").toBool());
			mode_policy->setChecked(ini->value("mode_policy").toBool());
			grid->addWidget(box, row, 0, 1, 2);
			QVBoxLayout* vbox = new QVBoxLayout;
			box->setLayout(vbox);
			vbox->addWidget(mode_s_vn);
			vbox->addWidget(mode_target);
			vbox->addWidget(mode_transition);
			vbox->addWidget(mode_policy);
			row++;
		}

		timefilter_checkbox = new QCheckBox("Time filter");
		timefilter_t1 = new QSpinBox();
		timefilter_t1->setMaximum(3000);
		timefilter_t2 = new QSpinBox();
		timefilter_t2->setMaximum(3000);
		timefilter_checkbox->setChecked(ini->value("timefilter_on").toBool());
		timefilter_t1->setValue(ini->value("timefilter_t1").toInt());
		timefilter_t2->setValue(ini->value("timefilter_t2").toInt());
		grid->addWidget(timefilter_checkbox, row, 0, 1, 2);
		row++;
		grid->addWidget(timefilter_t1, row, 0);
		grid->addWidget(timefilter_t2, row, 1);
		row++;

		grid->setRowStretch(row, 1);
		grid->setColumnStretch(2, 1);

		viz_widget = new Viz();
		grid->addWidget(viz_widget, 0, 2, row+1, 1);
		viz_widget->reopen(dir);

		timeout();
	}

	~VizWindow()
	{
		ini->setValue("xy_range", xy_range->value());
		ini->setValue("z_range", z_range->value());
		ini->setValue("mode_s_vn", mode_s_vn->isChecked());
		ini->setValue("mode_target", mode_target->isChecked());
		ini->setValue("mode_transition", mode_transition->isChecked());
		ini->setValue("mode_policy", mode_policy->isChecked());
		ini->setValue("xy_axis_n1", axis[0]->value());
		ini->setValue("xy_axis_n2", axis[1]->value());
		ini->setValue("xy_axis_n3", axis[2]->value());
		ini->setValue("xy_axis_n4", axis[3]->value());
		ini->setValue("timefilter_on", timefilter_checkbox->isChecked());
		ini->setValue("timefilter_t1", timefilter_t1->value());
		ini->setValue("timefilter_t2", timefilter_t2->value());
	}

	std::string dir;

public slots:
	void timeout()
	{
		if (viz_widget) {
			viz_widget->reprocess(
				xy_range->value(),
				z_range->value(),
				axis[0]->value(), axis[1]->value(), axis[2]->value(), axis[3]->value(),
				timefilter_checkbox->isChecked() ? timefilter_t1->value() : -1, timefilter_t2->value(),
				mode_transition->isChecked(), mode_policy->isChecked(), mode_target->isChecked()
				);
			viz_widget->actions_reprocess(
				1.0/z_range->value()
				);
		}
		viz_widget->xrot += 0.01;
		viz_widget->updateGL();
	}

	void timeout_long()
	{
		if (page_3d->isVisible())
			viz_widget->reopen(dir);
		if (page_progress->isVisible())
			progress_widget_rescan_dir(page_progress);
	}
};

int main(int argc, char *argv[])
{
	if (argc<2) {
		fprintf(stderr, "Usage:\n%s DIR\n", argv[0]);
		return 1;
	}
	QApplication app(argc, argv);
	VizWindow window(argv[1]);
	try {
		window.timeout_long();

	} catch (const std::exception& e) {
		fprintf(stderr, "ERROR: %s\n", e.what());
		return 1;
	}

	window.showMaximized();
	return app.exec();
}

#include "../.generated/viz.moc"

