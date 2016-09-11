#include "viz-progress.h"
#include <boost/thread/mutex.hpp>
#include <boost/filesystem.hpp>
#include <jsoncpp/value.h>
#include <jsoncpp/reader.h>
#include "miniutils.h"
#include <QtGui/QWidget>
#include <QtGui/QApplication>
#include <QtGui/QKeyEvent>
#include <QtGui/QPainter>
#include <QtGui/QTabWidget>
#include <QtGui/QFileDialog>
#include <QtCore/QSettings>
#include <ctime>

static boost::mutex progress_feed_mutex;

using std::string;

struct Progress {
	string file_folder;
	string my_filename;
	string my_name;
	Json::Value my;
	std::vector<Json::Value> others;
	std::vector<string> other_names;

	int prev_epoch_int = 0;
	double prev_epoch = 0;
	unsigned int learnset_cursor = 0;
	unsigned int testset_cursor = 0;
	int condensate_counter = 0;

	void save_logs();
};

boost::shared_ptr<Progress> progress_init(const std::string& task,
	const std::string& file_folder,
	const std::string& hint)
{
	boost::shared_ptr<Progress> p(new Progress);
	p->my["history"] = Json::Value(Json::arrayValue);
	p->my["history_testset"] = Json::Value(Json::arrayValue);
	p->my["task"] = task;

	time_t rawtime;
	time( &rawtime );
	char buf[80];
	struct tm* l = localtime( &rawtime );
	strftime(buf, 80, "%Y%m%d_%H%M%S", l);

	std::string fn = buf;
	boost::filesystem::path h = hint;
	fn += "_" + h.stem().string();
	fn += ".json";

	boost::filesystem::path t;
	t  = file_folder;
	t /= fn;

	p->my_filename = t.string();
	p->my_name = fn;

	std::map<string, Json::Value> tmp;
	boost::filesystem::directory_iterator end;
	for (boost::filesystem::directory_iterator i(file_folder); i!=end; ++i) {
		boost::filesystem::path fn = i->path();
		string ext = fn.extension().string();
		if (boost::filesystem::is_regular_file(fn) && (ext==".json")) {
			try {
				Json::Value v;
				Json::Reader r;
				r.parse(miniutils::read_file(fn.string()), v, false);
				string errs = r.getFormattedErrorMessages();
				if (!errs.empty()) throw std::logic_error(errs);
				tmp[fn.filename().string()] = v;

			} catch (const std::exception& e) {
				fprintf(stderr, "ERROR READING '%s': %s\n", fn.string().c_str(), e.what());
			}
		}
	}

	std::vector<QColor> other_colors;
	for (const auto& pair: tmp) {
		p->other_names.push_back(pair.first);
		const Json::Value& v = pair.second;
		p->others.push_back(v);
		QColor color(v["color"].asString().c_str());
		other_colors.push_back(color);
	}
	other_colors.push_back(Qt::black);
	other_colors.push_back(Qt::white);

	QColor best_my_color;
	double best_score = 0;
	for (int tries=0; tries<30; ++tries) {
		QColor t = qRgb(rand() % 256, rand() % 256, rand() % 256);
		double score = 65535;
		for (const QColor& x: other_colors) {
			score = std::min(score, sqrt(
				(t.red()-x.red())*(t.red()-x.red()) +
				(t.green()-x.green())*(t.green()-x.green()) +
				(t.blue()-x.blue())*(t.blue()-x.blue())
				));
		}
		if (score > best_score) {
			best_score = score;
			best_my_color = t;
		}
	}

	p->my["color"] = best_my_color.name().toAscii().data();

	//fprintf(stderr, "progress: %s, color %s, loaded %i others\n",
	//	p->my_filename.c_str(),
	//	best_my_color.name().toAscii().data(),
	//	(int)p->others.size());
	return p;
}

void progress_feed(
	const boost::shared_ptr<Progress>& p,
	double epoch, double time, double lr,
	const std::vector<double>& losses,
	bool testset,
	double sample_progress_every)
{
	p->condensate_counter++;

	boost::mutex::scoped_lock lock(progress_feed_mutex);
	Json::Value& rec =
		testset ?
		p->my["history_testset"][p->testset_cursor] :
		p->my["history"][p->learnset_cursor];
	rec[0U] = epoch;
	rec[1U] = time;
	rec[2U] = lr;

	if (testset) {
		Json::Value& tuple = rec[3U];
		double loss = 0;
		int L = losses.size();
		for (int i=0; i<L; i++) {
			assert(losses[i] >= 0);
			assert(losses[i]  < 1e10f);
			loss += losses[i];
			tuple[i] = losses[i];
		}
		tuple[L] = loss;
		p->testset_cursor++;

	} else {
		Json::Value& loss_min = rec[3U];
		Json::Value& loss_max = rec[4U];
		double loss = 0;
		int L = losses.size();
		for (int i=0; i<L; i++) {
			loss += losses[i];
			if (p->condensate_counter==1) {
				loss_min[i] = losses[i];
				loss_max[i] = losses[i];
			} else {
				loss_min[i] = std::min(loss_min[i].asDouble(), losses[i]);
				loss_max[i] = std::max(loss_max[i].asDouble(), losses[i]);
			}
		}
		if (p->condensate_counter==1) {
			loss_min[L] = loss;
			loss_max[L] = loss;
		} else {
			loss_min[L] = std::min(loss_min[L].asDouble(), loss);
			loss_max[L] = std::max(loss_max[L].asDouble(), loss);
		}

		bool save = (epoch - p->prev_epoch) > sample_progress_every;
		if (!save) return;

		p->prev_epoch = epoch;
		p->learnset_cursor++;
		p->condensate_counter = 0;
	}

	if (p->prev_epoch_int != int(epoch)) {
		p->prev_epoch_int = int(epoch);
		p->save_logs();
	}
}

void Progress::save_logs()
{
	boost::shared_ptr<FILE> f(miniutils::open_or_die(my_filename, "wb"), fclose);
	fprintf(f.get(), "%s", my.toStyledString().c_str());
}


class QTrainProgress: public QWidget {
public:
	struct LegendElement;

	boost::mutex progress_mutex;
	 boost::shared_ptr<Progress> progress_unsafe;

	boost::shared_ptr<Progress> progress_copy_qt_thread_only;
	bool outdated = true;

	QSettings settings;

	QTrainProgress(): settings("DSSL", "CaffeProgressPlot")
	{
		startTimer(1000);
		setMouseTracking(true);
	}

	void find_maximums(const Progress& s, double* max_epoch, double* first_loss, int* losses_count)
	{
		*max_epoch = 5.0;
		*first_loss = 1.0;
		*losses_count = 0;
		for (int c=0; c<=(int)s.others.size(); c++) {
			const Json::Value& log = c==(int)s.others.size() ? s.my : s.others[c];
			const Json::Value& h = log["history"];
			Json::Value last = h[h.size()-1U];
			Json::Value first = h[0U];
			*max_epoch = std::max(*max_epoch, last[0U].asDouble());
			const Json::Value& first_maxarray = first[4U];
			*losses_count = std::max(*losses_count, (int)first_maxarray.size());
			for (int l=0; l<*losses_count; l++) {
				if (!(losses_visible[c] & (1<<l))) continue;
				double loss = first_maxarray[l].asDouble();
				*first_loss = std::max(*first_loss, loss);
			}
		}
		*first_loss = fixed_scale;
	}

	QRect plot;
	double kx = 1;
	double ky = 1;
	int losses_count = 1;

	void paintEvent(QPaintEvent* pev) override
	{
		QPainter p(this);
		p.fillRect(rect(), Qt::black);
		if (!progress_copy_qt_thread_only) {
			return;
		}
		const Progress& s = *progress_copy_qt_thread_only;
		const int MARGIN = 10;
		double max_epoch;
		double first_loss;

		int N = s.others.size();
		if (losses_visible.empty()) {
			losses_visible.assign(N+1, 0);
			losses_visible[N] = 0xFFFF;
			for (int c=0; c<=N; c++) {
				QVariant v = settings.value(miniutils::stdprintf("losses-visible%i", c-N).c_str());
				if (v.isValid()) losses_visible[c] = (uint16_t) v.toInt();
			}
		}

		find_maximums(s, &max_epoch, &first_loss, &losses_count);

		double PLOTH = 650;
		plot = QRect(MARGIN, MARGIN, rect().width() - 2*MARGIN, PLOTH - 2*MARGIN);
		p.fillRect(plot, QColor(0x303030));
		kx = plot.width()  / max_epoch;
		ky = plot.height() / first_loss;
		p.setClipRect(plot);
		p.setPen(Qt::black);
		for (int e=1; e<1000; e++) {
			double x = plot.left() + kx*e;
			if (x > plot.right()) break;
			p.drawLine(QPointF(x,plot.top()), QPointF(x,plot.bottom()));
		}
		if (hl_shelf!=-1) {
			p.setPen(Qt::white);
			p.drawLine(QPointF(plot.left(),hl_shelf), QPointF(plot.right(),hl_shelf));
		}

		for (int n=0; n<=N; n++) {
			const Json::Value& log = n==N ? s.my : s.others[n];
			const Json::Value& h = log["history"];
			const Json::Value& t = log["history_testset"];
			QColor color(log["color"].asString().c_str());
			if (n==hl_n) color = color.lighter(140);
			std::vector<QPointF> drawme;
			std::vector<QPointF> drawme_test;
			int P = h.size();
			int T = t.size();
			drawme.resize(2*P);
			drawme_test.resize(T);
			for (int l=0; l<losses_count; l++) {
				if (!(losses_visible[n] & (1<<l))) continue;
				for (int i=0; i<P; ++i) {
					const Json::Value& pt = h[i];
					double pt_epoch       = pt[0U].asDouble();
					const Json::Value& pt_min = pt[3U];
					const Json::Value& pt_max = pt[4U];
					drawme[i      ].rx() = plot.left()   + kx*pt_epoch;
					drawme[i      ].ry() = plot.bottom() - ky*pt_max[l].asDouble();
					drawme[2*P-i-1].rx() = plot.left()   + kx*pt_epoch;
					drawme[2*P-i-1].ry() = plot.bottom() - ky*pt_min[l].asDouble();
				}
				for (int i=0; i<T; ++i) {
					const Json::Value& pt = t[i];
					double pt_epoch       = pt[0U].asDouble();
					const Json::Value& pt_val = pt[3U];
					drawme_test[i].rx() = plot.left()   + kx*pt_epoch;
					drawme_test[i].ry() = plot.bottom() - ky*pt_val[l].asDouble();;
				}
				p.setPen(color);
				p.setBrush(color);
				p.setOpacity(0.7);
				p.drawPolygon(drawme.data(), 2*P);
				p.setPen(color.lighter());
				p.setOpacity(1.0);
				p.drawPolyline(drawme_test.data(), T);
			}
		}

		p.setClipRect(QRect(), Qt::NoClip);
		int fh = p.fontMetrics().height();
		int textover = 0;
		for (int n=0; n<=N; n++) {
			const Json::Value& log = n==N ? s.my : s.others[n];
			string name = n==N ? s.my_name : s.other_names[n];

			QRect smallcolor(MARGIN,         PLOTH + MARGIN + (fh+3)*n,  fh, fh);
			QRect textrect(MARGIN+fh+MARGIN, PLOTH + MARGIN + (fh+3)*n, 300, fh);
			textover = 50 + textrect.right();

			QColor color(log["color"].asString().c_str());
			p.setBrush(color);
			p.setPen(color);
			p.setOpacity(0.7);
			p.drawRect(smallcolor);

			p.setOpacity(1.0);
			p.setPen(Qt::white);
			p.drawText(textrect, name.c_str());
			interactives.push_back({ textrect.adjusted(-2,-2,+2,+2), 0xFFFF, n });

			for (int l=0; l<losses_count; l++) {
				QRect r(textover + (fh+3)*l, PLOTH + MARGIN + (fh+3)*n, fh, fh);
				p.setPen(color);
				p.setBrush( (losses_visible[n]&(1<<l)) ? color : Qt::transparent);
				p.drawRect(r);
				interactives.push_back({ r.adjusted(-2,-2,+2,+2), uint16_t(1<<l), n });
			}
		}
		if (textover)
		for (int l=0; l<losses_count; l++) {
			QRect r(textover + (fh+3)*l, PLOTH + MARGIN + (fh+3)*(N+1), fh, fh);
			p.setPen(Qt::white);
			p.drawText(r, Qt::AlignCenter, l==losses_count-1 ? "S" : miniutils::stdprintf("%i", l).c_str());
			interactives.push_back({ r.adjusted(-2,-2,+2,+2), uint16_t(1<<l), -1 });
		}
		QRect desc = rect().adjusted(+MARGIN, +MARGIN, -MARGIN, -MARGIN);
		desc.setLeft(textover + losses_count*(fh+3) + MARGIN);
		desc.setTop(PLOTH + MARGIN);
		if (hl_n!=-1) {
			p.setPen(Qt::white);
			p.drawText(desc, Qt::AlignLeft|Qt::AlignTop, QString::fromUtf8(hl_desc.c_str()));
			QFontMetrics fm = p.fontMetrics();
			QString t = QString::fromUtf8(hl_pointdesc.c_str());
			QRect prect = QRect(QPoint(hl_x, hl_y), fm.size(0, t)).adjusted(0,-6-3,+6,-3);
			prect.moveBottom(hl_y);
			p.fillRect(prect, Qt::black);
			p.drawText(prect.adjusted(+3,+3,-3,-3), Qt::AlignLeft, t);
		}
	}

	std::vector<uint16_t> losses_visible;

	struct Interactive {
		QRect rect;
		uint16_t set_this;
		int here;
	};
	std::vector<Interactive> interactives;

	void interactive_apply(const Interactive& i)
	{
		const Progress& s = *progress_copy_qt_thread_only;
		int N = s.others.size();
		bool anything_changed = false;
		for (int c=0; c<=N; c++) {
			if (i.here==c || i.here==-1) {
				uint16_t new_version = losses_visible[c] | i.set_this;
				anything_changed = new_version != losses_visible[c];
				losses_visible[c] = new_version;
			}
		}
		if (!anything_changed)
		for (int c=0; c<=N; c++) {
			if (i.here==c || i.here==-1) {
				losses_visible[c] = losses_visible[c] & ~i.set_this;
			}
		}
		update();
		for (int c=0; c<=N; c++)
			settings.setValue(miniutils::stdprintf("losses-visible%i", c-N).c_str(), (int) losses_visible[c]);
		settings.sync();
	}

	void mouse_event(QMouseEvent* mev)
	{
		if (losses_visible.empty()) return;
		if (!progress_copy_qt_thread_only) return;
		const Progress& s = *progress_copy_qt_thread_only;

		for (const Interactive& i: interactives) {
			if (i.rect.contains(mev->pos()) && mev->type()==QEvent::MouseButtonPress) {
				interactive_apply(i);
				return;
			}
		}

		int new_hl_n = -1;
		int new_hl_l = -1;
		double closest = 50;
		std::string new_hl_pointdesc;
		int N = s.others.size();
		for (int n=0; n<=N; n++) {
			const Json::Value& log = n==N ? s.my : s.others[n];
			const Json::Value& h = log["history"];
			for (int l=0; l<losses_count; l++) {
				if (!(losses_visible[n] & (1<<l))) continue;
				int P = h.size();
				for (int i=0; i<P; ++i) {
					const Json::Value& pt = h[i];
					double x = plot.left() + kx*pt[0U].asDouble();
					if (abs(x - mev->x()) > 10) continue;
					const Json::Value& pt_min = pt[3U];
					const Json::Value& pt_max = pt[4U];
					double y1 = plot.bottom() - ky*pt_max[l].asDouble();
					double y2 = plot.bottom() - ky*pt_min[l].asDouble();
					assert(y1 <= y2);
					double dist = 0;
					if (mev->y() < y1) dist += abs(mev->y() - y1);
					if (mev->y() > y2) dist += abs(mev->y() - y2);
					dist += -n*0.1; // other being equal, use z-order
					if (dist < closest) {
						closest = dist;
						new_hl_n = n;
						new_hl_l = l;
						hl_color = QColor(log["color"].asString().c_str());
						hl_x = x + plot.left() + 10;
						hl_y = plot.bottom();
						new_hl_pointdesc = miniutils::stdprintf(
							"epoch = %0.3lf\n"
							"time  = %02i:%02i:%02i\n"
							"lr    = %0.4lf\n"
							"min   = %s"
							"max   = %s",
							pt[0U].asDouble(),
							int(pt[1U].asDouble()) / 60 / 60,
							int(pt[1U].asDouble()) / 60 % 60,
							int(pt[1U].asDouble()) % 60,
							pt[2U].asDouble(),
							pt[3U].toStyledString().c_str(),
							pt[4U].toStyledString().c_str()
							);
						hl_desc = log["task"].asString();
					}
				}
			}
		}

		if (hl_l != new_hl_l || hl_n != new_hl_n || hl_pointdesc != new_hl_pointdesc) {
			hl_n = new_hl_n;
			hl_l = new_hl_l;
			hl_pointdesc = new_hl_pointdesc;
			update();
		}
		if (mev->type()==QEvent::MouseButtonPress) {
			if (plot.contains(mev->pos())) {
				hl_shelf = mev->y();
			} else {
				hl_shelf = -1;
			}
			update();
		}
	}

	int hl_x, hl_y;
	int hl_n = -1;
	int hl_l = -1;
	int hl_shelf = -1;
	std::string hl_pointdesc;
	QColor hl_color;
	std::string hl_desc;

	void mouseReleaseEvent(QMouseEvent* mev)  { mouse_event(mev); }
	void mousePressEvent(QMouseEvent* mev)  { mouse_event(mev); }
	void mouseMoveEvent(QMouseEvent* mev)  { mouse_event(mev); }

	bool event(QEvent* ev) override
	{
		if (ev->type()==QEvent::User) {
			outdated = true;
		}
		if (ev->type()==QEvent::Timer && outdated) {
			boost::shared_ptr<Progress> p;
			{
				boost::mutex::scoped_lock lock(progress_mutex);
				p = progress_unsafe;
			}
			if (!p) return true;
			outdated = false;
			update();
			progress_copy_qt_thread_only.reset(new Progress);
			boost::mutex::scoped_lock lock(progress_feed_mutex);
			*progress_copy_qt_thread_only = *p; // by value
			return true;
		}
		return QWidget::event(ev);
	}

	double fixed_scale = 2.0;

	void wheelEvent(QWheelEvent* wev) override
	{
		fixed_scale *= pow(1.001, wev->delta());
		update();
	}
};

QWidget* progress_widget_create()
{
	return new QTrainProgress;
}

void progress_widget_update_thread_safe(QWidget* w_, const boost::shared_ptr<Progress>& p)
{
	QTrainProgress* w = (QTrainProgress*) w_;
	{
		boost::mutex::scoped_lock lock(w->progress_mutex);
		w->progress_unsafe = p;
	}
	QCoreApplication::postEvent(w, new QEvent(QEvent::User));
}
