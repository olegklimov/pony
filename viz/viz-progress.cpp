#include "viz-progress.h"
#include "miniutils.h"
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
#include <boost/iostreams/device/mapped_file.hpp>

using boost::shared_ptr;
using boost::iostreams::mapped_file_source;
using std::string;

static boost::mutex progress_feed_mutex;
static const int LOSSES_MAX = 6;
static const int HISTORY_STRIDE = 10;
static const int CONDENSATE_TO = 1000;
static const int FIRST_LOSS = 4;

struct Graph {
	std::string color;
	std::string name;
	std::string desc;
	std::vector<string> losses;
	std::vector<string> runind;

	std::string json_fn;
	Json::Value json;

	std::string file_log_fn;
	mapped_file_source file_log;
	int file_log_div;
	std::vector<float> file_log_min;
	std::vector<float> file_log_max;
	std::string file_tst_fn;
	mapped_file_source file_tst;
	std::string file_NT_fn;
	mapped_file_source file_NT;

	uint64_t reopened_ts;

	void reopen()
	{
		file_log.close();
		file_tst.close();
		file_NT.close();
		try {
			file_log.open(file_log_fn);
			file_tst.open(file_tst_fn);
			file_NT.open(file_NT_fn);
		} catch (const std::exception& e) {
			fprintf(stderr, "CANNOT OPEN %s: %s\n", file_log_fn.c_str(), e.what());
		}
		reopened_ts = miniutils::now();
	}

	void condense_log()
	{
		int N = ((uint32_t*)file_NT.data())[0];
		float* h = (float*) file_log.data();
		file_log_div = 1;
		while (N / file_log_div > CONDENSATE_TO)
			file_log_div += 1;
		int shorter = (N+file_log_div-1) / file_log_div;
		//int shorter = N / file_log_div;
		//shorter += 1;
		file_log_min.assign(HISTORY_STRIDE*shorter,  1e10);
		file_log_max.assign(HISTORY_STRIDE*shorter, -1e10);
		assert( (shorter-1)*HISTORY_STRIDE+(HISTORY_STRIDE-1) < (int)file_log_max.size() );
		for (int c=0; c<N; c++) {
			int s = c/file_log_div;
			assert(s < shorter);
			for (int l=FIRST_LOSS; l<HISTORY_STRIDE; l++) {
				float v = h[c*HISTORY_STRIDE + l];
				int idx = s*HISTORY_STRIDE + l;
				assert(idx < HISTORY_STRIDE*shorter);
				float& min = file_log_min[idx];
				float& max = file_log_max[idx];
				min = std::min(min, v);
				max = std::max(max, v);
			}
		}
	}
};

boost::shared_ptr<Graph> graph_init(const boost::filesystem::path& t, const std::string& prefix)
{
	boost::shared_ptr<Graph> g(new Graph);
	Json::Reader r;
	r.parse(miniutils::read_file( (t / prefix).string() + ".json" ), g->json, false);
	string errs = r.getFormattedErrorMessages();
	if (!errs.empty()) throw std::logic_error(errs);
	g->name = prefix;
	g->desc = g->json["desc"].asString();
	g->color = g->json["color"].asString();
	g->file_log_fn = g->json["mmapped_log"].asString();
	g->file_tst_fn = g->json["mmapped_tst"].asString();
	g->file_NT_fn = g->json["mmapped_NT"].asString();
	int lcnt = g->json["losses"].size();
	g->losses.resize(lcnt);
	for (int c=0; c<lcnt; c++)
		g->losses[c] = g->json["losses"][c].asString();
	int rcnt = g->json["runind"].size();
	g->runind.resize(rcnt);
	for (int c=0; c<rcnt; c++)
		g->runind[c] = g->json["runind"][c].asString();
	g->reopen();
	return g;
}

std::vector<shared_ptr<Graph>> rescan_directory(const std::string& folder)
{
	boost::filesystem::path t;
	t  = folder;

	std::vector<shared_ptr<Graph>> result;

	boost::filesystem::directory_iterator end;
	for (boost::filesystem::directory_iterator i(folder); i!=end; ++i) {
		boost::filesystem::path fn = i->path();
		string ext = fn.extension().string();
		if (boost::filesystem::is_regular_file(fn) && (ext==".json")) {
			try {
				//tmp[fn.filename().string()] = v;
				result.push_back(graph_init( folder, fn.stem().string() ));

			} catch (const std::exception& e) {
				fprintf(stderr, "ERROR READING '%s': %s\n", fn.string().c_str(), e.what());
			}
		}
	}

	return result;
}

class QTrainProgress: public QWidget {
public:
	std::vector<shared_ptr<Graph>> others;
	struct LegendElement;

	QSettings settings;
	std::string folder;

	QTrainProgress(): settings("OlegKlimov", "Pony")
	{
		QFont f = QFont("Courier");
                f.setFixedPitch(true);
                setFont(f);
		startTimer(1000);
		setMouseTracking(true);
	}

	void find_maximums(double* max_epoch, double* first_loss)
	{
		float e = 10;
		int graph_count = others.size();
		for (int n=0; n<graph_count; n++) {
			const shared_ptr<Graph> g = others[n];
			if (g->reopened_ts + 10*1000000ULL < miniutils::now())
				g->reopen();
			int T = ((uint32_t*)g->file_NT.data())[0];
			float* h = (float*) g->file_log.data();
			float epoch = h[(T-1)*HISTORY_STRIDE + 1];
			e = std::max(epoch, e);
		}
		*max_epoch = 1 + int(e);
		*first_loss = 1.0;
		*first_loss = fixed_scale;
	}

	QRect plot;
	double kx = 1;
	double ky = 1;

	void paintEvent(QPaintEvent* pev) override
	{
		QPainter p(this);
		p.fillRect(rect(), Qt::black);
		const int MARGIN = 10;
		double max_epoch;
		double first_loss;

		int graph_count = others.size();
		if (losses_visible.empty()) {
			losses_visible.assign(graph_count, 0);
			for (int c=0; c<graph_count; c++) {
				QVariant v = settings.value(miniutils::stdprintf("losses-visible%i", c-graph_count).c_str());
				if (v.isValid()) losses_visible[c] = (uint16_t) v.toInt();
			}
		}

		find_maximums(&max_epoch, &first_loss);

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

		for (int n=0; n<graph_count; n++) {
			const shared_ptr<Graph> g = others[n];
			QColor color( g->color.c_str() );
			if (n==hl_n) color = color.lighter(140);
			std::vector<QPointF> drawme_area;
			std::vector<QPointF> drawme_line;
			int N = ((uint32_t*)g->file_NT.data())[0];
			int T = ((uint32_t*)g->file_NT.data())[1];
			float* h = (float*) g->file_log.data();
			g->condense_log();
			int shorter_N = g->file_log_min.size() / HISTORY_STRIDE;
			drawme_area.resize(2*shorter_N);
			drawme_line.resize(T);
			int losses_count = g->losses.size();
			for (int l=0; l<losses_count; l++) {
				if (!(losses_visible[n] & (1<<l))) continue;
				for (int s=0; s<shorter_N; ++s) {
					int i = s*g->file_log_div;
					double pt_epoch     = h[HISTORY_STRIDE*i + 1];
					drawme_area[s              ].rx() = plot.left()   + kx*pt_epoch;
					drawme_area[s              ].ry() = plot.bottom() - ky*g->file_log_max[s*HISTORY_STRIDE + FIRST_LOSS+l];
					drawme_area[2*shorter_N-s-1].rx() = plot.left()   + kx*pt_epoch;
					drawme_area[2*shorter_N-s-1].ry() = plot.bottom() - ky*g->file_log_min[s*HISTORY_STRIDE + FIRST_LOSS+l];
				}
				p.setPen(color);
				p.setBrush(color);
				p.setOpacity(0.7);
				p.drawPolygon(drawme_area.data(), 2*shorter_N);
			}
			int runind_count = g->runind.size();
			float* t = (float*) g->file_tst.data();
			for (int l=0; l<runind_count; l++) {
				if (!(losses_visible[n] & (1<<(l+LOSSES_MAX)))) continue;
				for (int i=0; i<T; ++i) {
					double pt_epoch     = t[HISTORY_STRIDE*i + 1];
					double pt_val       = t[HISTORY_STRIDE*i + FIRST_LOSS+l];
					drawme_line[i].rx() = plot.left()   + kx*pt_epoch;
					drawme_line[i].ry() = plot.bottom() - ky*pt_val;
				}
				p.setPen(color.lighter());
				p.setOpacity(1.0);
				p.drawPolyline(drawme_line.data(), T);
			}
		}

		p.setClipRect(QRect(), Qt::NoClip);
		int fh = p.fontMetrics().height();
		int textover = 0;
		interactives.clear();
		for (int n=0; n<graph_count; n++) {
			const shared_ptr<Graph> g = others[n];
			string name = g->name;

			QRect smallcolor(MARGIN,         PLOTH + MARGIN + (fh+3)*n,  fh, fh);
			QRect textrect(MARGIN+fh+MARGIN, PLOTH + MARGIN + (fh+3)*n, 300, fh);
			textover = 50 + textrect.right();

			QColor color(g->color.c_str());
			p.setBrush(color);
			p.setPen(color);
			p.setOpacity(0.7);
			p.drawRect(smallcolor);

			p.setOpacity(1.0);
			p.setPen(Qt::white);
			p.drawText(textrect, name.c_str());
			interactives.push_back({ textrect.adjusted(-2,-2,+2,+2), 0xFFFF, n });

			int losses_count = g->losses.size();
			int runind_count = g->runind.size();
			for (int l=0; l<2*LOSSES_MAX; l++) {
				QRect r(textover + (fh+3)*l, PLOTH + MARGIN + (fh+3)*n, fh, fh);
				p.setPen(color);
				bool visible = false;
				int i = l-LOSSES_MAX;
				if (i<0 && l>=losses_count) continue;
				if (i>=runind_count) continue;
				visible = !!(losses_visible[n]&(1<<l));
				p.setBrush(visible ? color : Qt::transparent);
				p.drawRect(r);
				interactives.push_back({ r.adjusted(-2,-2,+2,+2), uint16_t(1<<l), n });
			}
		}
		if (textover)
		for (int l=0; l<2*LOSSES_MAX; l++) {
			QRect r(textover + (fh+3)*l, PLOTH + MARGIN + (fh+3)*(graph_count+1), fh, fh);
			p.setPen(Qt::white);
			p.drawText(r, Qt::AlignCenter, miniutils::stdprintf("%i", l%LOSSES_MAX).c_str());
			interactives.push_back({ r.adjusted(-2,-2,+2,+2), uint16_t(1<<l), -1 });
		}
		QRect desc = rect().adjusted(+MARGIN, +MARGIN, -MARGIN, -MARGIN);
		desc.setLeft(textover + 2*LOSSES_MAX*(fh+3) + MARGIN*2);
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
		int N = others.size();
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
		int graph_count = others.size();
		for (int n=0; n<graph_count; n++) {
			const shared_ptr<Graph> g = others[n];
			//int P = ((uint32_t*)g->file_NT.data())[0];
			int T = ((uint32_t*)g->file_NT.data())[1];
			float* h = (float*) g->file_log.data();
			float* t = (float*) g->file_tst.data();
			int shorter_N = g->file_log_min.size() / HISTORY_STRIDE;
			int losses_count = g->losses.size();
			int runind_count = g->runind.size();
			for (int l=0; l<2*LOSSES_MAX; l++) {
				if (!(losses_visible[n] & (1<<l))) continue;
				int q = l-LOSSES_MAX;
				if (q<0 && l>=losses_count) continue;
				if (q>=runind_count) continue;
				for (int s=0; s < (q<0 ? shorter_N:T); ++s) {
					double x, y1, y2;
					float* pointer;
					int i;
					if (q<0) {
						i = s*g->file_log_div;
						x  = plot.left()   + kx*h[i*HISTORY_STRIDE + 1];
						y1 = plot.bottom() - ky*g->file_log_max[n*HISTORY_STRIDE + FIRST_LOSS+l];
						y2 = plot.bottom() - ky*g->file_log_min[n*HISTORY_STRIDE + FIRST_LOSS+l];
						pointer = h;
					} else {
						i = s;
						x  = plot.left()   + kx*t[i*HISTORY_STRIDE + 1];
						y1 = plot.bottom() - ky*t[i*HISTORY_STRIDE + FIRST_LOSS+q];
						y2 = y1;
						pointer = t;
					}
					if (abs(x - mev->x()) > 10) continue;
					assert(y1 <= y2);
					double dist = 0;
					if (mev->y() < y1) dist += abs(mev->y() - y1);
					if (mev->y() > y2) dist += abs(mev->y() - y2);
					dist += -n*0.1; // other being equal, use z-order
					if (dist < closest) {
						closest = dist;
						new_hl_n = n;
						new_hl_l = l;
						hl_color = QColor(g->color.c_str());
						hl_x = x + plot.left() + 10;
						hl_y = plot.bottom();
						new_hl_pointdesc = miniutils::stdprintf(
							"loss[%i] = %0.4lf -- %s\n"
							"epoch    = %0.3lf\n"
							"time     = %02i:%02i:%02i\n"
							"lr       = %0.4lf\n"
							"iter     = %06i\n",
							l, double(pointer[i*HISTORY_STRIDE + FIRST_LOSS+(q<0 ? l:q)]),
							(q<0 ? g->losses[l].c_str() : g->runind[q].c_str()),
							double(pointer[i*HISTORY_STRIDE+1]),
							int(pointer[i*HISTORY_STRIDE+2]) / 60 / 60,
							int(pointer[i*HISTORY_STRIDE+2]) / 60 % 60,
							int(pointer[i*HISTORY_STRIDE+2]) % 60,
							double(pointer[i*HISTORY_STRIDE+3]),
							int(pointer[i*HISTORY_STRIDE+0])
							);
						hl_desc = g->desc;
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

	double fixed_scale = 2.0;

	void wheelEvent(QWheelEvent* wev) override
	{
		fixed_scale *= pow(1.001, wev->delta());
		update();
	}

	bool event(QEvent* ev)
        {
                if (ev->type()==QEvent::Timer)
                        update();
                return QWidget::event(ev);
        }
};

void progress_widget_rescan_dir(QWidget* w_)
{
	QTrainProgress* w = (QTrainProgress*) w_;
	w->others = rescan_directory(w->folder);
}

QWidget* progress_widget_create(const std::string& folder)
{
	QTrainProgress* w = new QTrainProgress;
	w->folder = folder;
	return w;
}
