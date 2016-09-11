#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

class QWidget;
struct Progress;

boost::shared_ptr<Progress> progress_init(
	const std::string& task,
	const std::string& file_folder,
	const std::string& hint);

void progress_feed(
	const boost::shared_ptr<Progress>& p,
	double epoch, double time, double lr,
	const std::vector<double>& losses,
	bool testset_separate_graph,
	double sample_progress_every =0.0);

QWidget* progress_widget_create();

void progress_widget_update_thread_safe(
	QWidget* w_,
	const boost::shared_ptr<Progress>& p);
