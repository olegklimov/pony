#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

class QWidget;

QWidget* progress_widget_create(const std::string& file_folder);
void progress_widget_rescan_dir(QWidget* w);
