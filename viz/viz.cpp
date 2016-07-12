#include <QtGui/QApplication>
#include <QtGui/QWidget>


int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QWidget window;
	window.showMaximized();
	return app.exec();
}

