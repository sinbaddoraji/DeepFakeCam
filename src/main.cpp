#include <QApplication>
#include <QMessageBox>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Set application info
    QApplication::setApplicationName("DeepFakeCam");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("DeepFakeCam");
    
    // Check for OpenCV and v4l2loopback
    try {
        // Create and show the main window
        MainWindow mainWindow;
        mainWindow.show();
        
        return app.exec();
    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, "Error", 
            QString("An error occurred: %1").arg(e.what()));
        return 1;
    }
} 