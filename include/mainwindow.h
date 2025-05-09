#pragma once

#include <QMainWindow>
#include <QTimer>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QSlider>
#include <QGridLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include "deepfake_model.h"
#include <QMap>

class VirtualWebcam;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_startCaptureButton_clicked();
    void on_stopCaptureButton_clicked();
    void on_deviceComboBox_currentIndexChanged(int index);
    void on_addFaceButton_clicked();
    void on_savePresetButton_clicked();
    void on_loadPresetButton_clicked();
    void on_blendSlider_valueChanged(int value);
    void on_faceSizeSlider_valueChanged(int value);
    void on_smoothnessSlider_valueChanged(int value);
    void on_modelTypeComboBox_currentIndexChanged(int index);
    void onTargetFaceButtonClicked();
    void updateFrame();

private:
    Ui::MainWindow *ui;
    QTimer *captureTimer;
    cv::VideoCapture webcam;
    std::unique_ptr<VirtualWebcam> virtualWebcam;
    std::unique_ptr<DeepFakeModel> deepfakeModel;
    QString currentTargetFace;
    
    // UI references
    QLabel *originalFeedLabel;
    QLabel *deepfakeFeedLabel;
    QComboBox *deviceComboBox;
    QComboBox *modelTypeComboBox;
    QPushButton *startCaptureButton;
    QPushButton *stopCaptureButton;
    QSlider *blendSlider;
    QSlider *faceSizeSlider;
    QSlider *smoothnessSlider;
    QGridLayout *targetFacesLayout;
    
    // Methods
    void initializeUI();
    void populateDeviceList();
    void populateModelTypes();
    void addTargetFace(const QString &path);
    bool loadFaceImage(const QString &path);
    void updateStatus(const QString &message);
    cv::Mat processFrame(const cv::Mat &frame);
    QMap<QString, QRect> faceRects;
    bool eventFilter(QObject* obj, QEvent* event) override;
    void setTargetFaceWithRect(const QString& path, const QRect& rect);
}; 