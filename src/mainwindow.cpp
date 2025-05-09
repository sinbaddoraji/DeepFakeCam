#include "mainwindow.h"
#include "virtualwebcam.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QStatusBar>
#include "webcammanager.h"
#include "targetfacemanager.h"
#include <filesystem>
#include "facepreviewdialog.h"
#include <QMap>
#include <QMouseEvent>

// Face detection cascade
cv::CascadeClassifier faceCascade;

// Add a map to store face rects for each image
QMap<QString, QRect> faceRects;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(nullptr), captureTimer(nullptr), webcam()
{
    initializeUI();
    
    // Create timer for frame capture
    captureTimer = new QTimer(this);
    connect(captureTimer, &QTimer::timeout, this, &MainWindow::updateFrame);
    
    // Initialize deepfake model
    std::string modelsDir = "models";
    
    // Create models directory if it doesn't exist
    if (!std::filesystem::exists(modelsDir)) {
        std::filesystem::create_directory(modelsDir);
    }
    
    deepfakeModel = std::make_unique<DeepFakeModel>(modelsDir + "/deepfake_model.onnx");
    
    // Load face cascade for face detection
    bool cascadeLoaded = false;
    
    // Try loading from OpenCV's samples directory
    cascadeLoaded = faceCascade.load(cv::samples::findFile("haarcascades/haarcascade_frontalface_alt2.xml"));
    
    // If that fails, try loading from our models directory
    if (!cascadeLoaded) {
        cascadeLoaded = faceCascade.load(modelsDir + "/haarcascade_frontalface_alt2.xml");
        
        // If that fails too, display a warning
        if (!cascadeLoaded) {
            QMessageBox::warning(this, "Warning", 
                "Could not load face detection model. Face detection will not work.\n"
                "Please copy haarcascade_frontalface_alt2.xml to the 'models' directory.");
        }
    }
    
    // Populate available webcam devices
    populateDeviceList();
    
    // Status bar initialization
    updateStatus("Ready");
}

MainWindow::~MainWindow()
{
    if (captureTimer) {
        captureTimer->stop();
    }
    
    // Stop webcam if running
    if (webcam.isOpened()) {
        webcam.release();
    }
}

void MainWindow::initializeUI()
{
    // Set window properties
    setWindowTitle("DeepFake Webcam");
    setMinimumSize(1000, 600);
    
    // Create central widget and main layout
    QWidget *centralWidget = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    
    // Create video feed layout
    QVBoxLayout *feedsLayout = new QVBoxLayout();
    
    // Original feed
    QGroupBox *originalFeedBox = new QGroupBox("Original Webcam Feed");
    QVBoxLayout *originalFeedLayout = new QVBoxLayout(originalFeedBox);
    originalFeedLabel = new QLabel("No webcam connected");
    originalFeedLabel->setMinimumSize(320, 240);
    originalFeedLabel->setAlignment(Qt::AlignCenter);
    originalFeedLabel->setStyleSheet("background-color: black; color: white;");
    originalFeedLayout->addWidget(originalFeedLabel);
    
    // Deepfake feed
    QGroupBox *deepfakeFeedBox = new QGroupBox("Deepfake Output Feed");
    QVBoxLayout *deepfakeFeedLayout = new QVBoxLayout(deepfakeFeedBox);
    deepfakeFeedLabel = new QLabel("No processing active");
    deepfakeFeedLabel->setMinimumSize(320, 240);
    deepfakeFeedLabel->setAlignment(Qt::AlignCenter);
    deepfakeFeedLabel->setStyleSheet("background-color: black; color: white;");
    deepfakeFeedLayout->addWidget(deepfakeFeedLabel);
    
    // Add feed boxes to the feeds layout
    feedsLayout->addWidget(originalFeedBox);
    feedsLayout->addWidget(deepfakeFeedBox);
    
    // Controls layout
    QVBoxLayout *controlsLayout = new QVBoxLayout();
    
    // Device selection
    QGroupBox *deviceBox = new QGroupBox("Webcam Device");
    QVBoxLayout *deviceLayout = new QVBoxLayout(deviceBox);
    deviceComboBox = new QComboBox();
    deviceLayout->addWidget(deviceComboBox);
    QHBoxLayout *captureButtonsLayout = new QHBoxLayout();
    startCaptureButton = new QPushButton("Start Capture");
    stopCaptureButton = new QPushButton("Stop Capture");
    stopCaptureButton->setEnabled(false);
    captureButtonsLayout->addWidget(startCaptureButton);
    captureButtonsLayout->addWidget(stopCaptureButton);
    deviceLayout->addLayout(captureButtonsLayout);
    
    // Target face selection
    QGroupBox *faceBox = new QGroupBox("Target Face");
    QVBoxLayout *faceLayout = new QVBoxLayout(faceBox);
    targetFacesLayout = new QGridLayout();
    faceLayout->addLayout(targetFacesLayout);
    QHBoxLayout *faceButtonsLayout = new QHBoxLayout();
    QPushButton *addFaceButton = new QPushButton("Add Face");
    faceButtonsLayout->addWidget(addFaceButton);
    faceLayout->addLayout(faceButtonsLayout);
    
    // Transformation settings
    QGroupBox *settingsBox = new QGroupBox("Transformation Settings");
    QVBoxLayout *settingsLayout = new QVBoxLayout(settingsBox);
    
    QHBoxLayout *blendLayout = new QHBoxLayout();
    blendLayout->addWidget(new QLabel("Blend Amount:"));
    blendSlider = new QSlider(Qt::Horizontal);
    blendSlider->setRange(0, 100);
    blendSlider->setValue(70);
    blendLayout->addWidget(blendSlider);
    settingsLayout->addLayout(blendLayout);
    
    QHBoxLayout *faceSizeLayout = new QHBoxLayout();
    faceSizeLayout->addWidget(new QLabel("Face Size:"));
    faceSizeSlider = new QSlider(Qt::Horizontal);
    faceSizeSlider->setRange(0, 100);
    faceSizeSlider->setValue(50);
    faceSizeLayout->addWidget(faceSizeSlider);
    settingsLayout->addLayout(faceSizeLayout);
    
    QHBoxLayout *smoothnessLayout = new QHBoxLayout();
    smoothnessLayout->addWidget(new QLabel("Smoothness:"));
    smoothnessSlider = new QSlider(Qt::Horizontal);
    smoothnessSlider->setRange(0, 100);
    smoothnessSlider->setValue(50);
    smoothnessLayout->addWidget(smoothnessSlider);
    settingsLayout->addLayout(smoothnessLayout);
    
    QHBoxLayout *modelTypeLayout = new QHBoxLayout();
    modelTypeLayout->addWidget(new QLabel("Model Type:"));
    modelTypeComboBox = new QComboBox();
    modelTypeComboBox->addItem("Face Swap");
    modelTypeComboBox->addItem("Face Animation");
    modelTypeComboBox->addItem("Face Attributes");
    modelTypeLayout->addWidget(modelTypeComboBox);
    settingsLayout->addLayout(modelTypeLayout);
    
    QHBoxLayout *presetButtonsLayout = new QHBoxLayout();
    QPushButton *savePresetButton = new QPushButton("Save Preset");
    QPushButton *loadPresetButton = new QPushButton("Load Preset");
    presetButtonsLayout->addWidget(savePresetButton);
    presetButtonsLayout->addWidget(loadPresetButton);
    settingsLayout->addLayout(presetButtonsLayout);
    
    // Add all control groups to the controls layout
    controlsLayout->addWidget(deviceBox);
    controlsLayout->addWidget(faceBox);
    controlsLayout->addWidget(settingsBox);
    controlsLayout->addStretch();
    
    // Add layouts to main layout
    mainLayout->addLayout(feedsLayout, 7);
    mainLayout->addLayout(controlsLayout, 3);
    
    // Set central widget
    setCentralWidget(centralWidget);
    
    // Connect signals to slots
    connect(startCaptureButton, &QPushButton::clicked, this, &MainWindow::on_startCaptureButton_clicked);
    connect(stopCaptureButton, &QPushButton::clicked, this, &MainWindow::on_stopCaptureButton_clicked);
    connect(deviceComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::on_deviceComboBox_currentIndexChanged);
    connect(addFaceButton, &QPushButton::clicked, this, &MainWindow::on_addFaceButton_clicked);
    connect(savePresetButton, &QPushButton::clicked, this, &MainWindow::on_savePresetButton_clicked);
    connect(loadPresetButton, &QPushButton::clicked, this, &MainWindow::on_loadPresetButton_clicked);
    connect(blendSlider, &QSlider::valueChanged, this, &MainWindow::on_blendSlider_valueChanged);
    connect(faceSizeSlider, &QSlider::valueChanged, this, &MainWindow::on_faceSizeSlider_valueChanged);
    connect(smoothnessSlider, &QSlider::valueChanged, this, &MainWindow::on_smoothnessSlider_valueChanged);
    connect(modelTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::on_modelTypeComboBox_currentIndexChanged);
}

void MainWindow::populateDeviceList()
{
    deviceComboBox->clear();
    
    // Use WebcamManager to get actual devices
    WebcamManager webcamManager;
    const std::vector<WebcamManager::WebcamDevice>& devices = webcamManager.getDevices();
    
    for (const auto& device : devices) {
        deviceComboBox->addItem(QString::fromStdString(device.name), QVariant(device.id));
    }
    
    // If no devices were found, show a warning
    if (devices.empty()) {
        updateStatus("No webcam devices found");
        QMessageBox::warning(this, "Warning", "No webcam devices were detected.");
    }
}

void MainWindow::populateModelTypes()
{
    // Already done in initializeUI for now
}

void MainWindow::on_startCaptureButton_clicked()
{
    // Open webcam
    int cameraIndex = deviceComboBox->currentData().toInt();
    webcam.open(cameraIndex);
    
    if (!webcam.isOpened()) {
        QMessageBox::critical(this, "Error", 
            QString("Could not open camera %1").arg(cameraIndex));
        return;
    }
    
    // Set resolution
    webcam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    webcam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    // Initialize virtual webcam if needed
    if (!virtualWebcam) {
        // Find a loopback device
        std::vector<std::string> loopbackDevices = VirtualWebcam::listLoopbackDevices();
        if (!loopbackDevices.empty()) {
            virtualWebcam = std::make_unique<VirtualWebcam>(loopbackDevices[0], 640, 480);
        } else {
            // Show a message about setting up v4l2loopback
            QMessageBox::information(this, "Virtual Webcam", 
                "No v4l2loopback devices found. Output to virtual webcam will be disabled.\n\n"
                "To enable this feature, run:\n"
                "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"DeepFake\" exclusive_caps=1");
        }
    }
    
    // Start the timer
    captureTimer->start(33); // ~30 fps
    
    // Update UI
    startCaptureButton->setEnabled(false);
    stopCaptureButton->setEnabled(true);
    
    // Update status
    updateStatus("Capturing from " + deviceComboBox->currentText());
}

void MainWindow::on_stopCaptureButton_clicked()
{
    // Stop timer
    captureTimer->stop();
    
    // Release webcam
    if (webcam.isOpened()) {
        webcam.release();
    }
    
    // Update UI
    startCaptureButton->setEnabled(true);
    stopCaptureButton->setEnabled(false);
    
    // Reset feed displays
    QPixmap emptyPixmap(640, 480);
    emptyPixmap.fill(Qt::black);
    originalFeedLabel->setPixmap(emptyPixmap);
    deepfakeFeedLabel->setPixmap(emptyPixmap);
    
    // Update status
    updateStatus("Capture stopped");
}

void MainWindow::on_deviceComboBox_currentIndexChanged(int index)
{
    // Currently just log the selection
    updateStatus("Selected device: " + deviceComboBox->currentText());
}

void MainWindow::on_addFaceButton_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, 
        "Select Face Image", 
        QString(), 
        "Images (*.png *.jpg *.jpeg)");
    
    if (!filePath.isEmpty()) {
        // First verify the image can be loaded
        cv::Mat faceImage = cv::imread(filePath.toStdString());
        if (faceImage.empty()) {
            QMessageBox::warning(this, "Error", 
                "Could not load image file: " + filePath + "\n\n"
                "Please select a valid image file.");
            return;
        }
        
        // Check that the image contains a face before adding it
        if (!deepfakeModel) {
            QMessageBox::warning(this, "Error", 
                "DeepFake model not initialized. Cannot check for faces.");
            return;
        }
        
        // Detect faces in the image
        std::vector<cv::Rect> faces = deepfakeModel->detectFaces(faceImage);
        
        // If no faces detected, show error and return
        if (faces.empty()) {
            QMessageBox::warning(this, "No Face Detected", 
                "No face was detected in the selected image.\n\n"
                "Please select an image with a clearly visible face.");
            return;
        }
        
        // Find the largest face (typically the main face in the image)
        cv::Rect bestFace = faces[0];
        for (const auto& face : faces) {
            if (face.area() > bestFace.area()) {
                bestFace = face;
            }
        }
        
        // If the face is too small relative to the image, it might not be suitable
        double faceAreaRatio = static_cast<double>(bestFace.area()) / (faceImage.cols * faceImage.rows);
        if (faceAreaRatio < 0.05) { // Face covers less than 5% of the image
            QMessageBox::StandardButton reply = QMessageBox::question(this, 
                "Small Face Detected",
                "The detected face is quite small in the image, which may result in poor quality.\n\n"
                "Would you like to use this image anyway?",
                QMessageBox::Yes | QMessageBox::No);
            
            if (reply == QMessageBox::No) {
                return;
            }
        }
        
        // Now try to add the target face
        addTargetFace(filePath);
    }
}

void MainWindow::on_savePresetButton_clicked()
{
    QString filePath = QFileDialog::getSaveFileName(this, 
        "Save Preset", 
        QString(), 
        "Preset Files (*.preset)");
    
    if (!filePath.isEmpty()) {
        // Save preset (to be implemented)
        updateStatus("Preset saved to " + filePath);
    }
}

void MainWindow::on_loadPresetButton_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, 
        "Load Preset", 
        QString(), 
        "Preset Files (*.preset)");
    
    if (!filePath.isEmpty()) {
        // Load preset (to be implemented)
        updateStatus("Preset loaded from " + filePath);
    }
}

void MainWindow::on_blendSlider_valueChanged(int value)
{
    // Update blend value
    updateStatus(QString("Blend amount set to %1%").arg(value));
}

void MainWindow::on_faceSizeSlider_valueChanged(int value)
{
    // Update face size value
    updateStatus(QString("Face size set to %1%").arg(value));
}

void MainWindow::on_smoothnessSlider_valueChanged(int value)
{
    // Update smoothness value
    updateStatus(QString("Smoothness set to %1%").arg(value));
}

void MainWindow::on_modelTypeComboBox_currentIndexChanged(int index)
{
    // Update model type
    updateStatus("Model type set to " + modelTypeComboBox->currentText());
}

void MainWindow::onTargetFaceButtonClicked()
{
    // Handle target face selection
    QPushButton *button = qobject_cast<QPushButton*>(sender());
    if (button) {
        QString faceId = button->property("faceId").toString();
        QString facePath = button->property("facePath").toString();
        currentTargetFace = faceId;
        
        // Load the face into the deepfake model
        if (deepfakeModel && !facePath.isEmpty()) {
            try {
                // Check if file exists
                QFileInfo fileInfo(facePath);
                if (!fileInfo.exists() || !fileInfo.isFile()) {
                    QMessageBox::warning(this, "Error", 
                        "The image file no longer exists: " + facePath + "\n\n"
                        "Please add the face again.");
                    
                    // Remove the button from the layout
                    button->deleteLater();
                    return;
                }
                
                if (!loadFaceImage(facePath)) {
                    QMessageBox::warning(this, "Error", 
                        "Failed to load face from image: " + facePath + "\n\n"
                        "No face could be detected in this image.");
                    return;
                }
                
                // Update UI to indicate selected face
                updateStatus("Selected face: " + facePath);
                
                // Highlight the selected button and unhighlight others
                for (int i = 0; i < targetFacesLayout->count(); ++i) {
                    QLayoutItem* item = targetFacesLayout->itemAt(i);
                    if (item && item->widget()) {
                        QPushButton* btn = qobject_cast<QPushButton*>(item->widget());
                        if (btn) {
                            if (btn->property("faceId").toString() == faceId) {
                                btn->setStyleSheet("border: 2px solid red;");
                            } else {
                                btn->setStyleSheet("");
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                QMessageBox::warning(this, "Error", 
                    QString("Error loading face: %1").arg(e.what()));
            }
        }
    }
}

void MainWindow::updateFrame()
{
    if (!webcam.isOpened()) {
        return;
    }
    
    // Capture frame
    cv::Mat frame;
    webcam >> frame;
    
    if (frame.empty()) {
        return;
    }
    
    // Make a copy of the original frame
    cv::Mat originalFrame = frame.clone();
    
    // Convert to Qt format for display
    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
    QImage originalImage(
        rgbFrame.data, 
        rgbFrame.cols, 
        rgbFrame.rows, 
        rgbFrame.step, 
        QImage::Format_RGB888
    );
    
    // Display original frame
    originalFeedLabel->setPixmap(QPixmap::fromImage(originalImage).scaled(
        originalFeedLabel->width(), 
        originalFeedLabel->height(), 
        Qt::KeepAspectRatio
    ));
    
    // Process frame with deepfake
    cv::Mat processedFrame = processFrame(frame);
    
    // Convert processed frame to Qt format
    cv::cvtColor(processedFrame, rgbFrame, cv::COLOR_BGR2RGB);
    QImage processedImage(
        rgbFrame.data, 
        rgbFrame.cols, 
        rgbFrame.rows, 
        rgbFrame.step, 
        QImage::Format_RGB888
    );
    
    // Display processed frame
    deepfakeFeedLabel->setPixmap(QPixmap::fromImage(processedImage).scaled(
        deepfakeFeedLabel->width(), 
        deepfakeFeedLabel->height(), 
        Qt::KeepAspectRatio
    ));
    
    // Send to virtual webcam if available
    if (virtualWebcam && virtualWebcam->isReady()) {
        // Convert back to BGR for virtual webcam
        virtualWebcam->writeFrame(processedFrame);
    }
}

void MainWindow::addTargetFace(const QString &path)
{
    try {
        // Load the face image in the deepfake model first to verify face can be detected
        if (!loadFaceImage(path)) {
            QMessageBox::warning(this, "Error", 
                "Failed to process the face image. Please try another image with a clearer face.");
            return;
        }
        // Load face image for the UI
        QPixmap facePixmap(path);
        if (facePixmap.isNull()) {
            QMessageBox::warning(this, "Error", "Could not load face image: " + path);
            return;
        }
        // Get detected face rect for preview/correction
        cv::Mat img = cv::imread(path.toStdString());
        std::vector<cv::Rect> faces = deepfakeModel->detectFaces(img);
        QRect faceRect;
        if (!faces.empty()) {
            cv::Rect bestFace = faces[0];
            for (const auto& f : faces) if (f.area() > bestFace.area()) bestFace = f;
            faceRect = QRect(bestFace.x, bestFace.y, bestFace.width, bestFace.height);
            faceRects[path] = faceRect;
        }
        // Create a unique ID for the face
        QString faceId = "face_" + QString::number(QDateTime::currentMSecsSinceEpoch());
        // Create a widget for the face thumbnail and remove button
        QWidget* thumbWidget = new QWidget();
        QVBoxLayout* thumbLayout = new QVBoxLayout(thumbWidget);
        thumbLayout->setContentsMargins(0,0,0,0);
        QPushButton *faceButton = new QPushButton();
        faceButton->setFixedSize(100, 100);
        faceButton->setIconSize(QSize(90, 90));
        faceButton->setIcon(QIcon(facePixmap));
        faceButton->setProperty("faceId", faceId);
        faceButton->setProperty("facePath", path);
        faceButton->setToolTip(path);
        // Double-click event for preview/correction
        faceButton->installEventFilter(this);
        // Remove button
        QPushButton* removeBtn = new QPushButton("X");
        removeBtn->setFixedSize(20, 20);
        removeBtn->setProperty("faceId", faceId);
        removeBtn->setProperty("facePath", path);
        connect(removeBtn, &QPushButton::clicked, this, [this, thumbWidget, faceId, path]() {
            // Remove from UI
            for (int i = 0; i < targetFacesLayout->count(); ++i) {
                QLayoutItem* item = targetFacesLayout->itemAt(i);
                if (item && item->widget() == thumbWidget) {
                    QWidget* w = item->widget();
                    targetFacesLayout->removeWidget(w);
                    w->deleteLater();
                    break;
                }
            }
            // Remove from map
            faceRects.remove(path);
            // If this was the selected face, clear selection
            if (currentTargetFace == faceId) currentTargetFace.clear();
            updateStatus("Removed face: " + path);
        });
        // Layout
        QHBoxLayout* topLayout = new QHBoxLayout();
        topLayout->addStretch();
        topLayout->addWidget(removeBtn);
        thumbLayout->addLayout(topLayout);
        thumbLayout->addWidget(faceButton);
        // Add to grid layout
        int row = targetFacesLayout->rowCount();
        int col = targetFacesLayout->count() % 2;
        targetFacesLayout->addWidget(thumbWidget, row, col);
        // Select this face
        currentTargetFace = faceId;
        // Update UI to highlight the selected face
        for (int i = 0; i < targetFacesLayout->count(); ++i) {
            QLayoutItem* item = targetFacesLayout->itemAt(i);
            if (item && item->widget()) {
                QWidget* w = item->widget();
                QPushButton* btn = w->findChild<QPushButton*>();
                if (btn) {
                    if (btn->property("faceId").toString() == faceId) {
                        btn->setStyleSheet("border: 2px solid red;");
                    } else {
                        btn->setStyleSheet("");
                    }
                }
            }
        }
        QMessageBox::information(this, "Face Added", 
            "Face was successfully added and selected.\n"
            "The image has been processed for use in face swapping.");
        updateStatus("Added and selected face: " + path);
    } catch (const std::exception& e) {
        qWarning("Exception in addTargetFace: %s", e.what());
        QMessageBox::warning(this, "Error", QString("Exception in addTargetFace: %1").arg(e.what()));
    } catch (...) {
        qWarning("Unknown exception in addTargetFace");
        QMessageBox::warning(this, "Error", "Unknown exception in addTargetFace.");
    }
}

bool MainWindow::loadFaceImage(const QString &path)
{
    if (!deepfakeModel) {
        return false;
    }
    
    try {
        // Load the image
        cv::Mat image = cv::imread(path.toStdString());
        if (image.empty()) {
            return false;
        }
        
        // Detect faces in the image
        std::vector<cv::Rect> faces = deepfakeModel->detectFaces(image);
        if (faces.empty()) {
            return false;
        }
        
        // Load target face (this will select the largest face)
        return deepfakeModel->loadTargetFace(path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", 
            QString("Error processing face image: %1").arg(e.what()));
        return false;
    }
}

void MainWindow::updateStatus(const QString &message)
{
    statusBar()->showMessage(message);
}

cv::Mat MainWindow::processFrame(const cv::Mat &frame)
{
    try {
        // If we don't have a deepfake model, return the original frame
        if (!deepfakeModel) {
            return frame.clone();
        }
        // Use the DeepFakeModel to detect faces
        std::vector<cv::Rect> faces = deepfakeModel->detectFaces(frame);
        // If no faces detected, return the original frame
        if (faces.empty()) {
            return frame.clone();
        }
        // Get the largest face (typically closest to camera)
        cv::Rect bestFace = faces[0];
        for (const auto& face : faces) {
            if (face.area() > bestFace.area()) {
                bestFace = face;
            }
        }
        // Apply deepfake transformation with parameters from sliders
        float blendAmount = blendSlider->value() / 100.0f;
        float faceSize = faceSizeSlider->value() / 50.0f; // 0-2 range
        float smoothness = smoothnessSlider->value() / 100.0f;
        // Process the frame with the deepfake model
        cv::Mat processedFrame = deepfakeModel->transform(frame, bestFace);
        // Remove debug face rectangle drawing for production
        // cv::rectangle(processedFrame, bestFace, cv::Scalar(0, 255, 0), 2);
        return processedFrame;
    } catch (const std::exception& e) {
        qWarning("Exception in processFrame: %s", e.what());
        QMessageBox::warning(this, "Error", QString("Exception in processFrame: %1").arg(e.what()));
        return frame.clone();
    } catch (...) {
        qWarning("Unknown exception in processFrame");
        QMessageBox::warning(this, "Error", "Unknown exception in processFrame.");
        return frame.clone();
    }
}

// Event filter for double-click preview/correction
bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
    if (event->type() == QEvent::MouseButtonDblClick) {
        QPushButton* btn = qobject_cast<QPushButton*>(obj);
        if (btn) {
            QString path = btn->property("facePath").toString();
            if (!faceRects.contains(path)) return false;
            QImage img(QString::fromStdString(path.toStdString()));
            QRect rect = faceRects[path];
            FacePreviewDialog dlg(img, rect, this);
            if (dlg.exec() == QDialog::Accepted) {
                QRect newRect = dlg.getCorrectedRect();
                faceRects[path] = newRect;
                // Update the target face in the model using the new rect
                setTargetFaceWithRect(path, newRect);
                updateStatus("Face region corrected for: " + path);
            }
            return true;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

// Helper to set target face with a custom rect
void MainWindow::setTargetFaceWithRect(const QString& path, const QRect& rect) {
    cv::Mat img = cv::imread(path.toStdString());
    if (img.empty()) return;
    cv::Rect cvRect(rect.x(), rect.y(), rect.width(), rect.height());
    cv::Mat faceROI = img(cvRect).clone();
    deepfakeModel->setTargetFace(faceROI);
} 