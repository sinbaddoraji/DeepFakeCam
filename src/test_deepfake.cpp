#include <iostream>
#include <opencv2/opencv.hpp>
#include "deepfake_model.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <webcam_index> <target_face_path>" << std::endl;
        return -1;
    }

    int cameraIndex = std::atoi(argv[1]);
    std::string targetFacePath = argv[2];

    // Initialize webcam
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera " << cameraIndex << std::endl;
        return -1;
    }

    // Set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Initialize deepfake model
    DeepFakeModel model("models/deepfake_model.onnx");
    
    // Load target face
    if (!model.loadTargetFace(targetFacePath)) {
        std::cerr << "Failed to load target face: " << targetFacePath << std::endl;
        return -1;
    }
    
    std::cout << "Face detection and deepfake model loaded successfully." << std::endl;
    std::cout << "Press ESC to exit." << std::endl;

    while (true) {
        // Capture frame
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Empty frame received from camera" << std::endl;
            break;
        }

        // Create a copy for display
        cv::Mat displayFrame = frame.clone();

        // Detect faces using improved detection
        std::vector<cv::Rect> faces = model.detectFaces(frame);

        // Process the largest face (closest to camera)
        cv::Mat transformedFrame = frame.clone();
        if (!faces.empty()) {
            // Find the largest face
            cv::Rect bestFace = faces[0];
            for (const auto& face : faces) {
                if (face.area() > bestFace.area()) {
                    bestFace = face;
                }
            }
            
            // Draw rectangle on display frame
            cv::rectangle(displayFrame, bestFace, cv::Scalar(0, 255, 0), 2);
            
            // Apply deepfake transformation
            transformedFrame = model.transform(frame, bestFace);
            
            // Draw rectangle on transformed frame too
            cv::rectangle(transformedFrame, bestFace, cv::Scalar(0, 0, 255), 2);
        }

        // Show original and transformed frames
        cv::imshow("Original", displayFrame);
        cv::imshow("Deepfake", transformedFrame);

        // Exit on ESC key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Clean up
    cap.release();
    cv::destroyAllWindows();

    return 0;
} 