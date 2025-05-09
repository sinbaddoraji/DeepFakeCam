#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

// Forward declaration
class DeepFakeModelImpl;

/**
 * @brief Class for applying deepfake transformations to images
 * 
 * This class provides an interface for loading a deepfake model and
 * applying transformations to faces in images. It supports multiple 
 * backend implementations.
 */
class DeepFakeModel {
public:
    /**
     * @brief Supported backend frameworks for the model
     */
    enum class Backend {
        ONNX,       ///< ONNX Runtime
        TENSORFLOW, ///< TensorFlow C++ API
        LIBTORCH,   ///< PyTorch C++ API (LibTorch)
        OPENVINO    ///< OpenVINO
    };

    /**
     * @brief Constructor with model path and optional backend
     * 
     * @param modelPath Path to the model file
     * @param backend Backend framework to use
     */
    DeepFakeModel(const std::string& modelPath, Backend backend = Backend::ONNX);
    
    /**
     * @brief Destructor
     */
    ~DeepFakeModel();
    
    /**
     * @brief Apply deepfake transformation to a face in an image
     * 
     * @param inputImage The original image containing a face
     * @param faceRect Rectangle containing the face
     * @return cv::Mat Transformed image with the deepfake applied
     */
    cv::Mat transform(const cv::Mat& inputImage, const cv::Rect& faceRect);
    
    /**
     * @brief Set the target face for transformation
     * 
     * @param targetFaceImage Image of the target face to transform into
     * @return bool True if target face was successfully loaded
     */
    bool setTargetFace(const cv::Mat& targetFaceImage);
    
    /**
     * @brief Load a target face from a file
     * 
     * @param targetFacePath Path to the target face image
     * @return bool True if target face was successfully loaded
     */
    bool loadTargetFace(const std::string& targetFacePath);

    /**
     * @brief Detect faces in an image
     * 
     * Uses advanced deep neural network (DNN) face detection with 
     * fallback to Haar cascade if DNN models are not available
     * 
     * @param image The image to detect faces in
     * @return std::vector<cv::Rect> Vector of detected face rectangles
     */
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);

private:
    // Implementation using the PIMPL idiom to hide backend-specific details
    std::unique_ptr<DeepFakeModelImpl> pImpl;
}; 