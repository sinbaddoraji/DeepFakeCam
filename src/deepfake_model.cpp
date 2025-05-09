#include "../include/deepfake_model.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <filesystem>
#include <set>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

// Add a static frame counter for debug output
static int debugFrameCount = 0;
static const int maxDebugFrames = 10;

// Helper: Get landmark indices for facial features (dlib 68-point)
namespace {
    const std::vector<int> MOUTH_IDX = {48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67};
    const std::vector<int> LEFT_EYE_IDX = {36,37,38,39,40,41};
    const std::vector<int> RIGHT_EYE_IDX = {42,43,44,45,46,47};
    const std::vector<int> LEFT_BROW_IDX = {17,18,19,20,21};
    const std::vector<int> RIGHT_BROW_IDX = {22,23,24,25,26};
}

// Helper: Warp a feature region from target to source using affine transform
cv::Mat warpFeatureRegion(const cv::Mat& target, const std::vector<cv::Point2f>& targetLM, const std::vector<cv::Point2f>& sourceLM, const std::vector<int>& indices, const cv::Size& size) {
    std::vector<cv::Point2f> srcPts, tgtPts;
    for (int idx : indices) {
        if (idx < targetLM.size() && idx < sourceLM.size()) {
            tgtPts.push_back(targetLM[idx]);
            srcPts.push_back(sourceLM[idx]);
        }
    }
    if (srcPts.size() < 3) return cv::Mat();
    cv::Rect bbox = cv::boundingRect(srcPts);
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    std::vector<cv::Point> srcPtsInt;
    for (const auto& pt : srcPts) srcPtsInt.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
    cv::fillConvexPoly(mask, srcPtsInt, 255);
    cv::Mat warpMat = cv::estimateAffinePartial2D(tgtPts, srcPts);
    cv::Mat warped;
    cv::warpAffine(target, warped, warpMat, size, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    cv::Mat feature = cv::Mat::zeros(size, target.type());
    warped.copyTo(feature, mask);
    return feature;
}

// Implementation class using PIMPL idiom
class DeepFakeModelImpl {
public:
    DeepFakeModelImpl(const std::string& modelPath, DeepFakeModel::Backend backend);
    ~DeepFakeModelImpl();
    
    cv::Mat transform(const cv::Mat& inputImage, const cv::Rect& faceRect);
    bool setTargetFace(const cv::Mat& targetFaceImage);
    bool loadTargetFace(const std::string& targetFacePath);
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    void setBlendAmount(float value) { blendAmount = value; }
    void setFaceSize(float value) { faceSize = value; }
    void setSmoothness(float value) { smoothness = value; }

private:
    DeepFakeModel::Backend backend;
    std::string modelPath;
    cv::Mat targetFace;
    bool modelLoaded;
    
    // Face detection and alignment
    cv::CascadeClassifier haarCascade;
    cv::dnn::Net faceDetector;
    bool hasDNNFaceDetector;
    float confidenceThreshold;
    cv::Ptr<cv::face::Facemark> facemark;
    bool hasLandmarkDetector;
    
    // Face processing parameters
    float blendAmount;
    float faceSize;
    float smoothness;
    
    // Face processing helpers
    cv::Mat extractFace(const cv::Mat& image, const cv::Rect& faceRect);
    cv::Mat preprocessFace(const cv::Mat& face);
    cv::Mat alignFace(const cv::Mat& face, std::vector<cv::Point2f>& landmarks);
    cv::Mat blendFaces(const cv::Mat& originalImage, const cv::Mat& transformedFace, const cv::Rect& faceRect);
    cv::Mat applyColorTransfer(const cv::Mat& source, const cv::Mat& target);
    bool detectLandmarks(const cv::Mat& face, std::vector<cv::Point2f>& landmarks);
    cv::Mat applySkintoneCorrection(const cv::Mat& sourceFace, const cv::Mat& targetFace);
    void preserveEyes(cv::Mat& transformedFace, const cv::Mat& originalFace, const std::vector<cv::Point2f>& landmarks);
    cv::Mat createFaceMask(const cv::Size& size, const std::vector<cv::Point2f>& landmarks);
    cv::Mat normalizeColorLab(const cv::Mat& src, const cv::Mat& ref, const cv::Mat& mask);
    
    // dlib members
    dlib::frontal_face_detector dlibFaceDetector;
    dlib::shape_predictor dlibShapePredictor;
    bool dlibInitialized = false;
};

// DeepFakeModelImpl implementation
DeepFakeModelImpl::DeepFakeModelImpl(const std::string& modelPath, DeepFakeModel::Backend backend)
    : modelPath(modelPath), backend(backend), modelLoaded(false), 
      blendAmount(0.7f), faceSize(1.0f), smoothness(0.5f), hasLandmarkDetector(false),
      hasDNNFaceDetector(false), confidenceThreshold(0.5f)
{
    // Load Haar cascade face detector as fallback
    if (!haarCascade.load(cv::samples::findFile("haarcascades/haarcascade_frontalface_alt2.xml"))) {
        std::cerr << "Error loading Haar cascade" << std::endl;
        // Fall back to a local path if the OpenCV path fails
        if (!haarCascade.load("models/haarcascade_frontalface_alt2.xml")) {
            std::cerr << "Could not load Haar cascade face detector model" << std::endl;
        }
    }
    
    // Try to load DNN face detector (preferred)
    try {
        // Check if model files exist
        std::ifstream prototxtFile("models/deploy.prototxt");
        std::ifstream caffeModelFile("models/res10_300x300_ssd_iter_140000.caffemodel");
        
        if (prototxtFile.good() && caffeModelFile.good()) {
            prototxtFile.close();
            caffeModelFile.close();
            
            // Load the DNN model
            faceDetector = cv::dnn::readNetFromCaffe(
                "models/deploy.prototxt",
                "models/res10_300x300_ssd_iter_140000.caffemodel"
            );
            
            if (!faceDetector.empty()) {
                hasDNNFaceDetector = true;
                std::cout << "DNN face detector loaded successfully" << std::endl;
                // Since we have DNN face detector, we can use it for landmarks too
                hasLandmarkDetector = true;
            }
        } else {
            std::cout << "DNN face detector model files not found, checking for TensorFlow model" << std::endl;
            
            // Try TensorFlow model as alternative
            std::ifstream pbFile("models/opencv_face_detector_uint8.pb");
            std::ifstream pbtxtFile("models/opencv_face_detector.pbtxt");
            
            if (pbFile.good() && pbtxtFile.good()) {
                pbFile.close();
                pbtxtFile.close();
                
                faceDetector = cv::dnn::readNetFromTensorflow(
                    "models/opencv_face_detector_uint8.pb",
                    "models/opencv_face_detector.pbtxt"
                );
                
                if (!faceDetector.empty()) {
                    hasDNNFaceDetector = true;
                    hasLandmarkDetector = true;
                    std::cout << "TensorFlow face detector loaded successfully" << std::endl;
                }
            } else {
                std::cout << "DNN face detector models not found, falling back to Haar cascade" << std::endl;
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading DNN face detector: " << e.what() << std::endl;
        std::cout << "Falling back to Haar cascade detector" << std::endl;
    }
    
    // Initialize dlib face detector and shape predictor
    try {
        dlibFaceDetector = dlib::get_frontal_face_detector();
        dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> dlibShapePredictor;
        dlibInitialized = true;
        hasLandmarkDetector = true;
        std::cout << "dlib 68-point face landmark detector loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load dlib shape predictor: " << e.what() << std::endl;
        dlibInitialized = false;
    }
    
    // In a real app, you would load the actual deepfake model here
    // For demonstration purposes, we'll set modelLoaded to true
    std::cout << "DeepFake model initialized with " << modelPath << std::endl;
    modelLoaded = true;
}

DeepFakeModelImpl::~DeepFakeModelImpl() {
    // Clean up resources
}

std::vector<cv::Rect> DeepFakeModelImpl::detectFaces(const cv::Mat& image) {
    std::vector<cv::Rect> faces;
    
    // Use DNN-based face detector if available (more accurate)
    if (hasDNNFaceDetector) {
        try {
            // Prepare image for DNN
            cv::Mat inputBlob = cv::dnn::blobFromImage(
                image, 1.0, cv::Size(300, 300), 
                cv::Scalar(104.0, 177.0, 123.0), false, false
            );
            
            faceDetector.setInput(inputBlob);
            
            // Forward pass through the network
            cv::Mat detections = faceDetector.forward();
            
            // Get dimensions of the original image
            int imageWidth = image.cols;
            int imageHeight = image.rows;
            
            // Process detections
            // The detections matrix is a 4D matrix with shape [1, 1, N, 7]
            // where N is the number of detections and the 7 columns are:
            // [image_id, label, confidence, x_min, y_min, x_max, y_max]
            for (int i = 0; i < detections.size[2]; i++) {
                const float* detection = detections.ptr<float>(0, 0, i);
                float confidence = detection[2];
                
                // Filter out weak detections
                if (confidence > confidenceThreshold) {
                    // Get coordinates (normalized between 0 and 1)
                    int x1 = static_cast<int>(detection[3] * imageWidth);
                    int y1 = static_cast<int>(detection[4] * imageHeight);
                    int x2 = static_cast<int>(detection[5] * imageWidth);
                    int y2 = static_cast<int>(detection[6] * imageHeight);
                    
                    // Ensure coordinates are within image bounds
                    x1 = std::max(0, std::min(x1, imageWidth - 1));
                    y1 = std::max(0, std::min(y1, imageHeight - 1));
                    x2 = std::max(0, std::min(x2, imageWidth - 1));
                    y2 = std::max(0, std::min(y2, imageHeight - 1));
                    
                    // Create rectangle
                    cv::Rect faceRect(x1, y1, x2 - x1, y2 - y1);
                    
                    // Skip invalid rectangles
                    if (faceRect.width <= 0 || faceRect.height <= 0) {
                        continue;
                    }
                    
                    faces.push_back(faceRect);
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Error in DNN face detection: " << e.what() << std::endl;
            // Fall back to Haar cascade
            hasDNNFaceDetector = false;
        }
    }
    
    // If DNN detector didn't find any faces or failed, try Haar cascade
    if (faces.empty()) {
        // Convert to grayscale for Haar cascade
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grayImage, grayImage);
        
        // Detect faces using Haar cascade
        haarCascade.detectMultiScale(
            grayImage, faces, 
            1.1, 5, 0, 
            cv::Size(30, 30)
        );
    }
    
    return faces;
}

// Helper: apply piecewise affine warping from target to source using Delaunay triangulation
cv::Mat warpFacePiecewise(const cv::Mat& target, const std::vector<cv::Point2f>& targetLM, const std::vector<cv::Point2f>& sourceLM, const cv::Size& outSize) {
    cv::Mat output = cv::Mat::zeros(outSize, target.type());
    cv::Rect rect(0, 0, outSize.width, outSize.height);
    cv::Subdiv2D subdiv(rect);
    // Insert all source landmarks (use all 68 points)
    for (const auto& pt : sourceLM) {
        // Clamp points to image bounds
        float x = std::min(std::max(pt.x, 0.0f), (float)(outSize.width-1));
        float y = std::min(std::max(pt.y, 0.0f), (float)(outSize.height-1));
        subdiv.insert(cv::Point2f(x, y));
    }
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    auto findIndex = [](const std::vector<cv::Point2f>& pts, const cv::Point2f& p) {
        for (size_t i = 0; i < pts.size(); ++i) {
            if (cv::norm(pts[i] - p) < 1.0) return (int)i;
        }
        return -1;
    };
    for (const auto& t : triangleList) {
        std::vector<cv::Point2f> srcTri, tgtTri;
        std::vector<cv::Point> srcTriInt;
        std::set<int> idxSet;
        bool validTriangle = true;
        for (int i = 0; i < 3; ++i) {
            cv::Point2f pt(t[i*2], t[i*2+1]);
            int idx = findIndex(sourceLM, pt);
            if (idx < 0) { validTriangle = false; break; }
            srcTri.push_back(sourceLM[idx]);
            tgtTri.push_back(targetLM[idx]);
            srcTriInt.push_back(cv::Point(cvRound(srcTri.back().x), cvRound(srcTri.back().y)));
            idxSet.insert(idx);
        }
        if (!validTriangle || idxSet.size() < 3) continue;
        cv::Mat warpMat = cv::getAffineTransform(tgtTri, srcTri);
        cv::Mat mask = cv::Mat::zeros(outSize, CV_8UC1);
        cv::fillConvexPoly(mask, srcTriInt, 255);
        cv::Rect roi = cv::boundingRect(srcTriInt);
        cv::Mat srcRoi;
        cv::warpAffine(target, srcRoi, warpMat, outSize, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        srcRoi.copyTo(output, mask);
    }
    return output;
}

// Helper: create a convex hull mask from landmarks
cv::Mat createFaceMask(const cv::Size& size, const std::vector<cv::Point2f>& landmarks) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    if (landmarks.size() >= 68) {
        std::vector<cv::Point> hull;
        for (const auto& pt : landmarks)
            hull.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        std::vector<cv::Point> hullPts;
        cv::convexHull(hull, hullPts);
        cv::fillConvexPoly(mask, hullPts, 255);
        int dilation_size = 32;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1));
        cv::dilate(mask, mask, kernel);
        cv::ellipse(mask, cv::Point(size.width/2, size.height/2),
                    cv::Size(size.width/1.7, size.height/1.5),
                    0, 0, 360, cv::Scalar(255), -1);
    } else if (landmarks.size() >= 5) {
        std::vector<cv::Point> hull;
        for (const auto& pt : landmarks) hull.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        std::vector<cv::Point> hullPts;
        cv::convexHull(hull, hullPts);
        cv::fillConvexPoly(mask, hullPts, 255);
    } else {
        cv::ellipse(mask, cv::Point(size.width/2, size.height/2),
                    cv::Size(size.width/2.2, size.height/2.2),
                    0, 0, 360, cv::Scalar(255), -1);
    }
    // Feather the mask edge for a softer, cleaner blend
    int feather_size = 41; // Must be odd, increase for softer edge
    cv::GaussianBlur(mask, mask, cv::Size(feather_size, feather_size), 0);
    return mask;
}

// Helper: color normalization (histogram matching in Lab)
cv::Mat normalizeColorLab(const cv::Mat& src, const cv::Mat& ref, const cv::Mat& mask) {
    cv::Mat srcLab, refLab;
    cv::cvtColor(src, srcLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(ref, refLab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> srcCh, refCh;
    cv::split(srcLab, srcCh);
    cv::split(refLab, refCh);
    for (int i = 0; i < 3; ++i) {
        cv::Scalar srcMean, srcStd, refMean, refStd;
        cv::meanStdDev(srcCh[i], srcMean, srcStd, mask);
        cv::meanStdDev(refCh[i], refMean, refStd, mask);
        srcCh[i].convertTo(srcCh[i], CV_32F);
        srcCh[i] = (srcCh[i] - srcMean[0]) * (refStd[0] / (srcStd[0] + 1e-6)) + refMean[0];
        srcCh[i].convertTo(srcCh[i], CV_8U);
    }
    cv::Mat outLab, outBGR;
    cv::merge(srcCh, outLab);
    cv::cvtColor(outLab, outBGR, cv::COLOR_Lab2BGR);
    return outBGR;
}

cv::Mat DeepFakeModelImpl::transform(const cv::Mat& inputImage, const cv::Rect& faceRect) {
    if (!modelLoaded || targetFace.empty()) {
        return inputImage.clone();
    }
    try {
        cv::Mat sourceFace = extractFace(inputImage, faceRect);
        if (sourceFace.empty() || sourceFace.cols < 10 || sourceFace.rows < 10) {
            std::cerr << "Extracted face is empty or too small." << std::endl;
            return inputImage.clone();
        }
        if (debugFrameCount < maxDebugFrames) {
            std::filesystem::create_directory("debug");
            cv::imwrite("debug/source_extracted_" + std::to_string(debugFrameCount) + ".png", sourceFace);
        }
        std::vector<cv::Point2f> sourceLandmarks;
        bool hasLandmarks = false;
        if (hasLandmarkDetector) {
            hasLandmarks = detectLandmarks(sourceFace, sourceLandmarks);
        }
        if (debugFrameCount < maxDebugFrames && hasLandmarks) {
            std::cerr << "[DEBUG] Landmarks for frame " << debugFrameCount << ": ";
            for (const auto& pt : sourceLandmarks) {
                std::cerr << "(" << pt.x << "," << pt.y << ") ";
            }
            std::cerr << std::endl;
        }
        cv::Mat warpedTargetFace;
        std::vector<cv::Point2f> targetLandmarks;
        bool targetHasLandmarks = false;
        if (hasLandmarks) {
            targetHasLandmarks = detectLandmarks(targetFace, targetLandmarks);
            if (targetHasLandmarks && targetLandmarks.size() == sourceLandmarks.size()) {
                warpedTargetFace = warpFacePiecewise(targetFace, targetLandmarks, sourceLandmarks, sourceFace.size());
                // Fill any black (unwarped) regions with resized target face for seamlessness
                cv::Mat fallbackTarget;
                cv::resize(targetFace, fallbackTarget, sourceFace.size());
                cv::Mat maskMissing;
                cv::inRange(warpedTargetFace, cv::Scalar(0,0,0), cv::Scalar(0,0,0), maskMissing);
                fallbackTarget.copyTo(warpedTargetFace, maskMissing);
                if (debugFrameCount < maxDebugFrames) {
                    cv::imwrite("debug/target_piecewise_" + std::to_string(debugFrameCount) + ".png", warpedTargetFace);
                }
                // --- Feature Animation ---
                // Animate mouth, eyes, eyebrows
                std::vector<std::pair<const std::vector<int>*, std::string>> features = {
                    {&MOUTH_IDX, "mouth"}, {&LEFT_EYE_IDX, "left_eye"}, {&RIGHT_EYE_IDX, "right_eye"}, {&LEFT_BROW_IDX, "left_brow"}, {&RIGHT_BROW_IDX, "right_brow"}
                };
                for (const auto& feature : features) {
                    cv::Mat animated = warpFeatureRegion(targetFace, targetLandmarks, sourceLandmarks, *feature.first, sourceFace.size());
                    if (!animated.empty()) {
                        // Use a mask for the feature
                        std::vector<cv::Point> regionPts;
                        for (int idx : *feature.first) {
                            if (idx < sourceLandmarks.size())
                                regionPts.push_back(cv::Point(cvRound(sourceLandmarks[idx].x), cvRound(sourceLandmarks[idx].y)));
                        }
                        if (regionPts.size() >= 3) {
                            cv::Mat featureMask = cv::Mat::zeros(sourceFace.size(), CV_8UC1);
                            cv::fillConvexPoly(featureMask, regionPts, 255);
                            animated.copyTo(warpedTargetFace, featureMask);
                        }
                    }
                }
            } else {
                cv::resize(targetFace, warpedTargetFace, sourceFace.size());
            }
        } else {
            cv::resize(targetFace, warpedTargetFace, sourceFace.size());
        }
        // Create face mask from source landmarks
        cv::Mat faceMask = hasLandmarks ? ::createFaceMask(sourceFace.size(), sourceLandmarks) : cv::Mat::ones(sourceFace.size(), CV_8UC1) * 255;
        if (warpedTargetFace.size() != sourceFace.size()) {
            cv::resize(warpedTargetFace, warpedTargetFace, sourceFace.size());
        }
        if (faceMask.size() != sourceFace.size()) {
            cv::resize(faceMask, faceMask, sourceFace.size());
        }
        // Color normalization: match warped target face to source face in the face region
        cv::Mat colorNormTarget = ::normalizeColorLab(warpedTargetFace, sourceFace, faceMask);
        if (debugFrameCount < maxDebugFrames) {
            cv::imwrite("debug/target_colornorm_" + std::to_string(debugFrameCount) + ".png", colorNormTarget);
            cv::imwrite("debug/face_mask_" + std::to_string(debugFrameCount) + ".png", faceMask);
        }
        // Preprocess the color-normalized target face
        cv::Mat processedTargetFace = preprocessFace(colorNormTarget);
        if (processedTargetFace.empty()) {
            std::cerr << "Processed target face is empty." << std::endl;
            return inputImage.clone();
        }
        if (processedTargetFace.size() != sourceFace.size()) {
            cv::resize(processedTargetFace, processedTargetFace, sourceFace.size());
        }
        if (faceMask.size() != sourceFace.size()) {
            cv::resize(faceMask, faceMask, sourceFace.size());
        }
        if (debugFrameCount < maxDebugFrames) {
            cv::imwrite("debug/target_processed_" + std::to_string(debugFrameCount) + ".png", processedTargetFace);
        }
        // Blend the processed target face into the source face using the mask
        cv::Mat blendedFace = sourceFace.clone();
        processedTargetFace.copyTo(blendedFace, faceMask);
        // Optionally, feather the mask for smoother blending
        if (hasLandmarks) {
            cv::Mat featheredMask;
            int blurSize = std::max(3, (int)(0.1 * std::min(faceMask.cols, faceMask.rows)) | 1);
            cv::GaussianBlur(faceMask, featheredMask, cv::Size(blurSize, blurSize), blurSize/2.0);
            for (int y = 0; y < blendedFace.rows; ++y) {
                for (int x = 0; x < blendedFace.cols; ++x) {
                    float alpha = featheredMask.at<uchar>(y, x) / 255.0f;
                    for (int c = 0; c < 3; ++c) {
                        blendedFace.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(
                            (1.0f - alpha) * sourceFace.at<cv::Vec3b>(y, x)[c] + alpha * processedTargetFace.at<cv::Vec3b>(y, x)[c]
                        );
                    }
                }
            }
        }
        // Beautification
        cv::Mat enhancedFace;
        cv::detailEnhance(blendedFace, enhancedFace, smoothness * 5.0f, 0.15f);
        if (hasLandmarks) {
            preserveEyes(enhancedFace, sourceFace, sourceLandmarks);
        }
        if (debugFrameCount < maxDebugFrames) {
            cv::imwrite("debug/final_face_" + std::to_string(debugFrameCount) + ".png", enhancedFace);
        }
        // --- Seamless blending using Poisson blending ---
        // Find center of faceRect in inputImage
        cv::Point center(faceRect.x + faceRect.width/2, faceRect.y + faceRect.height/2);
        cv::Mat result;
        try {
            cv::seamlessClone(enhancedFace, inputImage, faceMask, center, result, cv::NORMAL_CLONE);
        } catch (...) {
            // Fallback to old blend if seamlessClone fails
            result = blendFaces(inputImage, enhancedFace, faceRect);
        }
        if (debugFrameCount < maxDebugFrames) {
            cv::imwrite("debug/output_" + std::to_string(debugFrameCount) + ".png", result);
            std::cerr << "[DEBUG] Face rect for frame " << debugFrameCount << ": (" << faceRect.x << "," << faceRect.y << "," << faceRect.width << "," << faceRect.height << ")" << std::endl;
            ++debugFrameCount;
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Exception in transform: " << e.what() << std::endl;
        return inputImage.clone();
    } catch (...) {
        std::cerr << "Unknown exception in transform." << std::endl;
        return inputImage.clone();
    }
}

bool DeepFakeModelImpl::setTargetFace(const cv::Mat& targetFaceImage) {
    if (targetFaceImage.empty()) {
        return false;
    }
    
    // Store a copy of the original image
    cv::Mat originalImage = targetFaceImage.clone();
    
    // Detect face in the target image to ensure it's properly framed
    std::vector<cv::Rect> faces = detectFaces(originalImage);
    
    // If no face is detected, return false
    if (faces.empty()) {
        std::cerr << "No face detected in the target image" << std::endl;
        return false;
    }
    
    // Find the largest face (typically the main face in the image)
    cv::Rect bestFace = faces[0];
    for (const auto& face : faces) {
        if (face.area() > bestFace.area()) {
            bestFace = face;
        }
    }
    
    // Add some margin
    int margin = static_cast<int>(std::min(bestFace.width, bestFace.height) * 0.2);
    bestFace.x = std::max(0, bestFace.x - margin);
    bestFace.y = std::max(0, bestFace.y - margin);
    bestFace.width = std::min(originalImage.cols - bestFace.x, bestFace.width + 2 * margin);
    bestFace.height = std::min(originalImage.rows - bestFace.y, bestFace.height + 2 * margin);
    
    // Extract the face
    targetFace = originalImage(bestFace).clone();
    
    // Resize to a standard size
    cv::resize(targetFace, targetFace, cv::Size(256, 256));
    
    return true;
}

bool DeepFakeModelImpl::loadTargetFace(const std::string& targetFacePath) {
    // Load the target face from a file
    cv::Mat face = cv::imread(targetFacePath);
    if (face.empty()) {
        std::cerr << "Failed to load target face from: " << targetFacePath << std::endl;
        return false;
    }
    
    // Use setTargetFace to process the loaded image
    return setTargetFace(face);
}

cv::Mat DeepFakeModelImpl::extractFace(const cv::Mat& image, const cv::Rect& faceRect) {
    // Extract the face region with some margin
    float marginScale = 0.2f * faceSize; // Adjustable margin based on faceSize parameter
    cv::Rect expandedRect = faceRect;
    
    // Add margin
    int margin = static_cast<int>(std::min(faceRect.width, faceRect.height) * marginScale);
    expandedRect.x = std::max(0, expandedRect.x - margin);
    expandedRect.y = std::max(0, expandedRect.y - margin);
    expandedRect.width = std::min(image.cols - expandedRect.x, expandedRect.width + 2 * margin);
    expandedRect.height = std::min(image.rows - expandedRect.y, expandedRect.height + 2 * margin);
    
    // Extract the face
    return image(expandedRect).clone();
}

cv::Mat DeepFakeModelImpl::preprocessFace(const cv::Mat& face) {
    // Simple preprocessing: resize to model input size
    cv::Mat processedFace;
    cv::resize(face, processedFace, cv::Size(256, 256));
    
    // Apply any needed color correction
    cv::Mat enhancedFace;
    cv::detailEnhance(processedFace, enhancedFace, smoothness * 10.0f, 0.1f);
    
    return enhancedFace;
}

bool DeepFakeModelImpl::detectLandmarks(const cv::Mat& face, std::vector<cv::Point2f>& landmarks) {
    if (!dlibInitialized) return false;
    try {
        dlib::cv_image<dlib::bgr_pixel> dlibImg(face);
        std::vector<dlib::rectangle> dets = dlibFaceDetector(dlibImg);
        if (dets.empty()) return false;
        dlib::full_object_detection shape = dlibShapePredictor(dlibImg, dets[0]);
        landmarks.clear();
        for (unsigned int i = 0; i < shape.num_parts(); ++i) {
            landmarks.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "dlib landmark detection error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat DeepFakeModelImpl::alignFace(const cv::Mat& face, std::vector<cv::Point2f>& landmarks) {
    try {
        // Align the face so the eyes are horizontal and at a standard distance
        if (landmarks.size() < 17) {
            std::cerr << "Not enough landmarks for alignment." << std::endl;
            return face.clone();
        }
        // Use left eye (landmarks 5-8) and right eye (landmarks 9-12)
        cv::Point2f leftEye(0, 0), rightEye(0, 0);
        for (int i = 5; i <= 8; ++i) leftEye += landmarks[i];
        leftEye *= 0.25f;
        for (int i = 9; i <= 12; ++i) rightEye += landmarks[i];
        rightEye *= 0.25f;
        // Use mouth center (landmarks 15 and 16) as third point
        cv::Point2f mouthCenter = (landmarks[15] + landmarks[16]) * 0.5f;
        // Desired positions in output image
        float desiredLeftX = 0.35f;
        float desiredRightX = 0.65f;
        float desiredY = 0.4f;
        int outSize = 256;
        cv::Point2f desiredLeftEye(outSize * desiredLeftX, outSize * desiredY);
        cv::Point2f desiredRightEye(outSize * desiredRightX, outSize * desiredY);
        // Place the mouth center at 0.5, 0.75
        cv::Point2f desiredMouth(outSize * 0.5f, outSize * 0.75f);
        // Prepare triangles
        std::vector<cv::Point2f> srcTri = {leftEye, rightEye, mouthCenter};
        std::vector<cv::Point2f> dstTri = {desiredLeftEye, desiredRightEye, desiredMouth};
        cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
        cv::Mat aligned;
        cv::warpAffine(face, aligned, warpMat, cv::Size(outSize, outSize), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        return aligned;
    } catch (const std::exception& e) {
        std::cerr << "Exception in alignFace: " << e.what() << std::endl;
        return face.clone();
    } catch (...) {
        std::cerr << "Unknown exception in alignFace." << std::endl;
        return face.clone();
    }
}

cv::Mat DeepFakeModelImpl::applyColorTransfer(const cv::Mat& source, const cv::Mat& target) {
    // Implementation of color transfer algorithm
    // Convert to Lab color space
    cv::Mat sourceLab, targetLab;
    cv::cvtColor(source, sourceLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, targetLab, cv::COLOR_BGR2Lab);
    
    // Split channels
    std::vector<cv::Mat> sourceChannels, targetChannels;
    cv::split(sourceLab, sourceChannels);
    cv::split(targetLab, targetChannels);
    
    // Compute mean and std dev for each channel
    cv::Scalar sourceMean, targetMean, sourceStd, targetStd;
    cv::meanStdDev(sourceLab, sourceMean, sourceStd);
    cv::meanStdDev(targetLab, targetMean, targetStd);
    
    // Apply color transfer
    std::vector<cv::Mat> resultChannels;
    for (int i = 0; i < 3; i++) {
        cv::Mat channel;
        // (source - sourceMean) * (targetStd / sourceStd) + targetMean
        sourceChannels[i].convertTo(channel, CV_32F);
        channel = (channel - sourceMean[i]) * (targetStd[i] / sourceStd[i]) + targetMean[i];
        channel.convertTo(channel, CV_8U);
        resultChannels.push_back(channel);
    }
    
    // Merge channels
    cv::Mat resultLab;
    cv::merge(resultChannels, resultLab);
    
    // Convert back to BGR
    cv::Mat result;
    cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
    
    return result;
}

cv::Mat DeepFakeModelImpl::blendFaces(const cv::Mat& originalImage, const cv::Mat& transformedFace, const cv::Rect& faceRect) {
    cv::Mat result = originalImage.clone();
    
    // Apply margin similarly to extractFace
    float marginScale = 0.2f * faceSize;
    cv::Rect expandedRect = faceRect;
    int margin = static_cast<int>(std::min(faceRect.width, faceRect.height) * marginScale);
    expandedRect.x = std::max(0, expandedRect.x - margin);
    expandedRect.y = std::max(0, expandedRect.y - margin);
    expandedRect.width = std::min(originalImage.cols - expandedRect.x, expandedRect.width + 2 * margin);
    expandedRect.height = std::min(originalImage.rows - expandedRect.y, expandedRect.height + 2 * margin);
    
    // Resize the transformed face back to the original face size
    cv::Mat resizedFace;
    cv::resize(transformedFace, resizedFace, cv::Size(expandedRect.width, expandedRect.height));
    
    // Create a better mask for face blending
    cv::Mat mask = cv::Mat::zeros(expandedRect.height, expandedRect.width, CV_8UC1);
    
    // Create a more natural face shape using a combination of ellipse and polygon
    int centerX = expandedRect.width / 2;
    int centerY = expandedRect.height / 2;
    int faceWidth = expandedRect.width - 20; // Slightly narrower for better fit
    int faceHeight = expandedRect.height - 10;
    
    // Draw the main face ellipse
    cv::ellipse(mask, 
                cv::Point(centerX, centerY),
                cv::Size(faceWidth/2, faceHeight/2),
                0, 0, 360, cv::Scalar(255), -1);
    
    // Extend the chin area to look more natural
    std::vector<cv::Point> chinPolygon;
    chinPolygon.push_back(cv::Point(centerX - faceWidth/3, centerY + faceHeight/3));
    chinPolygon.push_back(cv::Point(centerX, centerY + faceHeight/2 + 15)); // Extend the chin
    chinPolygon.push_back(cv::Point(centerX + faceWidth/3, centerY + faceHeight/3));
    cv::fillConvexPoly(mask, chinPolygon, cv::Scalar(255));
    
    // Apply a more sophisticated feathering based on the smoothness parameter
    int featherAmount = static_cast<int>(51 * smoothness);
    if (featherAmount % 2 == 0) featherAmount++;  // Must be odd
    featherAmount = std::max(3, featherAmount);
    
    // Apply bilateral filtering for edge-aware blending
    cv::Mat blurredMask;
    cv::bilateralFilter(mask, blurredMask, featherAmount, featherAmount*2, featherAmount/2);
    
    // Additional Gaussian blur for smoother transitions
    cv::GaussianBlur(blurredMask, blurredMask, cv::Size(featherAmount, featherAmount), 0);
    
    // Create a color blended version of the original and transformed face
    cv::Mat colorBlendedFace = resizedFace.clone();
    
    // Create gradual alpha blend based on face mask
    cv::Mat subRegion = result(expandedRect);
    for (int y = 0; y < subRegion.rows; y++) {
        for (int x = 0; x < subRegion.cols; x++) {
            // Get blend amount from the mask (gradually decreases towards the edges)
            float alpha = blurredMask.at<uchar>(y, x) / 255.0f;
            
            // Apply additional edge-aware blending for natural look
            if (alpha > 0.05f && alpha < 0.95f) {
                // Calculate edge-aware blending for transition regions
                cv::Vec3b srcColor = subRegion.at<cv::Vec3b>(y, x);
                cv::Vec3b dstColor = resizedFace.at<cv::Vec3b>(y, x);
                
                // Calculate color similarity for edge-adaptive blending
                float colorDist = 0;
                for (int c = 0; c < 3; c++) {
                    colorDist += std::abs(srcColor[c] - dstColor[c]);
                }
                colorDist /= 765.0f; // Normalize (max diff = 255*3)
                
                // Adjust alpha based on color similarity to reduce seams
                alpha = alpha * (1.0f - colorDist*0.5f);
            }
            
            // Apply the final blending
            if (alpha > 0) {
                subRegion.at<cv::Vec3b>(y, x) = 
                    cv::Vec3b(
                        cv::saturate_cast<uchar>((1 - alpha) * subRegion.at<cv::Vec3b>(y, x)[0] + alpha * resizedFace.at<cv::Vec3b>(y, x)[0]),
                        cv::saturate_cast<uchar>((1 - alpha) * subRegion.at<cv::Vec3b>(y, x)[1] + alpha * resizedFace.at<cv::Vec3b>(y, x)[1]),
                        cv::saturate_cast<uchar>((1 - alpha) * subRegion.at<cv::Vec3b>(y, x)[2] + alpha * resizedFace.at<cv::Vec3b>(y, x)[2])
                    );
            }
        }
    }
    
    return result;
}

cv::Mat DeepFakeModelImpl::applySkintoneCorrection(const cv::Mat& sourceFace, const cv::Mat& targetFace) {
    // Create a mask for skin areas (simplified approach using color thresholding)
    cv::Mat skinMask;
    cv::Mat hsvTarget;
    cv::cvtColor(targetFace, hsvTarget, cv::COLOR_BGR2HSV);
    
    // Define skin tone range in HSV (this is approximate and can be improved)
    cv::Scalar lowerBound(0, 20, 70);
    cv::Scalar upperBound(20, 150, 255);
    
    // Create mask for skin tones
    cv::inRange(hsvTarget, lowerBound, upperBound, skinMask);
    
    // Dilate the mask to cover more skin area
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(skinMask, skinMask, kernel);
    
    // Compute average skin tone of source face
    cv::Mat hsvSource;
    cv::cvtColor(sourceFace, hsvSource, cv::COLOR_BGR2HSV);
    cv::Scalar avgSourceTone = cv::mean(hsvSource, skinMask);
    
    // Compute average skin tone of target face
    cv::Scalar avgTargetTone = cv::mean(hsvTarget, skinMask);
    
    // Create adjustment values
    cv::Scalar adjustment = avgSourceTone - avgTargetTone;
    
    // Apply adjustment to the target face where skin is detected
    cv::Mat result = targetFace.clone();
    cv::Mat hsvResult;
    cv::cvtColor(result, hsvResult, cv::COLOR_BGR2HSV);
    
    // Adjust only the Hue and Saturation channels in skin regions, not the Value (brightness)
    for (int y = 0; y < hsvResult.rows; y++) {
        for (int x = 0; x < hsvResult.cols; x++) {
            if (skinMask.at<uchar>(y, x) > 0) {
                cv::Vec3b& pixel = hsvResult.at<cv::Vec3b>(y, x);
                // Adjust Hue (H)
                float newH = pixel[0] + adjustment[0] * 0.5f; // Use partial adjustment for natural look
                if (newH > 180) newH -= 180;
                if (newH < 0) newH += 180;
                pixel[0] = static_cast<uchar>(newH);
                
                // Adjust Saturation (S)
                float newS = pixel[1] + adjustment[1] * 0.5f;
                pixel[1] = static_cast<uchar>(cv::saturate_cast<uchar>(newS));
            }
        }
    }
    
    // Convert back to BGR
    cv::cvtColor(hsvResult, result, cv::COLOR_HSV2BGR);
    
    return result;
}

void DeepFakeModelImpl::preserveEyes(cv::Mat& transformedFace, const cv::Mat& originalFace, const std::vector<cv::Point2f>& landmarks) {
    // Create masks for important facial features (eyes, eyebrows, lips)
    cv::Mat featureMask = cv::Mat::zeros(transformedFace.size(), CV_8UC1);
    
    if (landmarks.size() < 15) {
        // Not enough landmarks, fallback to basic eye preservation
        int eyeYPos = transformedFace.rows * 0.35;
        int leftX = transformedFace.cols * 0.25;
        int rightX = transformedFace.cols * 0.75;
        int eyeWidth = transformedFace.cols * 0.15;
        int eyeHeight = transformedFace.rows * 0.08;
        
        // Draw approximate eye regions
        cv::ellipse(featureMask, 
                   cv::Point(leftX, eyeYPos),
                   cv::Size(eyeWidth, eyeHeight),
                   0, 0, 360, cv::Scalar(255), -1);
        
        cv::ellipse(featureMask, 
                   cv::Point(rightX, eyeYPos),
                   cv::Size(eyeWidth, eyeHeight),
                   0, 0, 360, cv::Scalar(255), -1);
    } else {
        // We have enough landmarks to use them for feature preservation
        
        // Extract landmark points for eyes (assuming landmarks 5-12 are eyes based on our detection)
        std::vector<cv::Point> leftEyePoints, rightEyePoints;
        
        // Left eye (landmarks 5-8)
        for (int i = 5; i <= 8; i++) {
            if (i < landmarks.size()) {
                leftEyePoints.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
            }
        }
        
        // Right eye (landmarks 9-12)
        for (int i = 9; i <= 12; i++) {
            if (i < landmarks.size()) {
                rightEyePoints.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
            }
        }
        
        // Mouth points (landmarks 14-16)
        std::vector<cv::Point> mouthPoints;
        for (int i = 14; i <= 16; i++) {
            if (i < landmarks.size()) {
                mouthPoints.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
            }
        }
        
        // Draw eyes and mouth on the mask with some padding
        if (leftEyePoints.size() >= 4) {
            // Get min/max coordinates to create a bounding rectangle with padding
            int minX = transformedFace.cols, minY = transformedFace.rows;
            int maxX = 0, maxY = 0;
            
            for (const auto& pt : leftEyePoints) {
                minX = std::min(minX, pt.x);
                minY = std::min(minY, pt.y);
                maxX = std::max(maxX, pt.x);
                maxY = std::max(maxY, pt.y);
            }
            
            // Add padding
            int padding = std::max((maxX - minX) / 4, (maxY - minY) / 4);
            cv::ellipse(featureMask, 
                       cv::Point((minX + maxX) / 2, (minY + maxY) / 2),
                       cv::Size((maxX - minX) / 2 + padding, (maxY - minY) / 2 + padding),
                       0, 0, 360, cv::Scalar(255), -1);
        }
        
        if (rightEyePoints.size() >= 4) {
            // Similar approach for right eye
            int minX = transformedFace.cols, minY = transformedFace.rows;
            int maxX = 0, maxY = 0;
            
            for (const auto& pt : rightEyePoints) {
                minX = std::min(minX, pt.x);
                minY = std::min(minY, pt.y);
                maxX = std::max(maxX, pt.x);
                maxY = std::max(maxY, pt.y);
            }
            
            int padding = std::max((maxX - minX) / 4, (maxY - minY) / 4);
            cv::ellipse(featureMask, 
                       cv::Point((minX + maxX) / 2, (minY + maxY) / 2),
                       cv::Size((maxX - minX) / 2 + padding, (maxY - minY) / 2 + padding),
                       0, 0, 360, cv::Scalar(255), -1);
        }
        
        // Draw mouth with half intensity to partially preserve lips
        if (mouthPoints.size() >= 3) {
            int minX = transformedFace.cols, minY = transformedFace.rows;
            int maxX = 0, maxY = 0;
            
            for (const auto& pt : mouthPoints) {
                minX = std::min(minX, pt.x);
                minY = std::min(minY, pt.y);
                maxX = std::max(maxX, pt.x);
                maxY = std::max(maxY, pt.y);
            }
            
            int padding = std::max((maxX - minX) / 4, (maxY - minY) / 4);
            cv::rectangle(featureMask, 
                        cv::Rect(minX - padding, minY - padding, 
                                maxX - minX + 2*padding, maxY - minY + 2*padding),
                        cv::Scalar(128), -1);  // Half intensity for mouth
        }
    }
    
    // Apply feathering to the mask for smooth transitions
    cv::GaussianBlur(featureMask, featureMask, cv::Size(11, 11), 5.0);
    
    // Copy original features to transformed face based on mask
    for (int y = 0; y < transformedFace.rows; y++) {
        for (int x = 0; x < transformedFace.cols; x++) {
            float alpha = featureMask.at<uchar>(y, x) / 255.0f;
            if (alpha > 0) {
                transformedFace.at<cv::Vec3b>(y, x) = 
                    cv::Vec3b(
                        cv::saturate_cast<uchar>((1 - alpha) * transformedFace.at<cv::Vec3b>(y, x)[0] + alpha * originalFace.at<cv::Vec3b>(y, x)[0]),
                        cv::saturate_cast<uchar>((1 - alpha) * transformedFace.at<cv::Vec3b>(y, x)[1] + alpha * originalFace.at<cv::Vec3b>(y, x)[1]),
                        cv::saturate_cast<uchar>((1 - alpha) * transformedFace.at<cv::Vec3b>(y, x)[2] + alpha * originalFace.at<cv::Vec3b>(y, x)[2])
                    );
            }
        }
    }
}

// DeepFakeModel implementation (wrapper for the impl)
DeepFakeModel::DeepFakeModel(const std::string& modelPath, Backend backend) 
    : pImpl(new DeepFakeModelImpl(modelPath, backend)) {
}

DeepFakeModel::~DeepFakeModel() = default;

cv::Mat DeepFakeModel::transform(const cv::Mat& inputImage, const cv::Rect& faceRect) {
    return pImpl->transform(inputImage, faceRect);
}

bool DeepFakeModel::setTargetFace(const cv::Mat& targetFaceImage) {
    return pImpl->setTargetFace(targetFaceImage);
}

bool DeepFakeModel::loadTargetFace(const std::string& targetFacePath) {
    return pImpl->loadTargetFace(targetFacePath);
}

std::vector<cv::Rect> DeepFakeModel::detectFaces(const cv::Mat& image) {
    return pImpl->detectFaces(image);
}

void DeepFakeModel::setBlendAmount(float value) { pImpl->setBlendAmount(value); }
void DeepFakeModel::setFaceSize(float value) { pImpl->setFaceSize(value); }
void DeepFakeModel::setSmoothness(float value) { pImpl->setSmoothness(value); } 