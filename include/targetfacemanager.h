#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QString>

/**
 * @brief Manages target face images for deepfake transformations
 */
class TargetFaceManager {
public:
    struct FacePreset {
        std::string name;
        std::string imagePath;
        cv::Mat faceImage;
        int blendAmount;  // 0-100
        int faceSize;     // 0-100
        int smoothness;   // 0-100
        std::string modelType;
    };

    /**
     * @brief Constructor
     * @param storageDir Directory to store face images
     */
    TargetFaceManager(const std::string& storageDir = "faces");
    
    /**
     * @brief Destructor
     */
    ~TargetFaceManager();
    
    /**
     * @brief Adds a new target face from an image file
     * @param imagePath Path to the face image file
     * @param name Optional name for the face
     * @return ID of the new face or empty string if failed
     */
    std::string addFace(const std::string& imagePath, const std::string& name = "");
    
    /**
     * @brief Gets the list of all available faces
     * @return Vector of face IDs
     */
    std::vector<std::string> getFaceList() const;
    
    /**
     * @brief Gets information about a face
     * @param faceId Face ID
     * @return FacePreset structure
     */
    FacePreset getFaceInfo(const std::string& faceId) const;
    
    /**
     * @brief Gets a face image
     * @param faceId Face ID
     * @return OpenCV matrix with the face image
     */
    cv::Mat getFaceImage(const std::string& faceId) const;
    
    /**
     * @brief Gets a face thumbnail as QPixmap
     * @param faceId Face ID
     * @param maxWidth Maximum width of the thumbnail
     * @param maxHeight Maximum height of the thumbnail
     * @return QPixmap with the face thumbnail
     */
    QPixmap getFaceThumbnail(const std::string& faceId, int maxWidth = 128, int maxHeight = 128) const;
    
    /**
     * @brief Removes a face
     * @param faceId Face ID
     * @return True if successful
     */
    bool removeFace(const std::string& faceId);
    
    /**
     * @brief Saves a face preset to a file
     * @param faceId Face ID
     * @param presetPath Path to save the preset
     * @return True if successful
     */
    bool savePreset(const std::string& faceId, const std::string& presetPath) const;
    
    /**
     * @brief Loads a face preset from a file
     * @param presetPath Path to the preset file
     * @return Face ID of the loaded preset or empty string if failed
     */
    std::string loadPreset(const std::string& presetPath);
    
    /**
     * @brief Updates face parameters
     * @param faceId Face ID
     * @param blendAmount Blend amount (0-100)
     * @param faceSize Face size (0-100)
     * @param smoothness Smoothness (0-100)
     * @param modelType Model type
     * @return True if successful
     */
    bool updateFaceParameters(const std::string& faceId, int blendAmount, int faceSize, 
                             int smoothness, const std::string& modelType);
    
private:
    std::string storageDir;
    std::map<std::string, FacePreset> faces;
    
    /**
     * @brief Generates a unique ID for a face
     * @return Unique ID
     */
    std::string generateFaceId() const;
    
    /**
     * @brief Loads faces from the storage directory
     */
    void loadFacesFromStorage();
    
    /**
     * @brief Saves a face image to the storage directory
     * @param faceId Face ID
     * @param image Face image
     * @return True if successful
     */
    bool saveFaceImage(const std::string& faceId, const cv::Mat& image) const;
}; 