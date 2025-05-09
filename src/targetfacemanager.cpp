#include "targetfacemanager.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <random>
#include <chrono>
#include <sstream>
#include <filesystem>
#include <QDateTime>

TargetFaceManager::TargetFaceManager(const std::string& storageDir)
    : storageDir(storageDir)
{
    // Create storage directory if it doesn't exist
    struct stat info;
    if (stat(storageDir.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
        std::filesystem::create_directories(storageDir);
    }
    
    // Load existing faces
    loadFacesFromStorage();
}

TargetFaceManager::~TargetFaceManager()
{
    // Clean up resources if needed
}

std::string TargetFaceManager::addFace(const std::string& imagePath, const std::string& name)
{
    // Load the image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return "";
    }
    
    // Generate a unique ID for the face
    std::string faceId = generateFaceId();
    
    // Create face preset
    FacePreset preset;
    preset.name = name.empty() ? "Face " + faceId.substr(0, 8) : name;
    preset.imagePath = imagePath;
    preset.faceImage = image.clone();
    preset.blendAmount = 70;
    preset.faceSize = 50;
    preset.smoothness = 50;
    preset.modelType = "FaceSwap";
    
    // Save the face image to storage
    if (!saveFaceImage(faceId, image)) {
        return "";
    }
    
    // Add to face map
    faces[faceId] = preset;
    
    return faceId;
}

std::vector<std::string> TargetFaceManager::getFaceList() const
{
    std::vector<std::string> faceIds;
    for (const auto& pair : faces) {
        faceIds.push_back(pair.first);
    }
    return faceIds;
}

TargetFaceManager::FacePreset TargetFaceManager::getFaceInfo(const std::string& faceId) const
{
    auto it = faces.find(faceId);
    if (it != faces.end()) {
        return it->second;
    }
    
    // Return empty preset if not found
    return FacePreset();
}

cv::Mat TargetFaceManager::getFaceImage(const std::string& faceId) const
{
    auto it = faces.find(faceId);
    if (it != faces.end()) {
        return it->second.faceImage;
    }
    
    // Return empty image if not found
    return cv::Mat();
}

QPixmap TargetFaceManager::getFaceThumbnail(const std::string& faceId, int maxWidth, int maxHeight) const
{
    cv::Mat faceImage = getFaceImage(faceId);
    if (faceImage.empty()) {
        return QPixmap();
    }
    
    // Resize to thumbnail size
    double ratio = std::min(
        maxWidth / static_cast<double>(faceImage.cols),
        maxHeight / static_cast<double>(faceImage.rows)
    );
    
    cv::Mat thumbnail;
    cv::resize(
        faceImage, 
        thumbnail, 
        cv::Size(
            static_cast<int>(faceImage.cols * ratio),
            static_cast<int>(faceImage.rows * ratio)
        )
    );
    
    // Convert to QPixmap
    cv::Mat rgbImage;
    cv::cvtColor(thumbnail, rgbImage, cv::COLOR_BGR2RGB);
    
    QImage qImage(
        rgbImage.data,
        rgbImage.cols,
        rgbImage.rows,
        rgbImage.step,
        QImage::Format_RGB888
    );
    
    return QPixmap::fromImage(qImage);
}

bool TargetFaceManager::removeFace(const std::string& faceId)
{
    auto it = faces.find(faceId);
    if (it == faces.end()) {
        return false;
    }
    
    // Remove the face image file
    std::string imagePath = storageDir + "/" + faceId + ".jpg";
    remove(imagePath.c_str());
    
    // Remove from face map
    faces.erase(it);
    
    return true;
}

bool TargetFaceManager::savePreset(const std::string& faceId, const std::string& presetPath) const
{
    auto it = faces.find(faceId);
    if (it == faces.end()) {
        return false;
    }
    
    // Create JSON data (simple format for now)
    std::ofstream outFile(presetPath);
    if (!outFile.is_open()) {
        return false;
    }
    
    const FacePreset& preset = it->second;
    
    outFile << "{\n";
    outFile << "  \"faceId\": \"" << faceId << "\",\n";
    outFile << "  \"name\": \"" << preset.name << "\",\n";
    outFile << "  \"imagePath\": \"" << preset.imagePath << "\",\n";
    outFile << "  \"blendAmount\": " << preset.blendAmount << ",\n";
    outFile << "  \"faceSize\": " << preset.faceSize << ",\n";
    outFile << "  \"smoothness\": " << preset.smoothness << ",\n";
    outFile << "  \"modelType\": \"" << preset.modelType << "\"\n";
    outFile << "}\n";
    
    outFile.close();
    
    return true;
}

std::string TargetFaceManager::loadPreset(const std::string& presetPath)
{
    // In a real implementation, this would parse JSON
    // This is a simple stub implementation
    
    // For now, just return an empty string
    return "";
}

bool TargetFaceManager::updateFaceParameters(
    const std::string& faceId, 
    int blendAmount, 
    int faceSize, 
    int smoothness, 
    const std::string& modelType)
{
    auto it = faces.find(faceId);
    if (it == faces.end()) {
        return false;
    }
    
    // Update parameters
    it->second.blendAmount = blendAmount;
    it->second.faceSize = faceSize;
    it->second.smoothness = smoothness;
    it->second.modelType = modelType;
    
    return true;
}

std::string TargetFaceManager::generateFaceId() const
{
    // Generate a unique ID based on timestamp and random number
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99999);
    
    std::stringstream ss;
    ss << "face_" << timestamp << "_" << dis(gen);
    
    return ss.str();
}

void TargetFaceManager::loadFacesFromStorage()
{
    faces.clear();
    
    DIR *dir = opendir(storageDir.c_str());
    if (dir == nullptr) {
        return;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        // Check if it's a face image file
        if (name.find("face_") == 0 && name.find(".jpg") != std::string::npos) {
            // Extract face ID from filename
            std::string faceId = name.substr(0, name.length() - 4); // Remove .jpg
            
            // Load the face image
            std::string imagePath = storageDir + "/" + name;
            cv::Mat image = cv::imread(imagePath);
            
            if (!image.empty()) {
                // Create face preset
                FacePreset preset;
                preset.name = "Face " + faceId.substr(5, 8); // Skip "face_" prefix
                preset.imagePath = imagePath;
                preset.faceImage = image;
                preset.blendAmount = 70;
                preset.faceSize = 50;
                preset.smoothness = 50;
                preset.modelType = "FaceSwap";
                
                // Add to face map
                faces[faceId] = preset;
            }
        }
    }
    
    closedir(dir);
}

bool TargetFaceManager::saveFaceImage(const std::string& faceId, const cv::Mat& image) const
{
    std::string imagePath = storageDir + "/" + faceId + ".jpg";
    
    // Create a copy of the image with proper format
    cv::Mat saveImage;
    if (image.channels() != 3) {
        cv::cvtColor(image, saveImage, cv::COLOR_GRAY2BGR);
    } else {
        saveImage = image.clone();
    }
    
    // Save the image
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    
    bool success = cv::imwrite(imagePath, saveImage, compression_params);
    
    if (!success) {
        std::cerr << "Failed to save face image: " << imagePath << std::endl;
    }
    
    return success;
} 