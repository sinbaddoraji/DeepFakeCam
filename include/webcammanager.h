#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

/**
 * @brief Manages webcam devices and provides utilities for listing and opening them
 */
class WebcamManager {
public:
    struct WebcamDevice {
        int id;                 // Device identifier (index)
        std::string name;       // Human-readable name
        std::string path;       // Device path (e.g., /dev/video0)
        int width;              // Default width
        int height;             // Default height
        std::vector<double> supportedFps; // Supported frame rates
    };

    /**
     * @brief Constructor
     */
    WebcamManager();
    
    /**
     * @brief Destructor
     */
    ~WebcamManager();
    
    /**
     * @brief Refreshes the list of available webcam devices
     */
    void refresh();
    
    /**
     * @brief Gets the list of available webcam devices
     * @return Vector of WebcamDevice structures
     */
    const std::vector<WebcamDevice>& getDevices() const;
    
    /**
     * @brief Opens a webcam by its index
     * @param index Device index
     * @param width Desired width (optional)
     * @param height Desired height (optional)
     * @return OpenCV VideoCapture object
     */
    cv::VideoCapture openWebcam(int index, int width = 640, int height = 480);
    
    /**
     * @brief Opens a webcam by its path
     * @param path Device path
     * @param width Desired width (optional)
     * @param height Desired height (optional)
     * @return OpenCV VideoCapture object
     */
    cv::VideoCapture openWebcamByPath(const std::string& path, int width = 640, int height = 480);
    
    /**
     * @brief Gets information about the default webcam resolution
     * @param index Device index
     * @param width Reference to store the width
     * @param height Reference to store the height
     * @return True if successful
     */
    bool getDefaultResolution(int index, int& width, int& height);
    
    /**
     * @brief Gets the name of a webcam device
     * @param index Device index
     * @return Device name
     */
    std::string getDeviceName(int index) const;
    
private:
    std::vector<WebcamDevice> devices;
    
    /**
     * @brief Detects webcam devices on Linux
     */
    void detectDevicesLinux();
    
    /**
     * @brief Gets information about a V4L2 device
     * @param path Device path
     * @return WebcamDevice structure
     */
    WebcamDevice getV4L2DeviceInfo(const std::string& path);
}; 