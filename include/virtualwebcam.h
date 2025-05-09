#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Manages a virtual webcam using v4l2loopback
 */
class VirtualWebcam {
public:
    /**
     * @brief Constructor
     * @param devicePath Path to the virtual webcam device (e.g., /dev/video2)
     * @param width Frame width
     * @param height Frame height
     */
    VirtualWebcam(const std::string& devicePath, int width = 640, int height = 480);
    
    /**
     * @brief Destructor
     */
    ~VirtualWebcam();
    
    /**
     * @brief Writes a frame to the virtual webcam
     * @param frame Frame to write
     * @return True if successful
     */
    bool writeFrame(const cv::Mat& frame);
    
    /**
     * @brief Checks if the virtual webcam is ready
     * @return True if ready
     */
    bool isReady() const;
    
    /**
     * @brief Gets the device path
     * @return Device path
     */
    std::string getDevicePath() const;
    
    /**
     * @brief Gets the frame width
     * @return Frame width
     */
    int getWidth() const;
    
    /**
     * @brief Gets the frame height
     * @return Frame height
     */
    int getHeight() const;
    
    /**
     * @brief Sets the frame dimensions
     * @param width Frame width
     * @param height Frame height
     * @return True if successful
     */
    bool setDimensions(int width, int height);
    
    /**
     * @brief Lists available v4l2loopback devices
     * @return Vector of device paths
     */
    static std::vector<std::string> listLoopbackDevices();
    
    /**
     * @brief Checks if a device is a v4l2loopback device
     * @param devicePath Device path
     * @return True if it's a v4l2loopback device
     */
    static bool isLoopbackDevice(const std::string& devicePath);
    
    /**
     * @brief Creates a new v4l2loopback device
     * @param deviceNumber Device number (e.g., 2 for /dev/video2)
     * @param label Device label
     * @return Device path if successful, empty string otherwise
     */
    static std::string createLoopbackDevice(int deviceNumber, const std::string& label = "DeepFakeCam");
    
private:
    int fd;
    std::string devicePath;
    int width;
    int height;
    int frameSize;
    bool ready;
    
    /**
     * @brief Opens the virtual webcam device
     * @return True if successful
     */
    bool openDevice();
    
    /**
     * @brief Closes the virtual webcam device
     */
    void closeDevice();
}; 