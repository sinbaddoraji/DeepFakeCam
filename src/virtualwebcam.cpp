#include "virtualwebcam.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <dirent.h>
#include <cstring>

VirtualWebcam::VirtualWebcam(const std::string& devicePath, int width, int height)
    : devicePath(devicePath), width(width), height(height), fd(-1), frameSize(0), ready(false)
{
    openDevice();
}

VirtualWebcam::~VirtualWebcam()
{
    closeDevice();
}

bool VirtualWebcam::writeFrame(const cv::Mat& frame)
{
    if (!isReady() || frame.empty()) {
        return false;
    }
    
    // Convert the frame to RGB format if needed
    cv::Mat rgbFrame;
    if (frame.channels() == 3 && frame.type() == CV_8UC3) {
        // BGR to RGB conversion
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
    } else {
        frame.copyTo(rgbFrame);
    }
    
    // Resize the frame if needed
    if (rgbFrame.cols != width || rgbFrame.rows != height) {
        cv::resize(rgbFrame, rgbFrame, cv::Size(width, height));
    }
    
    // Write the frame to the virtual webcam
    ssize_t written = write(fd, rgbFrame.data, frameSize);
    return written == frameSize;
}

bool VirtualWebcam::isReady() const
{
    return ready;
}

std::string VirtualWebcam::getDevicePath() const
{
    return devicePath;
}

int VirtualWebcam::getWidth() const
{
    return width;
}

int VirtualWebcam::getHeight() const
{
    return height;
}

bool VirtualWebcam::setDimensions(int width, int height)
{
    if (isReady()) {
        closeDevice();
    }
    
    this->width = width;
    this->height = height;
    
    return openDevice();
}

std::vector<std::string> VirtualWebcam::listLoopbackDevices()
{
    std::vector<std::string> devices;
    DIR *dir = opendir("/dev");
    if (dir == nullptr) {
        return devices;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        if (name.find("video") == 0) {
            std::string path = "/dev/" + name;
            
            if (isLoopbackDevice(path)) {
                devices.push_back(path);
            }
        }
    }
    
    closedir(dir);
    return devices;
}

bool VirtualWebcam::isLoopbackDevice(const std::string& devicePath)
{
    int fd = open(devicePath.c_str(), O_RDWR);
    if (fd < 0) {
        return false;
    }
    
    struct v4l2_capability cap;
    bool isLoopback = false;
    
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) >= 0) {
        // v4l2loopback devices typically have "Loopback" in the driver name
        if (strstr(reinterpret_cast<const char*>(cap.driver), "loopback") != nullptr ||
            strstr(reinterpret_cast<const char*>(cap.card), "Loopback") != nullptr) {
            isLoopback = true;
        }
    }
    
    close(fd);
    return isLoopback;
}

std::string VirtualWebcam::createLoopbackDevice(int deviceNumber, const std::string& label)
{
    // This would require root privileges to execute modprobe
    // In practice, the user would need to create the loopback device separately
    // This is just a placeholder function
    
    std::string devicePath = "/dev/video" + std::to_string(deviceNumber);
    
    // Check if the device already exists and is a loopback device
    if (isLoopbackDevice(devicePath)) {
        return devicePath;
    }
    
    std::cerr << "Cannot create loopback device. Please run the following command as root:" << std::endl;
    std::cerr << "modprobe v4l2loopback devices=1 video_nr=" << deviceNumber 
              << " card_label=\"" << label << "\" exclusive_caps=1" << std::endl;
    
    return "";
}

bool VirtualWebcam::openDevice()
{
    // Close existing device if open
    if (fd >= 0) {
        closeDevice();
    }
    
    // Try to open the device
    fd = open(devicePath.c_str(), O_WRONLY);
    if (fd < 0) {
        std::cerr << "Failed to open virtual webcam device: " << devicePath << std::endl;
        ready = false;
        return false;
    }
    
    // Calculate frame size (RGB format)
    frameSize = width * height * 3;
    
    // Set format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = width * 3;
    fmt.fmt.pix.sizeimage = frameSize;
    
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "Failed to set format for virtual webcam" << std::endl;
        closeDevice();
        return false;
    }
    
    ready = true;
    return true;
}

void VirtualWebcam::closeDevice()
{
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
    ready = false;
} 