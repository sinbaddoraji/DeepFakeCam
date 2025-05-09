#include "webcammanager.h"
#include <iostream>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <algorithm>

WebcamManager::WebcamManager() {
    refresh();
}

WebcamManager::~WebcamManager() {
    // Clean up resources if needed
}

void WebcamManager::refresh() {
    devices.clear();
    detectDevicesLinux();
}

const std::vector<WebcamManager::WebcamDevice>& WebcamManager::getDevices() const {
    return devices;
}

cv::VideoCapture WebcamManager::openWebcam(int index, int width, int height) {
    cv::VideoCapture cap(index);
    if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    return cap;
}

cv::VideoCapture WebcamManager::openWebcamByPath(const std::string& path, int width, int height) {
    cv::VideoCapture cap(path);
    if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    return cap;
}

bool WebcamManager::getDefaultResolution(int index, int& width, int& height) {
    if (index >= 0 && index < devices.size()) {
        width = devices[index].width;
        height = devices[index].height;
        return true;
    }
    return false;
}

std::string WebcamManager::getDeviceName(int index) const {
    if (index >= 0 && index < devices.size()) {
        return devices[index].name;
    }
    return "Unknown Device";
}

void WebcamManager::detectDevicesLinux() {
    DIR *dir = opendir("/dev");
    if (dir == nullptr) {
        return;
    }
    
    struct dirent *entry;
    int id = 0;
    
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        // Check if it's a video device
        if (name.find("video") == 0) {
            std::string path = "/dev/" + name;
            
            // Try to open the device to check if it's a webcam
            int fd = open(path.c_str(), O_RDWR);
            if (fd < 0) {
                continue;
            }
            
            // Check capabilities
            struct v4l2_capability cap;
            if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
                close(fd);
                continue;
            }
            
            // Check if it's a capture device
            if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
                close(fd);
                continue;
            }
            
            // Get device info
            WebcamDevice device = getV4L2DeviceInfo(path);
            device.id = id++;
            devices.push_back(device);
            
            close(fd);
        }
    }
    
    closedir(dir);
    
    // If no devices were found, add a dummy device for testing
    if (devices.empty()) {
        WebcamDevice device;
        device.id = 0;
        device.name = "Default Camera";
        device.path = "/dev/video0";
        device.width = 640;
        device.height = 480;
        device.supportedFps = {30.0};
        devices.push_back(device);
    }
}

WebcamManager::WebcamDevice WebcamManager::getV4L2DeviceInfo(const std::string& path) {
    WebcamDevice device;
    device.path = path;
    
    int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) {
        device.name = "Unknown";
        device.width = 640;
        device.height = 480;
        return device;
    }
    
    // Get device name
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) >= 0) {
        device.name = reinterpret_cast<const char*>(cap.card);
    } else {
        device.name = "Unknown";
    }
    
    // Get default resolution
    struct v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) >= 0) {
        device.width = fmt.fmt.pix.width;
        device.height = fmt.fmt.pix.height;
    } else {
        device.width = 640;
        device.height = 480;
    }
    
    // Get supported frame rates
    device.supportedFps.push_back(30.0); // Default
    
    struct v4l2_frmivalenum frmival;
    struct v4l2_fmtdesc fmtdesc;
    fmtdesc.index = 0;
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) >= 0) {
        frmival.index = 0;
        frmival.pixel_format = fmtdesc.pixelformat;
        frmival.width = device.width;
        frmival.height = device.height;
        
        while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmival) >= 0) {
            if (frmival.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
                double fps = static_cast<double>(frmival.discrete.denominator) / 
                             static_cast<double>(frmival.discrete.numerator);
                
                // Add FPS if not already in the list
                if (std::find(device.supportedFps.begin(), device.supportedFps.end(), fps) == device.supportedFps.end()) {
                    device.supportedFps.push_back(fps);
                }
            }
            
            frmival.index++;
        }
        
        fmtdesc.index++;
    }
    
    close(fd);
    return device;
} 