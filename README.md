# DeepFake Webcam

A C++ application that applies deepfake transformations to webcam feeds in real-time and outputs to a virtual webcam.

![image](https://github.com/user-attachments/assets/890c5ed8-a2fb-496a-bf35-544f704680db)

![image](https://github.com/user-attachments/assets/a586c71b-a98f-4500-9bba-d325dd33739f)

![image](https://github.com/user-attachments/assets/f9a67c18-81c3-43d8-a7d7-aa57874a21af)



## Features

- Advanced face detection using Deep Neural Networks (DNN) with fallback to Haar cascades
- Multiple target face selection from image files
- Adjustable blend amount, face size, and smoothness settings
- Output to virtual webcam for use in video conferencing apps
- Qt-based GUI with original and transformed video feeds

## Requirements

- Linux operating system
- OpenCV with contrib modules (for face module)
- Qt6
- v4l2loopback (for virtual webcam output)

## Installation

### 1. Install dependencies

```bash
# Install OpenCV with contrib modules
sudo apt-get update
sudo apt-get install -y cmake build-essential
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y libopencv-contrib-dev

# Install Qt6
sudo apt-get install -y qt6-base-dev qt6-base-dev-tools

# Install v4l2loopback
sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils
```

### 2. Set up v4l2loopback

Create a virtual webcam device:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFake" exclusive_caps=1
```

### 3. Download required models

Run the provided script to download the face detection models:

```bash
# Make the script executable if needed
chmod +x download_models.sh

# Run the script
./download_models.sh
```

This script will download:
- Haar cascade face detector (fallback)
- Caffe DNN-based face detector model (primary)
- TensorFlow DNN-based face detector model (alternative)

### 4. Build the application

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Run the application:

```bash
./deepfakecam
```

## How to Use the Deepfake Features

1. **Start the Application**: Launch DeepFake Webcam and select your webcam device from the dropdown.

2. **Add Target Faces**: Click the "Add Face" button and select a clear face image you want to use for the transformation.

3. **Adjust Settings**:
   - **Blend Amount**: Controls how much of the target face is blended with your face
   - **Face Size**: Adjusts the size of the target face relative to detected faces
   - **Smoothness**: Controls the smoothness of the blending between your face and the target face

4. **Start Capturing**: Click "Start Capture" to begin the transformation process.

5. **Virtual Webcam Output**: The transformed video feed is automatically sent to the virtual webcam (if configured) which can be used in video conferencing applications.

## Face Detection Technology

The application uses a multi-stage face detection approach for robust detection in various conditions:

1. **Primary: Deep Neural Network (DNN) Detection** - Uses a pre-trained SSD (Single Shot MultiBox Detector) model based on ResNet-10 architecture, offering:
   - Better accuracy in various lighting conditions
   - Better detection at various face angles
   - More robust to occlusion (partially covered faces)
   - Better detection at different distances

2. **Fallback: Haar Cascade Detection** - If the DNN detector fails or doesn't find any faces, the app automatically falls back to the traditional Haar cascade detector.

## Advanced Settings

### Adjusting the Virtual Webcam

You can change the virtual webcam settings by modifying the v4l2loopback parameters:

```bash
# To remove the existing virtual webcam
sudo modprobe -r v4l2loopback

# To create a new one with different settings
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFake" exclusive_caps=1
```

### Using Your Own Models

For advanced users who want to implement a custom deepfake model:

1. Implement your own model in `DeepFakeModelImpl` class
2. Place your model files in the `models` directory
3. Update the model path in `MainWindow::MainWindow()`

## Troubleshooting

### Face Detection Issues

If face detection is not working well:

1. **Ensure proper lighting** - The DNN face detector works best with good, even lighting
2. **Keep your face clearly visible** - Try to face the camera directly
3. **Check model files** - Make sure all model files were downloaded properly
4. **Try different confidence threshold** - For advanced users, you can modify the `confidenceThreshold` parameter in the code

### Virtual Webcam Issues

If the virtual webcam doesn't work:
- Ensure v4l2loopback is properly installed and loaded
- Check if the virtual webcam device was created (`ls /dev/video*`)
- Run with elevated permissions if needed

