#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Download haarcascade_frontalface_alt2.xml (Haar cascade - fallback)
echo "Downloading Haar cascade face detection model..."
wget -O models/haarcascade_frontalface_alt2.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

# Download Caffe model for DNN face detection (more accurate)
echo "Downloading Caffe DNN face detection model..."
wget -O models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget -O models/res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# Download TensorFlow model for DNN face detection (alternative)
echo "Downloading TensorFlow DNN face detection model..."
wget -O models/opencv_face_detector.pbtxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
wget -O models/opencv_face_detector_uint8.pb https://github.com/opencv/opencv_3rdparty/raw/contrib_face_detection_20170818/opencv_face_detector_uint8.pb

# Download lbfmodel.yaml
echo "Downloading facial landmark model..."
# Since lbfmodel.yaml is large and might not be directly accessible,
# we'll provide information on how to get it
echo "LBF facial landmark model needs to be manually obtained."
echo "You can find it in the opencv_contrib repository under /modules/face/data/lbfmodel.yaml"
echo "or install opencv-contrib-python and copy it from the installation directory."

echo "Model download complete. Please check if the files are properly downloaded in the models directory."

# Make the script executable
chmod +x download_models.sh 