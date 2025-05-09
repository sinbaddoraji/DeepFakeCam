#!/bin/bash
set -e

# Install dependencies
sudo apt-get update
sudo apt-get install -y libboost-all-dev cmake libx11-dev libopenblas-dev wget

# Build and install dlib
cd /tmp
if [ ! -d dlib ]; then
    git clone https://github.com/davisking/dlib.git
fi
cd dlib
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig

# Download the 68-point shape predictor model
cd "$HOME/Desktop/Projects/DeepFakeCam/models"
if [ ! -f shape_predictor_68_face_landmarks.dat ]; then
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi

echo "dlib and the 68-point shape predictor model are installed." 