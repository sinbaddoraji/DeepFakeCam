cmake_minimum_required(VERSION 3.10)
project(DeepFakeCam VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect face)
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
find_package(dlib REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${dlib_INCLUDE_DIRS})

# Add Qt ui files
set(UI_FILES
    ui/mainwindow.ui
)

# Add source files
set(SOURCE_FILES
    src/main.cpp
    src/mainwindow.cpp
    src/deepfake_model.cpp
    src/webcammanager.cpp
    src/targetfacemanager.cpp
    src/virtualwebcam.cpp
    src/facepreviewdialog.cpp
)

# Add header files
set(HEADER_FILES
    include/deepfake_model.h
    include/mainwindow.h
    include/webcammanager.h
    include/targetfacemanager.h
    include/virtualwebcam.h
)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/ui)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)

# Add executable
add_executable(deepfakecam 
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${UI_FILES}
)

# Add test executable
add_executable(test_deepfake
    src/test_deepfake.cpp
    src/deepfake_model.cpp
)

# Link libraries
target_link_libraries(deepfakecam 
    ${OpenCV_LIBS}
    v4l2
    v4lconvert
    Qt6::Core
    Qt6::Widgets
    ${dlib_LIBRARIES}
)

# Link test executable
target_link_libraries(test_deepfake
    ${OpenCV_LIBS}
    ${dlib_LIBRARIES}
)

# Install targets
install(TARGETS deepfakecam DESTINATION bin)

# Copy required model files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

# Message to guide users about setting up face models
message(STATUS "DeepFakeCam requires facial models to function properly.")
message(STATUS "Please copy the haarcascade_frontalface_alt2.xml and lbfmodel.yaml files to the models directory.")
message(STATUS "These files can be found in OpenCV's data directory.") 