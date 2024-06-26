cmake_minimum_required(VERSION 3.0.0)
project(yolov5_deepsort VERSION 0.1.0)

set(TOOLCHAIN_DIR /home/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf) 
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/arm-linux-gnueabihf-gcc)

add_subdirectory(deepsort)

set(OpenCV_DIR /home/opencv/opencv-4.5.3/release)  # 填入OpenCVConfig.cmake
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(
    include_libs  
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/yolov5/include"
    "${PROJECT_SOURCE_DIR}/deepsort/include"
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/include
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/include
)

include_directories(${include_libs})

aux_source_directory(${PROJECT_SOURCE_DIR}/yolov5/src YOLO_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src YOLO_SRC_DIR)
add_executable(yolov5_deepsort yolov5_deepsort.cpp ${YOLO_SRC_DIR})

# 添加动态链接库
set(
    dynamic_libs  pthread
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/armhf/librknn_api.so
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/lib/librga.so
)

# 开启调试选项
add_definitions("-g")

target_link_libraries(yolov5_deepsort ${OpenCV_LIBS})
target_link_libraries(yolov5_deepsort ${dynamic_libs} deepsort)