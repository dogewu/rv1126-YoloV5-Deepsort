cmake_minimum_required(VERSION 3.0.0)

set(TOOLCHAIN_DIR /home/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf) 
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/arm-linux-gnueabihf-gcc)

set(OpenCV_DIR /home/opencv/opencv-4.5.3/release)   # 填入OpenCVConfig.cmake
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(
    include_libs  
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/deepsort/include"
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/include
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/include
)

set(
    dynamic_libs
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/armhf/librknn_api.so
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/lib/librga.so
)

include_directories(${include_libs})
aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src DEEPSORT_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DEEPSORT_SRC_DIR)

add_library(deepsort ${DEEPSORT_SRC_DIR})

target_link_libraries(deepsort ${OpenCV_LIBS})
target_link_libraries(deepsort ${dynamic_libs})