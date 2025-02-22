cmake_minimum_required(VERSION 3.10)
project(image-scan)

add_executable(image-scan main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(image-scan ${OpenCV_LIBS})

find_package(OpenCL REQUIRED)
target_link_libraries(image-scan OpenCL::OpenCL)
add_definitions(-DCL_TARGET_OPENCL_VERSION=300)