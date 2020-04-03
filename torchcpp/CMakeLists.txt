cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(light_pose_libtorch)

set(CMAKE_CXX_STANDARD 14)
add_definitions("-std=c++14")

# should change this
set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)
link_directories($ENV{HOME}/libtorch/lib)

find_package(Torch REQUIRED)
include_directories(include/)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "you should change CMAKE_PREFIX_PATH to libtorch dir, have you copied libtorch to home?")
endif()

find_package(OpenCV REQUIRED)

add_executable(hand hand.cc)
target_link_libraries(hand
        ${TORCH_LIBRARIES}
        glog
        ${OpenCV_LIBS}
        thor
        )
set_property(TARGET hand PROPERTY CXX_STANDARD 11)
