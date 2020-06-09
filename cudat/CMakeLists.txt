cmake_minimum_required(VERSION 2.8)
project(nms CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
add_definitions("-std=c++14")

file(GLOB COMMON common.h common.cpp)


find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__")

cuda_include_directories(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
include_directories(include
        ${CUDA_INCLUDE_DIRS}
        ${CUDNN_INCLUDE_DIR})

find_package(OpenCV 4 REQUIRED)

# find libtorch
#set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)
#link_directories($ENV{HOME}/libtorch/lib)
#find_package(Torch REQUIRED)
#include_directories(include/)
#if (NOT Torch_FOUND)
#    message(FATAL_ERROR "you should change CMAKE_PREFIX_PATH to libtorch dir, have you copied libtorch to home?")
#endif()


cuda_add_executable(
        main
        main.cu)

# cuda_add_executable(
#    nms
#        nms_test.cu
# )
# target_link_libraries(
#    nms
#    ${TORCH_LIBRARIES}
#    ${OpenCV_LIBS}
# )

cuda_add_executable(
        nmsss
        nms2.cu
)
target_link_libraries(
        nmsss
        ${OpenCV_LIBS}
)
