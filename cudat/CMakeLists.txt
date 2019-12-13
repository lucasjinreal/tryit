cmake_minimum_required(VERSION 2.8)
project(nms CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
add_definitions("-std=c++14")

find_package(CUDA QUIET REQUIRED)
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__" )

find_package(OpenCV REQUIRED)
 
# find libtorch
set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)
link_directories($ENV{HOME}/libtorch/lib)
find_package(Torch REQUIRED)
include_directories(include/)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "you should change CMAKE_PREFIX_PATH to libtorch dir, have you copied libtorch to home?")
endif()


cuda_add_executable(
    main
        main.cu)

cuda_add_executable(
   nms
       nms_test.cu
)
target_link_libraries(
   nms
   ${TORCH_LIBRARIES}
   ${OpenCV_LIBS}
)
