cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv needed but not found.")
    else()
    message(STATUS "found opencv: "${OpenCV_CONFIG_PATH}"   "${OpenCV_LIBS})
endif()



include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_read pcd_read.cpp)
target_link_libraries (pcd_read ${PCL_LIBRARIES})

add_executable(r r.cpp)
target_link_libraries(r ${PCL_LIBRARIES} ${OpenCV_LIBS} -lthor)

add_executable(a a.cpp)
target_link_libraries(a ${OpenCV_LIBS} -lglog)
