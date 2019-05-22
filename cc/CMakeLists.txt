cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pc_reader)

add_definitions('-std=c++11')

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

add_executable (pc_reader pc_reader.cc)
target_link_libraries (pc_reader ${PCL_LIBRARIES} -lthor)


add_executable (xml_reader xml_read.cc)
target_link_libraries (xml_reader -lthor -ltinyxml)

add_executable (eigent eigent.cc)
target_link_libraries (eigent -lthor)

add_executable (mapt mapt.cc)
target_link_libraries (mapt -lthor)