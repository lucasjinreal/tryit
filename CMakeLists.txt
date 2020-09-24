cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)

add_definitions("-std=c++14")

# find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv needed but not found.")
    else()
    message(STATUS "found opencv: "${OpenCV_CONFIG_PATH}"   "${OpenCV_LIBS})
endif()

include_directories(include)


link_directories(/usr/local/lib)

# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

find_package(PCL 1.8 REQUIRED)
if (PCL_FOUND)
  message(STATUS "Find PCL: " ${PCL_LIBRARIES})
  message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
  message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
  message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})
else()
  message(FATAL_ERROR "can not found PCL 1.8+ version.")
endif()

# add_executable (pcd_read pcd_read.cpp)
# target_link_libraries (pcd_read ${PCL_LIBRARIES}  ${OpenCV_LIBS})

# add_executable(r r.cpp)
# target_link_libraries(r ${PCL_LIBRARIES} ${OpenCV_LIBS} -lthor)

add_executable(a a.cpp)
target_link_libraries(a ${OpenCV_LIBS} -lglog -lthor)


add_executable(yaml y.cpp)
target_link_libraries(yaml yaml-cpp -lglog)