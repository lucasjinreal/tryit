cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pc_reader)

add_definitions('-std=c++11')

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "opencv needed but not found.")
else()
  message(STATUS "found
                 opencv:
                 "${OpenCV_CONFIG_PATH}"
                 "${OpenCV_LIBS})
endif()

find_package(Boost REQUIRED COMPONENTS system thread filesystem regex)
if (NOT Boost_FOUND)
  message(FATAL_ERROR "Boost not found, moca will not able to built without it.")
endif()
find_package(PythonLibs ${PY_VERSION} REQUIRED)
find_package(Boost REQUIRED COMPONENTS system python3)
if (NOT Boost_FOUND)
  message(FATAL_ERROR "Boost not found, moca will not able to built without it.")
endif()

# find_package(Torch REQUIRED)
# if(Torch_FOUND)
#   add_executable(torcht torcht.cc)
#   target_link_libraries(torcht "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
# else()
#   message(STATUS "libtorch not found, relate program will not built")
# endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pc_reader pc_reader.cc)
target_link_libraries(pc_reader ${PCL_LIBRARIES} -lthor)

add_executable(eigen eigen.cc)

add_executable(xml_reader xml_read.cc)
target_link_libraries(xml_reader -lthor -ltinyxml)

add_executable(eigent eigent.cc)
target_link_libraries(eigent -lthor)

add_executable (mapt mapt.cc)
target_link_libraries (mapt -lthor)

add_executable (boostt boostt.cc)
target_link_libraries (boostt -lthor ${Boost_LIBRARIES})

add_executable (sockett sockett.cc)
target_link_libraries (sockett -lthor ${OpenCV_LIBS} pthread glog ${Boost_LIBRARIES})

add_executable (clientt clientt.cc)
target_link_libraries (clientt -lthor ${OpenCV_LIBS} pthread glog)

add_executable (convt convt.cc)
target_link_libraries (convt -lthor ${OpenCV_LIBS} pthread glog)