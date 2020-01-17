CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(PROJECT_NAME kingcent_stereo_demo)
PROJECT(${PROJECT_NAME})

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


#AUX_SOURCE_DIRECTORY(src DIR_SRCS)
#MESSAGE(STATUS "Src file: ${DIR_SRCS}")


ADD_EXECUTABLE(binocular_calibration ./src/binocular_calibration.cpp)
TARGET_LINK_LIBRARIES(binocular_calibration ${OpenCV_LIBS})

ADD_EXECUTABLE(binocular_demo ./src/binocular_demo.cpp)
TARGET_LINK_LIBRARIES(binocular_demo ${OpenCV_LIBS})
