cmake_minimum_required(VERSION 3.24)

project(rubik_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
endif()

set(BUILD_SHARED_LIBS ON)

find_package(JNI REQUIRED COMPONENTS JVM)

set(OPENCV_VERSION "2027-4.13.0-3")
# also valid: windowsx86-64

# type can be "", "debug", "static", or "staticdebug"
set(OPENCV_TYPE "")

# Download opencv, and save the path
include(FetchContent)
FetchContent_Declare(
    opencv_lib
    URL
        https://frcmaven.wpi.edu/artifactory/release/org/wpilib/thirdparty/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(opencv_lib)

# download OpenCV headers
FetchContent_Declare(
    opencv_header
    URL
        https://frcmaven.wpi.edu/artifactory/release/org/wpilib/thirdparty/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(opencv_header)

# This probably doesn't work great with shared libraries, but I don't care about those right now
file(
    GLOB_RECURSE OPENCV_LIB_PATH
    "${opencv_lib_SOURCE_DIR}/**/*.lib"
    "${opencv_lib_SOURCE_DIR}/**/*.so*"
)
set(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on opencv ${OPENCV_LIB_PATH}")

# Download TensorFlow Lite headers
include(FetchContent)
FetchContent_Declare(
    tflite_header
    URL https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.19.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(tflite_header)

# Set TensorFlow Lite headers
set(TFLITE_INCLUDE_DIR
    "${tflite_header_SOURCE_DIR}"
    CACHE PATH
    "Path to TensorFlow Lite include directory"
)

# Set TensorFlow Lite library - use absolute path and verify it exists
set(LIB_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/lib/"
    CACHE FILEPATH
    "Path to TensorFlow Lite library"
)

# Check if the library path exists
if(NOT EXISTS ${LIB_PATH})
    message(FATAL_ERROR "Library directory not found at: ${LIB_PATH}")
else()
    message(STATUS "Found library directory: ${LIB_PATH}")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

file(GLOB NATIVE_SOURCES src/main/native/cpp/*.cpp)

add_library(${PROJECT_NAME} ${NATIVE_SOURCES})

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        Threads::Threads
        ${OPENCV_LIB_PATH}
        ${LIB_PATH}/libtensorflowlite_c.so
        ${LIB_PATH}/libtensorflowlite.so
        ${LIB_PATH}/libexternal_delegate.so
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
        ${JNI_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_PATH}
        ${TFLITE_INCLUDE_DIR}
)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
install(
    FILES
        ${LIB_PATH}/libtensorflowlite_c.so
        ${LIB_PATH}/libtensorflowlite.so
        ${LIB_PATH}/libexternal_delegate.so
    TYPE LIB
)
