cmake_minimum_required(VERSION 3.24)

project(tflite_jni LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_RPATH "$ORIGIN;$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

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

if(DEFINED ENV{JAVA_HOME})
    if(EXISTS "$ENV{JAVA_HOME}/include/jni.h")
        list(APPEND JNI_SEARCH_PATHS "$ENV{JAVA_HOME}/include")
    elseif(EXISTS "$ENV{JAVA_HOME}/lib/openjdk/include/jni.h")
        list(APPEND JNI_SEARCH_PATHS "$ENV{JAVA_HOME}/lib/openjdk/include")
    endif()
endif()

find_path(JNI_INCLUDE_DIR NAMES jni.h PATHS ${JNI_SEARCH_PATHS})

find_path(
    JNI_MD_INCLUDE_DIR
    NAMES jni_md.h
    PATHS ${JNI_SEARCH_PATHS}
    PATH_SUFFIXES linux darwin win32
)

if(NOT JNI_INCLUDE_DIR OR NOT JNI_MD_INCLUDE_DIR)
    message(
        FATAL_ERROR
        "Could not locate JNI headers. Set JAVA_HOME to a JDK installation."
    )
endif()

find_package(Threads REQUIRED)

set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
set(OPENCV_TYPE "")

include(FetchContent)
FetchContent_Declare(
    opencv_lib
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(opencv_lib)

FetchContent_Declare(
    opencv_header
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(opencv_header)

file(
    GLOB_RECURSE OPENCV_LIB_PATH
    "${opencv_lib_SOURCE_DIR}/**/*.lib"
    "${opencv_lib_SOURCE_DIR}/**/*.so*"
)

set(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on opencv ${OPENCV_LIB_PATH}")

include(GNUInstallDirs)

set(TFLITE_VERSION "v2.19.0")

# TensorFlow Lite's CMake build expects these options to be set before the
# subdirectory is added.
set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_LABEL_IMAGE OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_NNAPI OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_MMAP ON CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    tensorflowlite_src
    URL
        https://github.com/tensorflow/tensorflow/archive/refs/tags/${TFLITE_VERSION}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(tensorflowlite_src)

add_subdirectory(
    "${tensorflowlite_src_SOURCE_DIR}/tensorflow/lite/c"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite-c"
    EXCLUDE_FROM_ALL
)

# TensorFlow Lite's CMakeLists currently doesn't link Abseil's newer logging
# implementation into `tensorflow-lite` when building shared libraries, which
# can leave `absl::log_internal::*` symbols undefined at runtime.
#
# When the C API library depends on `tensorflow-lite`, the dynamic loader may
# fail with e.g.:
#   undefined symbol: absl::lts_20230802::log_internal::kCharNull
#
# Ensure the required Abseil logging objects are part of the link.
if(BUILD_SHARED_LIBS AND TARGET tensorflow-lite)
    target_link_libraries(tensorflow-lite PUBLIC absl::log_internal_message)
endif()

file(GLOB NATIVE_SOURCES CONFIGURE_DEPENDS src/main/native/cpp/*.cpp)

add_library(${PROJECT_NAME} ${NATIVE_SOURCES})

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPENCV_INCLUDE_PATH}
        ${JNI_INCLUDE_DIR}
        ${JNI_MD_INCLUDE_DIR}
        ${tensorflowlite_src_SOURCE_DIR}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        Threads::Threads
        ${OPENCV_LIB_PATH}
        tensorflowlite_c
        ${CMAKE_DL_LIBS}
)

install(TARGETS ${PROJECT_NAME})
