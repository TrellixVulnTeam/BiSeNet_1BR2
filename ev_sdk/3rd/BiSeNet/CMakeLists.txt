CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(segment)

set(CMAKE_CXX_FLAGS "-std=c++14 -O1")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/lib/python3.6/dist-packages/tensorrt/include)
link_directories(/usr/local/lib/python3.6/dist-packages/tensorrt)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(segment ${PROJECT_SOURCE_DIR}/src/segment.cpp ${PROJECT_SOURCE_DIR}/src/trt_dep.cpp)

target_include_directories(
    segment PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(segment
    /usr/local/lib/python3.6/dist-packages/tensorrt/libnvinfer.so.8
    /usr/local/lib/python3.6/dist-packages/tensorrt/libnvinfer_plugin.so.8
    /usr/local/lib/python3.6/dist-packages/tensorrt/libnvonnxparser.so.8
    /usr/local/lib/python3.6/dist-packages/tensorrt/libnvparsers.so.8
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )
