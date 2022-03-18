cmake_minimum_required(VERSION 3.5)
project(ji)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
find_package(PkgConfig REQUIRED)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/lib/python3.6/dist-packages/tensorrt/include)
link_directories(/usr/local/lib/python3.6/dist-packages/tensorrt)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/cJSON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/BiSeNet)

# 编译libji.so
list(APPEND libji_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ji.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/BiSeNetTRT.cpp)
add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC "/usr/local/lib/python3.6/dist-packages/tensorrt/include")
target_link_libraries(${PROJECT_NAME}
        "/usr/local/lib/python3.6/dist-packages/tensorrt/libnvinfer.so.8"
        ${OpenCV_LIBS}
        glog
        cjson
        cudart
        )

# 以下是EV_SDK所依赖和需要测试接口的程序&库，请勿修改！！！
# --- BEGIN EV_SDK DEPENDENCIES ---
# 编译 JI API 测试程序
add_executable(test-ji-api ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)
target_link_libraries(test-ji-api
        ${OpenCV_LIBS}
        cjson
        glog
        ${PROJECT_NAME}
        )
# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
# Install test-ji-api
install(TARGETS test-ji-api DESTINATION bin)
# ---  END EV_SDK DEPENDENCIES  ---