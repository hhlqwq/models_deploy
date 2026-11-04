cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_CURRENT_LIST_DIR}/version.txt" ASYNC_PERCEPTION_VERSION_RAW)
string(STRIP "${ASYNC_PERCEPTION_VERSION_RAW}" ASYNC_PERCEPTION_VERSION)
if(NOT ASYNC_PERCEPTION_VERSION MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)$")
    message(FATAL_ERROR "version.txt must contain a semantic version such as v1.0.0")
endif()
set(ASYNC_PERCEPTION_VERSION_NUMBER "${CMAKE_MATCH_1}")
project(async_perception VERSION "${ASYNC_PERCEPTION_VERSION_NUMBER}" LANGUAGES CXX)

set(ASYNC_PERCEPTION_BUILD_MODE "Release" CACHE STRING "Build mode: Normal, Test or Release")
set_property(CACHE ASYNC_PERCEPTION_BUILD_MODE PROPERTY STRINGS Normal Test Release)

set(ASYNC_PERCEPTION_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/output/linux/amd64/async_perception")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ASYNC_PERCEPTION_OUTPUT_DIR}")

# Orbbec SDK
set(ORBBEC_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/dep/linux-x86_64/orbbec_sdk" CACHE PATH "Orbbec SDK root")
set(OrbbecSDK_DIR "${ORBBEC_SDK_ROOT}/lib")
find_package(OrbbecSDK REQUIRED CONFIG)

# Async Infer SDK (Intel OpenVINO)
set(ASYNC_INFER_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/dep/linux-x86_64/async_infer_sdk" CACHE PATH "AsyncInfer SDK root")
set(ASYNC_INFER_INCLUDE_DIR "${ASYNC_INFER_SDK_ROOT}/sdk/include")
set(ASYNC_INFER_LIBRARY "${ASYNC_INFER_SDK_ROOT}/sdk/lib/libasync_infer.so")

if(NOT EXISTS "${ASYNC_INFER_INCLUDE_DIR}/async_infer.h"
   OR NOT EXISTS "${ASYNC_INFER_LIBRARY}")
    message(FATAL_ERROR "Missing AsyncInfer SDK. Run restore_sdk.sh linux-x86_64 first.")
endif()

# ROS2 Humble
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# OpenCV (system)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(async_perception
    src/main.cpp
    src/platform/linux/linux_application.cpp
    src/platform/linux/web_preview_server.cpp
    src/platform/linux/web_preview_server.hpp
    src/platform/linux/ros2/ros2_json_publisher.cpp
    src/platform/linux/ros2/ros2_json_publisher.hpp
    src/platform/linux/ros2/ros2_image_subscriber.cpp
    src/platform/linux/ros2/ros2_image_subscriber.hpp
    src/shared/core/perception_foundation.hpp
    src/shared/perception/async_inference_pipeline.hpp
    src/shared/reporting/perception_result_protocol.hpp
    src/shared/session/perception_session.hpp
    src/geometry/ipm.hpp
    src/geometry/ground_coverage.hpp
    src/assessment/dirt_score.hpp
)

set_target_properties(async_perception PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN")

target_compile_features(async_perception PRIVATE cxx_std_17)

target_compile_definitions(async_perception PRIVATE
    ASYNC_PERCEPTION_VERSION="${ASYNC_PERCEPTION_VERSION}"
    ASYNC_PERCEPTION_BUILD_MODE="${ASYNC_PERCEPTION_BUILD_MODE}"
    ASYNC_PERCEPTION_BUILD_TARGET="linux-amd64")

target_include_directories(async_perception PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${ASYNC_INFER_INCLUDE_DIR}")

ament_target_dependencies(async_perception rclcpp std_msgs sensor_msgs)

target_include_directories(async_perception PRIVATE
    "${OpenCV_INCLUDE_DIRS}")

target_link_libraries(async_perception
    ob::OrbbecSDK
    "${ASYNC_INFER_LIBRARY}"
    ${OpenCV_LIBS}
    pthread dl)

include(CTest)
if(BUILD_TESTING)
    add_executable(ground_coverage_test tests/ground_coverage_test.cpp)
    target_compile_features(ground_coverage_test PRIVATE cxx_std_17)
    target_include_directories(ground_coverage_test PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
    set_target_properties(ground_coverage_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
    add_test(NAME ground_coverage_test COMMAND ground_coverage_test)

endif()

set(MOBILENETV2_MODEL_DIR
    "${CMAKE_CURRENT_LIST_DIR}/models/mobilenetv2")
if(NOT EXISTS "${MOBILENETV2_MODEL_DIR}/mobilenetv2.xml"
   OR NOT EXISTS "${MOBILENETV2_MODEL_DIR}/mobilenetv2.bin")
    message(FATAL_ERROR "Missing MobileNetV2 XML/BIN in ${MOBILENETV2_MODEL_DIR}")
endif()

# Copy runtime libraries and config to output dir
file(GLOB ASYNC_INFER_RUNTIME "${ASYNC_INFER_SDK_ROOT}/sdk/lib/*.so*")
file(GLOB ORBBEC_RUNTIME "${ORBBEC_SDK_ROOT}/lib/*.so*")

add_custom_command(TARGET async_perception POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ASYNC_INFER_RUNTIME}
        "$<TARGET_FILE_DIR:async_perception>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORBBEC_RUNTIME}
        "$<TARGET_FILE_DIR:async_perception>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ORBBEC_SDK_ROOT}/lib/extensions"
        "$<TARGET_FILE_DIR:async_perception>/extensions"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORBBEC_SDK_ROOT}/lib/OrbbecSDKConfig.xml"
        "$<TARGET_FILE_DIR:async_perception>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ASYNC_INFER_SDK_ROOT}/example/files/models"
        "$<TARGET_FILE_DIR:async_perception>/models"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:async_perception>/models/mobilenetv2"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MOBILENETV2_MODEL_DIR}/mobilenetv2.xml"
        "${MOBILENETV2_MODEL_DIR}/mobilenetv2.bin"
        "$<TARGET_FILE_DIR:async_perception>/models/mobilenetv2"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:async_perception>/config"
    # 交付配置白名单：YOLOX 检测与 MobileNetV2 地面分类。
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/src/platform/linux/config/0_root_config.json"
        "$<TARGET_FILE_DIR:async_perception>/config/0_root_config.json"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/src/platform/linux/config/yolox_config.json"
        "$<TARGET_FILE_DIR:async_perception>/config/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/src/platform/linux/config/mobilenetv2_config.json"
        "$<TARGET_FILE_DIR:async_perception>/config/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/src/platform/linux/config/async_perception.ini"
        "$<TARGET_FILE_DIR:async_perception>/config/async_perception.ini"
)
