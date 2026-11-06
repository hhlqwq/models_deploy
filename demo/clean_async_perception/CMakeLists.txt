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

set(ASYNC_PERCEPTION_PLATFORM "linux-amd64" CACHE STRING "Target platform")
set_property(CACHE ASYNC_PERCEPTION_PLATFORM PROPERTY STRINGS linux-amd64 mtk-g720)
if(ASYNC_PERCEPTION_PLATFORM STREQUAL "mtk-g720")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        message(FATAL_ERROR "mtk-g720 requires an AArch64 Linux compiler/toolchain.")
    endif()
    set(ASYNC_PERCEPTION_PLATFORM_DIR "mtk-g720")
    set(ORBBEC_ARCH "linux-arm64")
elseif(ASYNC_PERCEPTION_PLATFORM STREQUAL "linux-amd64")
    set(ASYNC_PERCEPTION_PLATFORM_DIR "amd64")
    set(ORBBEC_ARCH "linux-x86_64")
else()
    message(FATAL_ERROR "Unknown ASYNC_PERCEPTION_PLATFORM: ${ASYNC_PERCEPTION_PLATFORM}")
endif()

set(ASYNC_PERCEPTION_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/output/linux/${ASYNC_PERCEPTION_PLATFORM_DIR}/async_perception")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ASYNC_PERCEPTION_OUTPUT_DIR}")

# Orbbec SDK
set(ORBBEC_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/dep/${ORBBEC_ARCH}/orbbec_sdk" CACHE PATH "Orbbec SDK root")
set(OrbbecSDK_DIR "${ORBBEC_SDK_ROOT}/lib")
if(ASYNC_PERCEPTION_PLATFORM STREQUAL "mtk-g720")
    find_package(OrbbecSDK REQUIRED CONFIG PATHS "${OrbbecSDK_DIR}"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
else()
    find_package(OrbbecSDK REQUIRED CONFIG)
endif()

if(ASYNC_PERCEPTION_PLATFORM STREQUAL "linux-amd64")
# Intel OpenVINO SDK.
set(ASYNC_INFER_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/dep/linux-x86_64/async_infer_sdk" CACHE PATH "AsyncInfer SDK root")
set(ASYNC_INFER_INCLUDE_DIR "${ASYNC_INFER_SDK_ROOT}/sdk/include")
set(ASYNC_INFER_LIBRARY "${ASYNC_INFER_SDK_ROOT}/sdk/lib/libasync_infer.so")

if(NOT EXISTS "${ASYNC_INFER_INCLUDE_DIR}/async_infer.h"
   OR NOT EXISTS "${ASYNC_INFER_LIBRARY}")
    message(FATAL_ERROR "Missing AsyncInfer SDK. Run restore_sdk.sh linux-x86_64 first.")
endif()
endif()

set(ASYNC_PERCEPTION_ENABLE_ROS2 ON)
if(ASYNC_PERCEPTION_PLATFORM STREQUAL "mtk-g720")
    option(G720_ENABLE_ROS2 "Publish G720 perception results over ROS 2" ON)
    set(ASYNC_PERCEPTION_ENABLE_ROS2 "${G720_ENABLE_ROS2}")
endif()
if(ASYNC_PERCEPTION_ENABLE_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    if(ASYNC_PERCEPTION_PLATFORM STREQUAL "linux-amd64")
        find_package(sensor_msgs REQUIRED)
    endif()
endif()

if(ASYNC_PERCEPTION_PLATFORM STREQUAL "mtk-g720" AND G720_BOARD_OPENCV)
    # 目标板 OpenCV 4.9 动态库及对应的开发头文件.
    set(OpenCV_INCLUDE_DIRS "${G720_SDK_DIR}/opencv/include")
    set(OpenCV_LIBS "")
    foreach(component IN ITEMS imgcodecs imgproc core)
        set(library "${G720_SDK_DIR}/board-libs/libopencv_${component}.so.409")
        foreach(resource IN ITEMS "${OpenCV_INCLUDE_DIRS}/opencv2/${component}.hpp" "${library}")
            if(NOT EXISTS "${resource}")
                message(FATAL_ERROR "Missing G720 OpenCV resource: ${resource}")
            endif()
        endforeach()
        list(APPEND OpenCV_LIBS "${library}")
    endforeach()
    if(NOT EXISTS "${OpenCV_INCLUDE_DIRS}/opencv2/opencv_modules.hpp")
        message(FATAL_ERROR "Missing G720 OpenCV resource: ${OpenCV_INCLUDE_DIRS}/opencv2/opencv_modules.hpp")
    endif()
else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
endif()

add_executable(async_perception
    src/main.cpp
    src/platform/linux/linux_application.cpp
    src/platform/linux/web_preview_server.cpp
    src/platform/linux/web_preview_server.hpp
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
    ASYNC_PERCEPTION_BUILD_TARGET="${ASYNC_PERCEPTION_PLATFORM}")

target_include_directories(async_perception PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")

if(ASYNC_PERCEPTION_ENABLE_ROS2)
    target_sources(async_perception PRIVATE src/platform/linux/ros2/ros2_json_publisher.cpp)
    if(ASYNC_PERCEPTION_PLATFORM STREQUAL "linux-amd64")
        target_sources(async_perception PRIVATE src/platform/linux/ros2/ros2_image_subscriber.cpp)
        ament_target_dependencies(async_perception rclcpp std_msgs sensor_msgs)
    else()
        ament_target_dependencies(async_perception rclcpp std_msgs)
    endif()
else()
    target_compile_definitions(async_perception PRIVATE ASYNC_PERCEPTION_NO_ROS2)
endif()

target_include_directories(async_perception PRIVATE
    "${OpenCV_INCLUDE_DIRS}")

if(ASYNC_PERCEPTION_PLATFORM STREQUAL "mtk-g720")
    # G720 仅链接目标板对应的 Neuron Runtime, 运行时使用板端系统库.
    set(G720_NEURON_INCLUDE_DIR "" CACHE PATH "Neuron Runtime API include directory")
    set(G720_NEURON_LIBRARY "" CACHE FILEPATH "AArch64 Neuron Runtime link library")
    set(G720_MODEL_DIR "" CACHE PATH "G720 demo model package")

    foreach(resource IN ITEMS
        "${G720_NEURON_INCLUDE_DIR}/neuron/api/RuntimeAPI.h"
        "${G720_NEURON_LIBRARY}"
        "${G720_MODEL_DIR}/yolox.dla"
        "${G720_MODEL_DIR}/mobilenetv2.dla"
        "${G720_MODEL_DIR}/tensors.json")
        if(NOT EXISTS "${resource}" OR IS_DIRECTORY "${resource}")
            message(FATAL_ERROR "Missing G720 resource: ${resource}")
        endif()
    endforeach()

    target_compile_definitions(async_perception PRIVATE ASYNC_PERCEPTION_MTK_G720)
    target_include_directories(async_perception SYSTEM PRIVATE "${G720_NEURON_INCLUDE_DIR}")
    set(ASYNC_PERCEPTION_INFERENCE_LIBRARY "${G720_NEURON_LIBRARY}")
    if(G720_BOARD_OPENCV)
        target_link_options(async_perception PRIVATE "LINKER:-rpath-link,${G720_SDK_DIR}/board-libs")
    endif()

    # 从共用配置派生 G720 默认配置, 保留 ROS 2 和预览设置.
    set(base_ini "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/linux/config/async_perception.ini")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${base_ini}")
    file(READ "${base_ini}" g720_ini)
    string(REPLACE "mode=ros" "mode=live" g720_ini "${g720_ini}")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/async_perception.ini" CONTENT "${g720_ini}")

    file(GLOB orbbec_runtime "${ORBBEC_SDK_ROOT}/lib/*.so*")
    if(NOT orbbec_runtime)
        message(FATAL_ERROR "Missing ARM64 Orbbec runtime libraries: ${ORBBEC_SDK_ROOT}/lib")
    endif()

    set(package_dir "$<TARGET_FILE_DIR:async_perception>")
    add_custom_target(g720_package ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${package_dir}/config" "${package_dir}/models"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${orbbec_runtime} "${package_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ORBBEC_SDK_ROOT}/lib/extensions" "${package_dir}/extensions"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORBBEC_SDK_ROOT}/lib/OrbbecSDKConfig.xml" "${package_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/async_perception.ini" "${package_dir}/config"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${G720_MODEL_DIR}/yolox.dla"
            "${G720_MODEL_DIR}/mobilenetv2.dla"
            "${G720_MODEL_DIR}/tensors.json"
            "${package_dir}/models"
        DEPENDS async_perception
        VERBATIM)
else()
    target_include_directories(async_perception PRIVATE "${ASYNC_INFER_INCLUDE_DIR}")
    set(ASYNC_PERCEPTION_INFERENCE_LIBRARY "${ASYNC_INFER_LIBRARY}")
endif()
target_link_libraries(async_perception
    ob::OrbbecSDK
    "${ASYNC_PERCEPTION_INFERENCE_LIBRARY}"
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

if(ASYNC_PERCEPTION_PLATFORM STREQUAL "linux-amd64")
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
endif()
