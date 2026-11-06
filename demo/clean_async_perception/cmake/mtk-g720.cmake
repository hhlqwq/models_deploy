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
