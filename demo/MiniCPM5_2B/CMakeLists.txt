cmake_minimum_required(VERSION 3.19)

# 工具链从唯一 config.json 读取编译设置,在 project 前选择编译器.
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/genio_720.cmake")
include("${CMAKE_TOOLCHAIN_FILE}")
list(PREPEND CMAKE_PREFIX_PATH "${DEMO_DEPENDENCIES}")
project(minicpm5_chat LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "当前平台后端需要 Linux,请使用目标板工具链配置")
endif()

# 仅查找已有依赖,不在配置阶段下载源码或安装软件.
find_package(nlohmann_json 3.10 QUIET CONFIG)
if(NOT TARGET nlohmann_json::nlohmann_json)
    # 单头文件依赖也可使用,不要求先向系统安装 CMake 包配置.
    find_path(JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
    file(READ "${JSON_INCLUDE_DIR}/nlohmann/json.hpp" json_header)
    foreach(component MAJOR MINOR PATCH)
        string(REGEX MATCH "NLOHMANN_JSON_VERSION_${component} +([0-9]+)" version_match "${json_header}")
        if(NOT version_match)
            message(FATAL_ERROR "无法识别 nlohmann/json 版本")
        endif()
        set(json_${component} "${CMAKE_MATCH_1}")
    endforeach()
    if("${json_MAJOR}.${json_MINOR}.${json_PATCH}" VERSION_LESS "3.10.0")
        message(FATAL_ERROR "nlohmann/json 版本需要至少 3.10")
    endif()
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JSON_INCLUDE_DIR}")
endif()
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_path(TOKENIZERS_INCLUDE_DIR tokenizers_cpp.h REQUIRED)
find_library(TOKENIZERS_LIBRARY NAMES tokenizers_cpp REQUIRED)
find_library(TOKENIZERS_C_LIBRARY NAMES tokenizers_c REQUIRED)
find_path(MINJA_INCLUDE_DIR minja/minja.hpp REQUIRED)
find_path(CNPY_INCLUDE_DIR cnpy.h REQUIRED)
find_library(CNPY_LIBRARY NAMES cnpy REQUIRED)
find_path(NEURON_INCLUDE_DIR neuron/api/RuntimeAPI.h REQUIRED)

add_library(chat_core STATIC core/chat.cpp)
target_compile_features(chat_core PUBLIC cxx_std_17)
target_include_directories(chat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

add_library(minicpm_mtk STATIC platforms/mtk/engine.cpp)
target_compile_features(minicpm_mtk PRIVATE cxx_std_17)
target_include_directories(minicpm_mtk PRIVATE
    ${TOKENIZERS_INCLUDE_DIR} ${MINJA_INCLUDE_DIR} ${CNPY_INCLUDE_DIR}
    ${NEURON_INCLUDE_DIR})
target_link_libraries(minicpm_mtk PRIVATE chat_core nlohmann_json::nlohmann_json
    ${TOKENIZERS_LIBRARY} ${TOKENIZERS_C_LIBRARY}
    ${CNPY_LIBRARY} ZLIB::ZLIB Threads::Threads ${CMAKE_DL_LIBS} m)

add_executable(minicpm5_chat main.cpp)
target_link_libraries(minicpm5_chat PRIVATE chat_core minicpm_mtk nlohmann_json::nlohmann_json)
install(TARGETS minicpm5_chat RUNTIME DESTINATION bin)
# 同一份本地配置进入当前构建和安装目录,不提交真实资源路径.
configure_file(config.json config.json COPYONLY)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/config.json" DESTINATION bin)
