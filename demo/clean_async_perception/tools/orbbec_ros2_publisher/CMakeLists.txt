cmake_minimum_required(VERSION 3.16)
project(orbbec_ros2_publisher LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

set(ORBBEC_SDK_ROOT "/opt/orbbec_sdk" CACHE PATH "Orbbec SDK root")
set(OrbbecSDK_DIR "${ORBBEC_SDK_ROOT}/lib")
find_package(OrbbecSDK REQUIRED CONFIG)

add_executable(orbbec_ros2_publisher main.cpp)
target_compile_features(orbbec_ros2_publisher PRIVATE cxx_std_17)
ament_target_dependencies(orbbec_ros2_publisher rclcpp sensor_msgs)
target_link_libraries(orbbec_ros2_publisher ob::OrbbecSDK)
set_target_properties(orbbec_ros2_publisher PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN/../lib")
