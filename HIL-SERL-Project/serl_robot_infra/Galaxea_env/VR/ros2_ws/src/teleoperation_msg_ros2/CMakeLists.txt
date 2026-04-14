cmake_minimum_required(VERSION 3.8)
project(teleoperation_msg_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 寻找依赖
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# 生成消息代码
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VrPose.msg"
  "srv/SwitchControlModeVR.srv"
  DEPENDENCIES std_msgs
)

ament_package()
