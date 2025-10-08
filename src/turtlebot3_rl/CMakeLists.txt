cmake_minimum_required(VERSION 3.5)
project(turtlebot3_rl)

find_package(ament_cmake REQUIRED)

# Optional: if you have any source files later
# find_package(rclcpp REQUIRED)
# add_executable(some_node src/some_node.cpp)
# ament_target_dependencies(some_node rclcpp)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py"
)

# Install world files
install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
