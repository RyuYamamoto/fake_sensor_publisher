cmake_minimum_required(VERSION 2.8.3)
project(fake_sensor_publisher)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  tf2
  tf2_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  /usr/include/eigen3
)

add_executable(fake_sensor_publisher_node src/fake_sensor_publisher_node.cpp)
target_link_libraries(fake_sensor_publisher_node ${catkin_LIBRARIES})