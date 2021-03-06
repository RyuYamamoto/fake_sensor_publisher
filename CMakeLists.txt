cmake_minimum_required(VERSION 3.5)
project(fake_sensor_publisher)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic )
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(Eigen3 REQUIRED)
ament_auto_find_build_dependencies()

include_directories(${EIGEN3_INCLUDE_DIRS})

ament_auto_add_library(fake_sensor_publisher SHARED
  src/fake_sensor_publisher_node.cpp
)

rclcpp_components_register_node(fake_sensor_publisher
  PLUGIN "FakeSensorPublisher"
  EXECUTABLE fake_sensor_publisher_node
)

target_link_libraries(fake_sensor_publisher ${PCL_LIBRARIES} ${EIGEN3_LIBRARIES})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  rviz
)

install(FILES DESTINATION share/${PROJECT_NAME})
