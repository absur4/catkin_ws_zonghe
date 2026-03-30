cmake_minimum_required(VERSION 3.0.2)
project(robocup_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  move_base_msgs
  geometry_msgs
  nav_msgs
  tf
  robocup_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  actionlib
  move_base_msgs
  geometry_msgs
  nav_msgs
  tf
  robocup_msgs
)

catkin_install_python(PROGRAMS
  scripts/navigation_action_server.py
  scripts/location_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
