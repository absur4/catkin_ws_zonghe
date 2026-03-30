cmake_minimum_required(VERSION 3.0.2)
project(robocup_manipulation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  moveit_commander
  moveit_ros_planning_interface
  control_msgs
  trajectory_msgs
  piper_msgs
  robocup_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  actionlib
  moveit_commander
  moveit_ros_planning_interface
  control_msgs
  trajectory_msgs
  piper_msgs
  robocup_msgs
)

catkin_install_python(PROGRAMS
  scripts/manipulation_manager.py
  scripts/pick_action_server.py
  scripts/place_action_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
