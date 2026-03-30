cmake_minimum_required(VERSION 3.0.2)
project(robocup_executive)

find_package(catkin REQUIRED COMPONENTS
  rospy
  smach
  smach_ros
  actionlib
  robocup_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  smach
  smach_ros
  actionlib
  robocup_msgs
)

catkin_install_python(PROGRAMS
  scripts/pick_place_task.py
  scripts/breakfast_task.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
