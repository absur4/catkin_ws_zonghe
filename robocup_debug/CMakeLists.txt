cmake_minimum_required(VERSION 3.0.2)
project(robocup_debug)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  robocup_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/debug_01_classification.py
  scripts/debug_02_perception.py
  scripts/debug_03_grasp_pose.py
  scripts/debug_04_pick.py
  scripts/debug_05_place.py
  scripts/debug_06_navigation.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
