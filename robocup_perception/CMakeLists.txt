cmake_minimum_required(VERSION 3.0.2)
project(robocup_perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  tf2_ros
  tf2_geometry_msgs
  image_transport
  robocup_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  tf2_ros
  tf2_geometry_msgs
  image_transport
  robocup_msgs
)

catkin_install_python(PROGRAMS
  scripts/vision_service_node.py
  scripts/object_detector_server.py
  scripts/shelf_detector_server.py
  scripts/grasp_pose_computer.py
  scripts/place_pose_computer.py
  scripts/object_classifier.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
