cmake_minimum_required(VERSION 3.0.2)
project(robocup_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)

## 消息文件
add_message_files(
  FILES
  DetectedObject.msg
  DetectedObjectArray.msg
  ShelfLayer.msg
  ShelfDetectionResult.msg
  GraspPose.msg
  TaskStatus.msg
)

## 服务文件
add_service_files(
  FILES
  DetectObjects.srv
  DetectShelf.srv
  ComputeGraspPose.srv
  ComputePlacePose.srv
  ClassifyObject.srv
)

## 动作文件
add_action_files(
  FILES
  NavigateToLocation.action
  PickObject.action
  PlaceObject.action
  ExecuteTask.action
)

## 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)
