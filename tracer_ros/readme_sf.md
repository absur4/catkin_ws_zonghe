第一部分：连接底盘并实现电脑控制底盘运动
# Basic usage of the ROS package

1. Install dependent packages

    ```
    $ sudo apt install ros-$ROS_DISTRO-teleop-twist-keyboard
    $ sudo apt install ros-$ROS_DISTRO-joint-state-publisher-gui
    $ sudo apt install ros-$ROS_DISTRO-ros-controllers
    ```
    
2. Clone the packages into your catkin workspace and compile

    (the following instructions assume your catkin workspace is at: ~/catkin_ws/src)

    ```
    $ cd ~/catkin_ws/src
    $ git clone https://github.com/agilexrobotics/ugv_sdk.git
    $ git clone https://github.com/agilexrobotics/tracer_ros.git
    $ cd ..
    $ catkin_make
    ```

3. Setup CAN-To-USB adapter
* Enable gs_usb kernel module(If you have already added this module, you do not need to add it)
    ```
    $ sudo modprobe gs_usb
    ```
* first time use tracer-ros package
    ```
    $rosrun tracer_bringup setup_can2usb.bash
    ```
* If not the first time use tracer-ros package(Run this command every time you turn on the power)
    ```
    $  rosrun tracer_bringup bringup_can2usb.bash
    ```
4. Launch ROS nodes

* Start the base node for the real robot whith can

    ```
    $ roslaunch tracer_bringup tracer_robot_base.launch
    ```
* Start the keyboard tele-op node

    ```
    $ roslaunch tracer_bringup tracer_teleop_keyboard.launch
    ```

**SAFETY PRECAUSION**: 

Always have your remote controller ready to take over the control whenever necessary. 
。
**重要**
记得把tracer_base_node.cpp文件替换自己的给别人


第二部分：连接雷达
给自己电脑设置静态端口:
ifconfig
找到e开头的，比如自己的是eno1,
sudo ifconfig eno1 192.168.1.50
ifconfig eno1进行确认是不是上面的地址
ping 192.168.1.167（67是雷达北面最后两个数字），如果能看到 time= 返回值，说明物理连线和静态 IP 设置均已成功。
上述命令是一次有效，电脑重启后需要在此设置
 **使用软件livox view2**
 前往 www.livoxtech.com 下载最新版本的 Livox Viewer 2，
 Windows 用户：解压文件，并于已解压的文件中打开文件名为 Livox Viewer 2 的程序。
Ubuntu 用户：需要在解压缩后文件的根目录下启动终端（或者直接启动终端后进入到解压缩后
文件夹的根目录），运行指令：./livox_viewer_2.sh 即可启动。
这个时候在软件里面就可以看到很多点云数据了
## ros&sdk##
在catkin_ws/src中 git clone https://github.com/Livox-SDK/livox_ros_driver2.git
 **安装Livox-SDK2 基础库**
 1. 安装必要的依赖库
在编译 SDK 之前，需要确保系统中安装了 cmake 和基本编译工具。

Bash

sudo apt update
sudo apt install cmake gcc g++
2. 下载并编译 Livox-SDK2
你需要从官方开源代码仓库下载源码 。


克隆源码： 建议在非 ROS 工作空间（如 ~/Downloads 或 ~/Documents）中进行安装。

Bash

git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
创建编译目录：

Bash

mkdir build && cd build
运行 CMake 和编译：

Bash

cmake ..
make -j$(nproc)
安装到系统： 这一步会将编译好的头文件和库文件安装到 /usr/local 路径下，方便后续 ROS 驱动调用。

Bash

sudo make install
**安装ros驱动**
cd ~/catkin_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git(之前执行过了，可以不执行)
cd livox_ros_driver2
# 针对 ROS 1 (Noetic/18.04) 执行以下脚本
./build.sh ROS1

在/home/songfei/catkin_ws/src/livox_ros_driver2/config/Mid360_config.json
{
  "lidar_summary_info" : {
    "lidar_type": 8
  },
  "MID360": {
    "lidar_net_info" : {
      "cmd_data_port": 56100,
      "push_msg_port": 56200,
      "point_data_port": 56300,
      "imu_data_port": 56400,
      "log_data_port": 56500
    },
    "host_net_info" : {
      "cmd_data_ip" : "192.168.1.50",
      "cmd_data_port": 56101,
      "push_msg_ip": "192.168.1.50", 
      "push_msg_port": 56201,
      "point_data_ip": "192.168.1.50",
      "point_data_port": 56301,
      "imu_data_ip" : "192.168.1.50",
      "imu_data_port": 56401,
      "log_data_ip" : "192.168.1.50",
      "log_data_port": 56501
    }
  },
  "lidar_configs" : [
    {
      "ip" : "192.168.1.167", 
      "pcl_data_type" : 1,
      "pattern_mode" : 0,
      "extrinsic_parameter" : {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "x": 0,
        "y": 0,
        "z": 0
      }
    }
  ]
}
上面最重要的是两个修改：
host_net_info 下所有的 IP 字段全部改为 192.168.1.50。
lidar_configs 中的 "ip": "192.168.1.167" 必须与雷达真实的 IP 一致。
然后
roslaunch livox_ros_driver2 rviz_MID360.launch
可以在rviz中查看3D点云了

/home/songfei/catkin_ws/src/livox_ros_driver2/launch_ROS1/msg_MID360.launch里面的	
<arg name="xfer_format" default="2"/>要是这个
然后执行 roslaunch livox_ros_driver2 mid360_to_laserscan.launch
，添加laserscan，就可以看到白点了
得是
songfei@songfei-OMEN-by-HP-Gaming-Laptop-16-k0xxx:~/catkin_ws$ rostopic type /livox/lidar
sensor_msgs/PointCloud2
ongfei@songfei-OMEN-by-HP-Gaming-Laptop-16-k0xxx:~/catkin_ws$ rostopic hz /scan
subscribed to [/scan]
average rate: 9.983
	min: 0.099s max: 0.102s std dev: 0.00060s window: 10
^Caverage rate: 9.995
	min: 0.099s max: 0.102s std dev: 0.00060s window: 17

## 建图 ##
**hector_mapping**:Csongfei@songfei-OMEN-by-HP-Gaming-Laptop-16-k0xxx:~/catkin_ws$ roslaunch tracer_base hector_mapping_handheld.launch 
保存： rosrun map_server map_saver -f /home/{your_usr_name}/tracer_ws/src/tracer_ros/tracer_bringup/map/{map_name}

**gmapping**     rosrun gmapping slam_gmapping
**导航**
tracer_nav包实现的，roslaunch tracer_nav my_nav.launch就可以实现手动给目标点导航了
