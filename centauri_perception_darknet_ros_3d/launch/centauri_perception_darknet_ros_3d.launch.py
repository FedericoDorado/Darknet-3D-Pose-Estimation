# Copyright 2020 Intelligent Robotics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():

    # Load params
    pkg_dir = get_package_share_directory('centauri_perception_darknet_ros_3d')
    params_file = '/config/darknet_3d.yaml'
    config_file_path = pkg_dir + params_file

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1')

    # Include darknet_ros launch file
    darknet_ros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_package_share_directory('darknet_ros') + '/launch/darknet_ros.launch.py'
        ),
        launch_arguments={
            'output': 'screen'
        }.items()
    )

    # RViz2 Node
    rviz_config_file = '/home/user/ros2_ws/src/gb_visual_detection_3d/centauri_perception_darknet_ros_3d/rviz/Perception_for_centauri.rviz'  # Specify the path to your .rviz file
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    # Create darknet3d_node
    darknet3d_node = Node(
        package='centauri_perception_darknet_ros_3d',
        executable='darknet3d_node',
        node_name='darknet3d_node',
        output='screen',
        parameters=[config_file_path]
    )

    ld = LaunchDescription()
    ld.add_action(darknet_ros_launch)
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(rviz_node)
    ld.add_action(darknet3d_node)

    return ld