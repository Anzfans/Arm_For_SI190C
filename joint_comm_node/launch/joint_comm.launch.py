from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('joint_comm_node')
    config_file = os.path.join(pkg_dir, 'config', 'motor_config.yaml')

    joint_comm_node = Node(
        package='joint_comm_node',
        executable='joint_comm_node',
        name='joint_serial_comm',
        parameters=[config_file],
        output='screen',
    )

    return LaunchDescription([
        joint_comm_node,
    ])
