from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

def generate_launch_description():
    warehouse_world = os.path.join(
        os.getenv('HOME'),
        'ros2_ws',
        'src',
        'turtlebot3_rl',
        'worlds',
        'warehouse_world.wbt'
    )

    return LaunchDescription([
        # Launch Webots with your world
        ExecuteProcess(
            cmd=['webots', '--batch', '--world', warehouse_world],
            output='screen'
        ),
        # Launch your robot driver node
        Node(
            package='webots_ros2_turtlebot',
            executable='robot_driver',
            name='turtlebot_driver',
            output='screen'
        )
    ])
