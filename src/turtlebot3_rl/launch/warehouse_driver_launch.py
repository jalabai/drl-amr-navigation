from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
import os

def generate_launch_description():
    # Define the path to your Webots world file
    warehouse_world = os.path.join(
        os.getenv('HOME'),
        'ros2_ws',
        'src',
        'turtlebot3_rl',
        'worlds',
        'warehouse_world.wbt'
    )

    return LaunchDescription([
        # Launch Webots with your custom world in real-time mode
        ExecuteProcess(
            cmd=['webots', '--mode=realtime', warehouse_world],
            output='screen'
        ),

        # Just a friendly log to confirm launch
        LogInfo(
            msg='🚀 Webots warehouse simulation is now running in realtime mode!'
        ),
    ])
