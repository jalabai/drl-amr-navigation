from setuptools import setup, find_packages # Ensure find_packages is imported
import os
# from glob import glob # glob is not strictly needed for this specific setup

package_name = 'turtlebot3_rl_env'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']), # Finds 'scripts' and 'scripts/rl_agents' if they have __init__.py
    data_files=[
        ('share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        # If you have launch files, add them here, e.g.:
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools', 'gymnasium', 'numpy', 'stable-baselines3'], # Added stable-baselines3
    zip_safe=True,
    maintainer='jala', # Your name
    maintainer_email='jala@example.com', # Your email
    description='ROS 2 RL Environment for TurtleBot3 in Webots',
    license='Apache License 2.0', # Or your preferred license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Entry point for the RL environment node (in scripts/)
            'turtlebot_rl_env_node = scripts.turtlebot_rl_env:main',
            # Entry point for the supervisor reset script (in scripts/)
            'webots_supervisor_reset_node = scripts.webots_supervisor_reset:main',
            # Corrected entry point for the training script in scripts/rl_agents/
            # This tells ros2 run: create 'train_agent', run main() from
            # 'train_agent.py' module inside the 'scripts.rl_agents' package path.
            'train_agent = scripts.rl_agents.train_agent:main',
            'ppo_evaluation_agent = scripts.rl_agents.ppo_evaluation_agent:main',
            'sac_evaluation_agent = scripts.rl_agents.sac_evaluation_agent:main',
        ],
    },
)
