#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from controller import Supervisor  # Webots Supervisor API
import sys

# Using a standard simple service type for a basic trigger
from std_srvs.srv import Empty

# Name of your TurtleBot3 node in the Webots scene tree (the DEF name)
# IMPORTANT: Make sure this matches the DEF name of your TurtleBot3.
TURTLEBOT_DEF_NAME = "TurtleBot3Burger"  # <<< CHANGE THIS IF YOURS IS DIFFERENT

# Define the target reset pose [x, y, z] and rotation [axis_x, axis_y, axis_z, angle]
RESET_POSITION = [-1.0, -2.0, 0.035]  # Example: near origin, slightly above ground
RESET_ROTATION = [0.0, 0.0, 1.0, 0.785]  # Facing along positive X-axis (no rotation)

class WebotsSupervisorReset(Node):
    def __init__(self, supervisor_robot_instance):
        super().__init__('webots_supervisor_reset_node')
        self.supervisor = supervisor_robot_instance
        self.robot_node_to_reset = None
        self.translation_field = None
        self.rotation_field = None

        self.get_logger().info(f"Supervisor node initializing to reset robot DEF: '{TURTLEBOT_DEF_NAME}'")

        # Get the TurtleBot3 node from its DEF name
        self.robot_node_to_reset = self.supervisor.getFromDef(TURTLEBOT_DEF_NAME)
        if self.robot_node_to_reset is None:
            self.get_logger().error(f"Could not find robot node with DEF name '{TURTLEBOT_DEF_NAME}'. "
                                    "Check the DEF name in your Webots world and script.")
            return

        # Get the fields for translation and rotation
        self.translation_field = self.robot_node_to_reset.getField("translation")
        self.rotation_field = self.robot_node_to_reset.getField("rotation")

        if self.translation_field is None or self.rotation_field is None:
            self.get_logger().error(f"Could not access 'translation' or 'rotation' fields on '{TURTLEBOT_DEF_NAME}'.")
            return

        # Create the ROS 2 service
        self.reset_service = self.create_service(
            Empty,
            'reset_robot_simulation',
            self.handle_reset_request
        )
        self.get_logger().info("Service '/reset_robot_simulation' is ready.")
        self.get_logger().info(f"Will reset robot '{TURTLEBOT_DEF_NAME}' to Pos: {RESET_POSITION}, Rot: {RESET_ROTATION}")

    def handle_reset_request(self, request, response):
        self.get_logger().info(f"Reset request received for robot '{TURTLEBOT_DEF_NAME}'.")

        if self.robot_node_to_reset is None or self.translation_field is None or self.rotation_field is None:
            self.get_logger().error("Cannot reset: robot node or its fields were not properly initialized.")
            return response

        try:
            self.translation_field.setSFVec3f(RESET_POSITION)
            self.rotation_field.setSFRotation(RESET_ROTATION)
            self.robot_node_to_reset.resetPhysics()
            self.get_logger().info(f"Robot '{TURTLEBOT_DEF_NAME}' reset to position {RESET_POSITION}, rotation {RESET_ROTATION}.")
        except Exception as e:
            self.get_logger().error(f"Error during robot reset: {e}")

        return response

def main(args=None):
    rclpy.init(args=args)
    supervisor_robot_instance = Supervisor()
    supervisor_reset_ros_node = None

    try:
        supervisor_reset_ros_node = WebotsSupervisorReset(supervisor_robot_instance)
        timestep = int(supervisor_robot_instance.getBasicTimeStep())
        while supervisor_robot_instance.step(timestep) != -1:
            rclpy.spin_once(supervisor_reset_ros_node, timeout_sec=0.001)
            if not rclpy.ok():
                break
    except RuntimeError as e:
        print(f"Critical runtime error: {e}. Check if TURTLEBOT_DEF_NAME is correct.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Supervisor reset node interrupted by user (Ctrl+C).", file=sys.stderr)
    finally:
        if supervisor_reset_ros_node is not None and rclpy.ok():
            supervisor_reset_ros_node.get_logger().info("Shutting down supervisor reset ROS node...")
            supervisor_reset_ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Supervisor reset script finished.", file=sys.stderr)

if __name__ == '__main__':
    main()
