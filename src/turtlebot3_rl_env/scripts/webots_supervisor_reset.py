#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from controller import Supervisor # Pylance might show an error here, this is normal. Webots provides this module at runtime.
import sys

# Using a standard simple service type for a basic trigger
from std_srvs.srv import Empty

# Name of your TurtleBot3 node in the Webots scene tree (the DEF name)
# IMPORTANT: Make sure this matches the DEF name of your TurtleBot3.
# Common DEF names are "TurtleBot3Burger", "TB3Burger", etc. Check your Webots world file.
TURTLEBOT_DEF_NAME = "TurtleBot3Burger" # <<< CHANGE THIS IF YOURS IS DIFFERENT

# Define the target reset pose [x, y, z] and rotation [axis_x, axis_y, axis_z, angle]
# Adjust these to your desired starting position and orientation in your Webots world.
RESET_POSITION = [0.0, 0.0, 0.035]  # Example: near origin, slightly above ground for TB3 Burger
RESET_ROTATION = [0.0, 0.0, 1.0, 0.0] # Example: facing along positive X-axis (no rotation initially)

class WebotsSupervisorReset(Node):
    def __init__(self, supervisor_robot_instance): # Renamed parameter for clarity
        super().__init__('webots_supervisor_reset_node')
        self.supervisor = supervisor_robot_instance # Store the supervisor instance
        self.robot_node = None
        self.translation_field = None
        self.rotation_field = None
        self.get_logger().info(f"Supervisor node initializing for robot DEF: '{TURTLEBOT_DEF_NAME}'")

        # Get the TurtleBot3 node from its DEF name
        self.robot_node = self.supervisor.getFromDef(TURTLEBOT_DEF_NAME)
        if self.robot_node is None:
            self.get_logger().error(f"Could not find robot node with DEF name '{TURTLEBOT_DEF_NAME}'. "
                                    "Please check the DEF name in your Webots world file and in this script.")
            # This is a critical error, the node cannot function without the robot.
            # Consider raising an exception or causing the controller to exit if appropriate for your setup.
            # For now, we'll log and continue, but reset will fail.
            return # Exit __init__ if robot not found

        # Get the fields for translation and rotation
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        if self.translation_field is None or self.rotation_field is None:
            self.get_logger().error("Could not get 'translation' or 'rotation' field for the robot DEF "
                                    f"'{TURTLEBOT_DEF_NAME}'. Ensure the robot model has these fields.")
            return # Exit __init__ if fields not found

        # Create the ROS 2 service
        self.reset_service = self.create_service(
            Empty, # Using std_srvs.Empty for a simple trigger
            'reset_robot_simulation', # Service name
            self.handle_reset_request
        )
        self.get_logger().info(f"Webots Supervisor Reset Service '/reset_robot_simulation' is ready.")
        self.get_logger().info(f"Will reset robot '{TURTLEBOT_DEF_NAME}' to Pos: {RESET_POSITION}, Rot: {RESET_ROTATION}")

    def handle_reset_request(self, request, response):
        self.get_logger().info(f"Reset request received for robot '{TURTLEBOT_DEF_NAME}'.")
        if self.robot_node is None or self.translation_field is None or self.rotation_field is None:
            self.get_logger().error("Cannot reset: Robot node or its fields were not properly initialized.")
            return response # For std_srvs.Empty, just return the empty response object

        try:
            # Stop the robot's physics simulation temporarily before moving it
            self.robot_node.resetPhysics()

            # Set the new position and rotation
            self.translation_field.setSFVec3f(RESET_POSITION)
            self.rotation_field.setSFRotation(RESET_ROTATION)
            
            # Reset physics again after moving to stabilize
            self.robot_node.resetPhysics()
            
            self.get_logger().info(f"Robot '{TURTLEBOT_DEF_NAME}' has been reset to pose: P={RESET_POSITION}, R={RESET_ROTATION}.")
        except Exception as e:
            self.get_logger().error(f"Error during robot reset: {e}")
        return response # For std_srvs.Empty, just return the empty response object

def main(args=None):
    rclpy.init(args=args)

    # Create a Supervisor instance from Webots. This is the entry point for a Webots Python controller.
    supervisor_robot_instance = Supervisor() 
    
    supervisor_reset_node = None # Define in a broader scope for finally block
    try:
        # Pass the Webots supervisor instance to our ROS 2 node class
        supervisor_reset_node = WebotsSupervisorReset(supervisor_robot_instance)
        
        timestep = int(supervisor_robot_instance.getBasicTimeStep())

        # Main loop for the Webots controller
        while supervisor_robot_instance.step(timestep) != -1:
            # Process ROS 2 events (e.g., service calls)
            rclpy.spin_once(supervisor_reset_node, timeout_sec=0.001) 
            if not rclpy.ok(): # Check if ROS has been shut down (e.g., by Ctrl+C in another terminal)
                break
    except RuntimeError as e:
        # This can happen if the DEF name is wrong and __init__ tries to use a None robot_node
        print(f"Critical runtime error in Supervisor: {e}. Ensure TURTLEBOT_DEF_NAME is correct.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Supervisor reset node interrupted by user (Ctrl+C in Webots console).", file=sys.stderr)
    finally:
        # Cleanup ROS 2 node
        if supervisor_reset_node is not None and rclpy.ok(): # Check if node was created and rclpy is still running
            supervisor_reset_node.get_logger().info("Shutting down supervisor reset node...")
            supervisor_reset_node.destroy_node()
        if rclpy.ok(): # Only shutdown rclpy if it's still initialized
            rclpy.shutdown()
        print("Supervisor reset script finished.", file=sys.stderr)

if __name__ == '__main__':
    # This script is intended to be run as a Webots controller.
    # Webots will execute it, and the main() function will be called.
    main()
