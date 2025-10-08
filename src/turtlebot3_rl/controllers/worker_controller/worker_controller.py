"""
simple_worker_controller.py
A simple Webots controller for a "worker" robot (e.g., Pioneer 3-DX)
that randomly moves forward, left, or right for random durations.
This version does not use GPS or IMU.
"""

from controller import Robot
import random

# Initialize the Robot instance
robot = Robot()

# Get the simulation timestep
timestep = int(robot.getBasicTimeStep())

# Define a maximum motor speed (adjust if necessary)
max_speed = 6.28

# Get motor devices
# IMPORTANT: Verify these motor names match your robot model in Webots.
# Common names are 'left wheel motor' and 'right wheel motor'.
try:
    left_motor = robot.getDevice('left wheel')
    right_motor = robot.getDevice('right wheel')
except Exception as e:
    print(f"Error getting motors: {e}")
    print("Please ensure motor names ('left wheel', 'right wheel') are correct for your robot model.")
    exit()

# Set motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Initialize motor velocities to 0
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --- Random Movement Logic Parameters ---
current_action = 'forward'  # Initial action
action_step_count = 0       # Counter for current action duration

# Speed factors
forward_speed_factor = 0.5
turn_speed_factor = 0.5 # For turning, one wheel will be negative

print("Simple Worker Controller started: Moving randomly.")

# Main control loop
while robot.step(timestep) != -1:
    if action_step_count == 0:  # Time to choose a new action and its duration
        # Choose a random action
        current_action = random.choice(['forward', 'left', 'right', 'stop']) # Added 'stop' for more variability

        # Set a random duration for this action (in simulation steps)
        if current_action == 'forward':
            action_step_count = random.randint(50, 150) # Duration for moving forward
            print(f"Worker: New action: FORWARD for {action_step_count} steps.")
        elif current_action == 'stop':
            action_step_count = random.randint(20, 80) # Duration for stopping
            print(f"Worker: New action: STOP for {action_step_count} steps.")
        else:  # 'left' or 'right'
            action_step_count = random.randint(20, 70)  # Duration for turning
            print(f"Worker: New action: {current_action.upper()} for {action_step_count} steps.")

    # Execute the current action
    if current_action == 'forward':
        left_motor.setVelocity(forward_speed_factor * max_speed)
        right_motor.setVelocity(forward_speed_factor * max_speed)
    elif current_action == 'left': # Turn left (robot rotates counter-clockwise)
        left_motor.setVelocity(-turn_speed_factor * max_speed)
        right_motor.setVelocity(turn_speed_factor * max_speed)
    elif current_action == 'right': # Turn right (robot rotates clockwise)
        left_motor.setVelocity(turn_speed_factor * max_speed)
        right_motor.setVelocity(-turn_speed_factor * max_speed)
    elif current_action == 'stop':
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)


    # Decrement the counter for the current action's duration
    action_step_count -= 1
