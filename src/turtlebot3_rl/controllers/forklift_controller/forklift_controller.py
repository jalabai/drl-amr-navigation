"""
forklift_controller.py
A simple Webots controller for a "forklift" robot (e.g., Pioneer 3-DX)
that moves forward and then turns in a repeating sequence.
This version does not use GPS or IMU, relying on timed movements.
Place this script in: .../worlds/controllers/forklift_controller/forklift_controller.py
"""

from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())
motor_max_speed_rad_s = 6.28 # Nominal max speed of Pioneer motors

try:
    # Ensure these motor names match your Pioneer 3-DX model in Webots
    left_motor = robot.getDevice('left wheel') # Or 'left wheel motor'
    right_motor = robot.getDevice('right wheel') # Or 'right wheel motor'
except Exception as e:
    print(f"Forklift Error: Could not get motors. Check names. {e}")
    exit()

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Movement Logic Parameters
step_counter = 0
current_phase = 0  # 0 = move forward, 1 = turn

# Durations for each phase (in simulation steps)
# Adjust these based on your timestep and desired movement
# Example: If timestep is 32ms, 100 steps = 3.2 seconds
FORWARD_DURATION_STEPS = 125  # e.g., ~4 seconds if timestep = 32ms
TURN_DURATION_STEPS = 50      # e.g., ~1.6 seconds if timestep = 32ms

# Speed factors (relative to motor_max_speed_rad_s)
# Forklift path: [-4, 0.15, 4], [0, 0.15, 4], [4, 0.15, 4], [0, 0.15, 4]
# This is a 4m stretch, then turn, 4m stretch, then turn.
# To cover 4m in ~4s (FORWARD_DURATION_STEPS * timestep_s), speed needs to be ~1 m/s
# Wheel radius of Pioneer ~0.0975m. So, 1 m/s wheel speed is 1/0.0975 = ~10.2 rad/s.
# This is higher than motor_max_speed_rad_s. So, we use a factor.
# Let's aim for a robot speed of ~0.5 m/s. Wheel speed = 0.5 / 0.0975 = ~5.1 rad/s
# Speed factor = 5.1 / 6.28 = ~0.8
FORWARD_SPEED_FACTOR = 0.6 # Adjusted for more moderate speed
TURN_SPEED_FACTOR = 0.4    # For turning

print("Forklift Controller (Time-Based) started.")

while robot.step(timestep) != -1:
    if current_phase == 0:  # Forward phase
        left_motor.setVelocity(FORWARD_SPEED_FACTOR * motor_max_speed_rad_s)
        right_motor.setVelocity(FORWARD_SPEED_FACTOR * motor_max_speed_rad_s)
        step_counter += 1
        if step_counter > FORWARD_DURATION_STEPS:
            # print("Forklift: Switching to turn phase.")
            current_phase = 1
            step_counter = 0
    elif current_phase == 1:  # Turn phase (e.g., turn right)
        left_motor.setVelocity(TURN_SPEED_FACTOR * motor_max_speed_rad_s)
        right_motor.setVelocity(-TURN_SPEED_FACTOR * motor_max_speed_rad_s) # Opposite for turn
        step_counter += 1
        if step_counter > TURN_DURATION_STEPS:
            # print("Forklift: Switching to forward phase.")
            current_phase = 0
            step_counter = 0
