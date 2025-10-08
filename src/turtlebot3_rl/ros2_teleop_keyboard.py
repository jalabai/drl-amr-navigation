import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, select, termios, tty, fcntl, os # For keyboard input on Linux/WSL

# Instructions for key presses
msg = """
Control Your Robot! (Debug v2.3 - Increased Speeds for Simulation)
---------------------------
Moving around:
        ^ (Up Arrow)
        |
(Left Arrow) <--  s  --> (Right Arrow)
        |
        v (Down Arrow)

u/j : increase/decrease linear speed by 10%
i/k : increase/decrease angular speed by 10%

space key, s : force stop
q : quit
---------------------------
Ensure this terminal window is in focus.
Watch this console for key detection logs!
The robot stops if no movement key is held/pressed.
Note: Actual robot speed may be limited by Webots controller/model settings.
"""

# Key bindings for arrow key escape sequence final characters
key_bindings_arrows = {
    'A': (1, 0),   # Up arrow (after escape sequence \x1b[A) -> Positive linear
    'B': (-1, 0),  # Down arrow (\x1b[B) -> Negative linear
    'C': (0, -1),  # Right arrow (\x1b[C) -> Negative angular (turn right)
    'D': (0, 1),   # Left arrow (\x1b[D) -> Positive angular (turn left)
}

# Key bindings for single character keys
key_bindings_single = {
    's': (0, 0),   # 's' for stop
    ' ': (0, 0),   # space for stop
}

# Speed change bindings
speed_bindings = {
    'u': (1.1, 1.0), # Increase linear speed
    'j': (0.9, 1.0), # Decrease linear speed
    'i': (1.0, 1.1), # Increase angular speed
    'k': (1.0, 0.9), # Decrease angular speed
}

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard_debug_v2_3')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Increased speeds for simulation purposes
        self.speed = 0.30  # Initial linear speed (m/s)
        self.turn = 1.5   # Initial angular speed (rad/s)
        
        self.max_linear_speed = 0.50 # m/s (Target max for teleop)
        self.min_linear_speed = 0.05 # m/s
        self.max_angular_speed = 3.5 # rad/s (Target max for teleop)
        self.min_angular_speed = 0.2  # rad/s
        
        self.current_x_multiplier = 0
        self.current_th_multiplier = 0

        self.status_display_timer = self.create_timer(1.0, self.print_status)

        self.get_logger().info("Teleop Keyboard Node Started (Debug v2.3 - Increased Speeds).")
        self.get_logger().info(msg)

    def get_key(self, settings):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1) # 0.1s timeout
        key_input = ''
        if rlist:
            key_input = sys.stdin.read(1) # Read the first character
            if key_input == '\x1b': # Escape character (potential start of arrow key sequence)
                orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
                fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)
                try:
                    char2 = sys.stdin.read(1)
                    char3 = sys.stdin.read(1)
                    if char2 is not None: key_input += char2
                    if char3 is not None: key_input += char3
                except BlockingIOError: 
                    pass 
                except Exception as e:
                    self.get_logger().error(f"  Exception reading escape: {e}. Key so far: {repr(key_input)}")
                finally:
                    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl) 
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings) 
        return key_input

    def print_status(self):
        status_msg = (f"Status: TargetLinSpeed={self.speed:.2f} TargetAngSpeed={self.turn:.2f} | "
                      f"LastCmdMultipliers: X={self.current_x_multiplier} TH={self.current_th_multiplier}")
        self.get_logger().info(status_msg)

    def run_teleop(self):
        original_terminal_settings = termios.tcgetattr(sys.stdin)
        try:
            while rclpy.ok():
                key = self.get_key(original_terminal_settings)
                
                # Reset multipliers: if no key is pressed or an unassigned key, robot stops/doesn't move.
                self.current_x_multiplier = 0 
                self.current_th_multiplier = 0

                if key: 
                    # self.get_logger().info(f"--- Key Event --- Raw key data: {repr(key)}") # Verbose

                    if key.startswith('\x1b[') and len(key) == 3: 
                        actual_key_char = key[-1] 
                        if actual_key_char in key_bindings_arrows:
                            self.current_x_multiplier = key_bindings_arrows[actual_key_char][0]
                            self.current_th_multiplier = key_bindings_arrows[actual_key_char][1]
                        # else: self.get_logger().warn(f"  Unrecognized ARROW key sequence end: '{actual_key_char}' from {repr(key)}")
                    elif key in key_bindings_single:
                        self.current_x_multiplier = key_bindings_single[key][0]
                        self.current_th_multiplier = key_bindings_single[key][1]
                    elif key in speed_bindings:
                        old_speed, old_turn = self.speed, self.turn
                        self.speed *= speed_bindings[key][0]
                        self.turn *= speed_bindings[key][1]
                        # Clamp speeds to defined min/max values for teleop
                        self.speed = max(self.min_linear_speed, min(self.speed, self.max_linear_speed))
                        self.turn = max(self.min_angular_speed, min(self.turn, self.max_angular_speed))
                        self.get_logger().info(f"  SPEED CHANGE key: '{key}'. Target Linear: {self.speed:.2f}, Target Angular: {self.turn:.2f}")
                    elif key == 'q':
                        self.get_logger().info("Quitting teleop.")
                        break
                    else:
                        if key == '\x03': 
                            self.get_logger().info("Ctrl+C detected, raising KeyboardInterrupt.")
                            raise KeyboardInterrupt
                        # self.get_logger().warn(f"  Unrecognized key: {repr(key)}") # Verbose
                
                twist = Twist()
                twist.linear.x = float(self.current_x_multiplier * self.speed)
                twist.linear.y = 0.0
                twist.linear.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = float(self.current_th_multiplier * self.turn)
                
                if key or twist.linear.x != 0.0 or twist.angular.z != 0.0: 
                    self.get_logger().info(f"  Publishing Twist: LinX={twist.linear.x:.3f}, AngZ={twist.angular.z:.3f}",
                                           throttle_duration_sec=0.25) 
                self.publisher_.publish(twist)

        except KeyboardInterrupt:
            self.get_logger().info("Teleop interrupted by user (Ctrl+C). Gracefully stopping.")
        except Exception as e:
            self.get_logger().error(f"An error occurred in run_teleop: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.get_logger().info("Restoring terminal settings and sending stop command.")
            twist = Twist() 
            self.publisher_.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal_settings)

def main(args=None):
    rclpy.init(args=args)
    teleop_node = TeleopKeyboard()
    try:
        teleop_node.run_teleop()
    except Exception as e:
        teleop_node.get_logger().fatal(f"Unhandled exception in main: {e}")
        import traceback
        teleop_node.get_logger().error(traceback.format_exc())
    finally:
        if rclpy.ok(): 
            teleop_node.get_logger().info("Shutting down ROS 2 node.")
            teleop_node.destroy_node()
            rclpy.shutdown()
        else:
            teleop_node.get_logger().info("ROS 2 already shut down.")

if __name__ == '__main__':
    main()