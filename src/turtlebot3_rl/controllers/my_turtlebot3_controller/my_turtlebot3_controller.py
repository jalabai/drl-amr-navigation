#!/usr/bin/env python3
from controller import Robot
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Quaternion # Import Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
import sys

# Define the known maximum velocity for the wheel motors
MAX_WHEEL_MOTOR_VELOCITY = 6.67 # rad/s (Realistic for TurtleBot3 Burger)

class MyTurtlebot3Controller(Node):
    def __init__(self, robot):
        super().__init__('my_turtlebot3_controller') # Node name
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.get_logger().info(f"Controller using timestep: {self.timestep} ms")

        # Device Initialization
        self.lidar = self._initialize_device('LDS-01', 'Lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

        self.left_motor = self._initialize_device('left wheel motor', 'Motor')
        self.right_motor = self._initialize_device('right wheel motor', 'Motor')
        if self.left_motor and self.right_motor:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

        self.left_encoder = self._initialize_device('left wheel sensor', 'Encoder')
        self.right_encoder = self._initialize_device('right wheel sensor', 'Encoder')
        
        self.prev_left_encoder_val = 0.0
        self.prev_right_encoder_val = 0.0
        if self.left_encoder: 
            self.left_encoder.enable(self.timestep)
            try:
                 self.prev_left_encoder_val = self.left_encoder.getValue()
                 if math.isnan(self.prev_left_encoder_val) or math.isinf(self.prev_left_encoder_val):
                      self.get_logger().warn("Initial left encoder value is NaN/Inf, setting to 0.0")
                      self.prev_left_encoder_val = 0.0
            except Exception as e:
                 self.get_logger().error(f"Failed to get initial left encoder value: {e}")
                 self.prev_left_encoder_val = 0.0
                 
        if self.right_encoder: 
            self.right_encoder.enable(self.timestep)
            try:
                 self.prev_right_encoder_val = self.right_encoder.getValue()
                 if math.isnan(self.prev_right_encoder_val) or math.isinf(self.prev_right_encoder_val):
                      self.get_logger().warn("Initial right encoder value is NaN/Inf, setting to 0.0")
                      self.prev_right_encoder_val = 0.0
            except Exception as e:
                 self.get_logger().error(f"Failed to get initial right encoder value: {e}")
                 self.prev_right_encoder_val = 0.0

        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.x = 0.0; self.y = 0.0; self.th = 0.0

        self.wheel_radius = 0.033
        self.wheel_base = 0.160
        self.get_logger().info(f"Robot params: WheelRadius={self.wheel_radius}m, WheelBase={self.wheel_base}m")

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        default_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)

        if self.lidar: self.lidar_pub = self.create_publisher(LaserScan, 'scan', sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, 'odom', default_qos)
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, default_qos)
        self.tf_broadcaster = TransformBroadcaster(self, qos=default_qos)

    def _initialize_device(self, name, device_type):
        try:
            device = self.robot.getDevice(name)
            if device is None: self.get_logger().warn(f"{device_type} device '{name}' not found."); return None
            self.get_logger().info(f"{device_type} device '{name}' initialized.")
            return device
        except Exception as e: self.get_logger().error(f"Error initializing {device_type} device '{name}': {e}"); return None

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f"Controller Received cmd_vel: Lin={msg.linear.x:.3f}, Ang={msg.angular.z:.3f}", throttle_duration_sec=0.5)
        self.target_linear_vel = msg.linear.x
        self.target_angular_vel = msg.angular.z

    def _euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        q = Quaternion(); q.w = cr*cp*cy + sr*sp*sy; q.x = sr*cp*cy - cr*sp*sy
        q.y = cr*sp*cy + sr*cp*sy; q.z = cr*cp*sy - sr*sp*cy
        return q

    def step(self):
        if self.robot.step(self.timestep) == -1: self.get_logger().info("Webots simulation step returned -1."); return False

        if self.left_motor and self.right_motor:
            left_wheel_linear_speed_mps = self.target_linear_vel - (self.target_angular_vel * self.wheel_base / 2.0)
            right_wheel_linear_speed_mps = self.target_linear_vel + (self.target_angular_vel * self.wheel_base / 2.0)
            left_motor_rps = left_wheel_linear_speed_mps / self.wheel_radius if self.wheel_radius != 0 else 0.0
            right_motor_rps = right_wheel_linear_speed_mps / self.wheel_radius if self.wheel_radius != 0 else 0.0
            clamped_left_motor_rps = max(-MAX_WHEEL_MOTOR_VELOCITY, min(left_motor_rps, MAX_WHEEL_MOTOR_VELOCITY))
            clamped_right_motor_rps = max(-MAX_WHEEL_MOTOR_VELOCITY, min(right_motor_rps, MAX_WHEEL_MOTOR_VELOCITY))
            self.left_motor.setVelocity(clamped_left_motor_rps)
            self.right_motor.setVelocity(clamped_right_motor_rps)

        vx = 0.0; omega = 0.0
        if self.left_encoder and self.right_encoder:
            current_left_encoder_val = self.left_encoder.getValue()
            current_right_encoder_val = self.right_encoder.getValue()

            if math.isnan(current_left_encoder_val) or math.isinf(current_left_encoder_val) or \
               math.isnan(current_right_encoder_val) or math.isinf(current_right_encoder_val):
                self.get_logger().warn("NaN/Inf detected in CURRENT encoder values. Skipping odometry update this cycle.")
            else:
                dt_sec = self.timestep / 1000.0
                if dt_sec <= 1e-6:
                    self.get_logger().warn(f"Odometry dt is too small or zero ({dt_sec}). Skipping update.")
                else:
                    delta_left_rad = current_left_encoder_val - self.prev_left_encoder_val
                    delta_right_rad = current_right_encoder_val - self.prev_right_encoder_val
                    if math.isnan(delta_left_rad) or math.isinf(delta_left_rad) or \
                       math.isnan(delta_right_rad) or math.isinf(delta_right_rad):
                       self.get_logger().warn("NaN/Inf detected in calculated encoder deltas. Skipping odometry update.")
                    else:
                        dist_left_wheel = delta_left_rad * self.wheel_radius
                        dist_right_wheel = delta_right_rad * self.wheel_radius
                        dist_center = (dist_left_wheel + dist_right_wheel) / 2.0
                        delta_th = (dist_right_wheel - dist_left_wheel) / self.wheel_base if self.wheel_base != 0 else 0.0
                        if math.isnan(dist_center) or math.isinf(dist_center) or \
                           math.isnan(delta_th) or math.isinf(delta_th):
                            self.get_logger().warn("NaN/Inf detected in calculated displacement/rotation. Skipping odometry update.")
                        else:
                            mid_delta_th = self.th + delta_th / 2.0
                            self.x += dist_center * math.cos(mid_delta_th)
                            self.y += dist_center * math.sin(mid_delta_th)
                            self.th += delta_th
                            self.th = math.atan2(math.sin(self.th), math.cos(self.th))
                            vx = dist_center / dt_sec
                            omega = delta_th / dt_sec
                            self.prev_left_encoder_val = current_left_encoder_val
                            self.prev_right_encoder_val = current_right_encoder_val
            
            if math.isnan(vx) or math.isinf(vx): vx = 0.0
            if math.isnan(omega) or math.isinf(omega): omega = 0.0
            
            odom = Odometry(); now = self.get_clock().now().to_msg()
            odom.header.stamp = now; odom.header.frame_id = "odom"; odom.child_frame_id = "base_footprint"
            odom.pose.pose.position.x = self.x; odom.pose.pose.position.y = self.y; odom.pose.pose.position.z = 0.0
            q = self._euler_to_quaternion(0, 0, self.th); odom.pose.pose.orientation = q
            odom.twist.twist.linear.x = vx; odom.twist.twist.linear.y = 0.0; odom.twist.twist.angular.z = omega
            if not(any(math.isnan(v) or math.isinf(v) for v in [self.x, self.y, q.x, q.y, q.z, q.w, vx, omega])):
                 self.odom_pub.publish(odom)
            else: self.get_logger().error("CRITICAL: NaN/Inf in Odom msg NOT published!")

            t = TransformStamped(); t.header.stamp = now; t.header.frame_id = "odom"; t.child_frame_id = "base_footprint"
            t.transform.translation.x = self.x; t.transform.translation.y = self.y; t.transform.translation.z = 0.0
            t.transform.rotation = q
            if not(any(math.isnan(v) or math.isinf(v) for v in [self.x, self.y, q.x, q.y, q.z, q.w])):
                 self.tf_broadcaster.sendTransform(t)
            else: self.get_logger().error("CRITICAL: NaN/Inf in TF msg NOT published!")

        if self.lidar and self.lidar_pub.get_subscription_count() > 0:
            try:
                raw_ranges = self.lidar.getRangeImage()
                if raw_ranges:
                    scan = LaserScan(); scan.header.stamp = self.get_clock().now().to_msg(); scan.header.frame_id = 'base_scan'
                    scan.angle_min = -math.pi; scan.angle_max = math.pi
                    num_ranges = self.lidar.getHorizontalResolution()
                    scan.angle_increment = (scan.angle_max - scan.angle_min) / num_ranges if num_ranges > 0 else 0.0
                    scan.time_increment = 0.0; scan.scan_time = self.timestep / 1000.0
                    scan.range_min = float(self.lidar.getMinRange()); scan.range_max = float(self.lidar.getMaxRange())
                    processed_ranges = [float(r) if not (math.isinf(r) or math.isnan(r)) else scan.range_max + 0.1 for r in raw_ranges]
                    scan.ranges = processed_ranges
                    self.lidar_pub.publish(scan)
            except Exception as e: self.get_logger().error(f"Error processing/publishing Lidar data: {e}")
        return True

def main(args=None):
    rclpy.init(args=args)
    controller_node = None
    try:
        robot_instance = Robot()
        if not robot_instance: print("Failed to get Robot instance from Webots.", file=sys.stderr); rclpy.shutdown(); return
        controller_node = MyTurtlebot3Controller(robot_instance)
        while controller_node.step(): # This loop is driven by Webots simulation steps
            rclpy.spin_once(controller_node, timeout_sec=0) # Process ROS callbacks non-blockingly
            if not rclpy.ok(): break 
    except KeyboardInterrupt: print("Controller node interrupted.")
    except Exception as e: print(f"Unhandled exception in controller main: {e}"); import traceback; traceback.print_exc()
    finally:
        if controller_node and rclpy.ok(): controller_node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        print("Controller node shutdown.")

if __name__ == '__main__':
    main()
