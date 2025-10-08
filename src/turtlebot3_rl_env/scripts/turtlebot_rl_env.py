#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import SingleThreadedExecutor # For spinning in main test

import gymnasium as gym # Use Gymnasium
from gymnasium import spaces
import numpy as np
import math
import time
import sys # Import sys for shutdown hook

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty # For the reset service

# --- Constants ---
MAX_LINEAR_VELOCITY = 0.22
MAX_ANGULAR_VELOCITY = 2.0
NUM_LIDAR_SAMPLES = 26
LIDAR_MAX_RANGE = 3.5
MAX_GOAL_DISTANCE = 5.0
GOAL_X_DEFAULT = 0.0
GOAL_Y_DEFAULT = -2.0
MAX_STEPS_PER_EPISODE = 500
GOAL_REACHED_THRESHOLD = 0.25
COLLISION_LIDAR_THRESHOLD = 0.18
REWARD_GOAL_REACHED = 200.0
REWARD_COLLISION = -150.0
REWARD_PER_STEP = -0.1
REWARD_CLOSER_TO_GOAL_FACTOR = 2
RESET_SERVICE_NAME = '/reset_robot_simulation'
STEP_SENSOR_TIMEOUT_SEC = 1

class TurtleBot3RLEnv(Node, gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, goal_x=GOAL_X_DEFAULT, goal_y=GOAL_Y_DEFAULT):
        Node.__init__(self, 'turtlebot3_rl_environment_node')
        self.get_logger().info("Initializing TurtleBot3 RL Environment...")
        self.goal_position = np.array([goal_x, goal_y], dtype=np.float32)
        self.get_logger().info(f"Goal Position set to: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}]")

        # --- ROS 2 Publishers, Subscribers, and Service Client ---
        self.action_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.lidar_subscriber = self.create_subscription(LaserScan, 'scan', self.lidar_callback, sensor_qos)
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, sensor_qos)
        self.reset_service_client = self.create_client(Empty, RESET_SERVICE_NAME)
        self._wait_for_reset_service()

        # --- Gymnasium Action and Observation Spaces ---
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        obs_low = np.array([0.0] * NUM_LIDAR_SAMPLES + [0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0] * NUM_LIDAR_SAMPLES + [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- Environment State Variables ---
        self.current_lidar_ranges = np.ones(NUM_LIDAR_SAMPLES, dtype=np.float32)
        self.current_robot_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_robot_linear_vel = 0.0
        self.current_robot_angular_vel = 0.0
        self.previous_distance_to_goal = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.new_lidar_data_received_since_last_step = False
        self.new_odom_data_received_since_last_step = False
        self.last_sensor_update_time = self.get_clock().now()

        # --- Register Shutdown Hook ---
        # Correct way for Humble: Use the default context's on_shutdown method
        try:
            rclpy.get_default_context().on_shutdown(self._on_node_shutdown)
            self.get_logger().info("Registered shutdown hook.")
        except Exception as e:
             self.get_logger().error(f"Failed to register shutdown hook: {e}")


        self.get_logger().info("Environment Initialized. Waiting for first sensor data...")
        self._wait_for_initial_sensor_data()

    def _on_node_shutdown(self):
        """Callback executed when rclpy context is shutting down."""
        # This method is now registered globally for the context.
        # It might be called even if this specific node is already destroyed,
        # so we need to be careful accessing self.
        # It's safer to check if the publisher still exists and is valid.
        
        # Use get_logger() carefully as the node might be partially destroyed
        print("INFO: [turtlebot3_rl_environment_node] Node shutdown initiated via rclpy context hook. Sending stop command...")
        
        # Check if the publisher exists and its handle is valid
        if hasattr(self, 'action_publisher') and \
           self.action_publisher is not None and \
           self.action_publisher.handle is not None:
            try:
                stop_twist = Twist() # Zero velocities
                self.action_publisher.publish(stop_twist)
                print("INFO: [turtlebot3_rl_environment_node] Stop command published on shutdown.")
                # Avoid long sleeps here as shutdown needs to proceed
                # time.sleep(0.05)
            except Exception as e:
                # Context might already be invalid here
                print(f"WARN: [turtlebot3_rl_environment_node] Could not publish stop command during node shutdown callback: {e}")
        else:
            print("WARN: [turtlebot3_rl_environment_node] Action publisher not available during shutdown callback.")


    # --- Rest of the methods ---
    # _wait_for_reset_service, _wait_for_initial_sensor_data,
    # lidar_callback, odom_callback, _get_yaw_from_quaternion,
    # _get_observation, _check_done_conditions, _calculate_reward,
    # step, reset, render
    # (Keep the implementations from the previous version with NaN/Inf checks)
    # ... (previous method implementations omitted for brevity, assume they are here) ...
    def _wait_for_reset_service(self):
        # ... (implementation from previous version) ...
        tries = 0; max_tries = 10
        while not self.reset_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Service '{RESET_SERVICE_NAME}' not available (attempt {tries+1}/{max_tries}), waiting...")
            tries += 1
            if tries >= max_tries: self.get_logger().error(f"Service '{RESET_SERVICE_NAME}' not found."); return
        self.get_logger().info(f"Successfully connected to '{RESET_SERVICE_NAME}' service.")

    def _wait_for_initial_sensor_data(self, timeout_sec=5.0):
        # ... (implementation from previous version) ...
        start_time = self.get_clock().now()
        self.new_lidar_data_received_since_last_step = False
        self.new_odom_data_received_since_last_step = False
        while not (self.new_lidar_data_received_since_last_step and self.new_odom_data_received_since_last_step):
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now() - start_time).nanoseconds / 1e9 > timeout_sec:
                self.get_logger().warn("Timeout waiting for initial Lidar/Odom data. Proceeding with potentially stale state.")
                break
        if self.new_lidar_data_received_since_last_step and self.new_odom_data_received_since_last_step:
            self.get_logger().info("Initial Lidar and Odometry data received.")

    def lidar_callback(self, msg: LaserScan):
        # ... (implementation from previous version) ...
        try:
            if msg.range_max <= 0.0: self.get_logger().warn(f"Invalid range_max ({msg.range_max}) in LaserScan."); return
            raw_ranges = np.array(msg.ranges, dtype=np.float32)
            raw_ranges[np.isinf(raw_ranges)] = msg.range_max; raw_ranges[np.isnan(raw_ranges)] = msg.range_max
            raw_ranges = np.clip(raw_ranges, msg.range_min, msg.range_max)
            num_raw_ranges = len(raw_ranges)
            if num_raw_ranges >= NUM_LIDAR_SAMPLES:
                indices = np.linspace(0, num_raw_ranges - 1, NUM_LIDAR_SAMPLES, dtype=int); processed_ranges = raw_ranges[indices]
            elif num_raw_ranges > 0 : processed_ranges = np.full(NUM_LIDAR_SAMPLES, msg.range_max, dtype=np.float32); processed_ranges[:num_raw_ranges] = raw_ranges
            else: processed_ranges = np.full(NUM_LIDAR_SAMPLES, msg.range_max, dtype=np.float32)
            normalized_ranges = processed_ranges / msg.range_max
            if np.any(np.isnan(normalized_ranges)) or np.any(np.isinf(normalized_ranges)):
                self.get_logger().warn("NaN/Inf in normalized Lidar. Using default."); self.current_lidar_ranges = np.ones(NUM_LIDAR_SAMPLES, dtype=np.float32)
            else: self.current_lidar_ranges = normalized_ranges
            self.new_lidar_data_received_since_last_step = True
            self.last_sensor_update_time = self.get_clock().now()
        except Exception as e: self.get_logger().error(f"Error in lidar_callback: {e}"); self.current_lidar_ranges = np.ones(NUM_LIDAR_SAMPLES, dtype=np.float32)

    def odom_callback(self, msg: Odometry):
        # ... (implementation from previous version) ...
        try:
            pos = msg.pose.pose.position; orient_q = msg.pose.pose.orientation; lin_vel = msg.twist.twist.linear.x; ang_vel = msg.twist.twist.angular.z
            if any(math.isnan(v) or math.isinf(v) for v in [pos.x, pos.y, orient_q.x, orient_q.y, orient_q.z, orient_q.w, lin_vel, ang_vel]):
                self.get_logger().warn("NaN/Inf in incoming Odometry. Skipping update."); return
            yaw = self._get_yaw_from_quaternion(orient_q)
            if math.isnan(yaw) or math.isinf(yaw): yaw = self.current_robot_pose[2]; self.get_logger().warn(f"NaN/Inf yaw. Keeping previous.")
            self.current_robot_pose = np.array([pos.x, pos.y, yaw], dtype=np.float32)
            self.current_robot_linear_vel = float(lin_vel)
            self.current_robot_angular_vel = float(ang_vel)
            self.new_odom_data_received_since_last_step = True
            self.last_sensor_update_time = self.get_clock().now()
        except Exception as e: self.get_logger().error(f"Error in odom_callback: {e}")

    def _get_yaw_from_quaternion(self, q_geom_msg):
        # ... (implementation from previous version) ...
        q = [q_geom_msg.x, q_geom_msg.y, q_geom_msg.z, q_geom_msg.w]
        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1]); cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        if abs(cosy_cosp) < 1e-9 and abs(siny_cosp) < 1e-9: self.get_logger().warn("atan2 domain error in yaw calc."); return 0.0
        return math.atan2(siny_cosp, cosy_cosp)

    def _get_observation(self):
        # ... (implementation from previous version with NaN/Inf checks) ...
        lidar_obs = np.copy(self.current_lidar_ranges).astype(np.float32)
        current_pose = np.copy(self.current_robot_pose).astype(np.float32)
        lin_vel = float(self.current_robot_linear_vel); ang_vel = float(self.current_robot_angular_vel)
        robot_x, robot_y, robot_yaw = current_pose
        goal_x, goal_y = self.goal_position
        dist_to_goal = np.linalg.norm(self.goal_position - current_pose[:2])
        if np.isnan(dist_to_goal) or np.isinf(dist_to_goal): dist_to_goal = MAX_GOAL_DISTANCE; self.get_logger().warn("Invalid dist calc.")
        angle_to_goal_global = math.atan2(goal_y - robot_y, goal_x - robot_x)
        angle_to_goal_relative = angle_to_goal_global - robot_yaw
        angle_to_goal_relative = math.atan2(math.sin(angle_to_goal_relative), math.cos(angle_to_goal_relative))
        if np.isnan(angle_to_goal_relative) or np.isinf(angle_to_goal_relative): angle_to_goal_relative = 0.0; self.get_logger().warn("Invalid angle calc.")
        norm_lidar = lidar_obs
        norm_dist = min(dist_to_goal / MAX_GOAL_DISTANCE, 1.0) if MAX_GOAL_DISTANCE > 0 else 0.0
        norm_angle = angle_to_goal_relative / math.pi if math.pi > 0 else 0.0
        norm_lin_vel = lin_vel / MAX_LINEAR_VELOCITY if MAX_LINEAR_VELOCITY != 0 else 0.0
        norm_ang_vel = ang_vel / MAX_ANGULAR_VELOCITY if MAX_ANGULAR_VELOCITY != 0 else 0.0
        norm_lin_vel = np.clip(norm_lin_vel, -1.0, 1.0); norm_ang_vel = np.clip(norm_ang_vel, -1.0, 1.0)
        observation = np.concatenate([norm_lidar, np.array([norm_dist, norm_angle, norm_lin_vel, norm_ang_vel], dtype=np.float32)]).astype(np.float32)
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            self.get_logger().error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.get_logger().error(f"FATAL: NaN or Inf detected in final observation vector!")
            self.get_logger().error(f"Components: LidarMin={np.min(norm_lidar)}, Dist={norm_dist}, Angle={norm_angle}, LinV={norm_lin_vel}, AngV={norm_ang_vel}")
            observation = np.concatenate([np.ones(NUM_LIDAR_SAMPLES, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]).astype(np.float32)
            self.get_logger().error(f"Observation replaced with default safe values.")
            self.get_logger().error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return observation

    def _check_done_conditions(self):
        # ... (implementation from previous version) ...
        robot_x, robot_y, _ = self.current_robot_pose
        dist_to_goal = np.linalg.norm(self.goal_position - self.current_robot_pose[:2])
        done_by_goal = dist_to_goal < GOAL_REACHED_THRESHOLD
        normalized_collision_thresh = COLLISION_LIDAR_THRESHOLD / LIDAR_MAX_RANGE if LIDAR_MAX_RANGE > 0 else 0.01
        done_by_collision = np.any(self.current_lidar_ranges < normalized_collision_thresh)
        if done_by_collision: self.get_logger().warn(f"Collision detected! Min Lidar (normalized): {np.min(self.current_lidar_ranges):.3f} < {normalized_collision_thresh:.3f}")
        done_by_timeout = self.current_step >= MAX_STEPS_PER_EPISODE
        return done_by_goal, done_by_collision, done_by_timeout

    def _calculate_reward(self, done_by_goal, done_by_collision):
        # ... (implementation from previous version) ...
        reward = REWARD_PER_STEP
        if done_by_goal: reward += REWARD_GOAL_REACHED; self.get_logger().info(f"Goal Reached! +{REWARD_GOAL_REACHED} reward.")
        elif done_by_collision: reward += REWARD_COLLISION; self.get_logger().info(f"Collision! {REWARD_COLLISION} reward.")
        else:
            current_dist_to_goal = np.linalg.norm(self.goal_position - self.current_robot_pose[:2])
            if self.previous_distance_to_goal is not None and not (np.isnan(current_dist_to_goal) or np.isinf(current_dist_to_goal)):
                distance_delta = self.previous_distance_to_goal - current_dist_to_goal
                reward += distance_delta * REWARD_CLOSER_TO_GOAL_FACTOR
            if not (np.isnan(current_dist_to_goal) or np.isinf(current_dist_to_goal)): self.previous_distance_to_goal = current_dist_to_goal
            else: self.previous_distance_to_goal = None
        self.episode_reward += reward
        return reward

    def step(self, action):
        # ... (implementation from previous version with improved sync wait) ...
        self.current_step += 1
        linear_vel = action[0] * MAX_LINEAR_VELOCITY; angular_vel = action[1] * MAX_ANGULAR_VELOCITY
        twist_msg = Twist(); twist_msg.linear.x = float(linear_vel); twist_msg.angular.z = float(angular_vel)
        self.get_logger().info(f"Step {self.current_step}: Publishing Action Lin={twist_msg.linear.x:.3f}, Ang={twist_msg.angular.z:.3f}", throttle_duration_sec=0.2)
        self.new_lidar_data_received_since_last_step = False; self.new_odom_data_received_since_last_step = False
        self.action_publisher.publish(twist_msg)
        start_wait_time = self.get_clock().now()
        while not (self.new_lidar_data_received_since_last_step and self.new_odom_data_received_since_last_step):
            rclpy.spin_once(self, timeout_sec=0.01)
            elapsed_time = (self.get_clock().now() - start_wait_time).nanoseconds / 1e9
            if elapsed_time > STEP_SENSOR_TIMEOUT_SEC:
                self.get_logger().warn(f"Timeout ({STEP_SENSOR_TIMEOUT_SEC}s) waiting for new sensor data in step! Lidar New: {self.new_lidar_data_received_since_last_step}, Odom New: {self.new_odom_data_received_since_last_step}.")
                break
        observation = self._get_observation()
        done_by_goal, done_by_collision, done_by_timeout = self._check_done_conditions()
        terminated = done_by_goal or done_by_collision; truncated = done_by_timeout; done = terminated or truncated
        reward = self._calculate_reward(done_by_goal, done_by_collision)
        info = {};
        if done_by_goal: info['is_success'] = True; info['reason'] = 'Goal Reached'
        if done_by_collision: info['is_success'] = False; info['reason'] = 'Collision'
        if done_by_timeout: info['is_success'] = False; info['reason'] = 'Timeout'
        if truncated: info['TimeLimit.truncated'] = True
        if done: self.get_logger().info(f"Episode finished: Step={self.current_step}, Reason='{info.get('reason', 'Unknown')}', TotalReward={self.episode_reward:.2f}, Terminated={terminated}, Truncated={truncated}")
        return observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        # ... (implementation from previous version, including service call) ...
        super().reset(seed=seed)
        self.get_logger().info(f"Resetting environment (seed={seed})...")
        self.current_step = 0; self.episode_reward = 0.0
        if self.reset_service_client is not None and self.reset_service_client.service_is_ready():
            self.get_logger().info(f"Calling '{RESET_SERVICE_NAME}' service to reset robot pose...")
            req = Empty.Request(); future = self.reset_service_client.call_async(req)
            try:
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.done() and future.result() is not None: self.get_logger().info("Robot reset service call completed.")
                elif not future.done(): self.get_logger().error("Timeout waiting for reset service response.")
                else: self.get_logger().error("Reset service call failed (exception or no result).")
            except Exception as e: self.get_logger().error(f"Exception calling reset service: {e}")
        else: self.get_logger().error(f"Reset service client not ready. Cannot reset robot.")
        stop_twist = Twist(); self.action_publisher.publish(stop_twist); time.sleep(0.2)
        self.current_robot_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_robot_linear_vel = 0.0; self.current_robot_angular_vel = 0.0
        # Recalculate previous distance based on assumed reset pose
        self.previous_distance_to_goal = np.linalg.norm(self.goal_position - self.current_robot_pose[:2])
        self.new_lidar_data_received_since_last_step = False; self.new_odom_data_received_since_last_step = False
        self._wait_for_initial_sensor_data(timeout_sec=2.0)
        initial_observation = self._get_observation()
        info = {}
        self.get_logger().info("Environment reset complete. Ready for new episode.")
        return initial_observation, info

    def render(self): pass

    def close(self):
        """
        Clean up resources. This method is called by the training script's finally block
        OR potentially by Stable Baselines3 wrappers.
        The registered shutdown hook (_on_node_shutdown) should handle the final stop command.
        """
        self.get_logger().info("Close method called for TurtleBot3 RL Environment.")
        # No need to publish stop here, shutdown hook handles it.
        # Other cleanup could go here if needed.
        # Note: destroy_node() should be called externally after close()
        pass


# (main function remains the same)
def main(args=None):
    rclpy.init(args=args)
    env_node = None
    executor = SingleThreadedExecutor() # Use an executor if spinning the node directly
    try:
        env_node = TurtleBot3RLEnv()
        executor.add_node(env_node)
        print("RL Environment node spinning. Press Ctrl+C to exit.")
        executor.spin()
    except KeyboardInterrupt:
        print("RL Environment node interrupted by user.")
    except Exception as e:
        print(f"Unhandled exception in RL Environment main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if executor: executor.shutdown()
        # Node destruction is implicitly handled by rclpy.shutdown() when spinning,
        # but explicit destroy_node() is safer if the node object exists.
        # The shutdown hook should have already run if shutdown was graceful.
        if env_node and not env_node.context.is_shutdown():
             env_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("RL Environment node shutdown.")

if __name__ == '__main__':
    main()
