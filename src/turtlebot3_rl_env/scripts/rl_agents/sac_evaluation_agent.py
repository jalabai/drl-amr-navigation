#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from stable_baselines3 import SAC # Import SAC
import os
import time
import numpy as np
import sys

# Import your custom environment class
from scripts.turtlebot_rl_env import TurtleBot3RLEnv # Assuming this is the correct path

# --- Configuration ---
ALGORITHM_TO_EVALUATE = "SAC"
# Path to your SAC model trained for ~200k steps on the "hard" task
MODEL_PATH = "sb3_logs/SAC_turtlebot3_rl_200679_steps_final.zip" # From your SAC ~200k run

NUM_EVALUATION_EPISODES = 50 # Number of episodes to run for evaluation

def main(args=None):
    rclpy.init(args=args)
    env = None
    model = None

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in this script to point to your saved SAC model.")
        if rclpy.ok(): rclpy.shutdown()
        sys.exit(1) # Exit if model not found

    try:
        # --- Create the RL Environment ---
        print(f"Creating evaluation environment for SAC (Primary Task)...")
        # Ensure TurtleBot3RLEnv uses the "hard task" defaults from its constants
        # (goal at 3,3, max_steps 500, reward_factor 0.5)
        env = TurtleBot3RLEnv() 
        print("Environment created.")

        # --- Load the Trained Agent ---
        print(f"Loading model: {MODEL_PATH} using algorithm: {ALGORITHM_TO_EVALUATE}")
        if ALGORITHM_TO_EVALUATE == "SAC":
            model = SAC.load(MODEL_PATH, env=env)
        else: 
            print(f"Error: Script configured for SAC, but ALGORITHM_TO_EVALUATE is {ALGORITHM_TO_EVALUATE}.")
            if env is not None: env.destroy_node()
            if rclpy.ok(): rclpy.shutdown()
            sys.exit(1)
        print("Model loaded successfully.")

        # --- Run Evaluation Episodes ---
        total_successes = 0
        total_collisions = 0
        total_steps_taken_success = [] 
        total_rewards = []
        episode_lengths = [] 

        print(f"\nStarting SAC evaluation for {NUM_EVALUATION_EPISODES} episodes (Primary Task)...")

        for episode in range(NUM_EVALUATION_EPISODES):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            termination_reason = "Unknown"

            print(f"\n--- Episode {episode + 1}/{NUM_EVALUATION_EPISODES} (SAC - Primary Task) ---")

            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if info and 'reason' in info: 
                    termination_reason = info['reason']
            
            episode_lengths.append(episode_steps)
            total_rewards.append(episode_reward)
            if info and info.get('is_success', False):
                total_successes += 1
                total_steps_taken_success.append(episode_steps)
                print(f"Episode {episode + 1}: SUCCESS! Steps: {episode_steps}, Reward: {episode_reward:.2f}")
            elif termination_reason == 'Collision':
                total_collisions += 1
                print(f"Episode {episode + 1}: COLLISION. Steps: {episode_steps}, Reward: {episode_reward:.2f}")
            elif truncated: 
                print(f"Episode {episode + 1}: TIMEOUT. Steps: {episode_steps}, Reward: {episode_reward:.2f}")
            else: 
                 print(f"Episode {episode + 1}: Ended (Reason: {termination_reason}). Steps: {episode_steps}, Reward: {episode_reward:.2f}")

        # --- Print Evaluation Metrics ---
        print("\n--- SAC Evaluation Summary (Primary Task) ---")
        success_rate = (total_successes / NUM_EVALUATION_EPISODES) * 100
        collision_rate = (total_collisions / NUM_EVALUATION_EPISODES) * 100 
        avg_reward = np.mean(total_rewards) if total_rewards else 0
        std_reward = np.std(total_rewards) if total_rewards else 0
        avg_ep_len = np.mean(episode_lengths) if episode_lengths else 0
        
        print(f"Total Episodes: {NUM_EVALUATION_EPISODES}")
        print(f"Success Rate: {success_rate:.2f}% ({total_successes}/{NUM_EVALUATION_EPISODES})")
        print(f"Collision Rate: {collision_rate:.2f}% ({total_collisions}/{NUM_EVALUATION_EPISODES})")
        print(f"Average Episode Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
        print(f"Average Episode Length: {avg_ep_len:.2f} steps")

        if total_steps_taken_success:
            avg_steps_success = np.mean(total_steps_taken_success)
            avg_time_success = avg_steps_success * 0.1 # Assuming each step is ~0.1s
            print(f"Average Steps for Successful Episodes: {avg_steps_success:.2f}")
            print(f"Estimated Average Time for Successful Episodes: {avg_time_success:.2f} seconds")
        else:
            print("No successful episodes to calculate average steps/time.")

    except KeyboardInterrupt:
        print("\nSAC Evaluation interrupted by user.")
    except Exception as e:
        print(f"An error occurred during SAC evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up SAC evaluation resources...")
        if env is not None:
            print("Destroying environment node...")
            try:
                if isinstance(env, Node):
                     env.destroy_node()
                     print("Environment node destroyed.")
            except Exception as e:
                 print(f"Error destroying environment node: {e}")
        if rclpy.ok():
            print("Shutting down rclpy...")
            rclpy.shutdown()
            print("rclpy shutdown complete.")
        else:
             print("rclpy already shut down.")

if __name__ == '__main__':
    main()
