#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from stable_baselines3 import PPO, SAC # Import PPO
import os
import time
import numpy as np
import sys # For sys.exit

# Import your custom environment class
from scripts.turtlebot_rl_env import TurtleBot3RLEnv # Assuming this is the correct path

# --- Configuration ---
ALGORITHM_TO_EVALUATE = "PPO" # <<< SET TO "PPO" for this script
# IMPORTANT: UPDATE THIS PATH to your actual saved PPO model file
MODEL_PATH = "sb3_logs/PPO_turtlebot3_rl_202887_steps_final.zip" # <<< From your previous PPO ~200k run

NUM_EVALUATION_EPISODES = 50 

# Environment parameters (should match those used during training for fair evaluation)
# If your environment constructor takes goal_x, goal_y as params, you might need to pass them.
# GOAL_X_EVAL = 3.0
# GOAL_Y_EVAL = 3.0

def main(args=None):
    rclpy.init(args=args)
    env = None
    model = None

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in this script to point to your saved PPO model.")
        if rclpy.ok(): rclpy.shutdown()
        sys.exit(1) # Exit if model not found

    try:
        # --- Create the RL Environment ---
        print(f"Creating evaluation environment for PPO...")
        # Pass goal parameters if your environment constructor takes them and they differ from defaults
        # env = TurtleBot3RLEnv(goal_x=GOAL_X_EVAL, goal_y=GOAL_Y_EVAL)
        env = TurtleBot3RLEnv() # Assuming default goal is fine for evaluation or set in env
        print("Environment created.")

        # --- Load the Trained Agent ---
        print(f"Loading model: {MODEL_PATH} using algorithm: {ALGORITHM_TO_EVALUATE}")
        if ALGORITHM_TO_EVALUATE == "PPO":
            model = PPO.load(MODEL_PATH, env=env)
        # elif ALGORITHM_TO_EVALUATE == "SAC": # Not relevant for this PPO-specific script
        #     model = SAC.load(MODEL_PATH, env=env)
        else: # Should not happen if ALGORITHM_TO_EVALUATE is "PPO"
            print(f"Error: Algorithm {ALGORITHM_TO_EVALUATE} not recognized for loading.")
            if env is not None: env.destroy_node()
            if rclpy.ok(): rclpy.shutdown()
            sys.exit(1)
        print("Model loaded successfully.")

        # --- Run Evaluation Episodes ---
        total_successes = 0
        total_collisions = 0
        total_steps_taken_success = [] # List to store steps for successful episodes
        total_rewards = []
        episode_lengths = [] # To store all episode lengths

        print(f"\nStarting PPO evaluation for {NUM_EVALUATION_EPISODES} episodes...")

        for episode in range(NUM_EVALUATION_EPISODES):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            termination_reason = "Unknown"


            print(f"\n--- Episode {episode + 1}/{NUM_EVALUATION_EPISODES} (PPO) ---")

            while not (terminated or truncated):
                # Use deterministic=True for evaluation (no exploration noise)
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if info and 'reason' in info: # Check if info is not None and has 'reason'
                    termination_reason = info['reason']

                # Optional: Add a small delay to make visualization easier if needed
                # time.sleep(0.05) 
            
            episode_lengths.append(episode_steps)
            total_rewards.append(episode_reward)
            # Use the 'is_success' flag from the info dictionary returned by your environment
            if info and info.get('is_success', False):
                total_successes += 1
                total_steps_taken_success.append(episode_steps)
                print(f"Episode {episode + 1}: SUCCESS! Steps: {episode_steps}, Reward: {episode_reward:.2f}")
            elif termination_reason == 'Collision': # Check specific reason
                total_collisions += 1
                print(f"Episode {episode + 1}: COLLISION. Steps: {episode_steps}, Reward: {episode_reward:.2f}")
            elif truncated: # Timed out
                print(f"Episode {episode + 1}: TIMEOUT. Steps: {episode_steps}, Reward: {episode_reward:.2f}")
            else: # Other termination reasons or if 'is_success' wasn't set
                 print(f"Episode {episode + 1}: Ended (Reason: {termination_reason}). Steps: {episode_steps}, Reward: {episode_reward:.2f}")


        # --- Print Evaluation Metrics ---
        print("\n--- PPO Evaluation Summary ---")
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
            avg_time_success = avg_steps_success * 0.1 # If each step is roughly 0.1s due to time.sleep(0.1) in env
            print(f"Average Steps for Successful Episodes: {avg_steps_success:.2f}")
            print(f"Estimated Average Time for Successful Episodes: {avg_time_success:.2f} seconds")
        else:
            print("No successful episodes to calculate average steps/time.")

    except KeyboardInterrupt:
        print("\nPPO Evaluation interrupted by user.")
    except Exception as e:
        print(f"An error occurred during PPO evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up PPO evaluation resources...")
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
