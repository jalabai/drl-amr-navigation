#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env # Not using for single env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # Not using for single env
import os
import sys
import time

from scripts.turtlebot_rl_env import TurtleBot3RLEnv # Assumes your env is here

# --- Configuration ---

ALGORITHM = "PPO"
SHOULD_CONTINUE_TRAINING_GLOBALLY = True
MODEL_TO_LOAD_PATH = "sb3_logs/PPO_turtlebot3_rl_202887_steps_final.zip" # Your ~200k PPO model
TIMESTEPS_FOR_THIS_RUN = 20000 # Additional steps for demo


# --- Common Settings ---
LOG_DIR = "sb3_logs"
# MODEL_NAME_PREFIX will be based on ALGORITHM (e.g., "PPO_turtlebot3_rl" or "SAC_turtlebot3_rl")
# If continuing, SB3 usually appends to the existing log or creates a new numbered one.
MODEL_NAME_PREFIX = f"{ALGORITHM}_turtlebot3_rl"

CHECKPOINT_SAVE_FREQ = 10000 # Save a checkpoint during this short demo run
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints_demo_continue") # Use a distinct demo checkpoint dir

def main(args=None):
    rclpy.init(args=args)
    env = None
    model = None

    local_continue_training = SHOULD_CONTINUE_TRAINING_GLOBALLY
    actual_timesteps_for_this_run = TIMESTEPS_FOR_THIS_RUN

    if local_continue_training:
        if not os.path.exists(MODEL_TO_LOAD_PATH):
            print(f"Error: Model to load for continuing training not found at '{MODEL_TO_LOAD_PATH}'.")
            print("Please verify the path. Exiting.")
            if rclpy.ok(): rclpy.shutdown()
            return # Exit if model to load isn't found
        else:
            print(f"Attempting to continue training from: {MODEL_TO_LOAD_PATH}")
    
    if not local_continue_training:
        print(f"Starting fresh training for {ALGORITHM}.")

    try:
        # Ensure TurtleBot3RLEnv is configured for the task you are demonstrating (e.g., "hard" task)
        env = TurtleBot3RLEnv() 
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
          save_freq=CHECKPOINT_SAVE_FREQ,
          save_path=CHECKPOINT_DIR,
          name_prefix=MODEL_NAME_PREFIX 
        )

        if ALGORITHM == "PPO":
            if local_continue_training: 
                print(f"Loading PPO model from {MODEL_TO_LOAD_PATH}...")
                model = PPO.load(MODEL_TO_LOAD_PATH, env=env, tensorboard_log=LOG_DIR)
                model.set_env(env) 
                print(f"Model loaded. Current total timesteps for this agent: {model.num_timesteps}.")
                print(f"Will train for {actual_timesteps_for_this_run} additional timesteps.")
            else: 
                print("Creating new PPO model.")
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
                            n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                            gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                            vf_coef=0.5, max_grad_norm=0.5, learning_rate=3e-4)
        elif ALGORITHM == "SAC":
            if local_continue_training: 
                print(f"Loading SAC model from {MODEL_TO_LOAD_PATH}...")
                model = SAC.load(MODEL_TO_LOAD_PATH, env=env, tensorboard_log=LOG_DIR)
                model.set_env(env) 
                print(f"Model loaded. Current total timesteps for this agent: {model.num_timesteps}.")
                print(f"Will train for {actual_timesteps_for_this_run} additional timesteps.")
            else: 
                print("Creating new SAC model.") 
                model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
                            learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
                            batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                            gradient_steps=1, ent_coef='auto')
        else:
            print(f"Error: Algorithm {ALGORITHM} not recognized.")
            if env is not None: env.destroy_node()
            if rclpy.ok(): rclpy.shutdown()
            return

        print(f"Starting/Continuing training for {ALGORITHM} for {actual_timesteps_for_this_run} timesteps...")
        print(f"TensorBoard log directory: {LOG_DIR} (run prefix: {MODEL_NAME_PREFIX})")
        start_time = time.time()

        model.learn(total_timesteps=actual_timesteps_for_this_run, 
                    log_interval=1, 
                    tb_log_name=MODEL_NAME_PREFIX, 
                    callback=checkpoint_callback,
                    reset_num_timesteps=not local_continue_training 
                   )

        final_model_save_path = os.path.join(LOG_DIR, f"{MODEL_NAME_PREFIX}_{model.num_timesteps}_steps_final_democontinued")
        model.save(final_model_save_path)
        print(f"Final model saved to {final_model_save_path} (Agent total steps: {model.num_timesteps})")

        end_time = time.time()
        print(f"This training session finished. Session time: {(end_time - start_time)/60:.2f} minutes")

    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user.")
        if model is not None:
             interrupted_model_save_path = os.path.join(LOG_DIR, f"{MODEL_NAME_PREFIX}_{model.num_timesteps}_steps_interrupted_democontinued")
             model.save(interrupted_model_save_path)
             print(f"Interrupted model saved to {interrupted_model_save_path}")
    except Exception as e:
        print(f"An error occurred during training: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        if env is not None:
            print("Destroying environment node...");
            try:
                if isinstance(env, Node): env.destroy_node(); print("Environment node destroyed.")
            except Exception as e: print(f"Error destroying environment node: {e}")
        if rclpy.ok(): print("Shutting down rclpy..."); rclpy.shutdown(); print("rclpy shutdown complete.")
        else: print("rclpy already shut down.")

if __name__ == '__main__':
    main()
