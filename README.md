# drl-amr-navigation
Benchmarking Deep Reinforcement Learning algorithms (PPO and SAC) for real-time autonomous navigation of a TurtleBot3 Burger in a dynamic warehouse simulation. Implemented using Webots R2025a, ROS 2 Humble, Gymnasium, and Stable Baselines3. 
Benchmarking PPO and SAC for Real-Time Autonomous Navigation in Dynamic Warehouse Environments

This repository contains the code, simulation environment, and training setup for my bachelor thesis:
"Development of an AI-Based System for Real-Time Decision Making in Robotic Applications".

The project investigates and compares two Deep Reinforcement Learning (DRL) algorithms—Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC)—for safe, efficient, and adaptive navigation of an Autonomous Mobile Robot (TurtleBot3 Burger) in a simulated warehouse environment built with Webots R2025a and integrated with ROS 2 Humble.

Project Overview
Simulation Platform: Webots R2025a
Middleware: ROS 2 Humble (via WSL Ubuntu 22.04)
Robot: TurtleBot3 Burger
Algorithms: PPO & SAC (Stable Baselines3)
Training Interface: Gymnasium-compatible wrapper
Logging & Visualization: TensorBoard
The custom simulation environment models both static obstacles (shelves, walls) and dynamic agents (workers with random paths, forklifts with predefined trajectories). Agents are trained to reach dynamic goals while avoiding collisions.

Evaluation Metrics
The agents are benchmarked using:

Success Rate (goal reached without collisions)
Collision Rate
Task Completion Time
Path Efficiency
Learning Progression (via TensorBoard)
