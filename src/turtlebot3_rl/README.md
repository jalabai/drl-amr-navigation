# TurtleBot3 RL Benchmarking

Simulated benchmark of PPO and SAC algorithms on a TurtleBot3 Burger navigating a dynamic industrial warehouse in Webots + ROS 2 Humble (WSL).

## Structure

- `controllers/` – (empty) handled by ROS 2
- `models/` – custom 3D robot/obstacle models
- `protos/` – reusable model templates
- `rl_agents/` – PPO/SAC implementations
- `worlds/` – Webots `.wbt` simulation world
- `README.md` – you're reading it :)

## Simulation

Environment: `warehouse.wbt`  
Robot: TurtleBot3 Burger (w/ lidar)  
Obstacles: static shelves + dynamic forklifts/workers

## ROS 2 Integration

Simulation runs on Webots (Windows)  
RL and ROS 2 nodes run on WSL (`ros2_ws`)
