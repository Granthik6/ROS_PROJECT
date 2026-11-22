# Completed task checklist and commands

All requested perception, baseline, reinforcement-learning, and environment
design tasks are implemented and have been exercised in Gazebo.

Double DQN and PPO implementations, launch commands, and completed 10,000-step
and 50,000-step results are documented in `DDQN_PPO.md`. Their 50,000-step
models were evaluated against random policies; see that file for the measured
outcomes. The earlier 50,000-step result below belongs to DQN.

## Build once

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_tasks --symlink-install
source install/setup.bash
```

## Start the simulator

Keep this command running in terminal 1 for every live experiment below:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

## 1. YOLO confidence versus distance

The YOLO node publishes a `Float32` confidence in the range 0.0-1.0 on
`/yolo/confidence`. Its `/yolo/detections` JSON output contains confidence,
distance, bounding box, class, and 3D position in one record.

Terminal 2:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run robot_tasks yolo_node --ros-args \
  -p show_window:=true \
  -p conf_threshold:=0.10
```

Terminal 3, for manual movement:

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

Terminal 4, to inspect and record one target class:

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic echo /yolo/confidence
```

```bash
ros2 run robot_tasks confidence_logger --ros-args \
  -p target_class:=coke_can \
  -p output_csv:=/root/ros2_ws/results/distance_confidence.csv \
  -p min_interval_s:=0.25
```

After collecting samples, generate the plot:

```bash
ros2 run robot_tasks plot_distance_confidence \
  --input /root/ros2_ws/results/distance_confidence.csv \
  --output /root/ros2_ws/results/distance_vs_confidence.png \
  --class-name coke_can \
  --bin-width 0.25
```

Verified output: 65 plotted `coke_can` detections spanning approximately
1.23-1.93 m, with confidence values from approximately 0.10-0.76.

## 2. SLAM, detection, object locations, and random baseline

With only the simulator running in terminal 1, run:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_tasks slam_detection_baseline.launch.py \
  duration_sec:=180 \
  run_id:=random_baseline_180s \
  random_seed:=42 \
  min_confidence:=0.25 \
  rviz:=true \
  show_window:=true
```

This single launch starts SLAM Toolbox, YOLO, RViz markers, and the seeded
random exploration policy. It writes the mapped object image, occupancy map,
observation CSV, unique-object CSV, and JSON/Markdown baseline summaries under
`/root/ros2_ws/baseline_runs/<run_id>`.

Verified example: `verification_30s_4` found 7 spatially deduplicated objects
in 0.50 minutes and produced `map_with_objects.png`.

## 3. DQN training and random-policy evaluation

The requested 360-LiDAR, three-action DQN is trained for 50,000 steps under
the `coverage_objects` profile. Its first 10,000 steps and reward curve are in
`/root/ros2_ws/dqn_runs/coverage_objects_10000`; the resumed 50,000-step
model and full reward curve are in
`/root/ros2_ws/dqn_runs/coverage_objects_50000`. In ten fixed-seed evaluation
episodes per policy, it found 3.8 confirmed objects per episode versus 2.3 for
random. The result and comparison plot are in
`/root/ros2_ws/dqn_evaluations/coverage_objects_50000_vs_random`. See
`STATE_ACTION_REWARD.md` for the exact reward and demonstration command.

The completed legacy experiment is stored in
`/root/ros2_ws/dqn_runs/dqn_50000_final`. Its fixed-seed evaluation found 1.2
objects per episode versus 0.0 for random.

To repeat that evaluation with the simulator running:

```bash
ros2 launch robot_tasks dqn_evaluation.launch.py \
  model:=/root/ros2_ws/dqn_runs/dqn_50000_final/dqn_final.zip \
  environment_profile:=legacy \
  episodes:=10 \
  action_duration:=0.05 \
  run_id:=dqn_50000_vs_random_repeat \
  seed:=1000
```

## 4. Requested state, action, and reward design

The `coverage` profile has a 364-value state: 360 normalized LiDAR readings,
robot x/y/yaw, and detection confidence. It has exactly three actions:
forward, left, and right. Its reward adds new-area and confidence bonuses and
subtracts collision and revisit penalties.
The `coverage_objects` profile keeps this design and adds +10 for a newly
confirmed object so the training reward directly reflects object discovery.

Run the live `reset()` and `step()` contract test:

```bash
ros2 launch robot_tasks dqn_environment_test.launch.py \
  action_duration:=0.10 \
  seed:=42 \
  report:=/root/ros2_ws/dqn_environment_check.json
```

The exact profile has now been trained for 50,000 steps. The saved model and
reward curve are in `/root/ros2_ws/dqn_runs/coverage_dqn_50000_complete`.
Its ten-episode evaluation is in
`/root/ros2_ws/dqn_evaluations/coverage_dqn_50000_vs_random_verified`:
DQN averaged 3.1 confirmed detections and -77.224 reward; random averaged
3.5 detections and -99.097 reward. The model improved the reward but did not
beat random on object count in this test.

To train a new model with this exact profile:

```bash
ros2 launch robot_tasks dqn_training.launch.py \
  environment_profile:=coverage \
  timesteps:=50000 \
  run_id:=coverage_dqn_50000_repeat \
  action_duration:=0.05 \
  episode_steps:=250 \
  seed:=42
```

Do not run keyboard teleoperation during the SLAM random baseline or DQN
training because it would publish competing velocity commands.
