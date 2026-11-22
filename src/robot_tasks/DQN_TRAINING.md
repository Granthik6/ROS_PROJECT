# DQN object-exploration training

This document describes the older `legacy` DQN profile. The requested
360-LiDAR, three-action `coverage` design and the completed 50,000-step
`coverage_objects` experiment are documented in `STATE_ACTION_REWARD.md`.

The DQN uses Stable Baselines3 with a 16-value observation: twelve normalized
laser sectors, recent YOLO confidence, confirmed-object count, and the previous
linear and angular command. Its six actions are forward, curved left/right,
rotate left/right, and reverse. Rewards favor movement and newly confirmed
object locations; collisions and unnecessary time cost reward.

Objects use the same confidence, spatial association, and two-observation
confirmation rules as the random baseline. This makes the evaluation directly
comparable and prevents one-frame detections from inflating the score.

## Build and dependency check

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
python3 -c "import stable_baselines3, gymnasium, torch; print(stable_baselines3.__version__)"
colcon build --packages-select robot_tasks --symlink-install
```

If the import check fails:

```bash
python3 -m pip install stable-baselines3 gymnasium
```

## Start the simulator

Keep this running in terminal 1 for training and evaluation:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

## 10,000-step training run

In terminal 2:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export MPLCONFIGDIR=/tmp/robot_tasks_matplotlib
ros2 launch robot_tasks dqn_training.launch.py \
  timesteps:=10000 \
  run_id:=dqn_10000 \
  action_duration:=0.05 \
  seed:=42
```

The run writes `/root/ros2_ws/dqn_runs/dqn_10000/dqn_final.zip`, checkpoints,
the replay buffer, `episodes.csv`, `monitor.csv`, `training_summary.json`, and
`reward_curve.png`.

## 50,000-step overnight run

Start from scratch:

```bash
ros2 launch robot_tasks dqn_training.launch.py \
  timesteps:=50000 \
  run_id:=dqn_50000 \
  action_duration:=0.05 \
  seed:=42
```

Or continue from the 10,000-step model:

```bash
ros2 launch robot_tasks dqn_training.launch.py \
  timesteps:=40000 \
  run_id:=dqn_50000 \
  action_duration:=0.05 \
  seed:=42 \
  resume:=/root/ros2_ws/dqn_runs/dqn_10000/dqn_final.zip
```

Here `timesteps` is the number of additional steps. The saved summary reports
`model_total_timesteps`, which should be 50,000 after this command. Resuming a
final model also restores `dqn_final_replay_buffer.pkl` when it is present.

If an overnight run is interrupted, resume its newest checkpoint and subtract
the checkpoint step from 50,000. For example, to continue a 14,000-step
checkpoint:

```bash
ros2 launch robot_tasks dqn_training.launch.py \
  timesteps:=36000 \
  run_id:=dqn_50000_resume \
  action_duration:=0.05 \
  seed:=42 \
  resume:=/root/ros2_ws/dqn_runs/dqn_50000/checkpoints/dqn_14000_steps.zip
```

Checkpoint resumes restore the matching replay buffer and copy episode history
only through the checkpoint step, so the resulting reward curve remains a
continuous, non-overlapping 50,000-step record.

## Compare DQN with a random policy

The evaluator runs both policies from the same episode seeds:

```bash
ros2 launch robot_tasks dqn_evaluation.launch.py \
  model:=/root/ros2_ws/dqn_runs/dqn_50000/dqn_final.zip \
  episodes:=10 \
  action_duration:=0.05 \
  run_id:=dqn_50000_vs_random
```

Results are saved under `/root/ros2_ws/dqn_evaluations`. Each evaluation
contains `evaluation.csv`, `summary.json`, and `dqn_vs_random.png`. The summary
reports the mean confirmed objects and mean reward for both policies, plus a
boolean `dqn_finds_more_objects` result.

## Verified run in this workspace

The completed 50,000-step run is in
`/root/ros2_ws/dqn_runs/dqn_50000_final`. Its final 100 training episodes
average 4.40 reward and 1.34 confirmed objects, compared with 2.98 reward and
1.06 objects for the 10,000-step run's final 100 episodes.

The 10-seed evaluation is in
`/root/ros2_ws/dqn_evaluations/dqn_50000_vs_random`. The DQN averaged 1.2
confirmed objects and 3.929 reward per episode. The random policy averaged 0.0
objects and -1.910 reward. The generated `summary.json` therefore records
`dqn_finds_more_objects: true`.
