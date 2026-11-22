# Coverage exploration environment

Use the `coverage` environment profile for the requested state, action, and
four-term reward design. The `coverage_objects` profile keeps that state,
those actions, and those reward terms, and adds a reward for newly confirmed
objects. The older `legacy` profile remains available so models trained before
this change can still be evaluated.

## State space

Each observation is a `float32` vector with 364 values:

| Indices | Values | Range |
| --- | --- | --- |
| 0-359 | 360 evenly sampled LiDAR ranges | normalized to 0.0-1.0 |
| 360 | odometry x position | metres |
| 361 | odometry y position | metres |
| 362 | robot yaw | -pi to +pi radians |
| 363 | most recent YOLO confidence | 0.0-1.0 |

LiDAR scans with a different number of beams are resampled to exactly 360
values. Invalid or infinite ranges become the configured maximum range.

## Action space

The action space is `Discrete(3)`:

| Action | Name | Linear x | Angular z |
| --- | --- | --- | --- |
| 0 | forward | 0.18 m/s | 0.00 rad/s |
| 1 | left | 0.10 m/s | +0.80 rad/s |
| 2 | right | 0.10 m/s | -0.80 rad/s |

## Reward function

The workspace is divided into 0.5 m square cells. For every step:

```text
reward = new_area + confidence - collision - revisit

new_area  = +1.00 when the robot enters an unvisited cell, otherwise 0
confidence = +0.50 * latest YOLO confidence
collision = -2.00 when forward clearance is below 0.14 m
revisit   = -0.05 when the current cell has already been visited
```

A collision stops the current motion and applies the penalty, but the episode
continues so the three-action policy can steer away. An episode ends after the
configured maximum of 250 steps.

The `info` dictionary returned by `step()` includes each value under
`reward_components`, plus `new_area`, `visited_cells`, `confidence`, and the
selected `action_name`.

## Test reset and step in Gazebo

Start the simulator in terminal 1:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

Run the contract check in terminal 2:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_tasks dqn_environment_test.launch.py
```

The check calls `env.reset()` and `env.step()` for forward, left, and right. It
validates observation shape and bounds, finite rewards, and all reward
components. The report is written to
`/root/ros2_ws/dqn_environment_check.json`.

Gymnasium uses the current API:

```python
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## Train with this design

With the simulator running:

```bash
ros2 launch robot_tasks dqn_training.launch.py \
  environment_profile:=coverage \
  timesteps:=10000 \
  run_id:=coverage_dqn_10000 \
  action_duration:=0.10 \
  seed:=42
```

Evaluate a model trained with this profile by also passing
`environment_profile:=coverage` to `dqn_evaluation.launch.py`. A model's state
and action spaces must match the profile used for training.

## Completed 50,000-step coverage run

The model trained from scratch with the state, actions, and reward above is at
`/root/ros2_ws/dqn_runs/coverage_dqn_50000_complete/dqn_final.zip`. Its
`training_summary.json` reports `model_total_timesteps: 50000`, and loading
the saved model confirms observation shape `(364,)` and three actions.

The training reward graph is
`/root/ros2_ws/dqn_runs/coverage_dqn_50000_complete/reward_curve.png`.
The first 20 episodes averaged -138.911 reward and 2.20 confirmed detections;
the last 20 averaged -74.719 reward and 4.35 confirmed detections. These are
training episodes, so they are not an independent performance estimate.

The ten-episode-per-policy evaluation is in
`/root/ros2_ws/dqn_evaluations/coverage_dqn_50000_vs_random_verified`.
DQN averaged 3.1 confirmed detections and -77.224 reward; random averaged
3.5 detections and -99.097 reward. Thus DQN earned better reward under this
reward function, but it did not find more objects in this evaluation.

To repeat the evaluation with the simulator running:

```bash
ros2 launch robot_tasks dqn_evaluation.launch.py \
  model:=/root/ros2_ws/dqn_runs/coverage_dqn_50000_complete/dqn_final.zip \
  environment_profile:=coverage \
  episodes:=10 \
  episode_steps:=250 \
  action_duration:=0.05 \
  seed:=1000 \
  run_id:=coverage_dqn_repeat
```

## Completed object-discovery DQN experiment

The first `coverage` model improved its training reward but did not find more
objects than random in its evaluation above. For the image's object-discovery
goal, the `coverage_objects` profile adds **+10.0 per newly confirmed object**
to the four reward terms shown above. It keeps the same 364-value observation,
three actions, 0.5 m coverage cells, collision rule, and 250-step episodes.
An object is counted after the detector's spatial matching and confirmation
rules; this is a confirmed detection count, not ground-truth object recall.

Stable Baselines3 DQN trained for 10,000 steps and then resumed for another
40,000 steps with its replay buffer, reaching 50,000 total. Both training
summaries report `completed`. The model has observation shape `(364,)` and
`Discrete(3)` actions.

| Artifact | Path |
| --- | --- |
| 10,000-step reward curve and model | `/root/ros2_ws/dqn_runs/coverage_objects_10000/` |
| 50,000-step reward curve and final model | `/root/ros2_ws/dqn_runs/coverage_objects_50000/` |
| DQN versus random results and plot | `/root/ros2_ws/dqn_evaluations/coverage_objects_50000_vs_random/` |

In the complete training history, mean reward was -64.931 for the first 20
episodes and -27.376 for the last 20. The rolling reward rose substantially
during training but declined near the end, so this does not establish steady
improvement. The independent evaluation used ten 250-step episodes per policy,
with matching simulator seeds 1000-1009. DQN found **3.8 confirmed objects per
episode** on average, versus **2.3** for random; mean rewards were **-29.903**
and **-88.292**, respectively. The result meets the requested object-count
comparison for this test; ten episodes are a limited performance estimate.

To inspect the saved results without retraining:

```bash
cat /root/ros2_ws/dqn_runs/coverage_objects_10000/training_summary.json
cat /root/ros2_ws/dqn_runs/coverage_objects_50000/training_summary.json
cat /root/ros2_ws/dqn_evaluations/coverage_objects_50000_vs_random/summary.json
ls -lh /root/ros2_ws/dqn_runs/coverage_objects_50000/reward_curve.png \
  /root/ros2_ws/dqn_evaluations/coverage_objects_50000_vs_random/dqn_vs_random.png
```

To demonstrate the trained model live, start the simulator using the command
in “Test reset and step in Gazebo” above. Then, in another terminal, run:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_tasks dqn_evaluation.launch.py \
  model:=/root/ros2_ws/dqn_runs/coverage_objects_50000/dqn_final.zip \
  environment_profile:=coverage_objects \
  episodes:=10 \
  episode_steps:=250 \
  action_duration:=0.05 \
  seed:=1000 \
  run_id:=coverage_objects_professor_demo
```

The command runs both policies and writes a new `summary.json`,
`evaluation.csv`, and `dqn_vs_random.png` under
`/root/ros2_ws/dqn_evaluations/coverage_objects_professor_demo/`. It does not
train again. Do not use keyboard teleoperation at the same time because it
would publish competing movement commands.
