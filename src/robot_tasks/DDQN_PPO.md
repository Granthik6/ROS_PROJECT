# Double DQN and PPO exploration

The `robot_tasks` package supports two additional algorithms on the same
`coverage_objects` environment used by the completed DQN experiment. Both
receive 360 normalized LiDAR values, odometry x/y/yaw, and YOLO confidence
(364 values total). Both choose among forward, left, and right. The reward is
+1 for a new 0.5 m cell, +0.5 times confidence, +10 per newly confirmed object,
-2 for a collision, and -0.05 for a revisit.

Double DQN (`ddqn`) selects the next action using the online Q-network and
evaluates that action using the target Q-network. Stable Baselines3's ordinary
DQN uses the target network for both operations. PPO (`ppo`) is the library's
on-policy clipped policy-gradient algorithm. It does not use a replay buffer;
Double DQN does. The implementations are in `robot_tasks/double_dqn.py`,
`robot_tasks/train_exploration.py`, and `robot_tasks/evaluate_exploration.py`.

Build after editing the package:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_tasks --symlink-install
```

Keep the simulator running in terminal 1:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

In terminal 2, train one algorithm at a time. Each command starts YOLO and
ends automatically after its requested steps. Do not run teleoperation or
another training process at the same time because they would compete for
`/cmd_vel`.

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_tasks exploration_training.launch.py \
  algorithm:=ddqn timesteps:=10000 run_id:=ddqn_10000 \
  environment_profile:=coverage_objects seed:=42
```

To continue the saved DDQN model to 50,000 total steps, run **40,000
additional** steps and load the saved replay buffer:

```bash
ros2 launch robot_tasks exploration_training.launch.py \
  algorithm:=ddqn timesteps:=40000 run_id:=ddqn_50000 \
  environment_profile:=coverage_objects seed:=42 \
  resume:=/root/ros2_ws/rl_runs/ddqn_10000/model_final.zip
```

For PPO, use the same environment and step counts:

```bash
ros2 launch robot_tasks exploration_training.launch.py \
  algorithm:=ppo timesteps:=10000 run_id:=ppo_10000 \
  environment_profile:=coverage_objects seed:=42
```

```bash
ros2 launch robot_tasks exploration_training.launch.py \
  algorithm:=ppo timesteps:=40000 run_id:=ppo_50000 \
  environment_profile:=coverage_objects seed:=42 \
  resume:=/root/ros2_ws/rl_runs/ppo_10000/model_final.zip
```

Each training run creates `training_summary.json`, `episodes.csv`,
`reward_curve.png`, checkpoints, and `model_final.zip` in
`/root/ros2_ws/rl_runs/<run_id>/`. Reusing a run ID creates a numbered folder;
use the actual folder path in `resume:=`.

Evaluate each saved model against random actions with the simulator running:

```bash
ros2 launch robot_tasks exploration_evaluation.launch.py \
  algorithm:=ddqn \
  model:=/root/ros2_ws/rl_runs/ddqn_50000/model_final.zip \
  environment_profile:=coverage_objects \
  episodes:=10 episode_steps:=250 seed:=1000 \
  run_id:=ddqn_50000_vs_random
```

```bash
ros2 launch robot_tasks exploration_evaluation.launch.py \
  algorithm:=ppo \
  model:=/root/ros2_ws/rl_runs/ppo_50000/model_final.zip \
  environment_profile:=coverage_objects \
  episodes:=10 episode_steps:=250 seed:=1000 \
  run_id:=ppo_50000_vs_random
```

The comparison writes `evaluation.csv`, `summary.json`, and
`policy_vs_random.png` under `/root/ros2_ws/rl_evaluations/<run_id>/`.
Both algorithms completed the requested 10,000-step run and a resumed
50,000-step total run on 2026-10-10. The 50,000-step run adds 40,000 steps to
the 10,000-step model; the 50,000-step reward curve contains all 200
250-step training episodes.

| Algorithm | 10,000-step model | 50,000-step model | Reward curves |
| --- | --- | --- | --- |
| DDQN | `rl_runs/ddqn_10000/model_final.zip` | `rl_runs/ddqn_50000/model_final.zip` | `rl_runs/ddqn_{10000,50000}/reward_curve.png` |
| PPO | `rl_runs/ppo_10000/model_final.zip` | `rl_runs/ppo_50000/model_final.zip` | `rl_runs/ppo_{10000,50000}/reward_curve.png` |

The 50,000-step models were evaluated for ten 250-step episodes per policy
against a random policy, using seeds 1000–1009. These are observed results,
not a guarantee that either model will outperform random in another run:

| Algorithm | Objects per episode, model | Objects per episode, random | Mean episode reward, model | Mean episode reward, random |
| --- | ---: | ---: | ---: | ---: |
| DDQN | 3.1 | 2.9 | -51.638 | -87.693 |
| PPO | 3.0 | 3.0 | -21.498 | -79.502 |

DDQN found 0.2 more objects per episode than its random control in this
sample. PPO tied its random control on object count, although its reward was
higher. These are small, noisy ten-episode comparisons; a higher reward does
not establish more object discovery. The older DQN evaluation was made in a
different simulation run, so its numbers are not a controlled head-to-head
ranking. The training reward curves also fluctuate; they do not prove
monotonic policy improvement.

To inspect the completed results without retraining, run:

```bash
cat /root/ros2_ws/rl_runs/ddqn_10000/training_summary.json
cat /root/ros2_ws/rl_runs/ddqn_50000/training_summary.json
cat /root/ros2_ws/rl_runs/ppo_10000/training_summary.json
cat /root/ros2_ws/rl_runs/ppo_50000/training_summary.json
cat /root/ros2_ws/rl_evaluations/ddqn_50000_vs_random/summary.json
cat /root/ros2_ws/rl_evaluations/ppo_50000_vs_random/summary.json
explorer.exe "$(wslpath -w /root/ros2_ws/rl_runs/ddqn_50000/reward_curve.png)"
explorer.exe "$(wslpath -w /root/ros2_ws/rl_runs/ppo_50000/reward_curve.png)"
explorer.exe "$(wslpath -w /root/ros2_ws/rl_evaluations/ddqn_50000_vs_random/policy_vs_random.png)"
explorer.exe "$(wslpath -w /root/ros2_ws/rl_evaluations/ppo_50000_vs_random/policy_vs_random.png)"
```

The orchestrator `/root/ros2_ws/train_ddqn_ppo_50k.sh` runs these four
training stages and two evaluations sequentially with Gazebo running. It
skips completed run directories, so invoking it again does not retrain the
existing models.

The requested timestep count is a lower bound: DDQN can round up to a multiple
of its four-step rollout length, and PPO can round up to its 250-step rollout.
A fair comparison uses the same
environment profile, episode length, action duration, and test seeds for DQN,
DDQN, PPO, and random.
