# 26-object Gazebo detection world

The validated world is `src/robot_tasks/worlds/spaced_objects_reliable_26_v4.world`.
It contains 26 scored target models, including `bottle_pcd_object` and
`doll_pcd_object`. The cabinet is placed in an open east-side position.

Run a fresh DDQN trial from `/root/ros2_ws`:

```bash
cd /root/ros2_ws
bash scripts/run_all26_ddqn.sh 12
```

The command starts Gazebo, the saved YOLO detector, and the saved DDQN policy.
It uses LiDAR, SLAM, and camera observations to choose actions; known Gazebo
object positions are used only by the evaluator to verify detections. The
runner prints the verified count and exits successfully only at 26/26. The
result is saved under `rl_evaluations/<run_id>/summary.json`, with individual
matches in `world_coverage_report.json`. Pass a unique run ID as the second
argument if desired. A first argument of `0` removes the time limit; stop it
with Ctrl+C if the robot stalls.

The fresh validation run `ddqn_reliable26_v4_20261120` matched all 26 of 26
targets, including the two PCD models. The recorded time to all targets was
2.9 minutes. This is one observed run, so a new run may take longer or miss
objects; check its own `summary.json` before reporting success.

## Seventeen DDQN trials

The saved series `rl_evaluations/ddqn_26_17_20261120` counts the verified
seed-42 run above as trial 1. It uses seeds 43 through 58 for trials 2 through
17. Each run has the same 26-object world, detector, saved DDQN model, and
12-minute limit. Its `results.csv`, `summary.json`, and `objects_per_trial.png`
are rebuilt from actual per-run summaries.

To resume the series, run:

```bash
cd /root/ros2_ws
bash scripts/run_ddqn_26_17_trials.sh 12 ddqn_26_17_20261120
```

The runner skips trials with saved summaries and starts at the next missing
trial. It retains incomplete or failed runs and reports their actual counts.

## PPO check

The PPO model `rl_runs/ppo_search_10000_quota_20261102/model_final.zip` was
tested in the same world with the same detector, action duration, coverage
assistance, and 12-minute limit. The saved run
`rl_evaluations/ppo_reliable26_seed42_20261121` reached 26/26 in 1.772
minutes, including both PCD objects. To repeat it:

```bash
cd /root/ros2_ws
bash scripts/run_all26_ppo.sh 12
```

This is a single PPO trial. A new trial can vary; verify its saved summary.

To run the corresponding 17-trial PPO series, with that verified seed-42 run
counted as trial 1, run:

```bash
cd /root/ros2_ws
bash scripts/run_ppo_26_17_trials.sh 12 ppo_26_17_20261121
```

The runner starts with trial 2 (seed 43) and continues through trial 17
(seed 58). Its real per-run results appear in
`rl_evaluations/ppo_26_17_20261121/results.csv`, `summary.json`, and
`objects_per_trial.png`. Repeating the command resumes any unfinished trials.

## Uniform random exploration

The random baseline uses the same 26-object world, detector, action duration,
and 12-minute limit. It chooses forward, left, or right uniformly at random.
It has no map coverage planner, inspection sweep, target-position guidance,
or safety action override. Run all 17 independent seeds (42 through 58) with:

```bash
cd /root/ros2_ws
bash scripts/run_random_26_17_trials.sh 12 random_26_17_20261122
```

The runner saves real per-trial counts and times in
`rl_evaluations/random_26_17_20261122/results.csv`, `summary.json`, and
`objects_per_trial.png`. Repeating the command resumes incomplete trials.
