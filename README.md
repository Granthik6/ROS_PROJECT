# Vision-based robotic object discovery (ROS 2 + Gazebo)

This repository contains the **final 26-object simulation** used to evaluate three exploration configurations: Double Deep Q-Network (DDQN), Proximal Policy Optimization (PPO), and uniform Random exploration. The robot is a simulated TurtleBot3 Burger. An RGB-depth camera, LiDAR, odometry, ROS 2, Gazebo Classic, a fine-tuned YOLO detector, and world-frame object matching form the experimental pipeline.

The final experiment consists of **17 saved trials per configuration (51 total)**, with seeds 42–58 and a 12-minute limit per trial. The archived trial records all use `src/robot_tasks/worlds/spaced_objects_reliable_26_v4.world`. The 26 scored instances include both `bottle_pcd_object` and `doll_pcd_object`; their full inventory is in `26_OBJECT_SHOWCASE.md` and `paper/main.tex`.

The final world has exactly 26 scored object instances. The detector can recognize labels outside this target inventory, but those labels are not additional objects in the evaluated world.

## Results from the archived 51 trials

| Metric | DDQN + assistance | PPO + assistance | Random |
|---|---:|---:|---:|
| Trials | 17 | 17 | 17 |
| Mean matched objects (of 26) | 25.41 | 25.24 | 19.35 |
| Full 26/26 completions | 11 | 9 | 1 |
| Mean elapsed trial time (min) | 6.19 | 7.26 | 11.70 |
| Mean time among successful trials (min) | 3.02 | 3.05 | 6.94 |

**Interpretation:** DDQN and PPO were evaluated with SLAM-based global coverage, inspection turns, confirmation, and safety recovery. Random was evaluated without those routines. The table compares the *complete configurations*, not isolated algorithm quality. A new run may find fewer than 26 targets; the commands below do not guarantee 26/26. Object counts are class-and-position matches against the 26 known SDF instances used only by the evaluator.

## Repository contents

- `src/robot_tasks/`: ROS 2 package, nodes, launch files, SDF worlds, and local PCD models.
- `gazebo_models/`: the 27 external Gazebo model directories needed by the final world, with upstream attribution and license.
- `scripts/`: trial runners, report scripts, and supporting tools. The `run_all26_*` scripts are the easiest entry points.
- `rl_runs/`: final DDQN and PPO policy checkpoints.
- `realistic_sim_yolo_runs/`, `pcd_yolo_runs/`, `yolo11n.pt`: the four detector weights referenced by the final launch path.
- `rl_evaluations/`: the three 17-trial summaries and all 51 original per-trial result directories, including `summary.json`, object-match records, and step/visit histories.
- `paper/`: IEEE-style `main.tex` and the figures used in the manuscript. The Gazebo image is an illustration captured from the same 26-object SDF world, not a trial result.

Large training datasets, replay buffers, old candidate checkpoints, and ROS build output are not required for evaluating the final saved policies and are not included. Some older candidate worlds and utilities remain in the ROS package; the final 26-object path above identifies the experiment reported in the paper.

## Setup (tested on Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11)

The existing launch scripts expect the workspace at `/root/ros2_ws` and the ROS 2 installation at `/opt/ros/humble`. Install ROS 2 Humble, Gazebo ROS integration, TurtleBot3 Gazebo models, SLAM Toolbox, `colcon`, and the ROS dependencies declared in `src/robot_tasks/package.xml` before building. The Python package requirements are in `requirements.txt`. A system with a different ROS distribution or workspace path needs corresponding script-path changes.

```bash
git clone https://github.com/Granthik6/ROS_PROJECT.git /root/ros2_ws
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
python3 -m pip install -r requirements.txt
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select robot_tasks --symlink-install
source install/setup.bash
```

The final runner adds `src/robot_tasks/models` and `gazebo_models` to `GAZEBO_MODEL_PATH`, so the included final world assets can be resolved without a separate model download. The saved policy and detector paths in the runners are relative to the fixed `/root/ros2_ws` workspace.

## Run a new 26-object trial

Run one command at a time from `/root/ros2_ws`. The first argument is the maximum trial duration in minutes, the second is a **new** run ID, and the third is the random seed. Gazebo runs headlessly in these scripts. Each command writes `rl_evaluations/<run_id>/summary.json` and the per-trial records. It exits with a nonzero code if the run ends without 26/26; a saved summary may still be present.

```bash
cd /root/ros2_ws
bash scripts/run_all26_ddqn.sh 12 "ddqn_demo_$(date +%s)" 42
```

```bash
cd /root/ros2_ws
bash scripts/run_all26_ppo.sh 12 "ppo_demo_$(date +%s)" 42
```

```bash
cd /root/ros2_ws
bash scripts/run_all26_random.sh 12 "random_demo_$(date +%s)" 42
```

The DDQN/PPO runs use saved deterministic policies plus the assistance described above. Random samples forward, left, or right uniformly. The scripts set different ROS domain IDs and Gazebo ports, but sequential execution is simplest.

## Inspect the saved experiments

```bash
cat rl_evaluations/ddqn_26_17_20261120/summary.json
cat rl_evaluations/ppo_26_17_20261121/summary.json
cat rl_evaluations/random_26_17_20261122/summary.json
```

The corresponding `results.csv` files contain trial, seed, run ID, matched count, elapsed time, reward, and termination status. Each row points to a per-run directory with the raw summary, `world_coverage_report.json`, action logs, and visit history. The three `run_*_26_17_trials.sh` scripts resume these archived series and skip trials with saved summaries; running them against this repository does **not** create 51 new trials.

## Paper and model attribution

Open `paper/main.tex` together with `paper/figures/` in Overleaf. The manuscript reports the final 26-object experiment and distinguishes coverage-assisted DDQN/PPO from unassisted Random. No PDF is stored here.

The models in `gazebo_models/` originate from the [OSRF Gazebo model database](https://github.com/osrf/gazebo_models) and retain their individual `model.config` author metadata. Their upstream license is included at `gazebo_models/LICENSE`. The fine-tuned detector weights were produced from Ultralytics YOLO models; consult the upstream [Ultralytics licensing terms](https://www.ultralytics.com/license) for reuse. The code and paper are project work; no separate code license has been declared in this repository.
