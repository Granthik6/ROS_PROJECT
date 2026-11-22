# SLAM, detection, and random-policy baseline

This experiment runs SLAM Toolbox and YOLO at the same time, drives the robot
with a seeded obstacle-avoiding random policy, marks spatially deduplicated
objects in RViz, and records the baseline result. Each accepted detection is
transformed into the SLAM `map` frame. Detections of the same class within the
association radius are treated as observations of one object. The default
`association_radius_m:=1.5` tolerates depth-estimation noise; lower it when the
world contains separate objects of the same class less than 1.5 m apart.
An object must be observed twice by default (`min_confirmations:=2`) before it
is counted or drawn, which prevents isolated false detections from inflating
the baseline.

Build once:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_tasks --symlink-install
```

Run the simulation in terminal 1:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

Run a three-minute baseline in terminal 2:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_tasks slam_detection_baseline.launch.py \
  duration_sec:=180 \
  run_id:=random_baseline_01 \
  random_seed:=42 \
  min_confidence:=0.25 \
  rviz:=true \
  show_window:=true
```

The timer starts only after the map, laser scan, odometry, and YOLO publisher
are ready. The launch ends automatically when the timed run finishes.

For a headless run:

```bash
ros2 launch robot_tasks slam_detection_baseline.launch.py \
  duration_sec:=180 run_id:=random_baseline_headless \
  rviz:=false show_window:=false
```

For manual exploration with keyboard teleoperation, set `drive_robot:=false`
and run `turtlebot3_teleop teleop_keyboard` in another sourced terminal.

Each run creates a directory under `/root/ros2_ws/baseline_runs`. It contains:

- `map_with_objects.png`: SLAM map, robot path, object positions and labels.
- `slam_map.pgm` and `slam_map.yaml`: reusable occupancy map.
- `observations.csv`: every accepted positioned observation.
- `unique_objects.csv`: the deduplicated object database.
- `summary.json`: machine-readable baseline metrics.
- `summary.md`: the statement “Random policy found X objects in Y minutes” and
  its supporting metrics.

All runs are also appended to `/root/ros2_ws/baseline_results.csv`. The main
baseline metric is `unique_objects_found`. Supporting metrics include explored
map coverage, distance traveled, detections, unique classes, time to first
object, objects per minute, and mean confidence.
