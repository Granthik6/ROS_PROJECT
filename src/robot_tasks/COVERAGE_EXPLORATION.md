# Autonomous coverage and the all-object check

Use the `nav2` policy for an autonomous demonstration. It chooses frontiers
from the live SLAM occupancy grid, sends them to Nav2, and counts each 0.5 m
floor tile actually entered. It prefers unvisited tiles and larger unknown
regions. It sends short waypoints along distant frontier paths and tries a
LiDAR-clear goal every fifth navigation goal. It may transit across an old
tile to reach a new area. Nav2 handles obstacle avoidance; a camera turn
scans for objects after progress. If Nav2 rejects a LiDAR-clear probe because
the global SLAM map has no route, the robot checks that same direction again
and uses Nav2's local-costmap collision-checked `DriveOnHeading` action for a
short move. It also tries this recovery if no new floor tile is entered for
45 seconds. A backup recovery tries to release the robot when there is no
accessible frontier. After 90 seconds without a new tile, it tries a distant
unvisited tile in known-free SLAM space, even if a nearby frontier is still
available. This broad search uses the map and tile history, not object poses.
Gazebo's known object positions are used **only after detection** to score
the result and decide whether the all-object stopping criterion was met.
The evaluator uses Gazebo's robot pose to align odometry-frame detections
with the world coordinates in the scene file. Neither those positions nor
the scoring result supplies a navigation goal; the driving policy sees only
SLAM, LiDAR, odometry, and tile visit history.

The YOLO checkpoint in the example below is the locally trained 39-class
`realistic_sim_yolo11n_all_objects` model. Its recorded peak validation
mAP50-95 is about 0.461. This is a model-selection clue, not a guarantee
that every object in the room will be recognized. The autonomous runner also
uses the local COCO YOLO checkpoint every fifth camera frame for the `person`
class, because the custom model missed the standing person in a prior
near-complete trial.

The older `coverage` policy below uses a direct Twist controller. It is kept
for comparison; Nav2 is the recommended autonomous policy.

The Nav2 planner also prefers frontiers with fewer visited tiles nearby. If
one short waypoint repeatedly fails without the robot moving, that tile is
excluded from later planned routes so a different corridor can be tried.

The `coverage` exploration policy uses the live SLAM occupancy grid to choose
reachable frontiers. It records each 0.5 m floor tile the robot enters and
prefers frontiers in unvisited tiles. It may cross a visited tile to reach a
new area; forbidding revisits entirely can trap the robot behind a doorway.
Every three new tiles it rotates once to give the camera a view of objects
beside or behind its travel direction. LiDAR clearance and backup behavior
remain active. If a goal makes no progress, the controller tries another
goal and a short escape maneuver. This is an experimental controller and
does not guarantee that every tile or object will be reached.

Multiple camera observations of the same class near the same map position
are grouped. The `world_coverage_report.json` then compares confirmed
detections one-to-one with the 34 model instances in
`worlds/realistic_room.world` by class and horizontal position. The default
matching radius is 1.25 m. A raw detection count is **not** evidence that
every world object was found. The report lists each matched and missing
object. This evaluation uses the simulation's known model positions; the
navigation policy itself does not receive these positions.

Build once:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_tasks --symlink-install
source install/setup.bash
```

Start Gazebo in terminal 1:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

For a headless run, append `gui:=false` to the Gazebo launch command.

Restart Gazebo before each new trial so the robot begins at the initial pose;
running another baseline launch against the same Gazebo process leaves it at
the previous trial's ending pose.

Start one coverage run in terminal 2. The 3600 seconds below is an experiment
cap, not a promised completion time. The launch stops early if all 34 world
instances are matched. Do not use keyboard teleoperation at the same time.

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_tasks slam_detection_baseline.launch.py \
  exploration_policy:=nav2 \
  use_nav2:=true \
  model_path:=/root/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n_all_objects/weights/best.pt \
  person_fallback_model_path:=/root/ros2_ws/yolo11n.pt \
  tile_size_m:=0.5 \
  min_confidence:=0.15 \
  yolo_conf_threshold:=0.02 \
  class_confidence_overrides:=saucepan:0.03,hokuyo:0.03,postbox:0.02 \
  min_confirmations:=2 \
  frontier_gain_weight_m:=3 \
  frontier_visit_penalty_m:=0.4 \
  distant_goal_every:=6 \
  stop_when_all_world_objects:=true \
  world_match_radius_m:=1.25 \
  duration_sec:=3600 \
  run_id:=autonomous_all_objects \
  rviz:=true \
  show_window:=true
```

The launch creates `/root/ros2_ws/baseline_runs/autonomous_all_objects/` (or a
numbered suffix if the name already exists). Read:

```bash
cat /root/ros2_ws/baseline_runs/autonomous_all_objects/summary.json
ros2 run robot_tasks evaluate_world_coverage \
  --run-dir /root/ros2_ws/baseline_runs/autonomous_all_objects
explorer.exe "$(wslpath -w /root/ros2_ws/baseline_runs/autonomous_all_objects/map_with_objects.png)"
```

`summary.json` includes the motion policy, visited tiles, tile revisits,
camera sweeps, Nav2 goal outcomes, matched world objects, and
`time_to_all_world_objects_s` when
all are matched. `world_scoring_samples` counts detections with a usable
Gazebo-world alignment. `world_unique_objects.csv` contains the confirmed
world-coordinate clusters used by the scorer. `unique_objects.csv` and
`map_with_objects.png` remain in SLAM's map frame for visualization.
`world_coverage_report.json` includes a per-instance
`detected` field and lists unmatched detections. `robot_path.csv` records the
SLAM trajectory for diagnosing where exploration stalled;
`world_robot_path.csv` records the evaluation-only Gazebo robot trajectory.
`world_discovery_curve.png` shows matched world objects against elapsed time.
`missing_object_audit.csv` lists each unmatched model, the robot's closest
approach, and same-class observations near its location. These are evaluation
figures; they are never sent to the navigation policy.
A timeout is a partial result;
it must not be presented as full detection. The score depends on both YOLO
classification and depth/SLAM localization accuracy. Some small or occluded
objects may remain undetected even after the reachable map is explored.

Measured trials so far have **not** achieved complete detection. A fresh
45-second run (`coverage_fresh_45s`) matched 12/34 world objects while
visiting eight 0.5 m tiles. A ten-minute run (`coverage_trial_600s`) matched
14/34 despite 26 raw confirmed locations and 13 visited tiles. A later
three-minute goal-selection test (`coverage_v2_180s`) matched 6/34. The robot
can get stuck against furniture; a one-minute escape retest from such a pose
traveled only 0.01 m. These trials are separate simulator starts and are not
a controlled policy ranking. The first Nav2 trial (`nav2_test_120s`) matched
12/34 in two minutes, and `nav2_v2_180s` matched 14/34 in three minutes.
With the locally trained all-objects YOLO checkpoint,
`nav2_recovery_240s` matched 19/34 in four minutes. A 900-second run with a
0.15 confidence acceptance threshold (`nav2_long_900s`) matched **24/34**,
traveled 34.985 m, and visited 41 tiles. Its 83 confirmed grouped detections
include duplicates and mislocalized detections; they are not 83 world objects.
The run still missed objects near room edges and some small objects. These
measurements do **not** establish full 34/34 discovery. The information-gain
and occasional distant-frontier policy above was added after the 900-second
run; that run used the previous nearest-frontier policy. A second 900-second
run with the information-gain and distant-frontier policy (`nav2_gain_900s`)
matched **25/34** and traveled 40.901 m. It visited 39 tiles but had 12
failed navigation goals. The nine unmatched instances are listed in its
`world_coverage_report.json`. A subsequent 900-second run using short waypoints and
LiDAR goal probes (`nav2_short_probe_900s`) traveled 59.634 m and visited
64 tiles, but matched only **22/34**; extra motion did not improve detection
recall. A 30-minute trial with local-costmap drive recovery
(`nav2_targeted_probe_1800s`) matched **24/34**, traveled 55.126 m, and
visited 40 tiles. Its path never crossed north of y = -1.098 m; ten objects
remained unmatched. The larger-area sensor probe and stagnant-tile escape
were added after that trial. A following 10-minute trial with larger sensor
probes (`nav2_escape_600s`) matched **20/34** and stalled near the western
barrier. A 15-minute reroute trial (`nav2_reroute_900s_2`) crossed into the
northern and eastern areas and visited 79 tiles, but the original map-frame
evaluator matched only **25/34**. Its audit showed a real cabinet detection
placed more than 1 m away in the drifting SLAM frame. The world-coordinate
scorer and less-visited-region preference above were added after that run.
The first 30-minute run using world-coordinate scoring and the visit-density
preference (`nav2_aligned_1800s`) matched **32/34** placed objects, traveled
91.098 m, and visited 139 tiles. The two misses were `hokuyo_object` and
`saucepan_object`. The robot came within 0.332 m of the Hokuyo but obtained
only one accepted sighting, below the two-sighting rule. A saved camera frame
shows the saucepan; the detector scored it near 0.05, below both the 0.10
YOLO output threshold and the 0.15 experiment acceptance threshold. A
follow-up test can keep 0.15 for other classes and use a lower threshold for
those two classes:

```bash
ros2 launch robot_tasks slam_detection_baseline.launch.py \
  exploration_policy:=nav2 use_nav2:=true \
  model_path:=/root/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n_all_objects/weights/best.pt \
  min_confidence:=0.15 yolo_conf_threshold:=0.03 \
  class_confidence_overrides:=saucepan:0.03,hokuyo:0.03 \
  min_confirmations:=2 frontier_visit_penalty_m:=0.4 \
  stop_when_all_world_objects:=true duration_sec:=1800 \
  run_id:=autonomous_low_confidence rviz:=true show_window:=true
```

Use this setting as an experiment, not a guaranteed result: lower thresholds
may admit false detections, so inspect the per-object evidence and camera
views when presenting the result.
The first low-threshold run (`nav2_lowconf_verified_1800s`) was interrupted
when Gazebo crashed; rescoring its partial observations matched **32/34**.
The 30-minute LiDAR-escape run (`nav2_lidar_escape_1800s`) completed with
**31/34**, 99.901 m traveled, and 115 visited tiles. It did not reach within
3 m of the car wheel, saucepan, or fire hose, so those were coverage misses.
The following unblocked-map run (`nav2_unblocked_1800s`) completed with
**33/34**, 73.376 m traveled, and 126 visited tiles. It matched the car wheel,
saucepan, and fire hose; only `person_standing_main` remained. The robot came
within 0.896 m of that person, but YOLO produced no accepted person
observation near it. This is the strongest completed verified result so far,
**not** full 34/34 detection. Later interrupted trials matched 26/34 and
13/34; the latter showed a repeated local-goal loop that motivated the
distant unvisited-tile recovery.

With the COCO person fallback active every fifth frame, a later autonomous
trial (`nav2_ready_fast_3600s`) confirmed the standing person at 123.377 s,
with 0.8751 confidence and 0.139 m position error. That trial found **21/34**
in 1399.979 s before it was interrupted after navigation stalled at a narrow
obstacle. Combining its confirmed observations with the earlier 33/34 run
matches **all 34/34 world instances across two independent autonomous runs**.
The last missing object was confirmed after 1923.455 s of combined experiment
time, excluding simulator restart and setup. This is **not** a 34/34 result in
one exploration run. The reproducible per-object audit is generated with:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run robot_tasks aggregate_world_coverage \
  --runs baseline_runs/nav2_unblocked_1800s \
         baseline_runs/nav2_ready_fast_3600s \
  --output-dir baseline_runs/combined_34_object_audit
cat baseline_runs/combined_34_object_audit/combined_world_coverage.json
column -s, -t < baseline_runs/combined_34_object_audit/combined_world_objects.csv
```

The combined audit uses the world only to score previously recorded detections.
`combined_world_objects.csv` identifies the source run for each matched object.
The JSON explicitly separates `all_objects_detected_across_runs` from
`all_objects_detected_in_one_run`.

The current autonomous runner is:

```bash
cd /root/ros2_ws
bash run_autonomous_all_objects_trial.sh autonomous_all_objects_3600s 3600
cat baseline_runs/autonomous_all_objects_3600s/summary.json
cat baseline_runs/autonomous_all_objects_3600s/missing_object_audit.csv
```

It runs headless Gazebo, SLAM, Nav2, and YOLO, then stops at 34 matched objects
or the time limit. It uses 45-second periodic camera turns even when no new
tile is reached. The runner writes ROS and Gazebo logs under
`baseline_runs/<run_id>_launcher/`. The object coordinates in the world file
are used only for the evaluation report and early-stop condition, never to
choose motion goals.
`summary.json` retains the legacy `unique_objects_found`
field, but that field counts confirmed detection *clusters*, which can
include false or duplicate locations. Use `world_objects_matched` for the
34-instance result. The earlier `nav2_gain_900s` audit shows that the robot stayed at least
2.234 m from each of the nine missed objects (and over 5 m from several);
most misses had no accepted same-class observation near the object.

To score an older run with the same simulated world, run:

```bash
ros2 run robot_tasks evaluate_world_coverage \
  --run-dir /root/ros2_ws/baseline_runs/random_baseline_180s
```
