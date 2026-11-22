# YOLO distance-confidence experiment

The detector publishes one `std_msgs/msg/Float32` value in the range 0.0 to
1.0 on `/yolo/confidence` for every accepted detection. It also publishes the
measured range on `/yolo/distance` when camera depth is valid. The existing
`/yolo/detections` JSON topic carries the class, confidence, distance and 3D
coordinates together and is used by the CSV logger.

Build once after changing the package:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_tasks --symlink-install
```

Run each numbered section in a separate terminal.

1. Start Gazebo and the robot:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/root/ros2_ws/src/robot_tasks/models:${GAZEBO_MODEL_PATH}"
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch robot_tasks realistic_sim.launch.py
```

2. Start keyboard teleoperation:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

3. Start YOLO. Set `show_window:=false` when no desktop display is available:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run robot_tasks yolo_node --ros-args \
  -p show_window:=true \
  -p conf_threshold:=0.10
```

4. Check the confidence output and full detection record:

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic echo /yolo/confidence
```

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic echo /yolo/detections
```

5. Record paired distance-confidence samples. Replace `person` with the exact
class shown by `/yolo/detections`, or remove the `target_class` parameter to
record every class:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run robot_tasks confidence_logger --ros-args \
  -p target_class:=person \
  -p output_csv:=/root/ros2_ws/results/distance_confidence.csv \
  -p min_interval_s:=0.25
```

Drive the robot toward and away from the selected object. Collect samples
across the distance range, keeping the object visible. Stop the logger with
Ctrl-C when finished.

6. Generate the research plot and binned numerical results:

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run robot_tasks plot_distance_confidence \
  --input /root/ros2_ws/results/distance_confidence.csv \
  --output /root/ros2_ws/results/distance_vs_confidence.png \
  --class-name person \
  --bin-width 0.5
```

The plot command creates `distance_vs_confidence.png` and
`distance_vs_confidence_binned.csv`. Use the same class for collection and
plotting. Every CSV row is one detection with its confidence, Euclidean camera
range, camera XYZ coordinates, bounding box, and timestamps.
