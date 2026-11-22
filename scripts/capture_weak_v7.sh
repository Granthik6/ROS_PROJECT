#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
world="$workspace/src/robot_tasks/worlds/spaced_objects_accessible_v7.world"
targets="$workspace/dataset_tools/weak_v7_targets_20261116.json"
output="$workspace/realistic_sim_yolo_dataset_weak_v7_20261116"
if [[ -e "$output" ]]; then
  echo "Capture output already exists: $output" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
source /usr/share/gazebo-11/setup.sh
set -u
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$workspace/src/robot_tasks/models:/root/gazebo_models:/root/.gazebo/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MASTER_URI=http://127.0.0.1:11686
export ROS_DOMAIN_ID=123
export ROS_LOG_DIR="$workspace/ros_logs"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"

log="$workspace/rl_training_logs/weak_v7_capture_gazebo.log"
mkdir -p "$workspace/rl_training_logs"
setsid ros2 launch robot_tasks realistic_sim.launch.py gui:=false world:="$world" \
  > "$log" 2>&1 &
sim_pid=$!
cleanup() {
  kill -INT -- "-$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15
if grep -Eq 'Unable to start server|Address already in use|Spawn service failed|gzserver-1.*process has died|Calculated port number is too high' "$log"; then
  echo "Gazebo startup failed; inspect $log" >&2
  exit 2
fi
cd "$workspace"
python3 dataset_tools/capture_spaced_hard_classes.py \
  --world "$world" --missing-list "$targets" --output-root "$output" \
  --train-per-class 30 --val-per-class 6 --test-per-class 6
echo "Captured dataset: $output"
