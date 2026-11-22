#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
run_id=${1:-ppo_search_50000_current_$(date +%Y%m%d_%H%M%S)}
world=${2:-$workspace/src/robot_tasks/worlds/spaced_objects_ppo_accessible_v1.world}
if [[ -e "$workspace/rl_runs/$run_id" ]]; then
  echo "Training output already exists: $workspace/rl_runs/$run_id" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$workspace/src/robot_tasks/models:${GAZEBO_MODEL_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://127.0.0.1:11487}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}
export ROS_LOG_DIR="$workspace/ros_logs"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"

detector="$workspace/realistic_sim_yolo_runs/safe_stands_epoch4_snapshot.pt"
hose_detector="$workspace/realistic_sim_yolo_runs/pedestal_v2_20261102/weights/best.pt"
mkdir -p "$workspace/rl_training_logs"

ros2 launch robot_tasks realistic_sim.launch.py gui:=false world:="$world" \
  > "$workspace/rl_training_logs/${run_id}_gazebo.log" 2>&1 &
sim_pid=$!
detector_pid=''
cleanup() {
  if [[ -n "$detector_pid" ]]; then
    kill -INT "$detector_pid" 2>/dev/null || true
  fi
  kill -INT "$sim_pid" 2>/dev/null || true
  wait "$detector_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15
ros2 run robot_tasks yolo_node --ros-args \
  -p use_sim_time:=true \
  -p model_path:="$detector" \
  -p hose_fallback_model_path:="$hose_detector" \
  -p hose_fallback_interval_frames:=2 \
  -p person_fallback_model_path:="$workspace/yolo11n.pt" \
  -p person_fallback_interval_frames:=5 \
  -p show_window:=false \
  -p conf_threshold:=0.10 \
  -p pcd_conf_threshold:=0.20 \
  -p pcd_interval_frames:=3 \
  -p inference_imgsz:=512 \
  -p log_detections:=false \
  > "$workspace/rl_training_logs/${run_id}_yolo.log" 2>&1 &
detector_pid=$!

sleep 8
ros2 run robot_tasks train_exploration \
  --algorithm ppo --timesteps 50000 --run-id "$run_id" \
  --environment-profile coverage_search \
  --world "$world" --detector-model "$detector" \
  --action-duration 0.10 --episode-steps 5000 --seed 42 \
  --entropy-coef 0.02 --new-area-reward 10.0 \
  --collision-penalty 3.0 --revisit-penalty 0.5 \
  --turn-penalty 0.3
