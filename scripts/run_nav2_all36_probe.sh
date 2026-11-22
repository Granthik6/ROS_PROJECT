#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
run_id=${1:-nav2_all36_probe_$(date +%Y%m%d_%H%M%S)}
duration_sec=${2:-600}
world="$workspace/src/robot_tasks/worlds/spaced_objects_accessible_v7.world"
detector="$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt"
if [[ ! -s "$detector" || ! -f "$world" ]]; then
  echo "World or detector weights missing" >&2
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
export GAZEBO_MASTER_URI=http://127.0.0.1:11687
export ROS_DOMAIN_ID=124
export ROS_LOG_DIR="$workspace/ros_logs"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"

mkdir -p "$workspace/rl_evaluations"
sim_log="$workspace/rl_evaluations/${run_id}_gazebo.log"
experiment_log="$workspace/rl_evaluations/${run_id}_nav2.log"
activation_log="$workspace/rl_evaluations/${run_id}_activation.log"
setsid ros2 launch robot_tasks realistic_sim.launch.py gui:=false world:="$world" \
  > "$sim_log" 2>&1 &
sim_pid=$!
cleanup() {
  if [[ -n "${activation_pid:-}" ]]; then
    kill "$activation_pid" 2>/dev/null || true
    wait "$activation_pid" 2>/dev/null || true
  fi
  kill -INT -- "-$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15
if grep -Eq 'Unable to start server|Address already in use|Spawn service failed|gzserver-1.*process has died|Calculated port number is too high' "$sim_log"; then
  echo "Gazebo startup failed; inspect $sim_log" >&2
  exit 2
fi

bash "$workspace/scripts/activate_nav2_nodes.sh" > "$activation_log" 2>&1 &
activation_pid=$!
ros2 launch robot_tasks slam_detection_baseline.launch.py \
  duration_sec:="$duration_sec" run_id:="$run_id" \
  world_file:="$world" model_path:="$detector" inference_imgsz:=640 \
  exploration_policy:=nav2 use_nav2:=true \
  nav2_autostart:=false manual_nav2_activation:=true \
  stop_when_all_world_objects:=true drive_robot:=true \
  min_confidence:=0.25 min_confirmations:=2 \
  world_match_radius_m:=1.25 \
  rviz:=false show_window:=false \
  > "$experiment_log" 2>&1

summary="$workspace/baseline_runs/$run_id/summary.json"
if [[ ! -s "$summary" ]]; then
  echo "Nav2 diagnostic ended without a summary; inspect $experiment_log and $activation_log" >&2
  exit 1
fi
echo "Nav2 result: $summary"
