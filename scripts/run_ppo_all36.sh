#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
run_id=${1:-ppo_accessible_v4_$(date +%Y%m%d_%H%M%S)}
gui=${2:-false}
world=${3:-$workspace/src/robot_tasks/worlds/spaced_objects_ddqn_accessible_v4.world}
model=${4:-$workspace/rl_runs/ppo_search_10000_quota_20261102/model_final.zip}
max_minutes=${5:-0}
stochastic_policy=${6:-false}
output_dir="$workspace/rl_evaluations/$run_id"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$workspace/src/robot_tasks/models:${GAZEBO_MODEL_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://127.0.0.1:11486}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-82}
export ROS_LOG_DIR="$workspace/ros_logs"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"

detector="$workspace/realistic_sim_yolo_runs/safe_stands_epoch4_snapshot.pt"
hose_detector="$workspace/realistic_sim_yolo_runs/pedestal_v2_20261102/weights/best.pt"
mkdir -p "$workspace/rl_evaluations"

ros2 launch robot_tasks realistic_sim.launch.py \
  gui:="$gui" world:="$world" \
  > "$workspace/rl_evaluations/${run_id}_gazebo.log" 2>&1 &
sim_pid=$!
cleanup() {
  kill -INT "$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15
ros2 launch robot_tasks object_discovery.launch.py \
  algorithm:=ppo environment_profile:=coverage_search safety_recovery:=true \
  global_coverage:=true global_coverage_stall_seconds:=1 \
  inspection_sweep_every_m:=0.5 \
  model:="$model" stochastic_policy:="$stochastic_policy" \
  max_minutes:="$max_minutes" action_duration:=0.15 seed:=42 \
  world:="$world" detector_model:="$detector" \
  hose_fallback_model:="$hose_detector" output_dir:="$output_dir"

python3 - "$output_dir/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    result = json.load(stream)
print(f"Detected: {result['matched_objects']}/{result['target_objects']}")
print(f"Elapsed: {result['elapsed_minutes']:.3f} minutes")
print(f"Status: {result['status']}")
print(f"Results: {sys.argv[1]}")
PY
