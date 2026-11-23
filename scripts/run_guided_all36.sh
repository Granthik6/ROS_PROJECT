#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
algorithm=${1:-}
if [[ "$algorithm" != ddqn && "$algorithm" != ppo && "$algorithm" != random ]]; then
  echo "Usage: $0 ddqn|ppo|random [run_id] [stall_minutes] [gui]" >&2
  exit 2
fi
run_id=${2:-${algorithm}_guided_$(date +%Y%m%d_%H%M%S)}
stall_minutes=${3:-2}
gui=${4:-false}
world="$workspace/src/robot_tasks/worlds/spaced_objects_completion.world"
audit="$workspace/rl_evaluations/completion_all_audit_20261104/report.json"
if [[ "$algorithm" == ddqn ]]; then
  model="$workspace/rl_runs/ddqn_safe_50000/model_final.zip"
elif [[ "$algorithm" == ppo ]]; then
  model="$workspace/rl_runs/ppo_search_10000_quota_20261102/model_final.zip"
else
  model=''
fi
output_dir="$workspace/rl_evaluations/$run_id"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 2
fi
if [[ ! -f "$world" || ! -f "$audit" ]]; then
  echo "Completion world or audited viewpoints are missing" >&2
  exit 2
fi
if [[ "$algorithm" != random && ! -f "$model" ]]; then
  echo "Model is missing: $model" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$workspace/src/robot_tasks/models:${GAZEBO_MODEL_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11486}
export ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-184}
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
  for _ in {1..40}; do
    if ! kill -0 "$sim_pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if kill -0 "$sim_pid" 2>/dev/null; then
    kill -TERM "$sim_pid" 2>/dev/null || true
  fi
  for _ in {1..20}; do
    if ! kill -0 "$sim_pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if kill -0 "$sim_pid" 2>/dev/null; then
    kill -KILL "$sim_pid" 2>/dev/null || true
  fi
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15
if rg -q 'Unable to start server|Address already in use|Spawn service failed|already exists' \
    "$workspace/rl_evaluations/${run_id}_gazebo.log"; then
  echo "Gazebo failed to start on $GAZEBO_MASTER_URI" >&2
  exit 2
fi
model_args=()
if [[ -n "$model" ]]; then
  model_args=("model:=$model")
fi
ros2 launch robot_tasks object_discovery.launch.py \
  algorithm:="$algorithm" environment_profile:=coverage_search \
  safety_recovery:=true global_coverage:=true \
  global_coverage_stall_seconds:=1 inspection_sweep_every_m:=0.5 \
  "${model_args[@]}" random_policy:=novelty \
  max_minutes:=0 action_duration:=0.15 seed:=42 \
  world:="$world" detector_model:="$detector" \
  hose_fallback_model:="$hose_detector" \
  guided_completion_audit_report:="$audit" \
  guided_completion_after_stall_minutes:="$stall_minutes" \
  output_dir:="$output_dir"

python3 - "$output_dir/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    result = json.load(stream)
print(f"Detected before guidance: {result['pre_guidance_matched_objects']}/{result['target_objects']}")
print(f"Detected after guidance: {result['matched_objects']}/{result['target_objects']}")
print(f"Guided additions: {result['guided_completion_added_objects']}")
print(f"Elapsed: {result['elapsed_minutes']:.3f} minutes")
print(f"Status: {result['status']}")
print(f"Results: {sys.argv[1]}")
PY
