#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
algorithm=${1:-}
model=${2:-}
run_id=${3:-}
seed=${4:-42}
minutes=${5:-3}
stochastic_policy=${6:-false}
trial_profile=${ROBOT_TASKS_TRIAL_PROFILE:-policy_only}
if [[ "$algorithm" != ddqn && "$algorithm" != ppo && "$algorithm" != random \
      || -z "$run_id" ]]; then
  echo "Usage: $0 ddqn|ppo|random model.zip|- run_id seed minutes [stochastic]" >&2
  exit 2
fi
if [[ "$algorithm" == random ]]; then
  model=''
elif [[ ! -f "$model" ]]; then
  echo "Model not found: $model" >&2
  exit 2
fi
if [[ "$trial_profile" != policy_only && "$trial_profile" != autonomous_coverage ]]; then
  echo "Invalid ROBOT_TASKS_TRIAL_PROFILE: $trial_profile" >&2
  exit 2
fi
output_dir="$workspace/rl_evaluations/$run_id"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
source /usr/share/gazebo-11/setup.sh
set -u
export TURTLEBOT3_MODEL=burger
turtlebot3_prefix=$(ros2 pkg prefix turtlebot3_gazebo)
turtlebot3_models="$turtlebot3_prefix/share/turtlebot3_gazebo/models"
if [[ ! -d "$turtlebot3_models/turtlebot3_common" ]]; then
  echo "TurtleBot3 Gazebo models unavailable: $turtlebot3_models" >&2
  exit 2
fi
export GAZEBO_MODEL_PATH="$workspace/src/robot_tasks/models:$workspace/gazebo_models:$turtlebot3_models:/root/gazebo_models:/root/.gazebo/models:${GAZEBO_MODEL_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11486}
export GAZEBO_MODEL_DATABASE_URI=
export ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-184}
if [[ ! "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || (( ROS_DOMAIN_ID > 232 )); then
  echo "ROS_DOMAIN_ID must be an integer from 0 to 232: $ROS_DOMAIN_ID" >&2
  exit 2
fi
export ROS_LOG_DIR="$workspace/ros_logs"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"

if [[ "$trial_profile" == autonomous_coverage ]]; then
  world=${ROBOT_TASKS_TRIAL_WORLD:-$workspace/src/robot_tasks/worlds/spaced_objects_ddqn_accessible_v2.world}
  random_policy=novelty
  safety_recovery=true
  global_coverage=${ROBOT_TASKS_GLOBAL_COVERAGE:-true}
  global_coverage_stall_seconds=${ROBOT_TASKS_GLOBAL_COVERAGE_STALL_SECONDS:-1}
  breadcrumb_return=${ROBOT_TASKS_BREADCRUMB_RETURN:-false}
  inspection_sweep_every_m=${ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M:-0.5}
else
  world=${ROBOT_TASKS_TRIAL_WORLD:-$workspace/src/robot_tasks/worlds/spaced_objects_completion.world}
  random_policy=uniform
  safety_recovery=false
  global_coverage=false
  global_coverage_stall_seconds=12
  breadcrumb_return=false
  inspection_sweep_every_m=0
fi
detector=${ROBOT_TASKS_DETECTOR_MODEL:-$workspace/realistic_sim_yolo_runs/safe_stands_epoch4_snapshot.pt}
inference_imgsz=${ROBOT_TASKS_INFERENCE_IMGSZ:-512}
action_duration=${ROBOT_TASKS_ACTION_DURATION:-0.15}
hose_detector="$workspace/realistic_sim_yolo_runs/pedestal_v2_20261102/weights/best.pt"
python3 - "$world" "$GAZEBO_MODEL_PATH" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

world = ET.parse(sys.argv[1]).getroot().find('world')
roots = [Path(item) for item in sys.argv[2].split(':') if item]
missing = sorted({
    uri.removeprefix('model://').split('/')[0]
    for include in world.findall('include')
    for uri in [include.findtext('uri', '').strip()]
    if uri.startswith('model://') and not any(
        (root / uri.removeprefix('model://').split('/')[0]).is_dir()
        for root in roots)
})
if missing:
    raise SystemExit('Gazebo world models unavailable: ' + ', '.join(missing))
PY
mkdir -p "$workspace/rl_evaluations"
setsid ros2 launch robot_tasks realistic_sim.launch.py gui:=false world:="$world" \
  > "$workspace/rl_evaluations/${run_id}_gazebo.log" 2>&1 &
sim_pid=$!
cleanup() {
  kill -INT -- "-$sim_pid" 2>/dev/null || true
  for _ in {1..60}; do
    if ! kill -0 -- "-$sim_pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if kill -0 -- "-$sim_pid" 2>/dev/null; then
    kill -TERM -- "-$sim_pid" 2>/dev/null || true
  fi
  for _ in {1..20}; do
    if ! kill -0 -- "-$sim_pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if kill -0 -- "-$sim_pid" 2>/dev/null; then
    kill -KILL -- "-$sim_pid" 2>/dev/null || true
  fi
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15
if grep -Eq 'Unable to start server|Address already in use|Spawn service failed|already exists|Unable to get local interface addresses|HOME environment variable is missing|gzserver-1.*process has died|Unable to find uri\[model://|Calculated port number is too high' \
    "$workspace/rl_evaluations/${run_id}_gazebo.log"; then
  echo "Gazebo failed to start on $GAZEBO_MASTER_URI" >&2
  exit 2
fi
model_args=()
if [[ -n "$model" ]]; then
  model_args=("model:=$model")
fi
wall_timeout_s=$(python3 - "$minutes" <<'PY'
import math
import sys
print(math.ceil(float(sys.argv[1]) * 60 + 180))
PY
)
launcher=(timeout --signal=INT --kill-after=15s "${wall_timeout_s}s")
if [[ "$minutes" == 0 || "$minutes" == 0.0 ]]; then
  launcher=()
fi
"${launcher[@]}" \
  ros2 launch robot_tasks object_discovery.launch.py \
  algorithm:="$algorithm" random_policy:="$random_policy" \
  environment_profile:=coverage_search safety_recovery:="$safety_recovery" \
  global_coverage:="$global_coverage" \
  global_coverage_stall_seconds:="$global_coverage_stall_seconds" \
  breadcrumb_return:="$breadcrumb_return" \
  inspection_sweep_every_m:="$inspection_sweep_every_m" \
  stochastic_policy:="$stochastic_policy" \
  "${model_args[@]}" max_minutes:="$minutes" \
  action_duration:="$action_duration" seed:="$seed" world:="$world" \
  detector_model:="$detector" inference_imgsz:="$inference_imgsz" \
  hose_fallback_model:="$hose_detector" \
  output_dir:="$output_dir" \
  > "$workspace/rl_evaluations/${run_id}_discovery.log" 2>&1

python3 - "$output_dir" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
with (root / 'steps.csv').open(newline='') as stream:
    actions = Counter(int(row['action']) for row in csv.DictReader(stream))
print(f"{summary['algorithm'].upper()}: "
      f"{summary['matched_objects']}/{summary['target_objects']} objects, "
      f"{summary['unique_visited_cells']} visited cells, "
      f"actions forward/left/right={actions[0]}/{actions[1]}/{actions[2]}")
print(root / 'summary.json')
PY
