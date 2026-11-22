#!/usr/bin/env bash
set -euo pipefail

workspace=/root/ros2_ws
run_id=${1:-ddqn_36_candidate_$(date +%Y%m%d_%H%M%S)}
seed=${2:-42}
minutes=${3:-20}
world=${4:-$workspace/src/robot_tasks/worlds/spaced_objects_accessible_v13.world}
if [[ ! -f "$world" ]]; then
  echo "World not found: $world" >&2
  exit 2
fi

export ROBOT_TASKS_BREADCRUMB_RETURN=true
export ROBOT_TASKS_TRIAL_PROFILE=autonomous_coverage
export ROBOT_TASKS_TRIAL_WORLD="$world"
export ROBOT_TASKS_DETECTOR_MODEL=${ROBOT_TASKS_DETECTOR_MODEL:-$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt}
export ROBOT_TASKS_INFERENCE_IMGSZ=${ROBOT_TASKS_INFERENCE_IMGSZ:-640}
export ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M=${ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M:-2}
export ROBOT_TASKS_PLANNER_BOUNDS_ENABLED=${ROBOT_TASKS_PLANNER_BOUNDS_ENABLED:-false}
export ROBOT_TASKS_SECTOR_BALANCE_WEIGHT=${ROBOT_TASKS_SECTOR_BALANCE_WEIGHT:-0}
export ROBOT_TASKS_ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-131}
export ROBOT_TASKS_GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11694}
export ROBOT_TASKS_ACTION_DURATION=${ROBOT_TASKS_ACTION_DURATION:-0.10}
model=${ROBOT_TASKS_DDQN_MODEL:-$workspace/rl_runs/ddqn_v13_balanced_50000/model_final.zip}
stochastic_policy=${ROBOT_TASKS_STOCHASTIC_POLICY:-true}
if [[ ! -f "$model" ]]; then
  echo "DDQN model not found: $model" >&2
  exit 2
fi

bash "$workspace/scripts/run_policy_only_trial.sh" ddqn \
  "$model" \
  "$run_id" "$seed" "$minutes" "$stochastic_policy"

python3 - "$workspace/rl_evaluations/$run_id/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
count = summary['matched_objects']
target = summary['target_objects']
print(f"Verified DDQN result: {count}/{target} at "
      f"{summary['elapsed_minutes']:.3f} minutes")
print('Missing:', ', '.join(summary['missing_objects']) or 'none')
print('Summary:', path)
if count != target:
    raise SystemExit(1)
PY
