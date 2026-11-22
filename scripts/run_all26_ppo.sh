#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_all26_ppo.sh [max_minutes] [run_id] [seed]
workspace=/root/ros2_ws
max_minutes=${1:-12}
run_id=${2:-ppo_26_demo_$(date +%Y%m%d_%H%M%S)}
seed=${3:-42}
model="$workspace/rl_runs/ppo_search_10000_quota_20261102/model_final.zip"

export ROBOT_TASKS_TRIAL_PROFILE=autonomous_coverage
export ROBOT_TASKS_TRIAL_WORLD="$workspace/src/robot_tasks/worlds/spaced_objects_reliable_26_v4.world"
export ROBOT_TASKS_DETECTOR_MODEL="$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt"
export ROBOT_TASKS_INFERENCE_IMGSZ=640
export ROBOT_TASKS_ACTION_DURATION=0.15
export ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M=1
export ROBOT_TASKS_CONFIRM_NEW_DETECTIONS=true
export ROBOT_TASKS_BREADCRUMB_RETURN=true
export ROBOT_TASKS_GLOBAL_COVERAGE=true
export ROBOT_TASKS_PLANNER_BOUNDS_ENABLED=false
export ROBOT_TASKS_SECTOR_BALANCE_WEIGHT=0
export ROBOT_TASKS_ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-172}
export ROBOT_TASKS_GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11772}

bash "$workspace/scripts/run_policy_only_trial.sh" ppo \
  "$model" "$run_id" "$seed" "$max_minutes" false

python3 - "$workspace/rl_evaluations/$run_id/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text(encoding='utf-8'))
count = result['matched_objects']
target = result['target_objects']
print(f'Verified PPO result: {count}/{target} in '
      f"{result['elapsed_minutes']:.3f} minutes")
print('Missing:', ', '.join(result['missing_objects']) or 'none')
print('Summary:', path)
if (count, target) != (26, 26):
    raise SystemExit(1)
PY
