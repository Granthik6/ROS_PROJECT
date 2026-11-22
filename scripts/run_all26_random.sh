#!/usr/bin/env bash
set -euo pipefail

# Uniform random forward/left/right exploration in the fixed 26-object world.
# Usage: bash scripts/run_all26_random.sh [max_minutes] [run_id] [seed]
workspace=/root/ros2_ws
max_minutes=${1:-12}
run_id=${2:-random_26_demo_$(date +%Y%m%d_%H%M%S)}
seed=${3:-42}

export ROBOT_TASKS_TRIAL_PROFILE=policy_only
export ROBOT_TASKS_TRIAL_WORLD="$workspace/src/robot_tasks/worlds/spaced_objects_reliable_26_v4.world"
export ROBOT_TASKS_DETECTOR_MODEL="$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt"
export ROBOT_TASKS_INFERENCE_IMGSZ=640
export ROBOT_TASKS_ACTION_DURATION=0.15
export ROBOT_TASKS_CONFIRM_NEW_DETECTIONS=false
export ROBOT_TASKS_ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-173}
export ROBOT_TASKS_GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11773}

# policy_only selects uniform random actions, with no global coverage planner,
# inspection sweeps, breadcrumb return, or safety action overrides.
bash "$workspace/scripts/run_policy_only_trial.sh" random - \
  "$run_id" "$seed" "$max_minutes" false

python3 - "$workspace/rl_evaluations/$run_id/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text(encoding='utf-8'))
count = result['matched_objects']
target = result['target_objects']
print(f'Verified random result: {count}/{target} in '
      f"{result['elapsed_minutes']:.3f} minutes")
print('Missing:', ', '.join(result['missing_objects']) or 'none')
print('Summary:', path)
if (count, target) != (26, 26):
    raise SystemExit(1)
PY
