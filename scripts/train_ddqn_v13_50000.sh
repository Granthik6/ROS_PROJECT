#!/usr/bin/env bash
set -euo pipefail

workspace=/root/ros2_ws
run_id=${1:-ddqn_v13_balanced_50000_$(date +%Y%m%d_%H%M%S)}
export ROBOT_TASKS_TRAIN_WORLD="$workspace/src/robot_tasks/worlds/spaced_objects_accessible_v13.world"
export ROBOT_TASKS_TRAIN_DETECTOR_MODEL="$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt"
export ROBOT_TASKS_TRAIN_INFERENCE_IMGSZ=640

bash "$workspace/scripts/train_coverage_improved.sh" ddqn "$run_id" 50000

python3 - "$workspace/rl_runs/$run_id/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
steps = int(summary['model_total_timesteps'])
print(f"DDQN training: {summary['status']}, {steps} steps")
print('Model:', summary['model_path'])
if summary['status'] != 'completed' or steps < 50000:
    raise SystemExit('Training did not complete 50,000 steps')
PY
