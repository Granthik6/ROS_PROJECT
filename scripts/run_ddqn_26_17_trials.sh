#!/usr/bin/env bash
set -euo pipefail

# Run 16 new independent seeds and include the existing verified seed-42 run.
# Usage: bash scripts/run_ddqn_26_17_trials.sh [max_minutes] [series_id]
workspace=/root/ros2_ws
max_minutes=${1:-12}
series_id=${2:-ddqn_26_17_20261120}
series_dir="$workspace/rl_evaluations/$series_id"
reference=ddqn_reliable26_v4_20261120

# Fix the experimental setup even if the shell has other trial overrides.
export ROBOT_TASKS_DDQN_MODEL="$workspace/rl_runs/ddqn_safe_50000/model_final.zip"
export ROBOT_TASKS_DETECTOR_MODEL="$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt"
export ROBOT_TASKS_ACTION_DURATION=0.15
export ROBOT_TASKS_STOCHASTIC_POLICY=false
export ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M=1
export ROBOT_TASKS_CONFIRM_NEW_DETECTIONS=true
export ROBOT_TASKS_GLOBAL_COVERAGE=true

if [[ ! -f "$workspace/rl_evaluations/$reference/summary.json" ]]; then
  echo "Verified reference run missing: $reference" >&2
  exit 2
fi
mkdir -p "$series_dir"
printf '%s\n' "$reference" > "$series_dir/trial_01.run_id"
python3 "$workspace/scripts/report_ddqn_26_17_trials.py" "$series_dir"

for trial in $(seq 2 17); do
  trial_label=$(printf '%02d' "$trial")
  seed=$((41 + trial))
  pointer="$series_dir/trial_${trial_label}.run_id"
  if [[ -f "$pointer" ]]; then
    run_id=$(cat "$pointer")
    if [[ -f "$workspace/rl_evaluations/$run_id/summary.json" ]]; then
      echo "Trial $trial_label already recorded: $run_id"
      continue
    fi
  fi

  attempt=1
  while :; do
    run_id="${series_id}_trial_${trial_label}_seed_${seed}_attempt_${attempt}"
    if [[ ! -e "$workspace/rl_evaluations/$run_id" ]]; then
      break
    fi
    attempt=$((attempt + 1))
  done
  printf '%s\n' "$run_id" > "$pointer"
  echo "Starting DDQN trial $trial_label/17 seed=$seed run_id=$run_id"
  if bash "$workspace/scripts/run_all26_ddqn.sh" "$max_minutes" "$run_id" "$seed" \
      > "$series_dir/trial_${trial_label}.log" 2>&1; then
    echo "Trial $trial_label: 26/26 confirmed"
  else
    echo "Trial $trial_label finished without 26/26; checking saved summary"
  fi
  if [[ ! -f "$workspace/rl_evaluations/$run_id/summary.json" ]]; then
    echo "No summary for trial $trial_label. Rerun this script to retry it." >&2
    exit 1
  fi
  python3 "$workspace/scripts/report_ddqn_26_17_trials.py" "$series_dir"
done

python3 "$workspace/scripts/report_ddqn_26_17_trials.py" "$series_dir"
echo "17-trial report: $series_dir/summary.json"
