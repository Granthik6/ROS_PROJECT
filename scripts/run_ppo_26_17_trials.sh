#!/usr/bin/env bash
set -euo pipefail

# Count the verified seed-42 PPO run as trial 1, then run seeds 43..58.
# Usage: bash scripts/run_ppo_26_17_trials.sh [max_minutes] [series_id]
workspace=/root/ros2_ws
max_minutes=${1:-12}
series_id=${2:-ppo_26_17_20261121}
series_dir="$workspace/rl_evaluations/$series_id"
reference=ppo_reliable26_seed42_20261121
if [[ "$max_minutes" != 12 && "$max_minutes" != 12.0 ]]; then
  echo 'This series uses the same 12-minute limit as trial 1.' >&2
  exit 2
fi

if [[ ! -f "$workspace/rl_evaluations/$reference/summary.json" ]]; then
  echo "Verified reference run missing: $reference" >&2
  exit 2
fi
mkdir -p "$series_dir"
printf '%s\n' "$reference" > "$series_dir/trial_01.run_id"
python3 "$workspace/scripts/report_ppo_26_17_trials.py" "$series_dir"

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
  echo "Starting PPO trial $trial_label/17 seed=$seed run_id=$run_id"
  if bash "$workspace/scripts/run_all26_ppo.sh" "$max_minutes" "$run_id" "$seed" \
      > "$series_dir/trial_${trial_label}.log" 2>&1; then
    echo "Trial $trial_label: 26/26 confirmed"
  else
    echo "Trial $trial_label finished without 26/26; checking saved summary"
  fi
  if [[ ! -f "$workspace/rl_evaluations/$run_id/summary.json" ]]; then
    echo "No summary for trial $trial_label. Rerun this script to retry it." >&2
    exit 1
  fi
  python3 "$workspace/scripts/report_ppo_26_17_trials.py" "$series_dir"
done

python3 "$workspace/scripts/report_ppo_26_17_trials.py" "$series_dir"
echo "17-trial report: $series_dir/summary.json"
