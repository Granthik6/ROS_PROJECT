#!/usr/bin/env bash
set -euo pipefail


workspace=/root/ros2_ws
experiment_id=${1:-}
minutes_per_trial=${2:-10}
first_seed=${3:-42}
number_of_seeds=${4:-17}
trial_profile=${5:-policy_only}

if [[ -z "$experiment_id" || ! "$experiment_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'Usage: bash scripts/run_real_17_trial_comparison.sh EXPERIMENT_ID [minutes_per_trial] [first_seed] [number_of_seeds] [policy_only|autonomous_coverage]' >&2
  echo 'EXPERIMENT_ID must contain only letters, digits, underscores, or hyphens.' >&2
  exit 2
fi
if [[ "$trial_profile" != policy_only && "$trial_profile" != autonomous_coverage ]]; then
  echo "Unknown trial profile: $trial_profile" >&2
  exit 2
fi
if ! [[ "$minutes_per_trial" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! [[ "$first_seed" =~ ^[0-9]+$ ]] ||
   ! [[ "$number_of_seeds" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Use a positive duration, a nonnegative first seed, and a positive seed count.' >&2
  exit 2
fi
root="$workspace/rl_evaluations/$experiment_id"
if [[ "$trial_profile" == autonomous_coverage ]]; then
  world=${ROBOT_TASKS_TRIAL_WORLD:-$workspace/src/robot_tasks/worlds/spaced_objects_ddqn_accessible_v2.world}
else
  world=${ROBOT_TASKS_TRIAL_WORLD:-$workspace/src/robot_tasks/worlds/spaced_objects_completion.world}
fi
ddqn_model="$workspace/rl_runs/ddqn_safe_50000/model_final.zip"
ppo_model="$workspace/rl_runs/ppo_search_10000_quota_20261102/model_final.zip"
detector_model=${ROBOT_TASKS_DETECTOR_MODEL:-$workspace/realistic_sim_yolo_runs/safe_stands_epoch4_snapshot.pt}
inference_imgsz=${ROBOT_TASKS_INFERENCE_IMGSZ:-512}
if ! [[ "$inference_imgsz" =~ ^[0-9]+$ ]] ||
   (( inference_imgsz < 320 || inference_imgsz > 1280 )); then
  echo 'ROBOT_TASKS_INFERENCE_IMGSZ must be an integer between 320 and 1280.' >&2
  exit 2
fi
for required in \
  "$workspace/scripts/run_policy_only_trial.sh" \
  "$workspace/scripts/report_real_17_trial_comparison.py" \
  "$world" \
  "$detector_model" \
  "$workspace/realistic_sim_yolo_runs/pedestal_v2_20261102/weights/best.pt" \
  "$ddqn_model" "$ppo_model"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
detector_model=$(realpath "$detector_model")

mkdir -p "$root"
manifest="$root/experiment.json"
python3 - "$manifest" "$experiment_id" "$minutes_per_trial" "$first_seed" "$number_of_seeds" "$ddqn_model" "$ppo_model" "$trial_profile" "$world" "$detector_model" "$inference_imgsz" <<'PY'
import json
import sys
from pathlib import Path

path, identifier, minutes, seed, count, ddqn, ppo, profile, world, detector, imgsz = sys.argv[1:]
path = Path(path)
record = {
    'experiment_id': identifier,
    'minutes_per_trial': float(minutes),
    'first_seed': int(seed),
    'number_of_seeds': int(count),
    'algorithms': ['ppo', 'ddqn', 'random'],
    'world': world,
    'detector': str(Path(detector).resolve()),
    'detector_inference_imgsz': int(imgsz),
    'models': {'ppo': ppo, 'ddqn': ddqn, 'random': None},
    'control_mode': profile,
    'guidance': False,
    'global_coverage': profile == 'autonomous_coverage',
    'inspection_sweeps': profile == 'autonomous_coverage',
    'safety_action_overrides': profile == 'autonomous_coverage',
    'random_policy': 'novelty' if profile == 'autonomous_coverage' else 'uniform',
}
if path.exists():
    current = json.loads(path.read_text(encoding='utf-8'))
    if current != record:
        raise SystemExit(f'Experiment ID already has different settings: {path}')
else:
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(f'Experiment: {path}')
PY

write_report() {
  MPLCONFIGDIR="$workspace/.matplotlib-cache" \
    python3 "$workspace/scripts/report_real_17_trial_comparison.py" "$root"
}
verify_trial() {
  python3 - "$1" "$2" "$3" "$trial_profile" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert summary['algorithm'] == sys.argv[2]
assert summary['seed'] == int(sys.argv[3])
assert summary['target_objects'] == 36
assert summary['status'] in ('all_objects_found', 'completed_time_limit')
if float(summary['time_limit_minutes'] or 0) == 0:
    assert summary['status'] == 'all_objects_found'
    assert summary['matched_objects'] == 36
    assert summary['all_objects_detected']
assert not summary['guided_completion_enabled']
if sys.argv[4] == 'autonomous_coverage':
    assert summary['global_coverage_enabled']
    assert summary['safety_recovery']
    assert summary['inspection_sweep_every_m'] == 0.5
else:
    assert not summary['global_coverage_enabled']
    assert summary['inspection_sweep_events'] == 0
    assert not summary['safety_recovery']
PY
}
archive_failed_attempt() {
  local trial_name=$1
  local archive="$root/failed_attempts/${trial_name}_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -p "$archive"
  for path in "$root/$trial_name" \
              "$root/${trial_name}_gazebo.log" \
              "$root/${trial_name}_discovery.log"; do
    if [[ -e "$path" ]]; then
      mv "$path" "$archive/"
    fi
  done
  echo "Archived incomplete attempt: $archive" | tee -a "$root/progress.log"
}
on_interrupt() {
  echo 'Operator interrupted the experiment; preserving completed trials.' | tee -a "$root/progress.log"
  exit 130
}
trap on_interrupt INT TERM
trap 'write_report || true' EXIT

last_seed=$((first_seed + number_of_seeds - 1))
for ((seed=first_seed; seed<=last_seed; seed++)); do
  for algorithm in ppo ddqn random; do
    trial_name="seed_${seed}_${algorithm}"
    trial_dir="$root/$trial_name"
    if [[ -f "$trial_dir/summary.json" ]]; then
      if verify_trial "$trial_dir/summary.json" "$algorithm" "$seed"; then
        echo "Already complete: $algorithm seed $seed"
        continue
      fi
      archive_failed_attempt "$trial_name"
    fi
    if [[ -e "$trial_dir" ]]; then
      archive_failed_attempt "$trial_name"
    fi
    if [[ "$algorithm" == ppo ]]; then
      model="$ppo_model"
    elif [[ "$algorithm" == ddqn ]]; then
      model="$ddqn_model"
    else
      model=-
    fi
    method_offset=0
    if [[ "$algorithm" == ddqn ]]; then method_offset=1; fi
    if [[ "$algorithm" == random ]]; then method_offset=2; fi
    trial_offset=$(((seed - first_seed) * 3 + method_offset))
    gazebo_port=$(( ${ROBOT_TASKS_GAZEBO_PORT_BASE:-11486} + trial_offset ))
    ros_domain=$(( ${ROBOT_TASKS_ROS_DOMAIN_ID_BASE:-100} + trial_offset ))
    if (( ros_domain > 232 )); then
      echo "Computed ROS_DOMAIN_ID $ros_domain exceeds 232; lower ROBOT_TASKS_ROS_DOMAIN_ID_BASE" >&2
      exit 2
    fi
    success=false
    for attempt in 1 2 3; do
      if [[ "$minutes_per_trial" == 0 || "$minutes_per_trial" == 0.0 ]]; then
        limit_text='until 36/36 (no time limit)'
      else
        limit_text="cap=${minutes_per_trial}min"
      fi
      echo "Starting $algorithm seed=$seed attempt=$attempt profile=$trial_profile $limit_text Gazebo port=$gazebo_port ROS domain=$ros_domain" | tee -a "$root/progress.log"
      if ROBOT_TASKS_GAZEBO_MASTER_URI="http://127.0.0.1:$gazebo_port" \
         ROBOT_TASKS_ROS_DOMAIN_ID="$ros_domain" \
         ROBOT_TASKS_TRIAL_PROFILE="$trial_profile" \
         ROBOT_TASKS_TRIAL_WORLD="$world" \
         ROBOT_TASKS_DETECTOR_MODEL="$detector_model" \
         ROBOT_TASKS_INFERENCE_IMGSZ="$inference_imgsz" \
         bash "$workspace/scripts/run_policy_only_trial.sh" \
         "$algorithm" "$model" "$experiment_id/$trial_name" "$seed" "$minutes_per_trial" false \
         && verify_trial "$trial_dir/summary.json" "$algorithm" "$seed"; then
        success=true
        echo "Finished $algorithm seed=$seed" | tee -a "$root/progress.log"
        write_report
        break
      fi
      if [[ -f "$trial_dir/summary.json" ]] && \
         python3 - "$trial_dir/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as stream:
    summary = json.load(stream)
raise SystemExit(0 if summary.get('status') == 'interrupted' else 1)
PY
      then
        on_interrupt
      fi
      echo "Attempt $attempt failed: $algorithm seed=$seed" | tee -a "$root/progress.log" >&2
      archive_failed_attempt "$trial_name"
      if (( attempt < 3 )); then sleep 5; fi
    done
    if [[ "$success" != true ]]; then
      echo "Trial failed after three attempts: $algorithm seed=$seed" | tee -a "$root/progress.log" >&2
      exit 1
    fi
  done
done

echo "All $((number_of_seeds * 3)) requested trials completed. Report: $root/report.pdf"
