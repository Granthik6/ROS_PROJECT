#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
comparison_id=${1:-policy_only_3min_3seeds_$(date +%Y%m%d_%H%M%S)}
duration_minutes=${2:-3}
output_root="$workspace/rl_evaluations/$comparison_id"
world="$workspace/src/robot_tasks/worlds/spaced_objects_completion.world"
ddqn_model="$workspace/rl_runs/ddqn_safe_50000/model_final.zip"
ppo_model="$workspace/rl_runs/ppo_search_10000_quota_20261102/model_final.zip"
detector="$workspace/realistic_sim_yolo_runs/safe_stands_epoch4_snapshot.pt"
hose_detector="$workspace/realistic_sim_yolo_runs/pedestal_v2_20261102/weights/best.pt"

for required in "$world" "$ddqn_model" "$ppo_model" "$detector" "$hose_detector"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
mkdir -p "$output_root"

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

sim_pid=''
cleanup_sim() {
  if [[ -z "$sim_pid" ]]; then
    return
  fi
  kill -INT "$sim_pid" 2>/dev/null || true
  for _ in {1..40}; do
    if ! kill -0 "$sim_pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if kill -0 "$sim_pid" 2>/dev/null; then
    kill -TERM "$sim_pid" 2>/dev/null || true
  fi
  for _ in {1..20}; do
    if ! kill -0 "$sim_pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if kill -0 "$sim_pid" 2>/dev/null; then
    kill -KILL "$sim_pid" 2>/dev/null || true
  fi
  wait "$sim_pid" 2>/dev/null || true
  sim_pid=''
}
trap cleanup_sim EXIT

for seed in 42 43 44; do
  for algorithm in ddqn ppo random; do
    trial_dir="$output_root/seed${seed}_${algorithm}"
    summary="$trial_dir/summary.json"
    if [[ -f "$summary" ]]; then
      echo "Skipping completed trial: $trial_dir"
      continue
    fi
    if [[ -e "$trial_dir" ]]; then
      echo "Incomplete trial directory needs review: $trial_dir" >&2
      exit 2
    fi
    echo "Starting $algorithm seed=$seed duration=${duration_minutes}m"
    ros2 launch robot_tasks realistic_sim.launch.py gui:=false world:="$world" \
      > "$output_root/seed${seed}_${algorithm}_gazebo.log" 2>&1 &
    sim_pid=$!
    sleep 15
    if rg -q 'Unable to start server|Address already in use|Spawn service failed|already exists' \
        "$output_root/seed${seed}_${algorithm}_gazebo.log"; then
      echo "Gazebo failed to start on $GAZEBO_MASTER_URI" >&2
      exit 2
    fi

    model_args=()
    if [[ "$algorithm" == ddqn ]]; then
      model_args=("model:=$ddqn_model")
    elif [[ "$algorithm" == ppo ]]; then
      model_args=("model:=$ppo_model")
    fi
    ros2 launch robot_tasks object_discovery.launch.py \
      algorithm:="$algorithm" random_policy:=uniform \
      environment_profile:=coverage_search \
      safety_recovery:=false global_coverage:=false \
      inspection_sweep_every_m:=0 \
      "${model_args[@]}" \
      max_minutes:="$duration_minutes" action_duration:=0.15 seed:="$seed" \
      world:="$world" detector_model:="$detector" \
      hose_fallback_model:="$hose_detector" output_dir:="$trial_dir" \
      > "$output_root/seed${seed}_${algorithm}_discovery.log" 2>&1
    cleanup_sim

    python3 - "$summary" "$algorithm" "$seed" <<'PY'
import json
import sys

summary_path, algorithm, seed = sys.argv[1:]
with open(summary_path, encoding='utf-8') as stream:
    row = json.load(stream)
assert row['algorithm'] == algorithm and row['seed'] == int(seed)
assert row['status'] in ('completed_time_limit', 'all_objects_found')
assert not row['guided_completion_enabled']
assert not row['global_coverage_enabled']
assert row['inspection_sweep_events'] == 0
assert not row['safety_recovery']
print(f"Finished {algorithm} seed={seed}: {row['matched_objects']}/36 "
      f"in {row['elapsed_minutes']:.3f} min; reward={row['total_reward']:.2f}",
      flush=True)
PY
  done
done

python3 "$workspace/scripts/summarize_policy_only_comparison.py" "$output_root"
