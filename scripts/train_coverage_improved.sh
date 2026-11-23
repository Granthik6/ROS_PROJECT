#!/usr/bin/env bash
set -eo pipefail

workspace=/root/ros2_ws
algorithm=${1:-}
run_id=${2:-}
timesteps=${3:-10000}
resume_model=${4:-}
if [[ "$algorithm" != ddqn && "$algorithm" != ppo || -z "$run_id" ]]; then
  echo "Usage: $0 ddqn|ppo run_id [timesteps]" >&2
  exit 2
fi
if [[ -e "$workspace/rl_runs/$run_id" ]]; then
  echo "Training output already exists: $workspace/rl_runs/$run_id" >&2
  exit 2
fi
if [[ -n "$resume_model" && ! -f "$resume_model" ]]; then
  echo "Resume model not found: $resume_model" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$workspace/src/robot_tasks/models:/root/gazebo_models:/root/.gazebo/models:${GAZEBO_MODEL_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11487}
export ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-185}
export ROS_LOG_DIR="$workspace/ros_logs"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"

world=${ROBOT_TASKS_TRAIN_WORLD:-$workspace/src/robot_tasks/worlds/spaced_objects_completion.world}
detector=${ROBOT_TASKS_TRAIN_DETECTOR_MODEL:-$workspace/realistic_sim_yolo_runs/safe_stands_epoch4_snapshot.pt}
inference_imgsz=${ROBOT_TASKS_TRAIN_INFERENCE_IMGSZ:-512}
if [[ ! -f "$world" || ! -f "$detector" ]]; then
  echo "Training world or detector model is missing: $world ; $detector" >&2
  exit 2
fi
hose_detector="$workspace/realistic_sim_yolo_runs/pedestal_v2_20261102/weights/best.pt"
mkdir -p "$workspace/rl_training_logs"
ros2 launch robot_tasks realistic_sim.launch.py gui:=false world:="$world" \
  > "$workspace/rl_training_logs/${run_id}_gazebo.log" 2>&1 &
sim_pid=$!
detector_pid=''
stop_child() {
  local pid=$1
  if [[ -z "$pid" ]]; then return; fi
  kill -INT "$pid" 2>/dev/null || true
  for _ in {1..40}; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  stop_child "$detector_pid"
  stop_child "$sim_pid"
}
trap cleanup EXIT

sleep 15
if rg -q 'Unable to start server|Address already in use|Spawn service failed|already exists' \
    "$workspace/rl_training_logs/${run_id}_gazebo.log"; then
  echo "Gazebo failed to start on $GAZEBO_MASTER_URI" >&2
  exit 2
fi
ros2 run robot_tasks yolo_node --ros-args \
  -p use_sim_time:=true \
  -p model_path:="$detector" \
  -p hose_fallback_model_path:="$hose_detector" \
  -p hose_fallback_interval_frames:=2 \
  -p person_fallback_model_path:="$workspace/yolo11n.pt" \
  -p person_fallback_interval_frames:=5 \
  -p show_window:=false -p conf_threshold:=0.10 \
  -p pcd_conf_threshold:=0.20 -p pcd_interval_frames:=3 \
  -p inference_imgsz:="$inference_imgsz" -p log_detections:=false \
  > "$workspace/rl_training_logs/${run_id}_yolo.log" 2>&1 &
detector_pid=$!
sleep 8
resume_args=()
if [[ -n "$resume_model" ]]; then
  resume_args=(--resume "$resume_model")
fi

ros2 run robot_tasks train_exploration \
  --algorithm "$algorithm" --timesteps "$timesteps" --run-id "$run_id" \
  --environment-profile coverage_search --feature-extractor scan_sectors \
  --world "$world" \
  --detector-model "$detector" --action-duration 0.10 \
  --episode-steps 1200 --seed 42 \
  --entropy-coef 0.05 --new-area-reward 15.0 \
  --new-object-reward 2.0 --collision-penalty 5.0 \
  --revisit-penalty 0.2 --turn-penalty 0.1 \
  --progress-reward-scale 4.0 --blocked-forward-penalty 2.0 \
  --spin-penalty 1.0 \
  "${resume_args[@]}"
