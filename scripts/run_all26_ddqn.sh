#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_all26_ddqn.sh [max_minutes] [run_id] [seed]
# max_minutes=0 waits for 26/26 or Ctrl+C; the default bounds a demo run.
workspace=/root/ros2_ws
max_minutes=${1:-12}
run_id=${2:-ddqn_26_demo_$(date +%Y%m%d_%H%M%S)}
seed=${3:-42}

export ROBOT_TASKS_DDQN_MODEL=${ROBOT_TASKS_DDQN_MODEL:-$workspace/rl_runs/ddqn_safe_50000/model_final.zip}
export ROBOT_TASKS_DETECTOR_MODEL=${ROBOT_TASKS_DETECTOR_MODEL:-$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt}
export ROBOT_TASKS_ACTION_DURATION=${ROBOT_TASKS_ACTION_DURATION:-0.15}
export ROBOT_TASKS_STOCHASTIC_POLICY=${ROBOT_TASKS_STOCHASTIC_POLICY:-false}
export ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M=${ROBOT_TASKS_INSPECTION_SWEEP_EVERY_M:-1}
export ROBOT_TASKS_CONFIRM_NEW_DETECTIONS=${ROBOT_TASKS_CONFIRM_NEW_DETECTIONS:-true}
export ROBOT_TASKS_ROS_DOMAIN_ID=${ROBOT_TASKS_ROS_DOMAIN_ID:-171}
export ROBOT_TASKS_GAZEBO_MASTER_URI=${ROBOT_TASKS_GAZEBO_MASTER_URI:-http://127.0.0.1:11771}

exec bash "$workspace/scripts/run_ddqn_36_candidate.sh" \
  "$run_id" "$seed" "$max_minutes" \
  "$workspace/src/robot_tasks/worlds/spaced_objects_reliable_26_v4.world"
