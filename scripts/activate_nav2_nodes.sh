#!/usr/bin/env bash
set -euo pipefail

nodes=(controller_server smoother_server planner_server behavior_server bt_navigator waypoint_follower velocity_smoother)
for node in "${nodes[@]}"; do
  state=
  for attempt in {1..60}; do
    state=$(timeout 5 ros2 lifecycle get "/$node" 2>/dev/null || true)
    if [[ "$state" == *"unconfigured"* || "$state" == *"inactive"* || "$state" == *"active"* ]]; then
      break
    fi
    sleep 1
  done
  if [[ "$state" == *"unconfigured"* ]]; then
    timeout 60 ros2 lifecycle set "/$node" configure
    state=inactive
  fi
  if [[ "$state" == *"inactive"* ]]; then
    timeout 60 ros2 lifecycle set "/$node" activate
    state=active
  fi
  if [[ "$state" != *"active"* ]]; then
    echo "Nav2 node $node did not activate (state: $state)" >&2
    exit 1
  fi
  echo "Nav2 $node: active"
done
