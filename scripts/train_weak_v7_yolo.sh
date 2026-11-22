#!/usr/bin/env bash
set -euo pipefail

workspace=/root/ros2_ws
run_name=all36_weak_v7_finetune_20261116
run_dir="$workspace/realistic_sim_yolo_runs/$run_name"
if [[ -e "$run_dir" ]]; then
  echo "Training output already exists: $run_dir" >&2
  exit 2
fi

cd "$workspace"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"
python3 dataset_tools/train_yolo_spaced_finetune.py \
  --model "$workspace/realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt" \
  --data "$workspace/realistic_sim_yolo_weak_v7_20261116.yaml" \
  --name "$run_name" --epochs 12 --batch 8 --workers 2 --imgsz 640

weights="$run_dir/weights/best.pt"
test -s "$weights"
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo_weak_v7_20261116.yaml" \
  --name all36_weak_v7_newviews_test_20261116
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo.yaml" \
  --name all36_weak_v7_original_test_20261116
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo_hardviews_20261115.yaml" \
  --name all36_weak_v7_hardviews_test_20261116
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo_spaced.yaml" \
  --name all36_weak_v7_spaced_test_20261116

python3 - <<'PY'
import json
from pathlib import Path

root = Path('/root/ros2_ws/realistic_sim_yolo_runs')
for label, old, new in (
    ('original', 'all36_balanced_original_test_20261115',
     'all36_weak_v7_original_test_20261116'),
    ('hardviews', 'all36_balanced_hardviews_test_20261115',
     'all36_weak_v7_hardviews_test_20261116'),
    ('spaced', 'all36_balanced_spaced_test_20261115',
     'all36_weak_v7_spaced_test_20261116'),
):
    before = json.loads((root / old / 'metrics.json').read_text())
    after = json.loads((root / new / 'metrics.json').read_text())
    key = 'metrics/mAP50(B)'
    print(f'{label}: mAP50 {before["overall"][key]:.4f} -> '
          f'{after["overall"][key]:.4f}')
    for cls in ('cricket_ball', 'monkey_wrench', 'saucepan',
                'ball_bearing', 'gear_part'):
        a, b = before['per_class'].get(cls), after['per_class'].get(cls)
        if a and b:
            print(f'  {cls}: AP50 {a["ap50"]:.4f} -> {b["ap50"]:.4f}')
PY

echo "Training and held-out evaluations finished. Weights: $weights"
