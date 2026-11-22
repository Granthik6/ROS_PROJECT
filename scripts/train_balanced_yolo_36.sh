#!/usr/bin/env bash
set -euo pipefail

workspace=/root/ros2_ws
run_name=all36_balanced_finetune_20261115
run_dir="$workspace/realistic_sim_yolo_runs/$run_name"
if [[ -e "$run_dir" ]]; then
  echo "Training output already exists: $run_dir" >&2
  echo "Use the saved last.pt checkpoint to resume an interrupted run." >&2
  exit 2
fi

cd "$workspace"
export MPLCONFIGDIR="$workspace/.matplotlib-cache"
python3 dataset_tools/train_yolo_spaced_finetune.py \
  --model "$workspace/realistic_sim_yolo_runs/all36_finetune_20261115-2/weights/best.pt" \
  --data "$workspace/realistic_sim_yolo_hardviews_balanced_20261115.yaml" \
  --name "$run_name" \
  --epochs 10 --batch 8 --workers 2 --imgsz 640

weights="$run_dir/weights/best.pt"
test -s "$weights"
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo.yaml" \
  --name all36_balanced_original_test_20261115
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo_hardviews_20261115.yaml" \
  --name all36_balanced_hardviews_test_20261115
python3 dataset_tools/evaluate_spaced_detector.py \
  --model "$weights" --data "$workspace/realistic_sim_yolo_spaced.yaml" \
  --name all36_balanced_spaced_test_20261115

python3 - "$workspace/realistic_sim_yolo_runs" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tests = (
    ('original', 'all36_finetune_v2_original_test_20261115',
     'all36_balanced_original_test_20261115'),
    ('new views', 'hardviews_old_model_test_20261115',
     'all36_balanced_hardviews_test_20261115'),
    ('spaced', 'all36_finetune_v2_spaced_test_20261115',
     'all36_balanced_spaced_test_20261115'),
)
for label, old_name, new_name in tests:
    old = json.loads((root / old_name / 'metrics.json').read_text())
    new = json.loads((root / new_name / 'metrics.json').read_text())
    key = 'metrics/mAP50(B)'
    print(f'{label}: mAP50 {old["overall"][key]:.4f} -> '
          f'{new["overall"][key]:.4f}')
    for class_name in ('coke_can', 'fire_hose_long', 'stop_light',
                       'table_marble'):
        before = old['per_class'].get(class_name)
        after = new['per_class'].get(class_name)
        if before and after:
            print(f'  {class_name}: AP50 {before["ap50"]:.4f} -> '
                  f'{after["ap50"]:.4f}')
PY

echo "Training and held-out evaluations finished. Weights: $weights"
