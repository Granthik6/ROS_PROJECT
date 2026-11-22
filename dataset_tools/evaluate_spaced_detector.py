"""Evaluate the saved YOLO weights on held-out spaced-world camera views."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
import yaml


ROOT = Path('/root/ros2_ws')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--data', default=str(ROOT / 'realistic_sim_yolo_spaced.yaml'))
    args = parser.parse_args()
    dataset = yaml.safe_load(Path(args.data).read_text())
    test_dir = Path(dataset['test'])
    if not test_dir.is_absolute():
        test_dir = Path(dataset.get('path', ROOT)) / test_dir
    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split='test', imgsz=640, batch=8, workers=2,
        device='cpu', project=str(ROOT / 'realistic_sim_yolo_runs'),
        name=args.name, exist_ok=False, plots=True, verbose=False,
    )
    names = metrics.names
    per_class = {}
    for position, class_id in enumerate(metrics.box.ap_class_index):
        name = names[int(class_id)]
        per_class[name] = {
            'ap50': round(float(metrics.box.ap50[position]), 4),
            'map50_95': round(float(metrics.box.ap[position]), 4),
        }
    report = {
        'model': str(Path(args.model).resolve()),
        'test_images': len(list(test_dir.glob('*.png'))),
        'overall': {key: round(float(value), 4)
                    for key, value in metrics.results_dict.items()
                    if isinstance(value, (int, float))},
        'per_class': per_class,
    }
    output = ROOT / 'realistic_sim_yolo_runs' / args.name / 'metrics.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
