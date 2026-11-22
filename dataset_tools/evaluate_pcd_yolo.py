"""Evaluate a PCD detector on views excluded from training."""

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / 'pcd_yolo_runs/pcd_bottle_doll/weights/best.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'pcd_yolo_dataset.yaml')
    parser.add_argument('--name', default='held_out_test')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    if not args.weights.exists():
        raise SystemExit(f'No trained checkpoint at {args.weights}')
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=640,
        batch=8,
        workers=0,
        device='cpu',
        project=str(ROOT / 'pcd_yolo_runs'),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    per_class = {}
    for class_id, name in model.names.items():
        precision, recall, ap50, ap5095 = metrics.box.class_result(class_id)
        per_class[name] = dict(precision=float(precision),
                               recall=float(recall),
                               ap50=float(ap50),
                               ap50_95=float(ap5095))
    dataset = Path(yaml.safe_load(args.data.read_text())['path'])
    report = dict(weights=str(args.weights),
                  test_images=len(list((dataset / 'images' / args.split).glob('*.png'))),
                  map50=float(metrics.box.map50),
                  map50_95=float(metrics.box.map),
                  per_class=per_class)
    output = ROOT / 'pcd_yolo_runs' / args.name / 'summary.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
