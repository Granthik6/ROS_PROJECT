"""Count bottle/doll false detections on pre-PCD Gazebo room images."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT / 'realistic_sim_yolo_dataset/images/test'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=0.20)
    args = parser.parse_args()

    torch.set_num_threads(2)
    model = YOLO(str(args.weights))
    paths = sorted(BACKGROUND.glob('*.png'))
    found = Counter()
    images_with_predictions = 0
    for path in paths:
        result = model.predict(str(path), conf=args.confidence,
                               imgsz=640, device='cpu', verbose=False)[0]
        classes = [model.names[int(box.cls[0])] for box in result.boxes]
        found.update(classes)
        images_with_predictions += bool(classes)
    report = dict(weights=str(args.weights),
                  background_images=len(paths),
                  images_with_predictions=images_with_predictions,
                  detections=dict(found),
                  confidence_threshold=args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
