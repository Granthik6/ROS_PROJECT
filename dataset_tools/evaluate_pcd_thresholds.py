"""Compare detection recall and background false positives by confidence."""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
POSITIVE = ROOT / 'pcd_yolo_dataset_v2'
NEGATIVE = ROOT / 'realistic_sim_yolo_dataset/images/test'
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)


def iou(box, truth):
    x1 = max(box[0], truth[0]); y1 = max(box[1], truth[1])
    x2 = min(box[2], truth[2]); y2 = min(box[3], truth[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])
    area_b = max(0.0, truth[2]-truth[0]) * max(0.0, truth[3]-truth[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def predictions(model, path):
    result = model.predict(str(path), conf=0.10, imgsz=640,
                           device='cpu', verbose=False)[0]
    return [(int(box.cls[0]), float(box.conf[0]), box.xyxy[0].tolist())
            for box in result.boxes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = YOLO(str(args.weights))
    positives = sorted((POSITIVE / 'images/test').glob('*.png'))
    negatives = sorted(NEGATIVE.glob('*.png'))
    samples = []
    for path in positives:
        label = (POSITIVE / 'labels/test' / (path.stem + '.txt')).read_text().strip().split()
        cls = int(label[0]); xc, yc, w, h = map(float, label[1:])
        truth = ((xc-w/2)*640, (yc-h/2)*480,
                 (xc+w/2)*640, (yc+h/2)*480)
        samples.append((cls, truth, predictions(model, path)))
    negative_predictions = [predictions(model, path) for path in negatives]

    report = dict(weights=str(args.weights), positive_images=len(positives),
                  negative_images=len(negatives), thresholds={})
    for threshold in THRESHOLDS:
        detected = {0: 0, 1: 0}
        false_positive_test = 0
        for cls, truth, pred in samples:
            valid = [(pcls, conf, box) for pcls, conf, box in pred
                     if conf >= threshold]
            matches = [p for p in valid if p[0] == cls and iou(p[2], truth) >= 0.5]
            detected[cls] += bool(matches)
            false_positive_test += len(valid) - bool(matches)
        bg_with_predictions = sum(
            any(conf >= threshold for _cls, conf, _box in preds)
            for preds in negative_predictions
        )
        report['thresholds'][str(threshold)] = dict(
            bottle_detected=f'{detected[0]}/16',
            doll_detected=f'{detected[1]}/16',
            false_positive_test=false_positive_test,
            background_images_with_predictions=bg_with_predictions,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
