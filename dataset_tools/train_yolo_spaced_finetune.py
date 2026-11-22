"""Fine-tune the saved detector with new spaced-world hard examples."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path('/root/ros2_ws')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', default='spaced_hard_classes_20261101')
    parser.add_argument('--data', default=str(ROOT / 'realistic_sim_yolo_spaced.yaml'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument(
        '--model',
        default=str(ROOT / 'realistic_sim_yolo_runs/'
                    'realistic_sim_yolo11n/weights/best.pt'),
    )
    args = parser.parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device='cpu',
        patience=10,
        project=str(ROOT / 'realistic_sim_yolo_runs'),
        name=args.name,
        exist_ok=False,
        seed=42,
        deterministic=True,
        plots=True,
        val=True,
    )


if __name__ == '__main__':
    main()
