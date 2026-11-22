"""Fine-tune a two-class YOLO11n detector for the Gazebo PCD objects."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'pcd_yolo_dataset.yaml')
    parser.add_argument('--warm-start', type=Path,
                        default=ROOT / 'yolo11n.pt')
    parser.add_argument('--name', default='pcd_bottle_doll')
    args = parser.parse_args()

    model = YOLO(str(args.warm_start))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device='cpu',
        patience=8,
        seed=2026,
        deterministic=True,
        project=str(ROOT / 'pcd_yolo_runs'),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
    )


if __name__ == '__main__':
    main()
