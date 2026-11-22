"""Add older no-PCD Gazebo views as hard negatives to the diverse dataset."""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'pcd_yolo_dataset_v2'
DESTINATION = ROOT / 'pcd_yolo_dataset_v3'
OLDER = ROOT / 'realistic_sim_yolo_dataset/images'


def copy_positive_data():
    for kind in ('images', 'labels'):
        for split in ('train', 'val', 'test'):
            src = SOURCE / kind / split
            dst = DESTINATION / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            for path in src.glob('*'):
                if path.suffix not in ('.png', '.txt'):
                    continue
                target = dst / path.name
                if not target.exists():
                    shutil.copy2(path, target)


def select_negatives(model, split, pool_size, hard_count, random_count, seed):
    rng = random.Random(seed)
    paths = sorted((OLDER / split).glob('*.png'))
    rng.shuffle(paths)
    candidates = paths[:min(pool_size, len(paths))]
    ranked = []
    for index, path in enumerate(candidates, 1):
        result = model.predict(str(path), conf=0.10, imgsz=640,
                               device='cpu', verbose=False)[0]
        score = max((float(box.conf[0]) for box in result.boxes),
                    default=0.0)
        ranked.append((score, path))
        if index % 100 == 0:
            print(split, 'scanned', index, '/', len(candidates), flush=True)
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    hard = ranked[:hard_count]
    remaining = ranked[hard_count:]
    ordinary = rng.sample(remaining, min(random_count, len(remaining)))
    chosen = [(score, path, 'hard') for score, path in hard]
    chosen.extend((score, path, 'random') for score, path in ordinary)
    return chosen


def add_negatives(split, chosen):
    records = []
    for score, source, kind in chosen:
        stem = 'negative_' + source.stem
        image = DESTINATION / 'images' / split / (stem + '.png')
        label = DESTINATION / 'labels' / split / (stem + '.txt')
        if not image.exists():
            shutil.copy2(source, image)
        label.write_text('')
        records.append(dict(image=image.name, score=score,
                            selection=kind, source=str(source)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path,
                        default=ROOT / 'pcd_yolo_runs/pcd_bottle_doll_v2/weights/best.pt')
    args = parser.parse_args()
    torch.set_num_threads(2)
    copy_positive_data()
    model = YOLO(str(args.weights))
    train = select_negatives(model, 'train', pool_size=600,
                             hard_count=80, random_count=120, seed=624)
    val = select_negatives(model, 'val', pool_size=200,
                           hard_count=20, random_count=20, seed=625)
    report = dict(weights=str(args.weights),
                  train=add_negatives('train', train),
                  val=add_negatives('val', val))
    output = DESTINATION / 'negative_selection.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Added', len(train), 'training negatives and', len(val),
          'validation negatives')


if __name__ == '__main__':
    main()
