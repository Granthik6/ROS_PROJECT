"""Select a small, class-balanced sample of the original Gazebo images.

The new hard-view images are all retained. This keeps fine-tuning practical
on a CPU while preserving examples of classes absent from the hard-view set.
"""

from collections import defaultdict
from pathlib import Path
import random

import yaml


ROOT = Path('/root/ros2_ws')
ORIGINAL = ROOT / 'realistic_sim_yolo_dataset'
SPACED = ROOT / 'realistic_sim_yolo_dataset_spaced_20261101'
OUTPUT = ROOT / 'realistic_sim_yolo_spaced_balanced.yaml'


def main():
    rng = random.Random(42)
    by_class = defaultdict(list)
    background = []
    for label in sorted((ORIGINAL / 'labels/train').glob('*.txt')):
        classes = {int(line.split()[0]) for line in label.read_text().splitlines()
                   if line.strip()}
        if not classes:
            background.append(label)
        for class_id in classes:
            by_class[class_id].append(label)

    selected = set()
    for class_id in sorted(by_class):
        choices = by_class.get(class_id, [])
        selected.update(rng.sample(choices, min(20, len(choices))))
    selected.update(rng.sample(background, min(80, len(background))))

    image_output = SPACED / 'images/original_subset'
    label_output = SPACED / 'labels/original_subset'
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    count = 0
    for label in sorted(selected):
        image = next((ORIGINAL / 'images/train' /
                      (label.stem + extension) for extension in
                      ('.png', '.jpg', '.jpeg') if
                      (ORIGINAL / 'images/train' /
                       (label.stem + extension)).exists()), None)
        if image is None:
            raise FileNotFoundError(f'Image for {label}')
        for source, destination in ((image, image_output / image.name),
                                    (label, label_output / label.name)):
            if not destination.exists():
                destination.symlink_to(source)
        count += 1

    original = yaml.safe_load((ROOT / 'realistic_sim_yolo_spaced.yaml').read_text())
    original['train'] = [
        str(image_output.relative_to(ROOT)),
        str((SPACED / 'images/train').relative_to(ROOT)),
    ]
    OUTPUT.write_text(yaml.safe_dump(original, sort_keys=False))
    print(f'Selected {count} original images plus '
          f'{len(list((SPACED / "images/train").glob("*.png")))} hard-view '
          f'images; wrote {OUTPUT}')


if __name__ == '__main__':
    main()
