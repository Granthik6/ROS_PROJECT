"""Keep older training images that do not contain poorly labeled tiny classes."""

from pathlib import Path

import yaml


ROOT = Path('/root/ros2_ws')
OLD_SOURCES = (
    ROOT / 'realistic_sim_yolo_dataset_spaced_20261101/images/original_subset',
    ROOT / 'realistic_sim_yolo_dataset_spaced_20261101/images/train',
)
WEAK_CLASS_IDS = {15, 16, 29, 32, 36, 37}
OUTPUT_IMAGES = ROOT / 'realistic_sim_yolo_dataset_pedestal_v2_20261102/images/clean_legacy'
OUTPUT_LABELS = ROOT / 'realistic_sim_yolo_dataset_pedestal_v2_20261102/labels/clean_legacy'


def main():
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)
    retained = 0
    removed = 0
    for image_dir in OLD_SOURCES:
        label_dir = Path(str(image_dir).replace('/images/', '/labels/'))
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            label = label_dir / (image.stem + '.txt')
            if not label.exists():
                continue
            classes = {int(line.split()[0]) for line in label.read_text().splitlines()
                       if line.strip()}
            if classes & WEAK_CLASS_IDS:
                removed += 1
                continue
            prefix = 'original' if image_dir.name == 'original_subset' else 'spaced'
            destination_image = OUTPUT_IMAGES / f'{prefix}_{image.name}'
            destination_label = OUTPUT_LABELS / f'{prefix}_{label.name}'
            if not destination_image.exists():
                destination_image.symlink_to(image.resolve())
            if not destination_label.exists():
                destination_label.symlink_to(label.resolve())
            retained += 1

    source = yaml.safe_load((ROOT / 'realistic_sim_yolo_pedestal_v2.yaml').read_text())
    source['train'] = [
        str(OUTPUT_IMAGES.relative_to(ROOT)),
        'realistic_sim_yolo_dataset_pedestal_v2_20261102/images/train',
    ]
    output = ROOT / 'realistic_sim_yolo_pedestal_clean.yaml'
    output.write_text(yaml.safe_dump(source, sort_keys=False))
    print(f'Retained {retained} older images, excluded {removed} images with '
          f'weak-class old labels, and kept all new pedestal views. {output}')


if __name__ == '__main__':
    main()
