import os
import random
import shutil

SRC_IMAGES = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_dataset/raw_images")
SRC_LABELS = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_dataset/raw_labels")
DEST = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_dataset")

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

random.seed(7)


def main():
    images = sorted(f for f in os.listdir(SRC_IMAGES) if f.endswith(".png"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    split_map = {}
    for f in images[:n_train]:
        split_map[f] = "train"
    for f in images[n_train:n_train + n_val]:
        split_map[f] = "val"
    for f in images[n_train + n_val:]:
        split_map[f] = "test"

    for split in SPLITS:
        os.makedirs(os.path.join(DEST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DEST, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": 0, "val": 0, "test": 0}
    empty_counts = {"train": 0, "val": 0, "test": 0}

    for fname, split in split_map.items():
        base = os.path.splitext(fname)[0]
        src_img = os.path.join(SRC_IMAGES, fname)
        src_lbl = os.path.join(SRC_LABELS, base + ".txt")

        dst_img = os.path.join(DEST, "images", split, fname)
        dst_lbl = os.path.join(DEST, "labels", split, base + ".txt")

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            with open(src_lbl) as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            if lines:
                label_counts[split] += len(lines)
            else:
                empty_counts[split] += 1
        else:
            open(dst_lbl, "w").close()
            empty_counts[split] += 1

        counts[split] += 1

    print("Split sizes:", counts)
    print("Label (box) counts per split:", label_counts)
    print("Background (no-object) images per split:", empty_counts)


if __name__ == "__main__":
    main()
