import os
import random

import cv2

SRC_IMAGES = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_dataset/raw_images")
SRC_LABELS = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_dataset/raw_labels")
DEST = os.path.expanduser("~/ros2_ws/realistic_sim_hog_dataset")

CLASS_NAMES = ["cup", "bowl", "coke_can", "cordless_drill", "hammer"]
WINDOW = (128, 128)
PAD_FRAC = 0.15  # a bit of context around the tight yolo box helps HOG

random.seed(3)


def yolo_box_to_pixels(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def main():
    for name in CLASS_NAMES + ["background"]:
        os.makedirs(os.path.join(DEST, name), exist_ok=True)

    counts = {name: 0 for name in CLASS_NAMES + ["background"]}

    label_files = sorted(f for f in os.listdir(SRC_LABELS) if f.endswith(".txt"))

    for lf in label_files:
        base = os.path.splitext(lf)[0]
        img_path = os.path.join(SRC_IMAGES, base + ".png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        with open(os.path.join(SRC_LABELS, lf)) as fh:
            lines = [l.split() for l in fh.read().splitlines() if l.strip()]

        boxes_px = []

        for parts in lines:
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_box_to_pixels(xc, yc, w, h, img_w, img_h)
            boxes_px.append((x1, y1, x2, y2))

            pad_x = (x2 - x1) * PAD_FRAC
            pad_y = (y2 - y1) * PAD_FRAC
            x1p = max(0, int(x1 - pad_x))
            y1p = max(0, int(y1 - pad_y))
            x2p = min(img_w, int(x2 + pad_x))
            y2p = min(img_h, int(y2 + pad_y))

            if x2p - x1p < 8 or y2p - y1p < 8:
                continue

            crop = img[y1p:y2p, x1p:x2p]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, WINDOW)

            class_name = CLASS_NAMES[cls_id]
            counts[class_name] += 1
            out_path = os.path.join(
                DEST, class_name, f"{class_name}_{counts[class_name]:05d}.png"
            )
            cv2.imwrite(out_path, resized)

        # background crops: random windows that don't overlap any real box
        if not lines:
            n_bg = 2
        else:
            n_bg = 1

        attempts = 0
        placed = 0
        while placed < n_bg and attempts < 20:
            attempts += 1
            bw = random.randint(60, min(180, img_w))
            bh = random.randint(60, min(180, img_h))
            bx = random.randint(0, max(1, img_w - bw))
            by = random.randint(0, max(1, img_h - bh))

            overlap = False
            for x1, y1, x2, y2 in boxes_px:
                if bx < x2 and bx + bw > x1 and by < y2 and by + bh > y1:
                    overlap = True
                    break

            if overlap:
                continue

            crop = img[by:by + bh, bx:bx + bw]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, WINDOW)
            counts["background"] += 1
            out_path = os.path.join(
                DEST, "background", f"background_{counts['background']:05d}.png"
            )
            cv2.imwrite(out_path, resized)
            placed += 1

    print("Crop counts:", counts)


if __name__ == "__main__":
    main()
