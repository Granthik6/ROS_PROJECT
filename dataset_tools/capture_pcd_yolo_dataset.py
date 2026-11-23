"""Capture labelled Gazebo camera views of the bottle and doll PCD models.

Start realistic_sim.launch.py first. This script moves only the simulated robot.
The labels come from the known model bounds and camera pose; depth rejects
views where the object is hidden by another surface.
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from xml.etree import ElementTree

import cv2
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_yolo_dataset as base


WORKSPACE = Path(__file__).resolve().parents[1]
WORLD = WORKSPACE / 'src/robot_tasks/worlds/realistic_room.world'
DATASET = WORKSPACE / 'pcd_yolo_dataset'
CLASSES = ('bottle', 'doll')


def world_pose(entity):
    root = ElementTree.parse(WORLD).getroot()
    for include in root.findall('.//include'):
        if include.findtext('name') == entity:
            return tuple(float(value) for value in include.findtext('pose').split())
    raise ValueError(f'{entity} is absent from {WORLD}')


def targets():
    specs = (
        ('bottle_pcd_object', 'bottle', (0.048, 0.032, 0.112),
         (0.00145, 0.00150, 0.112)),
        ('doll_pcd_object', 'doll', (0.084, 0.052, 0.114),
         (0.00045, 0.00050, 0.114)),
    )
    objects = []
    for entity, cls, half, offset in specs:
        x, y, z, _r, _p, yaw = world_pose(entity)
        if abs(z) > 0.001:
            raise ValueError(f'Update bounds for changed {entity} height: {z}')
        objects.append(dict(entity=entity, cls=cls, pos=(x, y), yaw=yaw,
                            half=half, offset=offset,
                            distances=[1.15, 1.35, 1.60, 1.90, 2.20]))
    return objects


def free_pose(x, y, target_objects, other_objects):
    if not (base.ROOM_X_MIN + 0.18 <= x <= base.ROOM_X_MAX - 0.18 and
            base.ROOM_Y_MIN + 0.18 <= y <= base.ROOM_Y_MAX - 0.18):
        return False
    for obj in (*target_objects, *other_objects):
        ox, oy = obj['pos']
        if math.hypot(x - ox, y - oy) < base.object_radius(obj) + 0.24:
            return False
    for ox, oy, radius in base.OBSTACLES:
        if math.hypot(x - ox, y - oy) < radius + 0.18:
            return False
    return True


def capture(args):
    random.seed(args.seed)
    original_objects = tuple(base.OBJECTS)
    target_objects = targets()
    base.OBJECTS = target_objects

    for split in ('train', 'val', 'test'):
        (DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)

    manifest_path = DATASET / 'manifest.csv'
    existing = set()
    if manifest_path.exists():
        with manifest_path.open(newline='') as stream:
            existing = {row['image'] for row in csv.DictReader(stream)}

    rclpy.init()
    node = base.DatasetGenerator()
    counts = {name: 0 for name in CLASSES}
    try:
        with manifest_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                'image', 'focus_class', 'split', 'robot_x', 'robot_y',
                'robot_yaw', 'label_count',
            ])
            if not existing:
                writer.writeheader()

            for obj in target_objects:
                viewpoints = base.sample_object_viewpoints(
                    obj, n_angles=args.angles
                )
                viewpoints = [vp for vp in viewpoints if free_pose(
                    vp[0], vp[1], target_objects, original_objects
                )]
                random.shuffle(viewpoints)
                node.get_logger().info(
                    f'{obj["cls"]}: {len(viewpoints)} free candidate views'
                )
                for x, y, yaw, _focus in viewpoints:
                    if counts[obj['cls']] >= args.per_class:
                        break
                    index = counts[obj['cls']]
                    stem = f'{obj["cls"]}_{index:04d}'
                    if stem + '.png' in existing:
                        counts[obj['cls']] += 1
                        continue
                    try:
                        acquired = node.acquire(x, y, yaw)
                    except RuntimeError as error:
                        node.get_logger().warn(str(error))
                        continue
                    if acquired is None:
                        continue
                    bx, by, _bz, byaw, frame, depth = acquired
                    cam_pos, rotation = base.compute_camera_pose(
                        bx, by, byaw, _bz
                    )
                    boxes = base.compute_visible_boxes(
                        cam_pos, rotation, depth
                    )
                    if not any(box[0]['cls'] == obj['cls'] for box in boxes):
                        continue

                    labels = []
                    for found, x1, y1, x2, y2, _distance in boxes:
                        cls_id = CLASSES.index(found['cls'])
                        labels.append((cls_id, (x1+x2)/2/base.IMG_W,
                                       (y1+y2)/2/base.IMG_H,
                                       (x2-x1)/base.IMG_W,
                                       (y2-y1)/base.IMG_H))
                    split = ('test' if index % 10 == 0 else
                             'val' if index % 10 == 1 else 'train')
                    image_path = DATASET / 'images' / split / (stem + '.png')
                    label_path = DATASET / 'labels' / split / (stem + '.txt')
                    if not cv2.imwrite(str(image_path), frame):
                        raise RuntimeError(f'Cannot write {image_path}')
                    label_path.write_text(''.join(
                        f'{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n'
                        for c, xc, yc, w, h in labels
                    ))
                    writer.writerow(dict(image=image_path.name,
                                         focus_class=obj['cls'], split=split,
                                         robot_x=f'{bx:.4f}',
                                         robot_y=f'{by:.4f}',
                                         robot_yaw=f'{byaw:.4f}',
                                         label_count=len(labels)))
                    stream.flush()
                    counts[obj['cls']] += 1
                    if counts[obj['cls']] % 10 == 0:
                        node.get_logger().info(
                            f'{obj["cls"]}: {counts[obj["cls"]]}/{args.per_class}'
                        )
                if counts[obj['cls']] < args.per_class:
                    raise RuntimeError(
                        f'Only captured {counts[obj["cls"]]} {obj["cls"]} views; '
                        'add viewpoints or inspect camera visibility.'
                    )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print('Capture complete:', counts, 'dataset:', DATASET)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--angles', type=int, default=48)
    parser.add_argument('--seed', type=int, default=2026)
    capture(parser.parse_args())
