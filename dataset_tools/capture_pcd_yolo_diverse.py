"""Capture bottle/doll views at multiple temporary Gazebo world positions.

The saved world is never edited. Objects are restored to their world-file poses
on exit. Validation and test each use object positions absent from training.
"""

import argparse
import csv
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_pcd_yolo_dataset as old
import generate_yolo_dataset as base


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'pcd_yolo_dataset_v2'
SITES = {
    'train': [
        (-3.5, -3.5, 0.0), (3.5, -1.5, 0.7), (-1.5, 0.5, 1.4),
        (0.5, 3.5, 2.1), (4.5, 0.5, -0.8),
    ],
    'val': [(-4.5, 1.5, 2.8)],
    'test': [(4.5, -3.5, -1.6)],
}
DISTANCES = (1.05, 1.30, 1.55, 1.85, 2.20)


class Capture(Node):
    def __init__(self):
        super().__init__('pcd_diverse_capture')
        self.bridge = CvBridge()
        self.images = deque(maxlen=5)
        self.depths = deque(maxlen=5)
        self.create_subscription(Image, '/camera/rgb/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw',
                                 self.on_depth, qos_profile_sensor_data)
        self.setter = self.create_client(SetEntityState,
                                         '/gazebo/set_entity_state')
        self.getter = self.create_client(GetEntityState,
                                         '/gazebo/get_entity_state')
        if not self.setter.wait_for_service(timeout_sec=30):
            raise RuntimeError('Gazebo set_entity_state service is unavailable')
        if not self.getter.wait_for_service(timeout_sec=30):
            raise RuntimeError('Gazebo get_entity_state service is unavailable')

    def on_image(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.images.append((stamp, time.monotonic(), msg))

    def on_depth(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.depths.append((stamp, time.monotonic(), msg))

    def move(self, entity, x, y, z, yaw):
        state = EntityState()
        state.name = entity
        state.reference_frame = 'world'
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.z = math.sin(yaw / 2)
        state.pose.orientation.w = math.cos(yaw / 2)
        future = self.setter.call_async(SetEntityState.Request(state=state))
        rclpy.spin_until_future_complete(self, future, timeout_sec=10)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError(f'Could not move {entity}')

    def pose(self, entity):
        future = self.getter.call_async(
            GetEntityState.Request(name=entity, reference_frame='world')
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=10)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError(f'Could not read pose of {entity}')
        p = future.result().state.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                         1 - 2 * (q.y*q.y + q.z*q.z))
        return p.position.x, p.position.y, p.position.z, yaw

    def fresh_pair(self, moved_at, timeout=2.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            recent_images = [item for item in self.images
                             if item[1] >= moved_at + 0.20]
            recent_depths = [item for item in self.depths
                             if item[1] >= moved_at + 0.20]
            for image in reversed(recent_images):
                if not recent_depths:
                    break
                depth = min(recent_depths,
                            key=lambda item: abs(item[0] - image[0]))
                if abs(image[0] - depth[0]) <= 0.10:
                    return (self.bridge.imgmsg_to_cv2(image[2], 'bgr8'),
                            self.bridge.imgmsg_to_cv2(depth[2], '32FC1'))
        return None


def free_robot_pose(x, y, focus, other, background):
    if not (base.ROOM_X_MIN + 0.18 <= x <= base.ROOM_X_MAX - 0.18 and
            base.ROOM_Y_MIN + 0.18 <= y <= base.ROOM_Y_MAX - 0.18):
        return False
    for obj in (*background, focus, other):
        ox, oy = obj['pos']
        if math.hypot(x-ox, y-oy) < base.object_radius(obj) + 0.27:
            return False
    return True


def copy_old_training():
    old_root = ROOT / 'pcd_yolo_dataset'
    for kind in ('images', 'labels'):
        destination = DATASET / kind / 'train'
        destination.mkdir(parents=True, exist_ok=True)
        source = old_root / kind / 'train'
        for path in source.glob('*'):
            if path.suffix not in ('.png', '.txt'):
                continue
            target = destination / ('original_' + path.name)
            if not target.exists():
                target.write_bytes(path.read_bytes())


def generate(args):
    copy_old_training()
    for kind in ('images', 'labels'):
        for split in SITES:
            (DATASET / kind / split).mkdir(parents=True, exist_ok=True)

    original = {obj['cls']: obj for obj in old.targets()}
    background = tuple(base.OBJECTS)
    manifest = DATASET / 'manifest.csv'
    existing = set()
    if manifest.exists():
        with manifest.open(newline='') as stream:
            existing = {row['image'] for row in csv.DictReader(stream)}

    rclpy.init()
    node = Capture()
    try:
        with manifest.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                'image', 'split', 'focus_class', 'site_index',
                'object_x', 'object_y', 'object_yaw',
                'robot_x', 'robot_y', 'robot_yaw',
            ))
            if not existing:
                writer.writeheader()
            for cls_id, cls in enumerate(old.CLASSES):
                focus = dict(original[cls])
                other = dict(original[old.CLASSES[1-cls_id]])
                for split, sites in SITES.items():
                    for site_index, (ox, oy, oyaw) in enumerate(sites):
                        node.move(focus['entity'], ox, oy, 0.0, oyaw)
                        focus['pos'] = (ox, oy)
                        focus['yaw'] = oyaw
                        focus['distances'] = DISTANCES
                        base.OBJECTS = [focus, other]
                        random.seed(args.seed + 1000*cls_id +
                                    100*('train','val','test').index(split) +
                                    site_index)
                        viewpoints = base.sample_object_viewpoints(
                            focus, n_angles=args.angles
                        )
                        viewpoints = [vp for vp in viewpoints if
                                      free_robot_pose(vp[0], vp[1], focus,
                                                      other, background)]
                        random.shuffle(viewpoints)
                        saved = 0
                        for candidate, (x, y, yaw, _) in enumerate(viewpoints):
                            if saved >= args.per_site:
                                break
                            stem = f'{cls}_{split}_s{site_index}_v{candidate:03d}'
                            filename = stem + '.png'
                            if filename in existing:
                                saved += 1
                                continue
                            node.move('burger', x, y, 0.02, yaw)
                            moved_at = time.monotonic()
                            pair = node.fresh_pair(moved_at)
                            if pair is None:
                                continue
                            frame, depth = pair
                            bx, by, bz, byaw = node.pose('burger')
                            if math.hypot(bx-x, by-y) > 0.06:
                                continue
                            cam_pos, rotation = base.compute_camera_pose(
                                bx, by, byaw, bz
                            )
                            boxes = base.compute_visible_boxes(
                                cam_pos, rotation, depth
                            )
                            if len(boxes) != 1 or boxes[0][0]['cls'] != cls:
                                continue
                            _obj, x1, y1, x2, y2, _distance = boxes[0]
                            label = (f'{cls_id} {(x1+x2)/2/base.IMG_W:.6f} '
                                     f'{(y1+y2)/2/base.IMG_H:.6f} '
                                     f'{(x2-x1)/base.IMG_W:.6f} '
                                     f'{(y2-y1)/base.IMG_H:.6f}\n')
                            image_path = DATASET / 'images' / split / filename
                            label_path = DATASET / 'labels' / split / (stem + '.txt')
                            if not cv2.imwrite(str(image_path), frame):
                                raise RuntimeError(f'Cannot write {image_path}')
                            label_path.write_text(label)
                            writer.writerow(dict(
                                image=filename, split=split, focus_class=cls,
                                site_index=site_index, object_x=ox, object_y=oy,
                                object_yaw=oyaw, robot_x=bx, robot_y=by,
                                robot_yaw=byaw,
                            ))
                            stream.flush()
                            saved += 1
                            if saved % 8 == 0:
                                node.get_logger().info(
                                    f'{cls} {split} site {site_index}: '
                                    f'{saved}/{args.per_site}'
                                )
                        if saved < args.per_site:
                            raise RuntimeError(
                                f'{cls} {split} site {site_index}: only '
                                f'{saved}/{args.per_site} visible views '
                                f'from {len(viewpoints)} free candidates'
                            )
    finally:
        for obj in original.values():
            x, y = obj['pos']
            node.move(obj['entity'], x, y, 0.0, obj['yaw'])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-site', type=int, default=16)
    parser.add_argument('--angles', type=int, default=36)
    parser.add_argument('--seed', type=int, default=4401)
    generate(parser.parse_args())
