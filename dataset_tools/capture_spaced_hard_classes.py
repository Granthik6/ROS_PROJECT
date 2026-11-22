"""Capture automatically labeled Gazebo views for weak detector classes."""

import argparse
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState

sys.path.insert(0, str(Path(__file__).parent))
import generate_yolo_dataset as gyd
from audit_planned_views import world_specs


ROOT = Path('/root/ros2_ws')


class StableDatasetGenerator(gyd.DatasetGenerator):
    """Use ROS Gazebo services instead of thousands of `gz model` processes."""

    def __init__(self):
        super().__init__()
        self.robot_pose = None
        self.set_state_client = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self._model_states, 1)

    def _model_states(self, message):
        try:
            self.robot_pose = message.pose[message.name.index('burger')]
        except ValueError:
            pass

    def acquire(self, robot_x, robot_y, robot_yaw):
        if not self.set_state_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('/gazebo/set_entity_state unavailable')
        request = SetEntityState.Request()
        request.state.name = 'burger'
        request.state.reference_frame = 'world'
        request.state.pose.position.x = robot_x
        request.state.pose.position.y = robot_y
        request.state.pose.position.z = 0.02
        request.state.pose.orientation.z = math.sin(robot_yaw / 2)
        request.state.pose.orientation.w = math.cos(robot_yaw / 2)
        future = self.set_state_client.call_async(request)
        deadline = time.monotonic() + 5
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('Gazebo robot reset failed')

        previous = None
        stable = 0
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            pose = self.robot_pose
            if pose is None:
                continue
            dx = abs(pose.position.x - robot_x)
            dy = abs(pose.position.y - robot_y)
            if dx > 0.15 or dy > 0.15:
                continue
            if previous is not None:
                shift = math.hypot(pose.position.x - previous.position.x,
                                   pose.position.y - previous.position.y)
                stable = stable + 1 if shift < 0.01 else 0
            previous = pose
            if stable >= 2:
                break
        else:
            return None

        self.flush_callbacks(duration=0.12)
        previous_frame_stamp = self.latest_rgb_stamp
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.latest_rgb_stamp is not None and
                    self.latest_rgb_stamp != previous_frame_stamp):
                break
        else:
            return None
        pose = self.robot_pose
        if pose is None or self.latest_rgb is None:
            return None
        orientation = pose.orientation
        yaw = math.atan2(
            2 * (orientation.w * orientation.z +
                 orientation.x * orientation.y),
            1 - 2 * (orientation.y ** 2 + orientation.z ** 2),
        )
        frame = self.latest_rgb.copy()
        depth = self.latest_depth.copy() if self.latest_depth is not None else None
        return (pose.position.x, pose.position.y, pose.position.z,
                yaw, frame, depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', default=str(
        ROOT / 'src/robot_tasks/worlds/spaced_objects.world'))
    parser.add_argument('--missing-list', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=
                        ROOT / 'realistic_sim_yolo_dataset_spaced_20261101')
    parser.add_argument('--train-per-class', type=int, default=30)
    parser.add_argument('--val-per-class', type=int, default=6)
    parser.add_argument('--test-per-class', type=int, default=6)
    args = parser.parse_args()
    specs, _ = world_specs(args.world)
    gyd.OBJECTS = [spec for spec in specs
                   if spec['cls'] in gyd.CLASS_ID]
    world_models = ET.parse(args.world).getroot().find('world').findall('model')
    if any(model.get('name', '').startswith('search_boundary_')
           for model in world_models):
        gyd.ROOM_X_MIN, gyd.ROOM_X_MAX = -8.75, 8.75
        gyd.ROOM_Y_MIN, gyd.ROOM_Y_MAX = -10.75, 6.75
    else:
        gyd.ROOM_X_MIN, gyd.ROOM_X_MAX = -15.0, 15.0
        gyd.ROOM_Y_MIN, gyd.ROOM_Y_MAX = -15.0, 10.0
    gyd.OBSTACLES = []
    missing = set(json.loads(args.missing_list.read_text())['missing_objects'])
    targets = [spec for spec in gyd.OBJECTS if spec['entity'] in missing]
    if len(targets) != len(missing):
        raise ValueError('Missing-list names do not match detector classes')

    rclpy.init()
    node = StableDatasetGenerator()
    if not node.set_state_client.wait_for_service(timeout_sec=60.0):
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        raise RuntimeError('Gazebo reset service did not become ready')
    summary = []
    try:
        for class_number, target in enumerate(targets):
            random.seed(1000 + class_number)
            viewpoints = gyd.sample_object_viewpoints(target, n_angles=24)
            pools = {'train': [], 'val': [], 'test': []}
            for index, (x, y, yaw, _) in enumerate(viewpoints):
                if not gyd.is_free(x, y):
                    continue
                camera_pos, rotation = gyd.compute_camera_pose(x, y, yaw)
                boxes = gyd.compute_visible_boxes(camera_pos, rotation)
                if not any(obj['entity'] == target['entity']
                           for obj, *_ in boxes):
                    continue
                angle_index = index % 24
                split = ('test' if angle_index % 6 == 0 else
                         'val' if angle_index % 6 == 1 else 'train')
                pools[split].append((x, y, yaw))
            counts = {}
            for split, limit in (('train', args.train_per_class),
                                 ('val', args.val_per_class),
                                 ('test', args.test_per_class)):
                candidates = pools[split]
                random.shuffle(candidates)
                gyd.RAW_IMAGES = str(args.output_root / 'images' / split)
                gyd.RAW_LABELS = str(args.output_root / 'labels' / split)
                Path(gyd.RAW_IMAGES).mkdir(parents=True, exist_ok=True)
                Path(gyd.RAW_LABELS).mkdir(parents=True, exist_ok=True)
                target_id = str(gyd.CLASS_ID[target['cls']])
                valid_target = 0
                attempted = 0
                for view_number, (x, y, yaw) in enumerate(candidates):
                    if valid_target >= limit:
                        break
                    if not node.set_state_client.service_is_ready():
                        raise RuntimeError('Gazebo stopped during dataset capture')
                    attempted += 1
                    image_index = 950000 + class_number * 1000 + (
                        {'train': 0, 'val': 100, 'test': 200}[split]
                    ) + view_number
                    node.capture(x, y, yaw, target['cls'], image_index)
                    label_path = (args.output_root / 'labels' / split /
                                  f'{target["cls"]}_{image_index:05d}.txt')
                    image_path = (args.output_root / 'images' / split /
                                  f'{target["cls"]}_{image_index:05d}.png')
                    is_valid = label_path.exists() and any(
                        line.split()[0] == target_id
                        for line in label_path.read_text().splitlines()
                        if line.strip()
                    )
                    if is_valid:
                        valid_target += 1
                    else:
                        label_path.unlink(missing_ok=True)
                        image_path.unlink(missing_ok=True)
                counts[split] = {'target_labeled': valid_target,
                                 'attempted': attempted,
                                 'candidate_pool': len(candidates)}
            summary.append({'name': target['entity'], 'class': target['cls'],
                            'counts': counts})
            print(f'{class_number + 1}/{len(targets)} '
                  f'{target["entity"]}: {counts}', flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'capture_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
