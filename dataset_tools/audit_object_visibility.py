"""Check each placed object from a nearby camera pose (diagnostic only).

This uses known Gazebo locations to test the detector. It is not an
autonomous exploration result and must never be scored as one.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from robot_tasks.world_coverage_report import read_world
from robot_tasks.world_frame import detection_in_world


def spin_for(node, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def move_robot(node, client, x, y, yaw):
    request = SetEntityState.Request()
    request.state.name = 'burger'
    request.state.reference_frame = 'world'
    request.state.pose.position.x = float(x)
    request.state.pose.position.y = float(y)
    request.state.pose.position.z = 0.02
    request.state.pose.orientation.z = math.sin(yaw / 2)
    request.state.pose.orientation.w = math.cos(yaw / 2)
    future = client.call_async(request)
    deadline = time.monotonic() + 5
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    return bool(future.done() and future.result() and future.result().success)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--view-seconds', type=float, default=2.0)
    parser.add_argument('--only-missing-from', type=Path)
    parser.add_argument('--image-dir', type=Path)
    args = parser.parse_args()
    objects = read_world(args.world)
    if args.only_missing_from:
        previous = json.loads(args.only_missing_from.read_text())
        missing = set(previous['missing_objects'])
        objects = [obj for obj in objects if obj['name'] in missing]
    center = (0.0, -2.0)
    state = {'odom': None, 'robot': None, 'target': None,
             'raw': [], 'nearby': [], 'image': None}

    rclpy.init()
    node = Node('object_visibility_audit')
    publisher = node.create_publisher(Twist, '/cmd_vel', 10)
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')
    bridge = CvBridge()

    def on_odom(message):
        state['odom'] = message.pose.pose

    def on_models(message):
        try:
            state['robot'] = message.pose[message.name.index('burger')]
        except ValueError:
            pass

    def on_detection(message):
        target = state['target']
        if target is None:
            return
        try:
            row = json.loads(message.data)
            if row['class'] != target['class']:
                return
            confidence = float(row['confidence'])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return
        threshold = 0.20 if target['class'] in ('bottle', 'doll') else 0.25
        if confidence < threshold:
            return
        state['raw'].append({'stamp': str(row.get('stamp', '')),
                             'confidence': confidence})
        xyz = row.get('odom_xyz')
        if (not isinstance(xyz, list) or len(xyz) != 3 or
                state['odom'] is None or state['robot'] is None):
            return
        try:
            point = detection_in_world(xyz, state['odom'], state['robot'])
        except (ValueError, TypeError):
            return
        distance = math.hypot(point[0] - target['x'], point[1] - target['y'])
        if distance <= 1.25:
            state['nearby'].append({'stamp': str(row.get('stamp', '')),
                                    'confidence': confidence,
                                    'distance_error_m': distance})

    node.create_subscription(Odometry, '/odom', on_odom, 10)
    node.create_subscription(ModelStates, '/gazebo/model_states', on_models, 1)
    node.create_subscription(String, '/yolo/detections', on_detection, 50)
    node.create_subscription(
        Image, '/camera/rgb/image_raw',
        lambda message: state.__setitem__('image', message),
        qos_profile_sensor_data,
    )
    if args.image_dir:
        args.image_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (state['odom'] is not None and state['robot'] is not None and
                    node.count_publishers('/yolo/detections') and
                    client.service_is_ready()):
                break
        else:
            raise RuntimeError('Gazebo, odometry, or YOLO did not become ready')

        for index, target in enumerate(objects, 1):
            dx, dy = target['x'] - center[0], target['y'] - center[1]
            distance = math.hypot(dx, dy)
            if distance < 0.01:
                dx, dy, distance = 1.0, 0.0, 1.0
            ux, uy = dx / distance, dy / distance
            confirmed = False
            attempted = []
            for viewing_distance in (0.9, 1.5):
                x = target['x'] - viewing_distance * ux
                y = target['y'] - viewing_distance * uy
                yaw = math.atan2(target['y'] - y, target['x'] - x)
                publisher.publish(Twist())
                if not move_robot(node, client, x, y, yaw):
                    attempted.append({'view_distance_m': viewing_distance,
                                      'error': 'teleport failed'})
                    continue
                state['target'] = None
                spin_for(node, 0.8)
                state['raw'], state['nearby'] = [], []
                state['target'] = target
                deadline = time.monotonic() + args.view_seconds
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if len({row['stamp'] for row in state['nearby']}) >= 2:
                        break
                state['target'] = None
                if args.image_dir and state['image'] is not None:
                    image = bridge.imgmsg_to_cv2(state['image'], 'bgr8')
                    cv2.imwrite(str(args.image_dir /
                                    f'{target["name"]}_{viewing_distance:.1f}m.png'),
                                image)
                distinct_stamps = len({row['stamp'] for row in state['nearby']})
                confirmed = distinct_stamps >= 2
                attempted.append({
                    'view_distance_m': viewing_distance,
                    'raw_class_hits': len(state['raw']),
                    'localized_nearby_hits': len(state['nearby']),
                    'distinct_nearby_frames': distinct_stamps,
                    'confirmed': confirmed,
                })
                if confirmed:
                    break
            results.append({'name': target['name'], 'class': target['class'],
                            'confirmed': confirmed, 'attempts': attempted})
            print(f'{index}/{len(objects)} {target["name"]}: '
                  f'{"visible" if confirmed else "MISS"}', flush=True)
    finally:
        publisher.publish(Twist())
        state['target'] = None
        node.destroy_node()
        rclpy.shutdown()

    report = {
        'diagnostic': 'known_position_close_view_not_autonomous_exploration',
        'world_file': str(Path(args.world).resolve()),
        'target_objects': len(objects),
        'confirmed_objects': sum(row['confirmed'] for row in results),
        'missing_objects': [row['name'] for row in results if not row['confirmed']],
        'objects': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'objects'}, indent=2))


if __name__ == '__main__':
    main()
