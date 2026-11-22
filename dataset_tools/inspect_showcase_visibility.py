"""Diagnostic: rotate at the start pose and check camera visibility."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from robot_tasks.world_coverage_report import match_instances, read_world
from robot_tasks.world_frame import detection_in_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--angular-speed', type=float, default=0.65)
    args = parser.parse_args()
    targets = read_world(args.world)
    rclpy.init()
    node = Node('showcase_visibility_diagnostic')
    pose = {'odom': None, 'world': None}
    samples = []
    start = [None]

    def on_odom(message):
        pose['odom'] = message.pose.pose

    def on_models(message):
        try:
            pose['world'] = message.pose[message.name.index('burger')]
        except ValueError:
            pass

    def on_detection(message):
        if start[0] is None or pose['odom'] is None or pose['world'] is None:
            return
        try:
            row = json.loads(message.data)
            xyz = row['odom_xyz']
            world_xyz = detection_in_world(xyz, pose['odom'], pose['world'])
            confidence = float(row['confidence'])
            name = str(row['class'])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return
        if confidence < (0.20 if name in ('bottle', 'doll') else 0.25):
            return
        samples.append({
            'time': time.monotonic() - start[0], 'class': name,
            'confidence': confidence, 'x': world_xyz[0], 'y': world_xyz[1],
            'stamp': str(row.get('stamp', '')),
        })

    node.create_subscription(Odometry, '/odom', on_odom, 10)
    node.create_subscription(ModelStates, '/gazebo/model_states', on_models, 1)
    node.create_subscription(String, '/yolo/detections', on_detection, 50)
    publisher = node.create_publisher(Twist, '/cmd_vel', 10)
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (pose['odom'] is not None and pose['world'] is not None
                    and node.count_publishers('/yolo/detections') > 0
                    and client.service_is_ready()):
                break
        else:
            raise RuntimeError('Waiting for robot, YOLO, and Gazebo timed out')
        request = SetEntityState.Request()
        request.state.name = 'burger'
        request.state.reference_frame = 'world'
        request.state.pose.position.x = 0.0
        request.state.pose.position.y = -2.0
        request.state.pose.position.z = 0.02
        request.state.pose.orientation.w = 1.0
        future = client.call_async(request)
        while not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        if not future.result().success:
            raise RuntimeError('Robot reset failed')
        start[0] = time.monotonic()
        end = start[0] + args.seconds
        command = Twist()
        command.angular.z = args.angular_speed
        while time.monotonic() < end:
            publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

    clusters = []
    for row in samples:
        nearest = None
        best = 0.75
        for cluster in clusters:
            if cluster['class'] != row['class']:
                continue
            distance = math.hypot(cluster['x'] - row['x'],
                                  cluster['y'] - row['y'])
            if distance < best:
                best, nearest = distance, cluster
        if nearest is None:
            clusters.append({
                'id': str(len(clusters) + 1), 'class': row['class'],
                'x': row['x'], 'y': row['y'],
                'confidence': row['confidence'], 'observations': 1,
                'last_stamp': row['stamp'], 'first_seen_s': row['time'],
                'confirmed_s': None,
            })
        elif row['stamp'] != nearest['last_stamp']:
            count = nearest['observations']
            if nearest['confirmed_s'] is None:
                nearest['x'] = (nearest['x'] * count + row['x']) / (count + 1)
                nearest['y'] = (nearest['y'] * count + row['y']) / (count + 1)
            nearest['observations'] += 1
            nearest['last_stamp'] = row['stamp']
            nearest['confidence'] = max(nearest['confidence'], row['confidence'])
            if nearest['observations'] >= 2 and nearest['confirmed_s'] is None:
                nearest['confirmed_s'] = row['time']
    matches, _ = match_instances(
        targets, [c for c in clusters if c['confirmed_s'] is not None], 1.25)
    result = {
        'diagnostic': 'fixed_position_camera_rotation_not_ddqn',
        'seconds': args.seconds, 'target': len(targets),
        'matched': sum(item['detected'] for item in matches),
        'missing': [item['name'] for item in matches if not item['detected']],
        'observations': len(samples),
        'matched_names': [item['name'] for item in matches if item['detected']],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    with Path(args.output).with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=samples[0].keys()
                                if samples else ('time', 'class', 'confidence',
                                                'x', 'y', 'stamp'))
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
