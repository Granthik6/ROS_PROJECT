"""Check one detector result against the corresponding Gazebo object pose."""

import argparse
import json
import math
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from robot_tasks.world_frame import detection_in_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report', required=True)
    parser.add_argument('--target', default='stop_light_object')
    parser.add_argument('--seconds', type=float, default=12)
    args = parser.parse_args()
    report = json.load(open(args.audit_report, encoding='utf-8'))
    target = next(row for row in report['objects']
                  if row['name'] == args.target)
    x, y, yaw = target['views'][0]['pose']
    world_x, world_y = (-4.2, 5.0) if args.target == 'stop_light_object' else (None, None)

    rclpy.init()
    node = Node('world_detection_diagnostic')
    state = {'odom': None, 'robot': None, 'rows': []}
    node.create_subscription(
        Odometry, '/odom',
        lambda msg: state.__setitem__('odom', msg.pose.pose), 10)

    def models(msg):
        if 'burger' in msg.name:
            state['robot'] = msg.pose[msg.name.index('burger')]

    def detections(msg):
        try:
            row = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if row.get('class') == target['class']:
            state['rows'].append(row)

    node.create_subscription(ModelStates, '/gazebo/model_states', models, 1)
    node.create_subscription(String, '/yolo/detections', detections, 50)
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')
    if not client.wait_for_service(timeout_sec=10):
        raise RuntimeError('Gazebo reset service unavailable')
    request = SetEntityState.Request()
    request.state.name = 'burger'
    request.state.reference_frame = 'world'
    request.state.pose.position.x = x
    request.state.pose.position.y = y
    request.state.pose.position.z = 0.02
    request.state.pose.orientation.z = math.sin(yaw / 2)
    request.state.pose.orientation.w = math.cos(yaw / 2)
    future = client.call_async(request)
    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.05)
    print('teleport succeeded:', future.result().success, flush=True)
    until = time.monotonic() + args.seconds
    printed = 0
    while time.monotonic() < until and printed < 8:
        rclpy.spin_once(node, timeout_sec=0.05)
        while state['rows'] and printed < 8:
            detection = state['rows'].pop(0)
            printed += 1
            if (state['odom'] is None or state['robot'] is None or
                    'odom_xyz' not in detection):
                print('no odom location:', detection, flush=True)
                continue
            point = detection_in_world(
                detection['odom_xyz'], state['odom'], state['robot'])
            error = (math.hypot(point[0] - world_x, point[1] - world_y)
                     if world_x is not None else None)
            print({'confidence': detection['confidence'],
                   'stamp': detection.get('stamp'),
                   'world_point': [round(v, 3) for v in point],
                   'position_error_m': round(error, 3) if error is not None else None,
                   'robot_world': [state['robot'].position.x,
                                   state['robot'].position.y],
                   'robot_odom': [state['odom'].position.x,
                                  state['odom'].position.y]}, flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
