"""Check bottle and doll detections from the ROS node in Gazebo."""

import csv
import json
import math
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[1]


def test_pose(name):
    with (ROOT / 'pcd_yolo_dataset/manifest.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['focus_class'] == name and row['split'] == 'test':
                return tuple(float(row[key]) for key in (
                    'robot_x', 'robot_y', 'robot_yaw'))
    raise ValueError(f'No held-out {name} camera pose in manifest')


def main():
    rclpy.init()
    node = Node('verify_pcd_yolo_live')
    seen = []
    node.create_subscription(
        String, '/yolo/detections',
        lambda message: seen.append((time.monotonic(), json.loads(message.data))),
        10,
    )
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')
    if not client.wait_for_service(timeout_sec=20):
        raise RuntimeError('Gazebo set_entity_state service unavailable')

    report = {}
    for name in ('bottle', 'doll'):
        x, y, yaw = test_pose(name)
        state = EntityState()
        state.name = 'burger'
        state.reference_frame = 'world'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.02
        state.pose.orientation.z = math.sin(yaw / 2)
        state.pose.orientation.w = math.cos(yaw / 2)
        request = SetEntityState.Request()
        request.state = state
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15)
        if not future.done() or not future.result().success:
            raise RuntimeError(f'Could not place robot for {name}')
        start = time.monotonic()
        while time.monotonic() - start < 14:
            rclpy.spin_once(node, timeout_sec=0.1)
        detections = [item for timestamp, item in seen
                      if timestamp >= start + 1 and item.get('class') == name]
        report[name] = dict(count=len(detections),
                            with_camera_xyz=sum('camera_xyz' in item for item in detections),
                            with_map_xyz=sum('map_xyz' in item for item in detections),
                            best_confidence=max(
                                (item['confidence'] for item in detections),
                                default=None,
                            ),
                            test_pose=[x, y, yaw])
        print(name, report[name], flush=True)

    output = ROOT / 'pcd_yolo_runs/live_ros_check.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    node.destroy_node()
    rclpy.shutdown()
    if any(item['count'] == 0 for item in report.values()):
        raise SystemExit('One or both classes were not observed on /yolo/detections')


if __name__ == '__main__':
    main()
