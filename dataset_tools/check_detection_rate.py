"""Count live YOLO detections and their image timestamp age."""

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    rclpy.init()
    node = Node('check_detection_rate')
    node.set_parameters([rclpy.parameter.Parameter(
        'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    rows = []
    ages = []

    def callback(message):
        try:
            item = json.loads(message.data)
        except json.JSONDecodeError:
            return
        rows.append(item)
        stamp = item.get('stamp')
        if stamp is not None:
            ages.append(node.get_clock().now().nanoseconds * 1e-9 -
                        float(stamp))

    node.create_subscription(String, '/yolo/detections', callback, 50)
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    now = node.get_clock().now().nanoseconds * 1e-9
    print('detections', len(rows),
          'classes', sorted({r.get('class') for r in rows}),
          'last_age_s', round(now - float(rows[-1].get('stamp', 0)), 3)
          if rows else None,
          'arrival_age_range_s',
          [round(min(ages), 3), round(max(ages), 3)] if ages else None,
          'localized', sum('odom_xyz' in r for r in rows))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
