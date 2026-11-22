import csv
import json
import math
import os
import time

import rclpy

from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_OUTPUT = '/root/ros2_ws/results/distance_confidence.csv'


class ConfidenceLogger(Node):
    """Record paired YOLO confidence and camera distance samples to CSV."""

    FIELDNAMES = [
        'ros_time_s',
        'wall_time_s',
        'class',
        'confidence',
        'distance_m',
        'camera_x_m',
        'camera_y_m',
        'camera_z_m',
        'bbox_x1',
        'bbox_y1',
        'bbox_x2',
        'bbox_y2',
    ]

    def __init__(self):
        super().__init__('confidence_logger')

        self.declare_parameter('output_csv', DEFAULT_OUTPUT)
        self.declare_parameter('target_class', '')
        self.declare_parameter('min_interval_s', 0.25)

        self.output_csv = os.path.abspath(
            os.path.expanduser(
                str(self.get_parameter('output_csv').value)
            )
        )
        self.target_class = str(
            self.get_parameter('target_class').value
        ).strip()
        self.min_interval_s = max(
            0.0,
            float(self.get_parameter('min_interval_s').value)
        )

        output_dir = os.path.dirname(self.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        file_exists = os.path.exists(self.output_csv)
        self.output_file = open(
            self.output_csv,
            'a',
            newline='',
            encoding='utf-8'
        )
        self.writer = csv.DictWriter(
            self.output_file,
            fieldnames=self.FIELDNAMES
        )

        if not file_exists or os.path.getsize(self.output_csv) == 0:
            self.writer.writeheader()
            self.output_file.flush()

        self.last_sample_time = {}
        self.sample_count = 0

        self.subscription = self.create_subscription(
            String,
            '/yolo/detections',
            self.detection_callback,
            10
        )

        class_description = self.target_class or 'all classes'
        self.get_logger().info(
            f'Logging {class_description} to {self.output_csv} '
            f'(minimum interval {self.min_interval_s:.2f}s per class)'
        )

    def detection_callback(self, msg):
        try:
            detection = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as error:
            self.get_logger().warning(f'Ignored invalid detection JSON: {error}')
            return

        class_name = str(detection.get('class', ''))
        if self.target_class and class_name != self.target_class:
            return

        if 'confidence' not in detection:
            return

        camera_xyz = detection.get('camera_xyz')
        if not isinstance(camera_xyz, list) or len(camera_xyz) != 3:
            return

        try:
            camera_x, camera_y, camera_z = map(float, camera_xyz)
            confidence = float(detection['confidence'])
            distance = float(
                detection.get(
                    'distance_m',
                    math.sqrt(
                        camera_x ** 2 +
                        camera_y ** 2 +
                        camera_z ** 2
                    )
                )
            )
        except (TypeError, ValueError):
            return

        now = time.monotonic()
        previous = self.last_sample_time.get(class_name)
        if previous is not None and now - previous < self.min_interval_s:
            return
        self.last_sample_time[class_name] = now

        bbox = detection.get('bbox', ['', '', '', ''])
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = ['', '', '', '']

        self.writer.writerow({
            'ros_time_s': detection.get('stamp', ''),
            'wall_time_s': f'{time.time():.6f}',
            'class': class_name,
            'confidence': f'{confidence:.6f}',
            'distance_m': f'{distance:.6f}',
            'camera_x_m': f'{camera_x:.6f}',
            'camera_y_m': f'{camera_y:.6f}',
            'camera_z_m': f'{camera_z:.6f}',
            'bbox_x1': bbox[0],
            'bbox_y1': bbox[1],
            'bbox_x2': bbox[2],
            'bbox_y2': bbox[3],
        })
        self.output_file.flush()
        self.sample_count += 1

        if self.sample_count == 1 or self.sample_count % 20 == 0:
            self.get_logger().info(
                f'Saved {self.sample_count} samples; latest: '
                f'{class_name}, {distance:.2f}m, confidence={confidence:.3f}'
            )

    def destroy_node(self):
        if not self.output_file.closed:
            self.output_file.flush()
            self.output_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ConfidenceLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
