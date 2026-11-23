import csv
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

ASSOCIATION_RADIUS_M = 1.2
LOG_PATH = os.path.expanduser('~/ros2_ws/realistic_sim_detections.csv')

CLASS_COLORS = {
    'cup': (0.9, 0.2, 0.2),
    'bowl': (0.2, 0.6, 0.9),
    'coke_can': (0.8, 0.1, 0.1),
    'cordless_drill': (0.9, 0.6, 0.1),
    'hammer': (0.6, 0.6, 0.6),
}


class ObjectMapper(Node):

    def __init__(self):
        super().__init__('object_mapper')

        self.objects = []

        self.subscription = self.create_subscription(
            String, '/yolo/detections', self.detection_callback, 20
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray, '/object_markers', 10
        )

        self.marker_timer = self.create_timer(1.0, self.publish_markers)
        self.summary_timer = self.create_timer(10.0, self.print_summary)

        new_file = not os.path.exists(LOG_PATH)
        self.csv_file = open(LOG_PATH, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if new_file:
            self.csv_writer.writerow([
                'stamp', 'class', 'confidence', 'map_x', 'map_y', 'map_z',
                'is_new_object', 'total_unique_objects',
            ])

        self.get_logger().info('object_mapper started, logging to ' + LOG_PATH)

    def detection_callback(self, msg):
        try:
            det = json.loads(msg.data)
        except Exception:
            return

        map_xyz = det.get('odom_xyz') or det.get('map_xyz')
        if map_xyz is None:
            return

        cls = det['class']
        conf = det.get('confidence', 0.0)
        x, y, z = map_xyz
        now = time.time()

        match = None
        for obj in self.objects:
            if obj['class'] != cls:
                continue
            dist = math.hypot(obj['x'] - x, obj['y'] - y)
            if dist < ASSOCIATION_RADIUS_M:
                match = obj
                break

        is_new = match is None

        if match is None:
            match = {
                'class': cls, 'x': x, 'y': y, 'z': z,
                'first_seen': now, 'last_seen': now,
                'confidence': conf, 'n_obs': 1,
            }
            self.objects.append(match)
            self.get_logger().info(
                f'NEW object #{len(self.objects)}: {cls} at '
                f'({x:.2f}, {y:.2f}, {z:.2f}) conf={conf:.2f}'
            )
        else:
            n = match['n_obs']
            match['x'] = (match['x'] * n + x) / (n + 1)
            match['y'] = (match['y'] * n + y) / (n + 1)
            match['z'] = (match['z'] * n + z) / (n + 1)
            match['n_obs'] += 1
            match['last_seen'] = now
            match['confidence'] = max(match['confidence'], conf)

        self.csv_writer.writerow([
            now, cls, round(conf, 4), round(x, 3), round(y, 3), round(z, 3),
            is_new, len(self.objects),
        ])
        self.csv_file.flush()

    def publish_markers(self):
        array = MarkerArray()

        for i, obj in enumerate(self.objects):
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'objects'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obj['x']
            marker.pose.position.y = obj['y']
            marker.pose.position.z = obj['z']
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            r, g, b = CLASS_COLORS.get(obj['class'], (1.0, 1.0, 1.0))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.9
            array.markers.append(marker)

            text = Marker()
            text.header.frame_id = 'odom'
            text.header.stamp = marker.header.stamp
            text.ns = 'labels'
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = obj['x']
            text.pose.position.y = obj['y']
            text.pose.position.z = obj['z'] + 0.2
            text.pose.orientation.w = 1.0
            text.scale.z = 0.15
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"{obj['class']} ({obj['n_obs']}x, {obj['confidence']:.2f})"
            array.markers.append(text)

        self.marker_publisher.publish(array)

    def print_summary(self):
        counts = {}
        for obj in self.objects:
            counts[obj['class']] = counts.get(obj['class'], 0) + 1

        self.get_logger().info(
            f'Object DB: {len(self.objects)} unique objects. '
            f'Per-class: {counts}'
        )

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMapper()
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
