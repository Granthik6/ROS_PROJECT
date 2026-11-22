import csv
import json
import math
import os
import random
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import String

CSV_PATH = os.path.expanduser('~/ros2_ws/exploration_results.csv')


class BaselineExploration(Node):

    def __init__(self, duration_sec, run_id):
        super().__init__('baseline_exploration')
        self.duration_sec = duration_sec
        self.run_id = run_id
        self.start_time = time.time()

        self.map = None
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.goals_attempted = 0
        self.goals_succeeded = 0
        self.goals_failed = 0

        self.last_odom_pos = None
        self.distance_traveled = 0.0

        self.unique_objects = set()
        self.total_detections = 0
        self.first_object_time = None
        self.confidences = []

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(String, '/yolo/detections', self.det_cb, 20)

        self.goal_in_flight = False
        self.done = False

        self.timer = self.create_timer(1.0, self.tick)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        if self.last_odom_pos is not None:
            d = math.hypot(p.x - self.last_odom_pos[0], p.y - self.last_odom_pos[1])
            self.distance_traveled += d
        self.last_odom_pos = (p.x, p.y)

    def det_cb(self, msg):
        try:
            det = json.loads(msg.data)
        except Exception:
            return
        self.total_detections += 1
        self.confidences.append(det.get('confidence', 0.0))
        key = det['class']
        if key not in self.unique_objects:
            self.unique_objects.add(key)
            if self.first_object_time is None:
                self.first_object_time = time.time() - self.start_time

    def map_cb(self, msg):
        if self.map is None:
            self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}')
        self.map = msg

    def tick(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.duration_sec:
            self.finish()
            return

        if self.map is None or self.goal_in_flight or self.done:
            return

        self.send_random_goal()

    def send_random_goal(self):
        if not self.client.wait_for_server(timeout_sec=2.0):
            return

        width = self.map.info.width
        height = self.map.info.height
        res = self.map.info.resolution
        free_cells = [
            (x, y) for y in range(height) for x in range(width)
            if self.map.data[y * width + x] == 0
        ]
        if not free_cells:
            return

        x_cell, y_cell = random.choice(free_cells)
        goal_x = self.map.info.origin.position.x + (x_cell + 0.5) * res
        goal_y = self.map.info.origin.position.y + (y_cell + 0.5) * res

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.w = 1.0

        self.goal_in_flight = True
        self.goals_attempted += 1
        self.get_logger().info(f'Goal #{self.goals_attempted}: ({goal_x:.2f},{goal_y:.2f})')

        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.goals_failed += 1
            self.goal_in_flight = False
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.goals_succeeded += 1
        else:
            self.goals_failed += 1
        self.goal_in_flight = False

    def finish(self):
        self.done = True
        self.timer.cancel()

        elapsed_min = (time.time() - self.start_time) / 60.0
        mean_conf = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

        row = {
            'run_id': self.run_id,
            'duration_sec': round(time.time() - self.start_time, 1),
            'distance_traveled_m': round(self.distance_traveled, 3),
            'goals_attempted': self.goals_attempted,
            'goals_succeeded': self.goals_succeeded,
            'goals_failed': self.goals_failed,
            'total_detections': self.total_detections,
            'unique_objects_found': len(self.unique_objects),
            'unique_classes': ';'.join(sorted(self.unique_objects)),
            'time_to_first_object_s': round(self.first_object_time, 1) if self.first_object_time else '',
            'objects_per_minute': round(len(self.unique_objects) / elapsed_min, 3) if elapsed_min > 0 else 0,
            'mean_detection_confidence': round(mean_conf, 4),
        }

        new_file = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        self.get_logger().info('BASELINE EXPLORATION RESULT: ' + json.dumps(row))


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    run_id = sys.argv[2] if len(sys.argv) > 2 else f'run_{int(time.time())}'

    rclpy.init()
    node = BaselineExploration(duration, run_id)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
