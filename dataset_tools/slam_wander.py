import math
import random
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class Wander(Node):

    def __init__(self, duration_sec):
        super().__init__('slam_wander')
        self.scan = None
        self.duration_sec = duration_sec
        self.start_time = time.time()

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = 'forward'
        self.state_until = 0.0
        self.turn_dir = 1.0

        self.last_pos = None
        self.stuck_check_pos = None
        self.stuck_check_time = time.time()
        self.stuck_count = 0

        self.timer = self.create_timer(0.1, self.step)

    def scan_cb(self, msg):
        self.scan = msg

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.last_pos = (p.x, p.y)

    def clearance(self, center_deg, width_deg):
        if self.scan is None or len(self.scan.ranges) == 0:
            return 10.0
        n = len(self.scan.ranges)
        center = int(n * (center_deg % 360) / 360.0)
        window = max(1, int(n * width_deg / 360.0))
        idx = [(center + i) % n for i in range(-window, window)]
        vals = [
            self.scan.ranges[i] for i in idx
            if math.isfinite(self.scan.ranges[i]) and self.scan.ranges[i] > 0.01
        ]
        return min(vals) if vals else 10.0

    def front_clearance(self):
        return self.clearance(0, 40)

    def side_clearance(self):
        left = self.clearance(70, 25)
        right = self.clearance(-70, 25)
        return left, right

    def enter_backup(self):
        self.state = 'backup'
        self.state_until = time.time() + random.uniform(0.8, 1.3)

    def enter_turn(self, hard=False):
        self.state = 'turn'
        self.turn_dir = random.choice([-1.0, 1.0])
        self.state_until = time.time() + random.uniform(1.5, 3.0 if hard else 2.2)

    def step(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.duration_sec:
            self.cmd_pub.publish(Twist())
            self.get_logger().info('Wander duration elapsed, stopping.')
            self.timer.cancel()
            return

        now = time.time()
        cmd = Twist()
        front = self.front_clearance()
        left, right = self.side_clearance()

        if front < 0.28 and self.state != 'backup':
            self.get_logger().warn(f'Too close (front={front:.2f}m) — backing up')
            self.enter_backup()

        if now - self.stuck_check_time > 4.0:
            if self.last_pos is not None and self.stuck_check_pos is not None:
                moved = math.hypot(
                    self.last_pos[0] - self.stuck_check_pos[0],
                    self.last_pos[1] - self.stuck_check_pos[1],
                )
                if moved < 0.15:
                    self.stuck_count += 1
                    self.get_logger().warn(
                        f'Stuck detected (moved {moved:.2f}m in 4s), '
                        f'forcing backup+turn (count={self.stuck_count})'
                    )
                    self.enter_backup()
                else:
                    self.stuck_count = 0
            self.stuck_check_pos = self.last_pos
            self.stuck_check_time = now

        if self.state == 'backup':
            cmd.linear.x = -0.12
            cmd.angular.z = 0.0
            if now > self.state_until:
                self.enter_turn(hard=self.stuck_count > 0)

        elif self.state == 'turn':
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_dir * 0.7
            if now > self.state_until:
                self.state = 'forward'

        else:
            if front < 0.55:
                self.enter_turn()
            elif left < 0.3 or right < 0.3:
                self.turn_dir = 1.0 if left < right else -1.0
                self.state = 'turn'
                self.state_until = now + random.uniform(0.8, 1.4)
            else:
                cmd.linear.x = 0.18
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main():
    import sys
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0

    rclpy.init()
    node = Wander(duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
