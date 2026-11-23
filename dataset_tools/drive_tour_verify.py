import json
import math
import sys
import time

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import generate_yolo_dataset as gyd

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def euler_from_quaternion(q):
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return 0.0, 0.0, yaw

CONF_THRESHOLD = 0.15
REMOVED_FROM_WORLD = {"door_handle", "kinect", "asus_xtion_pro_camera", "speed_limit_sign"}
GOAL_TOLERANCE = 0.5
DRIVE_TIMEOUT = 25.0
LOOK_SWEEP_DEG = [-35, -15, 0, 15, 35]
LOOK_SETTLE = 1.1
STUCK_WINDOW = 5.0
STUCK_DIST = 0.15


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class DriveTour(Node):
    def __init__(self):
        super().__init__("drive_tour_verify")
        self.pose = None
        self.scan = None
        self.best_conf = {}

        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(String, "/yolo/detections", self.on_detection, 50)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (p.x, p.y, yaw)

    def on_scan(self, msg):
        self.scan = msg

    def on_detection(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        cls = d.get("class")
        conf = d.get("confidence", 0.0)
        if cls and conf > self.best_conf.get(cls, 0.0):
            self.best_conf[cls] = conf

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def clearance(self, center_deg, width_deg=30):
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

    def stop(self):
        self.cmd_pub.publish(Twist())

    def drive_to(self, gx, gy, timeout=DRIVE_TIMEOUT):
        start = time.time()
        stuck_anchor = self.pose[:2] if self.pose else None
        stuck_anchor_t = time.time()
        recovering_until = 0.0
        recovery_turn_dir = 1.0

        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.pose is None:
                continue
            x, y, yaw = self.pose
            dist = math.hypot(gx - x, gy - y)
            if dist < GOAL_TOLERANCE:
                self.stop()
                return True

            now = time.time()
            if stuck_anchor is None:
                stuck_anchor = (x, y)
                stuck_anchor_t = now
            elif math.hypot(x - stuck_anchor[0], y - stuck_anchor[1]) > STUCK_DIST:
                stuck_anchor = (x, y)
                stuck_anchor_t = now
            elif now - stuck_anchor_t > STUCK_WINDOW and now >= recovering_until:
                recovering_until = now + 2.5
                recovery_turn_dir = 1.0 if self.clearance(90, 40) > self.clearance(-90, 40) else -1.0
                stuck_anchor = (x, y)
                stuck_anchor_t = now

            front = self.clearance(0, 40)
            left = self.clearance(60, 30)
            right = self.clearance(-60, 30)

            cmd = Twist()
            if now < recovering_until:
                cmd.linear.x = -0.15
                cmd.angular.z = 0.6 * recovery_turn_dir
            elif front < 0.4:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.6 if left > right else -0.6
            else:
                heading = math.atan2(gy - y, gx - x)
                err = wrap(heading - yaw)
                if abs(err) > 0.5:
                    cmd.linear.x = 0.0
                    cmd.angular.z = max(-0.8, min(0.8, 1.5 * err))
                else:
                    cmd.linear.x = min(0.22, 0.15 + 0.1 * dist)
                    cmd.angular.z = max(-0.6, min(0.6, 1.2 * err))
            self.cmd_pub.publish(cmd)
        self.stop()
        return False

    def look_around(self, cls, ox, oy):
        if self.pose is None:
            return
        x, y, _ = self.pose
        base_yaw = math.atan2(oy - y, ox - x)
        for off_deg in LOOK_SWEEP_DEG:
            target_yaw = wrap(base_yaw + math.radians(off_deg))
            t0 = time.time()
            while time.time() - t0 < 4.0:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.pose is None:
                    break
                _, _, yaw = self.pose
                err = wrap(target_yaw - yaw)
                if abs(err) < 0.08:
                    self.stop()
                    break
                cmd = Twist()
                cmd.angular.z = max(-0.5, min(0.5, 1.5 * err))
                self.cmd_pub.publish(cmd)
            self.spin_for(LOOK_SETTLE)


def approach_point(ox, oy, dist):
    for k in range(24):
        theta = 2 * math.pi * k / 24
        rx = ox + dist * math.cos(theta)
        ry = oy + dist * math.sin(theta)
        if gyd.is_free(rx, ry):
            return rx, ry
    return ox + dist, oy


def main():
    rclpy.init()
    node = DriveTour()
    node.spin_for(1.0)

    results = {}
    seen_cls = set()

    for obj in gyd.OBJECTS:
        cls = obj["cls"]
        if cls in REMOVED_FROM_WORLD:
            continue

        ox, oy = obj["pos"]
        dist = (obj.get("distances") or [1.5])[len(obj.get("distances") or [1.5]) // 2]
        gx, gy = approach_point(ox, oy, dist)

        node.get_logger().info(f"--- Driving toward {cls} @ ({gx:.2f},{gy:.2f}) ---")
        reached = node.drive_to(gx, gy)

        node.best_conf.pop(cls, None)
        node.look_around(cls, ox, oy)

        conf = node.best_conf.get(cls, 0.0)
        prev_detected, prev_best = results.get(cls, (False, 0.0))
        detected = prev_detected or conf >= CONF_THRESHOLD
        best = max(prev_best, conf)
        results[cls] = (detected, best)
        seen_cls.add(cls)

        status = "OK" if detected else "MISS"
        print(f"[{status}] {cls}: reached={reached} best_conf={best:.3f}", flush=True)

    node.stop()

    print()
    print("=== FINAL SUMMARY (drive-tour, teleop-style) ===")
    hits = [c for c, (d, _) in results.items() if d]
    misses = [c for c, (d, _) in results.items() if not d]
    print(f"Detected: {len(hits)}/{len(results)}")
    print("Missing:", misses)

    with open("/root/ros2_ws/drive_tour_verify_results.json", "w") as f:
        json.dump(
            {c: {"detected": d, "best_conf": b} for c, (d, b) in results.items()},
            f, indent=2
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
