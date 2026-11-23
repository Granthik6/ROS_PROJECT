import json
import math
import sys
import time

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
from slam_wander import Wander
import generate_yolo_dataset as gyd

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

CONF_THRESHOLD = 0.15
REMOVED_FROM_WORLD = {"door_handle", "kinect", "asus_xtion_pro_camera", "speed_limit_sign"}
ROBOT_RADIUS = 0.18


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

ALL_CLASSES = [
    "cup", "bowl", "coke_can", "cordless_drill", "hammer", "bookshelf", "cabinet",
    "table", "cafe_table", "person", "beer", "fire_hydrant", "construction_cone",
    "construction_barrel", "cardboard_box", "cricket_ball", "monkey_wrench",
    "mailbox", "postbox", "stop_sign", "lamp_post", "first_2015_trash_can",
    "utility_cart", "grey_tote", "car_wheel", "hokuyo", "table_marble",
    "wooden_case", "saucepan", "jersey_barrier", "stop_light", "oak_tree",
    "ball_bearing", "gear_part", "fire_hose_long",
]


class WanderAndLog(Wander):
    def __init__(self, duration_sec, report_path):
        super().__init__(duration_sec)
        self.best_conf = {}
        self.report_path = report_path
        self.yaw = 0.0
        self.create_subscription(String, "/yolo/detections", self.on_detection, 50)
        self.report_timer = self.create_timer(20.0, self.report)

    def odom_cb(self, msg):
        super().odom_cb(msg)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def step(self):
        if self.last_pos is not None:
            x, y = self.last_pos
            margin = 0.45
            near_wall = (
                x < gyd.ROOM_X_MIN + margin or x > gyd.ROOM_X_MAX - margin or
                y < gyd.ROOM_Y_MIN + margin or y > gyd.ROOM_Y_MAX - margin
            )
            if near_wall:
                heading = math.atan2(0.0 - y, 0.0 - x)
                err = wrap(heading - self.yaw)
                cmd = Twist()
                if abs(err) > 0.4:
                    cmd.angular.z = max(-0.5, min(0.5, 1.2 * err))
                else:
                    cmd.linear.x = 0.12
                    cmd.angular.z = max(-0.4, min(0.4, 1.0 * err))
                self.cmd_pub.publish(cmd)
                self.state = 'forward'
                return
        super().step()

    def object_clearance(self, center_deg, width_deg):
        if self.last_pos is None:
            return 10.0
        x, y = self.last_pos
        center = self.yaw + math.radians(center_deg)
        best = 10.0
        for obj in gyd.OBJECTS:
            if obj["cls"] in REMOVED_FROM_WORLD:
                continue
            ox, oy = obj["pos"]
            dx, dy = ox - x, oy - y
            dist = math.hypot(dx, dy)
            if dist < 0.02:
                continue
            rel = wrap(math.atan2(dy, dx) - center)
            if abs(rel) > math.radians(width_deg):
                continue
            margin = dist - (gyd.object_radius(obj) + ROBOT_RADIUS)
            if margin < best:
                best = margin
        return max(best, 0.0)

    def clearance(self, center_deg, width_deg=30):
        lidar = super().clearance(center_deg, width_deg)
        obj = self.object_clearance(center_deg, width_deg)
        return min(lidar, obj)

    def enter_backup(self):
        rear = self.clearance(180, 40)
        if rear < 0.3:
            self.enter_turn(hard=True)
            return
        self.state = 'backup'
        backup_time = min(0.8 + min(self.stuck_count, 10) * 0.15, 1.6)
        self.state_until = time.time() + backup_time

    def enter_turn(self, hard=False):
        self.state = 'turn'
        left = self.clearance(90, 45)
        right = self.clearance(-90, 45)
        self.turn_dir = 1.0 if left > right else -1.0
        turn_time = 1.5 + min(self.stuck_count, 10) * 0.35
        self.state_until = time.time() + turn_time

    def on_detection(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        cls = d.get("class")
        conf = d.get("confidence", 0.0)
        if cls and conf > self.best_conf.get(cls, 0.0):
            self.best_conf[cls] = conf
            self.get_logger().info(f"NEW BEST {cls}: {conf:.3f}")

    def report(self):
        hits = [c for c in ALL_CLASSES if self.best_conf.get(c, 0.0) >= CONF_THRESHOLD]
        elapsed = time.time() - self.start_time
        self.get_logger().info(
            f"[progress t={elapsed:.0f}s] {len(hits)}/{len(ALL_CLASSES)} detected so far. "
            f"Missing: {[c for c in ALL_CLASSES if c not in hits]}"
        )
        with open(self.report_path, "w") as f:
            json.dump(
                {c: {"detected": self.best_conf.get(c, 0.0) >= CONF_THRESHOLD,
                     "best_conf": self.best_conf.get(c, 0.0)} for c in ALL_CLASSES},
                f, indent=2
            )


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    report_path = sys.argv[2] if len(sys.argv) > 2 else "/root/ros2_ws/wander_verify_results.json"

    rclpy.init()
    node = WanderAndLog(duration, report_path)
    try:
        end = time.time() + duration + 3.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
