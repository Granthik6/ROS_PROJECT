import json
import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TARGETS = [
    ("cordless_drill", -1.0, 0.0),
    ("cricket_ball", 1.25, -4.5),
    ("jersey_barrier", -2.12, -0.45),
    ("stop_light", -0.77, -0.45),
    ("oak_tree", 0.58, -0.45),
    ("table_marble", 3.95, -1.8),
    ("wooden_case", 5.3, -1.8),
    ("door_handle", -1.45, -1.8),
    ("monkey_wrench", 2.6, -4.5),
    ("stop_sign", -4.83, -3.15),
]

DIST = 0.6


class Diag(Node):
    def __init__(self):
        super().__init__("diagnose_misses")
        self.all_seen = []
        self.create_subscription(String, "/yolo/detections", self.cb, 50)

    def cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self.all_seen.append((d.get("class"), round(d.get("confidence", 0), 3)))

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


def main():
    rclpy.init()
    node = Diag()

    for cls, ox, oy in TARGETS:
        bearing = math.pi / 2 if oy < 0 else -math.pi / 2
        robot_x = ox + DIST * math.cos(bearing)
        robot_y = oy + DIST * math.sin(bearing)
        yaw = math.atan2(oy - robot_y, ox - robot_x)
        node.all_seen.clear()
        set_pose(robot_x, robot_y, yaw)
        node.spin_for(2.0)
        print(f"{cls}: robot=({robot_x:.2f},{robot_y:.2f}) yaw={math.degrees(yaw):.0f} "
              f"seen={node.all_seen}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
