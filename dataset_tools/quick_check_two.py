import json
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CHECKS = [
    ("stop_sign", -4.83, 0.05, -90),
    ("speed_limit_sign", -2.12, 0.05, -90),
    ("speed_limit_sign", -2.12, -0.75, -90),
]


def set_pose(x, y, yaw_deg):
    import math
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(math.radians(yaw_deg))],
        capture_output=True, timeout=8.0
    )


class Check(Node):
    def __init__(self):
        super().__init__("quick_check_two")
        self.seen = []
        self.create_subscription(String, "/yolo/detections", self.cb, 50)

    def cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self.seen.append((d.get("class"), round(d.get("confidence", 0), 3)))

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    node = Check()
    for label, x, y, yaw_deg in CHECKS:
        node.seen.clear()
        set_pose(x, y, yaw_deg)
        node.spin_for(2.0)
        print(f"{label} @ ({x},{y},{yaw_deg}deg): seen={node.seen}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
