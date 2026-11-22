import math
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


class Grabber(Node):
    def __init__(self):
        super().__init__("capture_one_view")
        self.bridge = CvBridge()
        self.frame = None
        self.create_subscription(Image, "/camera/rgb/image_raw", self.cb, 10)

    def cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    name, ox, oy, dist, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    bearing = math.pi / 2 if oy < 0 else -math.pi / 2
    robot_x = ox + dist * math.cos(bearing)
    robot_y = oy + dist * math.sin(bearing)
    yaw = math.atan2(oy - robot_y, ox - robot_x)

    rclpy.init()
    node = Grabber()
    set_pose(robot_x, robot_y, yaw)
    node.spin_for(2.0)
    if node.frame is not None:
        cv2.imwrite(out, node.frame)
        print(f"saved {out}, robot=({robot_x:.2f},{robot_y:.2f}) yaw={math.degrees(yaw):.0f}")
    else:
        print("no frame received")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
