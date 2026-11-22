import csv
import math
import os
import subprocess
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

YOLO_MODEL_PATH = (
    '/root/ros2_ws/realistic_sim_yolo_runs/'
    'realistic_sim_yolo11n/weights/best.pt'
)
CSV_PATH = os.path.expanduser('~/ros2_ws/confidence_distance.csv')

# object entity name, class name, world (x,y), approx center height
TARGETS = [
    ('cup_object', 'cup', (-1.8, -2.5), 0.057),
    ('bowl_object', 'bowl', (-0.4, -2.8), 0.018),
    ('can_object', 'coke_can', (1.2, -2.5), 0.064),
    ('drill_object', 'cordless_drill', (-1.0, 0.0), 0.129),
    ('hammer_object', 'hammer', (1.8, 0.0), 0.017),
]

DISTANCES = [1.1, 1.4, 1.8, 2.2, 2.6, 3.0]
CAM_HEIGHT = 0.47


def set_gz_pose(x, y, z, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger",
         "-x", str(x), "-y", str(y), "-z", str(z),
         "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, text=True, timeout=8.0
    )


class ConfDistNode(Node):
    def __init__(self):
        super().__init__('confidence_distance_experiment')
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)
        self.latest_rgb = None
        self.create_subscription(Image, '/camera/rgb/image_raw', self.cb, 10)

    def cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def wait_frame(self, timeout=3.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_rgb is not None:
                return True
        return False


def main():
    rclpy.init()
    node = ConfDistNode()

    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, 'a', newline='')
    writer = csv.writer(f)
    if new_file:
        writer.writerow(['class', 'true_distance_m', 'confidence', 'detected'])

    results = []

    for entity, cls, (ox, oy), oz in TARGETS:
        for dist in DISTANCES:
            depression = math.atan2(CAM_HEIGHT - oz, dist)
            if depression > math.radians(22.0):
                continue  # would fall outside vertical FOV, skip (see camera geometry notes)

            robot_x = ox - dist
            robot_y = oy
            yaw = 0.0  # face +x, i.e. toward the object at (ox,oy) from the west

            set_gz_pose(robot_x, robot_y, 0.02, yaw)
            time.sleep(0.6)
            node.latest_rgb = None
            if not node.wait_frame():
                continue

            best_conf = 0.0
            detected = False
            for _ in range(3):
                yres = node.model(node.latest_rgb, conf=0.15, imgsz=640, verbose=False)
                boxes = yres[0].boxes
                if boxes is not None:
                    for b in boxes:
                        if node.model.names[int(b.cls[0].item())] == cls:
                            c = float(b.conf[0].item())
                            if c > best_conf:
                                best_conf = c
                            detected = True
                rclpy.spin_once(node, timeout_sec=0.1)

            writer.writerow([cls, dist, round(best_conf, 4), detected])
            f.flush()
            results.append((cls, dist, best_conf, detected))
            print(f'{cls:16s} dist={dist:.1f}m conf={best_conf:.3f} detected={detected}')

    f.close()
    node.destroy_node()
    rclpy.shutdown()

    print()
    print(f'Saved {len(results)} rows to {CSV_PATH}')


if __name__ == '__main__':
    main()
