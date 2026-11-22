import csv
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import generate_yolo_dataset as gyd

YOLO_MODEL_PATH = (
    '/root/ros2_ws/realistic_sim_yolo_runs/'
    'realistic_sim_yolo11n/weights/best.pt'
)
CSV_PATH = os.path.expanduser('~/ros2_ws/localization_results.csv')

TARGETS = [
    ('cup_object', 'cup'),
    ('bowl_object', 'bowl'),
    ('can_object', 'coke_can'),
    ('drill_object', 'cordless_drill'),
    ('hammer_object', 'hammer'),
]

DISTANCES = [1.3, 1.8, 2.3]


class LocNode(Node):
    def __init__(self):
        super().__init__('localization_error_experiment')
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)
        self.latest_rgb = None
        self.latest_pc = None
        self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_cb, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_cb, 10)

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def pc_cb(self, msg):
        self.latest_pc = msg

    def get_xyz(self, u, v):
        msg = self.latest_pc
        if msg is None or u < 0 or u >= msg.width or v < 0 or v >= msg.height:
            return None
        offset = v * msg.row_step + u * msg.point_step
        x, y, z = struct.unpack_from('<fff', msg.data, offset)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        if z <= 0.1 or z > 10.0:
            return None
        return x, y, z

    def get_box_xyz(self, x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        pts = []
        for fx in (0.3, 0.5, 0.7):
            for fy in (0.3, 0.5, 0.7):
                xyz = self.get_xyz(int(x1 + w * fx), int(y1 + h * fy))
                if xyz is not None:
                    pts.append(xyz)
        if len(pts) < 3:
            return None
        arr = np.array(pts)
        return tuple(np.median(arr, axis=0))

    def wait_fresh(self, timeout=3.0):
        self.latest_rgb = None
        self.latest_pc = None
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_rgb is not None and self.latest_pc is not None:
                return True
        return False


def main():
    rclpy.init()
    node = LocNode()

    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, 'a', newline='')
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            'class', 'true_distance_m', 'true_cam_x', 'true_cam_y', 'true_cam_z',
            'meas_cam_x', 'meas_cam_y', 'meas_cam_z',
            'error_x', 'error_y', 'error_z', 'euclidean_error_m',
        ])

    errors = []

    for entity, cls in TARGETS:
        obj_def = next(o for o in gyd.OBJECTS if o['cls'] == cls)
        ox, oy = obj_def['pos']

        for dist in DISTANCES:
            robot_x = ox - dist
            robot_y = oy
            gyd.set_gz_pose('burger', robot_x, robot_y, 0.02, 0.0)
            time.sleep(0.6)

            if not node.wait_fresh():
                continue

            try:
                bx, by, bz, broll, bpitch, byaw = gyd.get_gz_pose('burger')
            except Exception:
                continue

            cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw, bz)
            obj_center_world = gyd.object_center_world(obj_def)
            rel = obj_center_world - cam_pos
            true_cam = R_cam_world.T @ rel

            results = node.model(node.latest_rgb, conf=0.2, imgsz=640, verbose=False)
            boxes = results[0].boxes
            best_box = None
            best_conf = -1
            if boxes is not None:
                for b in boxes:
                    if node.model.names[int(b.cls[0].item())] != cls:
                        continue
                    c = float(b.conf[0].item())
                    if c > best_conf:
                        best_conf = c
                        best_box = b.xyxy[0].tolist()

            if best_box is None:
                print(f'{cls:16s} dist={dist:.1f}m NOT DETECTED')
                continue

            meas_xyz = node.get_box_xyz(*best_box)
            if meas_xyz is None:
                print(f'{cls:16s} dist={dist:.1f}m detected but no valid depth')
                continue

            ex = meas_xyz[0] - true_cam[0]
            ey = meas_xyz[1] - true_cam[1]
            ez = meas_xyz[2] - true_cam[2]
            euclid = math.sqrt(ex**2 + ey**2 + ez**2)

            writer.writerow([
                cls, dist, *[round(v, 4) for v in true_cam],
                *[round(v, 4) for v in meas_xyz],
                round(ex, 4), round(ey, 4), round(ez, 4), round(euclid, 4),
            ])
            f.flush()
            errors.append(euclid)
            print(f'{cls:16s} dist={dist:.1f}m true={tuple(round(v,2) for v in true_cam)} '
                  f'meas={tuple(round(v,2) for v in meas_xyz)} err={euclid:.3f}m')

    f.close()
    node.destroy_node()
    rclpy.shutdown()

    if errors:
        arr = np.array(errors)
        print()
        print(f'N = {len(arr)}')
        print(f'MAE (mean abs euclidean error): {arr.mean():.4f} m')
        print(f'RMSE: {np.sqrt((arr**2).mean()):.4f} m')
        print(f'Median error: {np.median(arr):.4f} m')
        print(f'Max error: {arr.max():.4f} m')


if __name__ == '__main__':
    main()
