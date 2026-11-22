import csv
import json
import math
import subprocess
import sys
import time

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import generate_yolo_dataset as gyd

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TARGET_CLASSES = {"cafe_table", "person", "cricket_ball", "monkey_wrench", "speed_limit_sign"}
DISTANCES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
CONF_THRESHOLD = 0.15
SETTLE_TIME = 1.3
ANGLE_OFFSETS_DEG = [-60, -30, -10, 10, 30, 60]
OUT_CSV = "/root/ros2_ws/detection_range_wideangle_results.csv"


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


class RangeMeasurer(Node):
    def __init__(self):
        super().__init__("measure_range_wide_angle")
        self.best_conf = {}
        self.create_subscription(String, "/yolo/detections", self.on_detection, 50)

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


def main():
    rclpy.init()
    node = RangeMeasurer()

    rows = []

    for obj in gyd.OBJECTS:
        cls = obj["cls"]
        if cls not in TARGET_CLASSES:
            continue
        ox, oy = obj["pos"]
        bearing = math.pi / 2 if oy < 0 else -math.pi / 2

        per_distance_conf = rows and next(
            (r["_per_dist"] for r in rows if r["class"] == cls), None
        ) or {}

        for dist in DISTANCES:
            node.best_conf.pop(cls, None)
            best_at_this_dist = per_distance_conf.get(dist, 0.0)

            for dtheta_deg in ANGLE_OFFSETS_DEG:
                theta = bearing + math.radians(dtheta_deg)
                robot_x = ox + dist * math.cos(theta)
                robot_y = oy + dist * math.sin(theta)

                if not gyd.is_free(robot_x, robot_y):
                    continue

                yaw = math.atan2(oy - robot_y, ox - robot_x)
                set_pose(robot_x, robot_y, yaw)
                node.spin_for(SETTLE_TIME)

                conf = node.best_conf.get(cls, 0.0)
                if conf > best_at_this_dist:
                    best_at_this_dist = conf

            per_distance_conf[dist] = best_at_this_dist

        detected_distances = [d for d, c in per_distance_conf.items() if c >= CONF_THRESHOLD]
        min_dist = min(detected_distances) if detected_distances else None
        max_dist = max(detected_distances) if detected_distances else None
        best_conf = max(per_distance_conf.values()) if per_distance_conf else 0.0

        status = "DETECTED" if detected_distances else "NOT DETECTED"
        range_str = f"{min_dist}m - {max_dist}m" if detected_distances else "n/a"
        print(f"[{status:12s}] {cls:22s} ({obj['entity']}) range={range_str:16s} best_conf={best_conf:.3f} "
              f"per_distance={ {d: round(c,2) for d, c in per_distance_conf.items()} }")

        rows.append({
            "class": cls,
            "entity": obj["entity"],
            "detected": bool(detected_distances),
            "min_detect_dist_m": min_dist,
            "max_detect_dist_m": max_dist,
            "best_confidence": round(best_conf, 4),
            "_per_dist": per_distance_conf,
            **{f"conf_at_{d}m": round(c, 4) for d, c in per_distance_conf.items()},
        })

    for r in rows:
        r.pop("_per_dist", None)

    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Full results written to {OUT_CSV}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
