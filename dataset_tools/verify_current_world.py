import json
import subprocess
import sys
import time

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import generate_yolo_dataset as gyd

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CONF_THRESHOLD = 0.15
SETTLE_TIME = 1.3
N_ANGLES = 15

REMOVED_FROM_WORLD = {"door_handle", "kinect", "asus_xtion_pro_camera", "speed_limit_sign"}


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


class Verifier(Node):
    def __init__(self):
        super().__init__("verify_current_world")
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
    node = Verifier()

    results = {}

    for obj in gyd.OBJECTS:
        cls = obj["cls"]
        if cls in REMOVED_FROM_WORLD:
            continue

        vps = gyd.sample_object_viewpoints(obj, n_angles=N_ANGLES)
        vps = [(x, y, yaw, c) for x, y, yaw, c in vps if gyd.is_free(x, y)]

        prev_detected, prev_best = results.get(cls, (False, 0.0))
        node.best_conf.pop(cls, None)
        best_seen = prev_best
        detected = prev_detected

        for x, y, yaw, _ in vps:
            set_pose(x, y, yaw)
            node.spin_for(SETTLE_TIME)
            conf = node.best_conf.get(cls, 0.0)
            if conf > best_seen:
                best_seen = conf
            if conf >= CONF_THRESHOLD:
                detected = True

        results[cls] = (detected, best_seen)
        status = "OK" if detected else "MISS"
        print(f"[{status}] {cls} (entity={obj['entity']}): best_conf={best_seen:.3f}, tries={len(vps)}", flush=True)

    print()
    print("=== FINAL SUMMARY ===")
    hits = [c for c, (d, _) in results.items() if d]
    misses = [c for c, (d, _) in results.items() if not d]
    print(f"Detected: {len(hits)}/{len(results)}")
    print("Missing:", misses)

    with open("/root/ros2_ws/verify_current_world_results.json", "w") as f:
        json.dump(
            {c: {"detected": d, "best_conf": b} for c, (d, b) in results.items()},
            f, indent=2
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
