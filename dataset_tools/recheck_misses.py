import json
import subprocess
import sys
import time

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import generate_yolo_dataset as gyd

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

MISSES = {
    "cafe_table", "person", "beer", "fire_hydrant", "construction_cone",
    "construction_barrel", "cardboard_box", "cricket_ball", "monkey_wrench",
    "mailbox", "postbox", "stop_sign", "speed_limit_sign", "lamp_post",
    "first_2015_trash_can", "utility_cart", "grey_tote", "car_wheel",
    "door_handle", "kinect", "asus_xtion_pro_camera", "hokuyo",
    "table_marble", "wooden_case", "saucepan", "jersey_barrier",
}

CONF_THRESHOLD = 0.15
SETTLE_TIME = 1.3


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


class Verifier(Node):
    def __init__(self):
        super().__init__("recheck_misses")
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
        if cls not in MISSES:
            continue

        vps = gyd.sample_object_viewpoints(obj, n_angles=16)
        vps = [(x, y, yaw, c) for x, y, yaw, c in vps if gyd.is_free(x, y)]

        node.best_conf.pop(cls, None)
        best_seen = results.get(cls, (False, 0.0))[1]
        detected = results.get(cls, (False, 0.0))[0]

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
        print(f"[{status}] {cls} (entity={obj['entity']}): best_conf={best_seen:.3f}, tries={len(vps)}")

    print()
    print("=== RECHECK SUMMARY ===")
    hits = [c for c, (d, _) in results.items() if d]
    misses = [c for c, (d, _) in results.items() if not d]
    print(f"Now detected: {len(hits)}/{len(results)}")
    print("Still missing:", misses)

    with open("/root/ros2_ws/recheck_misses_results.json", "w") as f:
        json.dump(
            {c: {"detected": d, "best_conf": b} for c, (d, b) in results.items()},
            f, indent=2
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
