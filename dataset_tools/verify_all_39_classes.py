import json
import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# (include_name, class_name, x, y)
OBJECTS = [
    ("bookshelf_main", "bookshelf", -4.5, 3.8),
    ("cabinet_main", "cabinet", 4.5, 3.8),
    ("table_main", "table", 0.0, 2.0),
    ("cafe_table_main", "cafe_table", -3.2, -1.8),
    ("person_standing_main", "person", -3.8, 0.3),
    ("cup_object", "cup", -1.8, -2.5),
    ("bowl_object", "bowl", -0.4, -2.8),
    ("can_object", "coke_can", 1.2, -2.5),
    ("drill_object", "cordless_drill", -1.0, 0.0),
    ("hammer_object", "hammer", 1.8, 0.0),
    ("beer_object", "beer", -5.5, -4.5),
    ("fire_hydrant_object", "fire_hydrant", -4.15, -4.5),
    ("construction_cone_object", "construction_cone", -2.8, -4.5),
    ("construction_barrel_object", "construction_barrel", -1.45, -4.5),
    ("cardboard_box_object", "cardboard_box", -0.1, -4.5),
    ("cricket_ball_object", "cricket_ball", 1.25, -4.5),
    ("monkey_wrench_object", "monkey_wrench", 2.6, -4.5),
    ("mailbox_object", "mailbox", 3.95, -4.5),
    ("postbox_object", "postbox", 5.3, -4.5),
    ("stop_sign_object", "stop_sign", -4.83, -3.15),
    ("speed_limit_sign_object", "speed_limit_sign", -2.12, -3.15),
    ("lamp_post_object", "lamp_post", -0.77, -3.15),
    ("first_2015_trash_can_object", "first_2015_trash_can", 0.58, -3.15),
    ("utility_cart_object", "utility_cart", 1.93, -3.15),
    ("grey_tote_object", "grey_tote", 3.28, -3.15),
    ("car_wheel_object", "car_wheel", -5.5, -1.8),
    ("door_handle_object", "door_handle", -1.45, -1.8),
    ("kinect_object", "kinect", -0.1, -1.8),
    ("asus_xtion_pro_camera_object", "asus_xtion_pro_camera", 1.25, -1.8),
    ("hokuyo_object", "hokuyo", 2.6, -1.8),
    ("table_marble_object", "table_marble", 3.95, -1.8),
    ("wooden_case_object", "wooden_case", 5.3, -1.8),
    ("saucepan_object", "saucepan", -4.83, -0.45),
    ("jersey_barrier_object", "jersey_barrier", -2.12, -0.45),
    ("stop_light_object", "stop_light", -0.77, -0.45),
    ("oak_tree_object", "oak_tree", 0.58, -0.45),
    ("ball_bearing_object", "ball_bearing", 3.28, -0.45),
    ("gear_part_object", "gear_part", 4.62, -0.45),
    ("fire_hose_long_object", "fire_hose_long", -5.5, 0.9),
]

DISTANCES = [0.8, 1.2, 1.6]
ANGLE_OFFSETS_DEG = [-20, 0, 20]
SETTLE_TIME = 1.3
CONF_THRESHOLD = 0.15


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


class Verifier(Node):

    def __init__(self):
        super().__init__("verify_all_39_classes")
        self.best_conf = {}
        self.create_subscription(
            String, "/yolo/detections", self.on_detection, 50
        )

    def on_detection(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        cls = d.get("class")
        conf = d.get("confidence", 0.0)
        if cls is None:
            return
        prev = self.best_conf.get(cls, 0.0)
        if conf > prev:
            self.best_conf[cls] = conf

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    node = Verifier()

    results = {}

    for include_name, class_name, ox, oy in OBJECTS:
        # approach from the side toward the open central aisle (y=0)
        bearing = math.pi / 2 if oy < 0 else -math.pi / 2
        node.best_conf.pop(class_name, None)
        detected = False
        best_seen = 0.0

        for dist in DISTANCES:
            for dtheta_deg in ANGLE_OFFSETS_DEG:
                theta = bearing + math.radians(dtheta_deg)
                robot_x = ox + dist * math.cos(theta)
                robot_y = oy + dist * math.sin(theta)
                yaw = math.atan2(oy - robot_y, ox - robot_x)
                set_pose(robot_x, robot_y, yaw)
                node.spin_for(SETTLE_TIME)

                conf = node.best_conf.get(class_name, 0.0)
                if conf > best_seen:
                    best_seen = conf
                if conf >= CONF_THRESHOLD:
                    detected = True

        results[class_name] = (detected, best_seen)
        status = "OK" if detected else "MISS"
        print(f"[{status}] {class_name}: best_conf={best_seen:.3f}")

    print()
    print("=== SUMMARY ===")
    hits = [c for c, (d, _) in results.items() if d]
    misses = [c for c, (d, _) in results.items() if not d]
    print(f"Detected: {len(hits)}/{len(results)}")
    print("Missing:", misses)

    with open("/root/ros2_ws/verify_all_39_results.json", "w") as f:
        json.dump(
            {c: {"detected": d, "best_conf": b} for c, (d, b) in results.items()},
            f, indent=2
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
