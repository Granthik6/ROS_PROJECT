import math
import os
import random
import subprocess
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Camera intrinsics (from robot_tasks camera spec)
# ---------------------------------------------------------------------------
FX = 554.254691191187
FY = 554.254691191187
CX = 320.5
CY = 240.5
IMG_W = 640
IMG_H = 480
NEAR_CLIP = 0.10
FAR_CLIP = 10.0

# camera_link relative to base_link (static_transform_publisher in
# realistic_sim.launch.py): x y z roll pitch yaw
CAM_OFFSET = np.array([0.105, 0.0, 0.45])
CAM_ROLL = -1.57079632679
CAM_PITCH = 0.0
CAM_YAW = -1.57079632679

DATASET_ROOT = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_dataset")
RAW_IMAGES = os.path.join(DATASET_ROOT, "raw_images")
RAW_LABELS = os.path.join(DATASET_ROOT, "raw_labels")

CLASS_NAMES = [
    "cup", "bowl", "coke_can", "cordless_drill", "hammer",
    "bookshelf", "cabinet", "table", "cafe_table", "person",
    "beer", "fire_hydrant", "construction_cone", "construction_barrel",
    "cardboard_box", "cricket_ball", "monkey_wrench", "mailbox", "postbox",
    "stop_sign", "speed_limit_sign", "lamp_post", "first_2015_trash_can",
    "utility_cart", "grey_tote", "car_wheel", "door_handle", "kinect",
    "asus_xtion_pro_camera", "hokuyo", "table_marble", "wooden_case",
    "saucepan", "jersey_barrier", "stop_light", "oak_tree", "ball_bearing",
    "gear_part", "fire_hose_long",
]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# gazebo entity name, class name, world (x, y, yaw), local half-extent (m),
# local center offset (m) in the model's own frame (rotates with object yaw).
# Small graspable objects use the default distance list in
# sample_object_viewpoints; furniture/person are much bigger and need to be
# viewed from farther away to fit in frame, so they carry their own
# "distances" override.
OBJECTS = [
    dict(entity="cup_object", cls="cup", pos=(-1.8, -2.5), yaw=0.0,
         half=(0.03725, 0.03725, 0.0651), offset=(0.0, 0.0079, 0.1301)),
    dict(entity="bowl_object", cls="bowl", pos=(-0.4, -2.8), yaw=0.5,
         half=(0.0978, 0.0978, 0.01745), offset=(0.0, 0.0, 0.0175)),
    dict(entity="can_object", cls="coke_can", pos=(1.2, -2.5), yaw=0.0,
         half=(0.0335, 0.0335, 0.06195), offset=(-0.0006, -0.0007, 0.0644)),
    dict(entity="drill_object", cls="cordless_drill", pos=(-1.0, 0.0), yaw=0.8,
         half=(0.0907, 0.116, 0.1273), offset=(-0.053, 0.0102, 0.1293)),
    dict(entity="hammer_object", cls="hammer", pos=(1.8, 0.0), yaw=1.2,
         half=(0.06675, 0.1674, 0.0168), offset=(0.0086, -0.0694, 0.0168)),
    dict(entity="bookshelf_main", cls="bookshelf", pos=(-4.5, 3.8), yaw=0.0,
         half=(0.46, 0.2025, 0.6), offset=(0.0, -0.1925, 0.6),
         distances=[1.5, 2.0, 2.7, 3.5]),
    dict(entity="cabinet_main", cls="cabinet", pos=(4.5, 3.8), yaw=3.1415926,
         half=(0.235, 0.245, 0.51), offset=(0.01, 0.0, 0.51),
         distances=[1.3, 1.8, 2.4, 3.2]),
    dict(entity="table_main", cls="table", pos=(0.0, 2.0), yaw=0.0,
         half=(0.75, 0.4, 0.5075), offset=(0.0, 0.0, 0.5075),
         distances=[1.6, 2.2, 3.0, 4.0]),
    dict(entity="cafe_table_main", cls="cafe_table", pos=(-3.2, -1.8), yaw=0.6,
         half=(0.4572, 0.4572, 0.3875), offset=(0.0, 0.0, 0.3875),
         distances=[1.5, 2.0, 2.7, 3.5]),
    dict(entity="person_standing_main", cls="person", pos=(-3.8, 0.3), yaw=2.5,
         half=(0.2721, 0.1928, 0.9654), offset=(-0.0053, -0.0822, 0.9654),
         distances=[1.5, 2.2, 3.0, 4.0]),
    dict(entity="person_walking_main", cls="person", pos=(3.7, 3.0), yaw=-1.5,
         half=(0.273, 0.441, 0.9379), offset=(-0.0008, 0.0359, 0.9379),
         distances=[1.5, 2.2, 3.0, 4.0]),
    dict(entity="beer_object", cls="beer", pos=(-5.5, -4.5), yaw=0.0,
         half=(0.04, 0.04, 0.13), offset=(0.0, 0.0, 0.2450),
         distances=[1.0, 1.4, 1.9, 2.5]),
    dict(entity="fire_hydrant_object", cls="fire_hydrant", pos=(-4.15, -4.5), yaw=0.0,
         half=(0.1566, 0.204, 0.3886), offset=(-0.0093, 0.0033, 0.4453),
         distances=[1.3, 1.8, 2.4, 3.2]),
    dict(entity="construction_cone_object", cls="construction_cone", pos=(-2.8, -4.5), yaw=0.0,
         half=(0.18, 0.18, 0.35), offset=(0.0, 0.0, 0.35),
         distances=[1.1, 1.5, 2.0, 2.7]),
    dict(entity="construction_barrel_object", cls="construction_barrel", pos=(-1.45, -4.5), yaw=0.0,
         half=(0.3, 0.3, 0.45), offset=(0.0, 0.0, 0.45),
         distances=[1.3, 1.8, 2.4, 3.2]),
    dict(entity="cardboard_box_object", cls="cardboard_box", pos=(-0.1, -4.5), yaw=0.0,
         half=(0.25, 0.2, 0.15), offset=(0.0, 0.0, 0.15),
         distances=[1.1, 1.5, 2.0, 2.7]),
    dict(entity="cricket_ball_object", cls="cricket_ball", pos=(1.25, -4.5), yaw=0.0,
         half=(0.04, 0.04, 0.04), offset=(0.0, 0.0, 0.0775),
         distances=[0.9, 1.2, 1.6, 2.1]),
    dict(entity="monkey_wrench_object", cls="monkey_wrench", pos=(2.6, -4.5), yaw=0.0,
         half=(0.1953, 0.0417, 0.0134), offset=(-0.0075, 0.0123, 0.0042),
         distances=[0.9, 1.2, 1.6, 2.1]),
    dict(entity="mailbox_object", cls="mailbox", pos=(3.95, -4.5), yaw=0.0,
         half=(0.15, 0.2, 0.4), offset=(0.0, 0.0, 0.5),
         distances=[1.1, 1.5, 2.0, 2.7]),
    dict(entity="postbox_object", cls="postbox", pos=(5.3, -4.5), yaw=0.0,
         half=(0.25, 0.25, 0.55), offset=(0.0, 0.0, 0.55),
         distances=[1.3, 1.8, 2.4, 3.2]),
    dict(entity="stop_sign_object", cls="stop_sign", pos=(-4.83, -3.15), yaw=0.0,
         half=(0.2965, 0.0114, 0.9088), offset=(0.0, 0.0, 0.9088),
         distances=[2.0, 2.6, 3.2, 3.8]),
    dict(entity="speed_limit_sign_object", cls="speed_limit_sign", pos=(-0.1, 0.6), yaw=0.0,
         half=(0.3892, 0.0148, 1.1801), offset=(0.0000, -0.0375, 1.1801),
         distances=[3.0, 3.6, 4.2, 4.8]),
    dict(entity="lamp_post_object", cls="lamp_post", pos=(-0.77, -3.15), yaw=0.0,
         half=(0.15, 0.15, 1.2), offset=(0.0, 0.0, 1.2),
         distances=[1.5, 2.0, 2.7, 3.5]),
    dict(entity="first_2015_trash_can_object", cls="first_2015_trash_can", pos=(0.58, -3.15), yaw=0.0,
         half=(0.2, 0.2, 0.35), offset=(0.0, 0.0, 0.35),
         distances=[1.1, 1.5, 2.0, 2.7]),
    dict(entity="utility_cart_object", cls="utility_cart", pos=(1.93, -3.15), yaw=0.0,
         half=(0.35, 0.25, 0.45), offset=(0.0, 0.0, 0.5500),
         distances=[1.3, 1.8, 2.4, 3.2]),
    dict(entity="grey_tote_object", cls="grey_tote", pos=(3.28, -3.15), yaw=0.0,
         half=(0.25, 0.2, 0.15), offset=(0.0, 0.0, 0.15),
         distances=[1.1, 1.5, 2.0, 2.7]),
    dict(entity="car_wheel_object", cls="car_wheel", pos=(-5.5, -1.8), yaw=0.0,
         half=(0.18, 0.1, 0.18), offset=(0.0, 0.0, 0.2606),
         distances=[1.0, 1.4, 1.9, 2.5]),
    dict(entity="door_handle_object", cls="door_handle", pos=(-5.9, 2.5), yaw=0.0,
         half=(0.1101, 0.0508, 0.0479), offset=(0.0593, 0.0000, 0.9479),
         distances=[1.2, 1.5, 1.8, 2.2]),
    dict(entity="kinect_object", cls="kinect", pos=(3.28, -3.05), yaw=0.0,
         half=(0.0520, 0.1380, 0.0445), offset=(-0.0155, 0.0000, 0.3445),
         distances=[0.6, 0.9, 1.3, 1.8]),
    dict(entity="asus_xtion_pro_camera_object", cls="asus_xtion_pro_camera", pos=(-0.1, -4.4), yaw=0.0,
         half=(0.0200, 0.0900, 0.0235), offset=(0.0000, 0.0000, 0.3235),
         distances=[0.6, 0.9, 1.3, 1.8]),
    dict(entity="hokuyo_object", cls="hokuyo", pos=(2.6, -1.8), yaw=0.0,
         half=(0.025, 0.025, 0.035), offset=(0.0, 0.0, 0.035),
         distances=[0.9, 1.2, 1.6, 2.1]),
    dict(entity="table_marble_object", cls="table_marble", pos=(-2.0, 3.9), yaw=0.0,
         half=(1.3062, 0.9515, 0.4816), offset=(0.0, -0.1226, 0.4832),
         distances=[2.4, 3.0, 3.6, 4.2]),
    dict(entity="wooden_case_object", cls="wooden_case", pos=(5.3, -1.8), yaw=0.0,
         half=(0.3, 0.2, 0.15), offset=(0.0, 0.0, 0.1740),
         distances=[1.1, 1.5, 2.0, 2.7]),
    dict(entity="saucepan_object", cls="saucepan", pos=(-4.83, -0.45), yaw=0.0,
         half=(0.3187, 0.3000, 0.0532), offset=(0.1917, 0.1730, 0.0533),
         distances=[0.9, 1.2, 1.6, 2.1]),
    dict(entity="jersey_barrier_object", cls="jersey_barrier", pos=(-2.12, -0.45), yaw=0.0,
         half=(0.4, 0.3, 0.4), offset=(0.0, 0.0, 0.4),
         distances=[1.3, 1.8, 2.4, 3.2]),
    dict(entity="stop_light_object", cls="stop_light", pos=(-0.77, -0.45), yaw=0.0,
         half=(0.127, 0.1658, 0.381), offset=(0.0, -0.1214, 0.381),
         distances=[1.5, 2.0, 2.7, 3.5]),
    dict(entity="oak_tree_object", cls="oak_tree", pos=(0.58, -0.45), yaw=0.0,
         half=(0.6, 0.6, 1.2), offset=(0.0, 0.0, 1.2703),
         distances=[1.6, 2.2, 3.0, 4.0]),
    dict(entity="ball_bearing_object", cls="ball_bearing", pos=(3.28, -0.45), yaw=0.0,
         half=(0.03, 0.03, 0.03), offset=(0.0, 0.0, 0.03),
         distances=[0.8, 1.1, 1.5, 2.0]),
    dict(entity="gear_part_object", cls="gear_part", pos=(4.62, -0.45), yaw=0.0,
         half=(0.0392, 0.0392, 0.0060), offset=(0.0000, 0.0000, 0.0059),
         distances=[0.9, 1.2, 1.6, 2.1]),
    dict(entity="fire_hose_long_object", cls="fire_hose_long", pos=(-5.5, 0.9), yaw=0.0,
         half=(0.3, 0.3, 0.15), offset=(0.0, 0.0, 0.2500),
         distances=[1.1, 1.5, 2.0, 2.7]),
]

# Non-target obstacles the robot must not be teleported into (washer is real
# but a broken/tiny mesh in this asset set - kept as background, not a
# detection target; see project memory).
OBSTACLES = [
    (4.2, -2.8, 0.3),   # washer
]

ROOM_X_MIN, ROOM_X_MAX = -5.6, 5.6
ROOM_Y_MIN, ROOM_Y_MAX = -4.6, 4.6


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rpy_to_matrix(roll, pitch, yaw):
    return rot_z(yaw) @ np.array([
        [math.cos(pitch), 0.0, math.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch), 0.0, math.cos(pitch)],
    ]) @ np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll), math.cos(roll)],
    ])


R_STATIC = rpy_to_matrix(CAM_ROLL, CAM_PITCH, CAM_YAW)


def object_world_corners(obj):
    hx, hy, hz = obj["half"]
    ox, oy, oz = obj["offset"]
    signs = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
             (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    local_corners = np.array([
        [ox + sx * hx, oy + sy * hy, oz + sz * hz] for sx, sy, sz in signs
    ])
    R = rot_z(obj["yaw"])
    world_xy = obj["pos"]
    world_corners = (R @ local_corners.T).T + np.array([world_xy[0], world_xy[1], 0.0])
    return world_corners


def get_gz_pose(entity, retries=3, timeout=8.0):
    last_error = None
    for _ in range(retries):
        try:
            out = subprocess.run(
                ["gz", "model", "-m", entity, "-p"],
                capture_output=True, text=True, timeout=timeout
            )
            vals = [float(v) for v in out.stdout.strip().split()]
            x, y, z, roll, pitch, yaw = vals
            return x, y, z, roll, pitch, yaw
        except (subprocess.TimeoutExpired, ValueError) as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"get_gz_pose failed after {retries} retries: {last_error}")


def set_gz_pose(entity, x, y, z, yaw, retries=3, timeout=8.0):
    last_error = None
    for _ in range(retries):
        try:
            subprocess.run(
                ["gz", "model", "-m", entity,
                 "-x", str(x), "-y", str(y), "-z", str(z),
                 "-R", "0", "-P", "0", "-Y", str(yaw)],
                capture_output=True, text=True, timeout=timeout
            )
            return
        except subprocess.TimeoutExpired as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"set_gz_pose failed after {retries} retries: {last_error}")


def project_points(world_points, cam_pos, R_cam_world):
    # world_points: (N,3)
    rel = world_points - cam_pos
    cam_frame = (R_cam_world.T @ rel.T).T  # (N,3) in camera optical frame
    z = cam_frame[:, 2]
    valid = z > NEAR_CLIP
    u = FX * cam_frame[:, 0] / np.clip(z, 1e-6, None) + CX
    v = FY * cam_frame[:, 1] / np.clip(z, 1e-6, None) + CY
    return u, v, z, valid


def compute_camera_pose(base_x, base_y, base_yaw, base_z=0.019739):
    R_base = rot_z(base_yaw)
    cam_pos = np.array([base_x, base_y, base_z]) + R_base @ CAM_OFFSET
    R_cam_world = R_base @ R_STATIC
    return cam_pos, R_cam_world


def object_center_world(obj):
    return np.array([obj["pos"][0], obj["pos"][1], obj["offset"][2]])


def object_radius(obj):
    hx, hy, _ = obj["half"]
    return math.hypot(hx, hy)


def compute_visible_boxes(cam_pos, R_cam_world, depth=None):
    """Returns list of (obj, x1, y1, x2, y2, expected_dist) for objects that
    are actually visible (not occluded by a closer target object, and, when
    a depth image is supplied, consistent with the rendered depth)."""

    # bearing (horizontal angle in camera frame) + distance for every object,
    # used for inter-object occlusion reasoning.
    info = {}
    for obj in OBJECTS:
        center = object_center_world(obj)
        rel = center - cam_pos
        cam_frame = R_cam_world.T @ rel
        dist = np.linalg.norm(cam_frame)
        bearing = math.atan2(cam_frame[0], cam_frame[2])  # horizontal angle
        half_width = math.atan2(object_radius(obj), max(cam_frame[2], 1e-3))
        info[obj["entity"]] = dict(dist=dist, bearing=bearing, half_width=half_width,
                                    z=cam_frame[2])

    results = []

    for obj in OBJECTS:
        this_info = info[obj["entity"]]

        if this_info["z"] <= NEAR_CLIP:
            continue

        occluded = False
        for other in OBJECTS:
            if other is obj:
                continue
            other_info = info[other["entity"]]
            if other_info["z"] <= NEAR_CLIP:
                continue
            if other_info["dist"] >= this_info["dist"] - 0.05:
                continue
            angular_gap = abs(other_info["bearing"] - this_info["bearing"])
            if angular_gap < (other_info["half_width"] + this_info["half_width"] * 0.6):
                occluded = True
                break

        if occluded:
            continue

        corners = object_world_corners(obj)
        u, v, z, valid = project_points(corners, cam_pos, R_cam_world)

        if not np.any(valid):
            continue

        u_valid = u[valid]
        v_valid = v[valid]

        if len(u_valid) < 3:
            continue

        x1, x2 = np.min(u_valid), np.max(u_valid)
        y1, y2 = np.min(v_valid), np.max(v_valid)

        x1c, x2c = np.clip([x1, x2], 0, IMG_W - 1)
        y1c, y2c = np.clip([y1, y2], 0, IMG_H - 1)

        box_w = x2c - x1c
        box_h = y2c - y1c

        if box_w < 8 or box_h < 8:
            continue

        orig_w = max(x2 - x1, 1e-3)
        orig_h = max(y2 - y1, 1e-3)
        visible_frac = (box_w * box_h) / (orig_w * orig_h)

        if visible_frac < 0.25:
            continue

        expected_dist = this_info["dist"]

        if depth is not None:
            pu0 = int(np.clip(x1c, 0, IMG_W - 1))
            pu1 = int(np.clip(x2c, 0, IMG_W - 1))
            pv0 = int(np.clip(y1c, 0, IMG_H - 1))
            pv1 = int(np.clip(y2c, 0, IMG_H - 1))
            pu_samples = np.linspace(pu0, pu1, 5).astype(int)
            pv_samples = np.linspace(pv0, pv1, 5).astype(int)
            patch = depth[np.ix_(pv_samples, pu_samples)]
            finite = patch[np.isfinite(patch) & (patch > 0.05)]
            if finite.size >= 5:
                tol = max(0.35, expected_dist * 0.35)
                good = np.abs(finite - expected_dist) <= tol
                if good.mean() < 0.35:
                    continue

        results.append((obj, x1c, y1c, x2c, y2c, expected_dist))

    return results


def sample_object_viewpoints(obj, n_distances=6, n_angles=10):
    # camera is mounted at ~0.45m with no downward tilt (fixed hardware,
    # see model.sdf) -> floor-level objects fall outside the ~23 deg
    # vertical half-FOV closer than roughly 0.9-1.0m, so start further out.
    distances = obj.get("distances") or [1.1, 1.4, 1.8, 2.3, 2.9, 3.5][:n_distances]
    viewpoints = []
    ox, oy = obj["pos"]
    for r in distances:
        for i in range(n_angles):
            theta = (2 * math.pi * i / n_angles) + random.uniform(-0.15, 0.15)
            rx = r + random.uniform(-0.08, 0.08)
            rx = max(rx, 0.3)
            robot_x = ox + rx * math.cos(theta)
            robot_y = oy + rx * math.sin(theta)
            face_yaw = math.atan2(oy - robot_y, ox - robot_x) + random.uniform(-0.18, 0.18)
            viewpoints.append((robot_x, robot_y, face_yaw, obj["cls"]))
    return viewpoints


def sample_background_viewpoints(n=150):
    viewpoints = []
    tries = 0
    while len(viewpoints) < n and tries < n * 20:
        tries += 1
        x = random.uniform(ROOM_X_MIN, ROOM_X_MAX)
        y = random.uniform(ROOM_Y_MIN, ROOM_Y_MAX)
        if not is_free(x, y):
            continue
        yaw = random.uniform(-math.pi, math.pi)
        viewpoints.append((x, y, yaw, "background"))
    return viewpoints


def is_free(x, y):
    if not (ROOM_X_MIN <= x <= ROOM_X_MAX and ROOM_Y_MIN <= y <= ROOM_Y_MAX):
        return False
    for ox, oy, r in OBSTACLES:
        if math.hypot(x - ox, y - oy) < r:
            return False
    for obj in OBJECTS:
        ox, oy = obj["pos"]
        footprint_r = object_radius(obj) + 0.2  # + robot clearance margin
        if math.hypot(x - ox, y - oy) < footprint_r:
            return False
    return True


class DatasetGenerator(Node):

    def __init__(self):
        super().__init__("yolo_dataset_generator")
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_rgb_stamp = None
        self.latest_depth = None

        self.create_subscription(Image, "/camera/rgb/image_raw", self.rgb_cb, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_cb, 10)

        os.makedirs(RAW_IMAGES, exist_ok=True)
        os.makedirs(RAW_LABELS, exist_ok=True)

        self.saved = 0
        self.skipped_no_object = 0

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_rgb_stamp = time.time()

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def flush_callbacks(self, duration=0.15):
        """Drain any backlog in the subscriber queues (frames published
        while we were still waiting for physics to settle) so the next
        'fresh frame' really is fresh."""
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_fresh_frame(self, timeout=2.0):
        start = time.time()
        before = self.latest_rgb_stamp
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_rgb_stamp is not None and self.latest_rgb_stamp != before:
                if self.latest_rgb is not None:
                    return True
        return self.latest_rgb is not None

    def wait_settled(self, timeout=2.0, pos_tol=0.01, yaw_tol=0.01, gap=0.12):
        """Poll gz pose until two readings `gap` seconds apart agree,
        so residual velocity from a prior teleport can't leave the
        ground-truth pose out of sync with the rendered frame."""
        start = time.time()
        try:
            prev = get_gz_pose("burger")
        except Exception:
            return None

        while time.time() - start < timeout:
            time.sleep(gap)
            try:
                cur = get_gz_pose("burger")
            except Exception:
                continue
            dx = abs(cur[0] - prev[0])
            dy = abs(cur[1] - prev[1])
            dyaw = abs(math.atan2(math.sin(cur[5] - prev[5]), math.cos(cur[5] - prev[5])))
            if dx < pos_tol and dy < pos_tol and dyaw < yaw_tol:
                return cur
            prev = cur

        return prev

    def acquire(self, robot_x, robot_y, robot_yaw):
        """Teleport the robot, wait for physics to actually settle at the
        commanded pose, then grab a synchronized (pose, rgb, depth) triple.
        Returns None if anything failed."""
        set_gz_pose("burger", robot_x, robot_y, 0.02, robot_yaw)
        time.sleep(0.2)

        pose = self.wait_settled()
        if pose is None:
            self.get_logger().warn("gz pose read failed")
            return None
        bx, by, bz, broll, bpitch, byaw = pose

        # discard any RGB/depth frames that were published while we were
        # still polling for the pose to settle (queue backlog), then wait
        # for a genuinely new one so it's guaranteed to reflect the
        # now-settled pose, not a mid-transition one.
        self.flush_callbacks()
        self.wait_fresh_frame()
        # re-read pose after the frame arrives; if the settle check above
        # was solid this should barely move, but this is what actually
        # corresponds to the grabbed frame's timing.
        try:
            bx, by, bz, broll, bpitch, byaw = get_gz_pose("burger")
        except Exception as error:
            self.get_logger().warn(f"gz pose read failed: {error}")
            return None

        if self.latest_rgb is None:
            return None

        frame = self.latest_rgb.copy()
        depth = self.latest_depth.copy() if self.latest_depth is not None else None

        return bx, by, bz, byaw, frame, depth

    def capture(self, robot_x, robot_y, robot_yaw, focus_cls, index):
        prefix = focus_cls if focus_cls else "scene"
        filename = f"{prefix}_{index:05d}"
        if os.path.exists(os.path.join(RAW_IMAGES, filename + ".png")):
            return  # already captured in a previous (interrupted) run

        try:
            result = self.acquire(robot_x, robot_y, robot_yaw)
        except RuntimeError as error:
            self.get_logger().warn(f"acquire failed at index {index}: {error}")
            return
        if result is None:
            return
        bx, by, bz, byaw, frame, depth = result

        cam_pos, R_cam_world = compute_camera_pose(bx, by, byaw, bz)

        labels = []

        for obj, x1c, y1c, x2c, y2c, expected_dist in compute_visible_boxes(
            cam_pos, R_cam_world, depth
        ):
            center_x = (x1c + x2c) / 2.0
            center_y = (y1c + y2c) / 2.0
            box_w = x2c - x1c
            box_h = y2c - y1c

            xc_n = center_x / IMG_W
            yc_n = center_y / IMG_H
            w_n = box_w / IMG_W
            h_n = box_h / IMG_H

            labels.append((CLASS_ID[obj["cls"]], xc_n, yc_n, w_n, h_n))

        cv2.imwrite(os.path.join(RAW_IMAGES, filename + ".png"), frame)

        with open(os.path.join(RAW_LABELS, filename + ".txt"), "w") as f:
            for cls_id, xc, yc, w, h in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        self.saved += 1
        if not labels:
            self.skipped_no_object += 1

        if self.saved % 25 == 0:
            self.get_logger().info(
                f"Captured {self.saved} images "
                f"({self.skipped_no_object} with no visible target object)"
            )


def main():
    rclpy.init()
    node = DatasetGenerator()

    random.seed(42)

    viewpoints = []
    for obj in OBJECTS:
        vps = sample_object_viewpoints(obj)
        vps = [(x, y, yaw, cls) for x, y, yaw, cls in vps if is_free(x, y)]
        viewpoints.extend(vps)

    viewpoints.extend(sample_background_viewpoints(n=150))

    random.shuffle(viewpoints)

    node.get_logger().info(f"Total planned viewpoints: {len(viewpoints)}")

    for index, (x, y, yaw, cls) in enumerate(viewpoints):
        node.capture(x, y, yaw, cls, index)

    node.get_logger().info(
        f"DONE. Saved {node.saved} images, "
        f"{node.skipped_no_object} had no visible target object."
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
