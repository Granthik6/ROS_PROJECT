import csv
import math
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd

NEW_OBJECTS = [
    ("beer_object", "beer", (-5.5, -4.5)),
    ("fire_hydrant_object", "fire_hydrant", (-4.15, -4.5)),
    ("construction_cone_object", "construction_cone", (-2.8, -4.5)),
    ("construction_barrel_object", "construction_barrel", (-1.45, -4.5)),
    ("cardboard_box_object", "cardboard_box", (-0.1, -4.5)),
    ("cricket_ball_object", "cricket_ball", (1.25, -4.5)),
    ("monkey_wrench_object", "monkey_wrench", (2.6, -4.5)),
    ("mailbox_object", "mailbox", (3.95, -4.5)),
    ("postbox_object", "postbox", (5.3, -4.5)),
    ("stop_sign_object", "stop_sign", (-4.83, -3.15)),
    ("speed_limit_sign_object", "speed_limit_sign", (-2.12, -3.15)),
    ("lamp_post_object", "lamp_post", (-0.77, -3.15)),
    ("first_2015_trash_can_object", "first_2015_trash_can", (0.58, -3.15)),
    ("utility_cart_object", "utility_cart", (1.93, -3.15)),
    ("grey_tote_object", "grey_tote", (3.28, -3.15)),
    ("car_wheel_object", "car_wheel", (-5.5, -1.8)),
    ("door_handle_object", "door_handle", (-1.45, -1.8)),
    ("kinect_object", "kinect", (-0.1, -1.8)),
    ("asus_xtion_pro_camera_object", "asus_xtion_pro_camera", (1.25, -1.8)),
    ("hokuyo_object", "hokuyo", (2.6, -1.8)),
    ("table_marble_object", "table_marble", (3.95, -1.8)),
    ("wooden_case_object", "wooden_case", (5.3, -1.8)),
    ("saucepan_object", "saucepan", (-4.83, -0.45)),
    ("jersey_barrier_object", "jersey_barrier", (-2.12, -0.45)),
    ("stop_light_object", "stop_light", (-0.77, -0.45)),
    ("oak_tree_object", "oak_tree", (0.58, -0.45)),
    ("ball_bearing_object", "ball_bearing", (3.28, -0.45)),
    ("gear_part_object", "gear_part", (4.62, -0.45)),
    ("fire_hose_long_object", "fire_hose_long", (-5.5, 0.9)),
    ("washer_main", "washer", (4.2, -2.8)),
]

TRY_DISTANCES = [1.8, 2.6, 3.6, 1.2]
DEPTH_MARGIN = 0.15
MAX_DEPTH = 6.0
BAND_TOLERANCE = 0.5


def calibrate_one(node, entity, cls, pos, dist):
    ox, oy = pos
    robot_x = ox - dist
    robot_y = oy
    result = node.acquire(robot_x, robot_y, 0.0)
    if result is None:
        return None
    bx, by, bz, byaw, frame, depth = result

    cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw, bz)

    band_lo = max(gyd.NEAR_CLIP, dist - BAND_TOLERANCE)
    band_hi = min(MAX_DEPTH, dist + BAND_TOLERANCE)

    valid = np.isfinite(depth) & (depth > band_lo) & (depth < band_hi)

    mask = np.zeros_like(valid)
    mask[gyd.IMG_H // 5: gyd.IMG_H - gyd.IMG_H // 5,
         gyd.IMG_W // 5: gyd.IMG_W - gyd.IMG_W // 5] = True
    candidate = (valid & mask).astype(np.uint8)

    if candidate.sum() < 30:
        return None

    roi_x0, roi_x1 = gyd.IMG_W // 5, gyd.IMG_W - gyd.IMG_W // 5
    roi_y0, roi_y1 = gyd.IMG_H // 5, gyd.IMG_H - gyd.IMG_H // 5
    border_margin = 4

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if n_labels <= 1:
        return None

    best_label = None
    best_dist = None
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        x, y, w, h = stats[lbl, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
        touches_border = (
            x <= roi_x0 + border_margin or y <= roi_y0 + border_margin or
            x + w >= roi_x1 - border_margin or y + h >= roi_y1 - border_margin
        )
        if touches_border:
            continue
        cx_blob, cy_blob = centroids[lbl]
        d = math.hypot(cx_blob - gyd.CX, cy_blob - gyd.CY)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_label = lbl

    if best_label is None:
        return None

    is_object = labels == best_label

    ys, xs = np.where(is_object)
    if len(xs) < 30:
        return None

    zs = depth[ys, xs]
    dxs = (xs - gyd.CX) / gyd.FX * zs
    dys = (ys - gyd.CY) / gyd.FY * zs
    cam_pts = np.stack([dxs, dys, zs], axis=-1)
    world_pts = cam_pts @ R_cam_world.T + cam_pos

    mins = world_pts.min(axis=0)
    maxs = world_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    half = (maxs - mins) / 2.0

    offset = (center[0] - ox, center[1] - oy, center[2])

    annotated = frame.copy()
    for uu, vv in zip(xs[::7], ys[::7]):
        cv2.circle(annotated, (int(uu), int(vv)), 2, (0, 255, 0), -1)

    return {
        "cls": cls, "pos": pos, "half": tuple(round(h, 4) for h in half),
        "offset": tuple(round(o, 4) for o in offset), "n_pts": int(len(xs)),
        "dist_used": dist, "annotated": annotated,
    }


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    results = []
    for entity, cls, pos in NEW_OBJECTS:
        got = None
        for dist in TRY_DISTANCES:
            got = calibrate_one(node, entity, cls, pos, dist)
            if got is not None:
                break
        if got is None:
            print(f"{cls:24s} CALIBRATION FAILED (no valid segmentation)")
            continue
        print(f"{cls:24s} half={got['half']} offset={got['offset']} "
              f"n_pts={got['n_pts']} dist={got['dist_used']}")
        cv2.imwrite(f"/root/ros2_ws/dataset_tools/calib_preview_{cls}.png", got["annotated"])
        results.append(got)

    with open("/root/ros2_ws/dataset_tools/new_object_bounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cls", "pos_x", "pos_y", "half_x", "half_y", "half_z",
                          "offset_x", "offset_y", "offset_z"])
        for r in results:
            writer.writerow([r["cls"], r["pos"][0], r["pos"][1],
                              *r["half"], *r["offset"]])

    print(f"\nCalibrated {len(results)}/{len(NEW_OBJECTS)} objects")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
