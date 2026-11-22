import os
import random
import sys

import cv2

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    random.seed(1)

    test_points = []
    for obj in gyd.OBJECTS:
        vps = gyd.sample_object_viewpoints(obj, n_distances=2, n_angles=2)
        vps = [v for v in vps if gyd.is_free(v[0], v[1])]
        test_points.extend(vps[:2])

    out_dir = "/root/ros2_ws/dataset_tools/preview"
    os.makedirs(out_dir, exist_ok=True)

    for i, (x, y, yaw, cls) in enumerate(test_points):
        result = node.acquire(x, y, yaw)
        if result is None:
            print(f"acquire failed for viewpoint {i}")
            continue
        bx, by, bz, byaw, frame, depth = result

        cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw, bz)

        n_boxes = 0
        for obj, x1c, y1c, x2c, y2c, expected_dist in gyd.compute_visible_boxes(
            cam_pos, R_cam_world, depth
        ):
            cv2.rectangle(frame, (int(x1c), int(y1c)), (int(x2c), int(y2c)),
                          (0, 255, 0), 2)
            cv2.putText(frame, f'{obj["cls"]} {expected_dist:.2f}m',
                        (int(x1c), max(int(y1c) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            n_boxes += 1

        fname = f"{out_dir}/preview_{i:02d}_{cls}_boxes{n_boxes}.png"
        cv2.imwrite(fname, frame)
        print(f"saved {fname} robot=({bx:.2f},{by:.2f},yaw={byaw:.2f}) boxes={n_boxes}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
