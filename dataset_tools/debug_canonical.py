import math
import sys

import cv2

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd


def shot(node, x, y, yaw, tag):
    result = node.acquire(x, y, yaw)
    if result is None:
        print(f"[{tag}] acquire failed")
        return
    bx, by, bz, byaw, frame, depth = result

    cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw, bz)
    print(f"[{tag}] robot=({bx:.3f},{by:.3f},yaw={byaw:.3f}) cam_pos={cam_pos.round(3)}")

    annotated = frame.copy()
    for obj, x1c, y1c, x2c, y2c, expected_dist in gyd.compute_visible_boxes(
        cam_pos, R_cam_world, depth
    ):
        cv2.rectangle(annotated, (int(x1c), int(y1c)), (int(x2c), int(y2c)),
                      (0, 255, 0), 2)
        cv2.putText(annotated, f'{obj["cls"]} {expected_dist:.2f}m',
                    (int(x1c), max(int(y1c) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        print(f'   box {obj["cls"]}: ({x1c:.1f},{y1c:.1f},{x2c:.1f},{y2c:.1f}) dist={expected_dist:.2f}')

    cv2.imwrite(f"/root/ros2_ws/dataset_tools/preview/canonical_{tag}.png", annotated)


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    for obj in gyd.OBJECTS:
        ox, oy = obj["pos"]
        robot_x = ox
        robot_y = oy - 1.5
        yaw = math.atan2(oy - robot_y, ox - robot_x)
        shot(node, robot_x, robot_y, yaw, f'{obj["cls"]}_faceon')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
