import math
import sys

import cv2

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd

CHECK_CLASSES = ["construction_cone", "lamp_post", "table_marble", "oak_tree",
                  "cardboard_box", "postbox", "jersey_barrier"]


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    for cls in CHECK_CLASSES:
        obj = next(o for o in gyd.OBJECTS if o["cls"] == cls)
        ox, oy = obj["pos"]
        dist = obj.get("distances", [1.8])[1]
        robot_x, robot_y = ox - dist, oy
        yaw = math.atan2(oy - robot_y, ox - robot_x)

        result = node.acquire(robot_x, robot_y, yaw)
        if result is None:
            print(f"{cls}: acquire failed")
            continue
        bx, by, bz, byaw, frame, depth = result
        cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw, bz)

        annotated = frame.copy()
        n = 0
        for o2, x1c, y1c, x2c, y2c, dist_meas in gyd.compute_visible_boxes(cam_pos, R_cam_world, depth):
            cv2.rectangle(annotated, (int(x1c), int(y1c)), (int(x2c), int(y2c)), (0, 255, 0), 2)
            cv2.putText(annotated, f'{o2["cls"]} {dist_meas:.2f}m', (int(x1c), max(int(y1c) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            n += 1
        cv2.imwrite(f"/root/ros2_ws/dataset_tools/quickcheck_{cls}.png", annotated)
        print(f"{cls}: {n} boxes drawn")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
