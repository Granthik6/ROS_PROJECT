import sys
import time
import math
import numpy as np

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    x, y, yaw = 2.28, -2.59, 2.73
    bx, by, bz, byaw, frame, depth = node.acquire(x, y, yaw)
    print("robot true pose:", bx, by, bz, byaw)

    cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw, bz)
    print("cam_pos:", cam_pos)

    for obj in gyd.OBJECTS:
        center = gyd.object_center_world(obj)
        rel = center - cam_pos
        cam_frame = R_cam_world.T @ rel
        dist = np.linalg.norm(cam_frame)
        bearing = math.atan2(cam_frame[0], cam_frame[2])
        half_width = math.atan2(gyd.object_radius(obj), max(cam_frame[2], 1e-3))
        print(f"{obj['cls']:16s} cam_frame={cam_frame.round(3)} dist={dist:.3f} "
              f"bearing_deg={math.degrees(bearing):.1f} "
              f"half_width_deg={math.degrees(half_width):.1f} z={cam_frame[2]:.3f}")

        corners = gyd.object_world_corners(obj)
        u, v, z, valid = gyd.project_points(corners, cam_pos, R_cam_world)
        print("   u:", u.round(1), "v:", v.round(1), "valid:", valid)

    print()
    print("compute_visible_boxes result:")
    for obj, x1c, y1c, x2c, y2c, expected_dist in gyd.compute_visible_boxes(
        cam_pos, R_cam_world, depth
    ):
        print(f"  {obj['cls']}: box=({x1c:.1f},{y1c:.1f},{x2c:.1f},{y2c:.1f}) "
              f"expected_dist={expected_dist:.2f}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
