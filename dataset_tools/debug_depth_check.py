import sys
import time

import cv2

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    x, y, yaw = -0.65, -2.63, 3.22
    gyd.set_gz_pose("burger", x, y, 0.02, yaw)
    time.sleep(0.5)
    node.wait_fresh_frame()

    # drain a bit more so the depth callback catches up with the new pose
    start = time.time()
    while time.time() - start < 1.5:
        rclpy.spin_once(node, timeout_sec=0.05)

    bx, by, bz, broll, bpitch, byaw = gyd.get_gz_pose("burger")
    print("robot true pose:", bx, by, bz, broll, bpitch, byaw)
    cam_pos, R_cam_world = gyd.compute_camera_pose(bx, by, byaw)

    depth = node.latest_depth
    import numpy as np

    annotated = node.latest_rgb.copy()
    for obj, x1c, y1c, x2c, y2c, expected_dist in gyd.compute_visible_boxes(
        cam_pos, R_cam_world, depth
    ):
        cv2.rectangle(annotated, (int(x1c), int(y1c)), (int(x2c), int(y2c)),
                      (0, 255, 0), 2)
        cv2.putText(annotated, f'{obj["cls"]} {expected_dist:.2f}m',
                    (int(x1c), max(int(y1c) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        print(f'box for {obj["cls"]}: ({x1c:.1f},{y1c:.1f},{x2c:.1f},{y2c:.1f}) dist={expected_dist:.2f}')
    cv2.imwrite("/root/ros2_ws/dataset_tools/preview/depth_check_annotated.png", annotated)
    print("shape:", depth.shape, "dtype:", depth.dtype)
    finite = np.isfinite(depth)
    print("finite count:", finite.sum(), "/", depth.size)
    if finite.any():
        print("min/max finite:", depth[finite].min(), depth[finite].max())
    print("center pixel (320,240):", depth[240, 320])
    print("depth at box center (444, 450):", depth[450, 444])

    print()
    print("grid (rows=v every 40px 0..480, cols=u every 40px 0..640):")
    for v in range(0, 480, 40):
        row_vals = []
        for u in range(0, 640, 40):
            d = depth[v, u]
            row_vals.append("  inf" if not np.isfinite(d) else f"{d:5.2f}")
        print(f"v={v:3d}: " + " ".join(row_vals))

    cv2.imwrite("/root/ros2_ws/dataset_tools/preview/depth_check_rgb.png", node.latest_rgb)

    print()
    print("fine scan v=370..420 step5, u=390..480 step5:")
    for v in range(370, 421, 5):
        row_vals = []
        for u in range(390, 481, 5):
            d = depth[v, u]
            row_vals.append("  inf" if not np.isfinite(d) else f"{d:5.2f}")
        print(f"v={v:3d}: " + " ".join(row_vals))

    vis = depth.copy()
    finite_mask = np.isfinite(vis)
    vis[~finite_mask] = 0
    if finite_mask.any():
        vmax = vis[finite_mask].max()
        vis_norm = (vis / vmax * 255).astype("uint8")
    else:
        vis_norm = vis.astype("uint8")
    vis_color = cv2.applyColorMap(vis_norm, cv2.COLORMAP_JET)
    vis_color[~finite_mask] = (0, 0, 0)
    cv2.imwrite("/root/ros2_ws/dataset_tools/preview/depth_check_depth.png", vis_color)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
