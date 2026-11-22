import random
import sys

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd

INDEX_OFFSET = 100000


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    random.seed(99)

    viewpoints = []
    for obj in gyd.OBJECTS:
        vps = gyd.sample_object_viewpoints(obj, n_distances=6, n_angles=14)
        vps = [(x, y, yaw, cls) for x, y, yaw, cls in vps if gyd.is_free(x, y)]
        viewpoints.extend(vps)

    viewpoints.extend(gyd.sample_background_viewpoints(n=120))

    random.shuffle(viewpoints)

    node.get_logger().info(f"Batch 2 planned viewpoints: {len(viewpoints)}")

    for i, (x, y, yaw, cls) in enumerate(viewpoints):
        node.capture(x, y, yaw, cls, INDEX_OFFSET + i)

    node.get_logger().info(
        f"BATCH2 DONE. Saved {node.saved} images, "
        f"{node.skipped_no_object} had no visible target object."
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
