import random
import sys

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd

INDEX_OFFSET = 500000

TARGET_CLASSES = {"stop_sign", "table_marble"}


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    random.seed(2026)

    viewpoints = []
    for obj in gyd.OBJECTS:
        if obj["cls"] not in TARGET_CLASSES:
            continue
        vps = gyd.sample_object_viewpoints(obj, n_angles=24)
        vps = [(x, y, yaw, cls) for x, y, yaw, cls in vps if gyd.is_free(x, y)]
        viewpoints.extend(vps)

    random.shuffle(viewpoints)

    node.get_logger().info(f"Batch 6 (fixed objects) planned viewpoints: {len(viewpoints)}")

    for i, (x, y, yaw, cls) in enumerate(viewpoints):
        node.capture(x, y, yaw, cls, INDEX_OFFSET + i)

    node.get_logger().info(
        f"BATCH6 DONE. Saved {node.saved} images, "
        f"{node.skipped_no_object} had no visible target object."
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
