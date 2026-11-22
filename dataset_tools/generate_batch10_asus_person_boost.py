import random
import sys

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd

INDEX_OFFSET = 900000

TARGET_CLASSES = {"asus_xtion_pro_camera"}
TARGET_ENTITIES = {"person_standing_main"}


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    random.seed(271828)

    viewpoints = []
    for obj in gyd.OBJECTS:
        if obj["cls"] not in TARGET_CLASSES and obj["entity"] not in TARGET_ENTITIES:
            continue
        vps = gyd.sample_object_viewpoints(obj, n_distances=4, n_angles=40)
        vps = [(x, y, yaw, cls) for x, y, yaw, cls in vps if gyd.is_free(x, y)]
        viewpoints.extend(vps)

    random.shuffle(viewpoints)

    node.get_logger().info(f"Batch 10 (asus boost) planned viewpoints: {len(viewpoints)}")

    for i, (x, y, yaw, cls) in enumerate(viewpoints):
        node.capture(x, y, yaw, cls, INDEX_OFFSET + i)

    node.get_logger().info(
        f"BATCH10 DONE. Saved {node.saved} images, "
        f"{node.skipped_no_object} had no visible target object."
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
