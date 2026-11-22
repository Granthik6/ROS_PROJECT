import random
import sys

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
import rclpy
import generate_yolo_dataset as gyd

INDEX_OFFSET = 400000

TARGET_CLASSES = {
    "monkey_wrench", "speed_limit_sign", "door_handle", "kinect",
    "asus_xtion_pro_camera", "saucepan", "gear_part",
}


def main():
    rclpy.init()
    node = gyd.DatasetGenerator()

    random.seed(789)

    viewpoints = []
    for obj in gyd.OBJECTS:
        if obj["cls"] not in TARGET_CLASSES:
            continue
        vps = gyd.sample_object_viewpoints(obj, n_distances=4, n_angles=24)
        vps = [(x, y, yaw, cls) for x, y, yaw, cls in vps if gyd.is_free(x, y)]
        viewpoints.extend(vps)

    random.shuffle(viewpoints)

    node.get_logger().info(f"Batch 5 (geom-fixed) planned viewpoints: {len(viewpoints)}")

    for i, (x, y, yaw, cls) in enumerate(viewpoints):
        node.capture(x, y, yaw, cls, INDEX_OFFSET + i)

    node.get_logger().info(
        f"BATCH5 DONE. Saved {node.saved} images, "
        f"{node.skipped_no_object} had no visible target object."
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
