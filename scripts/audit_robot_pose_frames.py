#!/usr/bin/env python3
"""Read the robot pose from Gazebo, odometry, and SLAM for diagnostics."""

import math
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def main():
    rclpy.init()
    node = rclpy.create_node('robot_pose_frame_diagnostic')
    poses = {}

    def on_models(message):
        try:
            position = message.pose[message.name.index('burger')].position
        except ValueError:
            return
        poses['world'] = (position.x, position.y)

    def on_odom(message):
        position = message.pose.pose.position
        poses['odom'] = (position.x, position.y)

    node.create_subscription(ModelStates, '/gazebo/model_states', on_models, 10)
    node.create_subscription(Odometry, '/odom', on_odom, 10)
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    transform = None
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if len(poses) < 2:
                continue
            try:
                transform = buffer.lookup_transform('map', 'odom', Time())
                break
            except TransformException:
                pass
        if transform is None:
            raise RuntimeError(f'Pose topics or map transform unavailable: {poses}')
        odom_x, odom_y = poses['odom']
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1 - 2 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        map_pose = (
            transform.transform.translation.x + math.cos(yaw) * odom_x
            - math.sin(yaw) * odom_y,
            transform.transform.translation.y + math.sin(yaw) * odom_x
            + math.cos(yaw) * odom_y,
        )
        for frame, value in (*poses.items(), ('map', map_pose)):
            print(f'{frame}: ({value[0]:.3f}, {value[1]:.3f}) m')
        for frame, value in (('odom', poses['odom']), ('map', map_pose)):
            print(f'{frame} error: {math.dist(poses["world"], value):.3f} m')
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
