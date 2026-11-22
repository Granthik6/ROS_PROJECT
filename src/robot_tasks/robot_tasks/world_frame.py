"""Evaluation-only alignment from robot odometry into Gazebo world space."""

import math


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


def detection_in_world(odom_xyz, odom_robot_pose, world_robot_pose):
    """Transform an odom-frame detection using the same robot in both frames.

    Gazebo's robot pose is used for result scoring only; this function must
    never feed an exploration goal or control command.
    """
    dx = float(odom_xyz[0]) - odom_robot_pose.position.x
    dy = float(odom_xyz[1]) - odom_robot_pose.position.y
    angle = (yaw_from_quaternion(world_robot_pose.orientation) -
             yaw_from_quaternion(odom_robot_pose.orientation))
    c, s = math.cos(angle), math.sin(angle)
    return (
        world_robot_pose.position.x + c * dx - s * dy,
        world_robot_pose.position.y + s * dx + c * dy,
        world_robot_pose.position.z + float(odom_xyz[2]) -
        odom_robot_pose.position.z,
    )
