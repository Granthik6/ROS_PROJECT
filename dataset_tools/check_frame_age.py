"""Print the age of robot camera images in ROS simulation time."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def main():
    rclpy.init()
    node = Node('check_camera_frame_age')
    node.set_parameters([rclpy.parameter.Parameter(
        'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    count = [0]

    def callback(message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        now = node.get_clock().now().nanoseconds * 1e-9
        print(f'image={stamp:.3f} now={now:.3f} age={now-stamp:.3f}s',
              flush=True)
        count[0] += 1

    node.create_subscription(Image, '/camera/rgb/image_raw', callback, 10)
    while count[0] < 10:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
