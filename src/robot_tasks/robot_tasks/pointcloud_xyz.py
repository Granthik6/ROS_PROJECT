import struct

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2


class PointCloudXYZ(Node):

    def __init__(self):
        super().__init__('pointcloud_xyz')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )

        self.pixels = [
            (160, 120),
            (320, 120),
            (480, 120),
            (160, 240),
            (320, 240),
            (480, 240),
            (160, 360),
            (320, 360),
            (480, 360),
        ]

        self.last_print_time = self.get_clock().now()

        self.get_logger().info(
            'Listening to /camera/depth/points'
        )

    def get_xyz(self, msg, u, v):

        offset = (
            v * msg.row_step +
            u * msg.point_step
        )

        x = struct.unpack_from(
            '<f',
            msg.data,
            offset
        )[0]

        y = struct.unpack_from(
            '<f',
            msg.data,
            offset + 4
        )[0]

        z = struct.unpack_from(
            '<f',
            msg.data,
            offset + 8
        )[0]

        return x, y, z

    def pointcloud_callback(self, msg):

        now = self.get_clock().now()

        if (
            now - self.last_print_time
        ).nanoseconds < 1_000_000_000:
            return

        self.last_print_time = now

        self.get_logger().info(
            f'PointCloud: {msg.width}x{msg.height}, '
            f'frame={msg.header.frame_id}'
        )

        for u, v in self.pixels:

            try:
                x, y, z = self.get_xyz(
                    msg,
                    u,
                    v
                )

                self.get_logger().info(
                    f'({u},{v}) -> '
                    f'X={x:.3f} m, '
                    f'Y={y:.3f} m, '
                    f'Z={z:.3f} m'
                )

            except Exception as error:

                self.get_logger().error(
                    f'Pixel ({u},{v}) error: {error}'
                )


def main(args=None):

    rclpy.init(args=args)

    node = PointCloudXYZ()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
