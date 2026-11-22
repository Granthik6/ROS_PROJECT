import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math


class OdomSubscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        angle = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

        self.get_logger().info(
            f'x: {x:.2f}, y: {y:.2f}, angle: {angle:.2f} rad'
        )


def main(args=None):
    rclpy.init(args=args)

    node = OdomSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
