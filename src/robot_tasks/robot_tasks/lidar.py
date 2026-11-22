import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg):
        front_angle = 0.0

        index = int(
            (front_angle - msg.angle_min) /
            msg.angle_increment
        )

        if 0 <= index < len(msg.ranges):
            front = msg.ranges[index]

            if math.isfinite(front):
                self.get_logger().info(
                    f'Front distance: {front:.2f} m'
                )

def main(args=None):
    rclpy.init(args=args)

    node = LidarNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
