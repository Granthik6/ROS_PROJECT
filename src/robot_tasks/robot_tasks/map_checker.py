import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class MapChecker(Node):
    def __init__(self):
        super().__init__('map_checker')

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        self.received = False

    def map_callback(self, msg):
        if self.received:
            return

        self.received = True

        data = list(msg.data)

        unknown = data.count(-1)
        free = data.count(0)
        occupied = data.count(100)
        other = len(data) - unknown - free - occupied

        print('\n===== OCCUPANCY GRID =====')
        print(f'Resolution : {msg.info.resolution:.3f} m')
        print(f'Width      : {msg.info.width}')
        print(f'Height     : {msg.info.height}')
        print(f'Total cells: {len(data)}')
        print(f'Unknown    : {unknown}')
        print(f'Free       : {free}')
        print(f'Occupied   : {occupied}')
        print(f'Other      : {other}')
        print('===========================\n')

        rclpy.shutdown()


def main():
    rclpy.init()
    node = MapChecker()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
