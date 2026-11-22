import random

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy


class RandomExplorer(Node):

    def __init__(self):
        super().__init__('random_explorer')

        self.client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.map = None
        self.goal_handle = None
        self.next_goal_timer = None

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        self.get_logger().info('Waiting for map...')

    def map_callback(self, msg):

        if self.map is None:

            self.map = msg

            self.get_logger().info(
                f'Map received: '
                f'{msg.info.width} x {msg.info.height}'
            )

            self.send_random_goal()

    def send_random_goal(self):

        if self.map is None:
            return

        self.get_logger().info(
            'Waiting for Nav2...'
        )

        if not self.client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'Nav2 is not available!'
            )

            self.schedule_next_goal(5.0)
            return

        width = self.map.info.width
        height = self.map.info.height
        resolution = self.map.info.resolution

        free_cells = []

        for y in range(height):

            for x in range(width):

                index = y * width + x

                if self.map.data[index] == 0:

                    free_cells.append(
                        (x, y)
                    )

        if not free_cells:

            self.get_logger().error(
                'No free cells found in map!'
            )

            return

        x_cell, y_cell = random.choice(
            free_cells
        )

        goal_x = (
            self.map.info.origin.position.x
            + (x_cell + 0.5) * resolution
        )

        goal_y = (
            self.map.info.origin.position.y
            + (y_cell + 0.5) * resolution
        )

        self.get_logger().info(
            f'Random goal: '
            f'x={goal_x:.2f}, '
            f'y={goal_y:.2f}'
        )

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = 'map'

        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y

        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):

        self.goal_handle = future.result()

        if not self.goal_handle.accepted:

            self.get_logger().warn(
                'Random goal rejected!'
            )

            self.schedule_next_goal(2.0)

            return

        self.get_logger().info(
            'Random goal accepted!'
        )

        result_future = (
            self.goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.result_callback
        )

    def feedback_callback(self, feedback):

        pass

    def result_callback(self, future):

        result = future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                'Random goal reached!'
            )

        elif result.status == GoalStatus.STATUS_ABORTED:

            self.get_logger().warn(
                'Random goal failed!'
            )

        elif result.status == GoalStatus.STATUS_CANCELED:

            self.get_logger().warn(
                'Random goal canceled!'
            )

        else:

            self.get_logger().warn(
                f'Goal finished with status: '
                f'{result.status}'
            )

        self.schedule_next_goal(2.0)

    def schedule_next_goal(self, delay):

        if self.next_goal_timer is not None:

            self.next_goal_timer.cancel()

        self.next_goal_timer = self.create_timer(
            delay,
            self.next_goal_callback
        )

    def next_goal_callback(self):

        if self.next_goal_timer is not None:

            self.next_goal_timer.cancel()
            self.next_goal_timer = None

        self.get_logger().info(
            'Choosing another random goal...'
        )

        self.send_random_goal()


def main(args=None):

    rclpy.init(args=args)

    node = RandomExplorer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
