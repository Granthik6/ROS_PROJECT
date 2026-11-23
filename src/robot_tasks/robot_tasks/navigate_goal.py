import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class NavigateGoal(Node):

    def __init__(self):
        super().__init__('navigate_goal')

        self.client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.send_goal()

    def send_goal(self):

        self.get_logger().info('Waiting for Nav2...')

        self.client.wait_for_server()

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = 1.0
        goal.pose.pose.position.y = 1.0

        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = 0.948
        goal.pose.pose.orientation.w = 0.318

        self.get_logger().info('Sending goal...')

        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.get_logger().info('Goal rejected!')

            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted!')

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.result_callback
        )

    def feedback_callback(self, feedback):

        pass

    def result_callback(self, future):
    
        result = future.result()
    
        if result.status == GoalStatus.STATUS_SUCCEEDED:
    
            self.get_logger().info(
                'Goal succeeded! Robot reached the goal.'
            )
    
        elif result.status == GoalStatus.STATUS_CANCELED:
    
            self.get_logger().info(
                'Goal canceled!'
            )
    
        elif result.status == GoalStatus.STATUS_ABORTED:
    
            self.get_logger().info(
                'Goal failed! Nav2 aborted the goal.'
            )
    
        else:
    
            self.get_logger().info(
                f'Goal finished with status: {result.status}'
            )
    
        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = NavigateGoal()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
