from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    action_duration = LaunchConfiguration('action_duration')
    seed = LaunchConfiguration('seed')
    report = LaunchConfiguration('report')

    yolo = Node(
        package='robot_tasks',
        executable='yolo_node',
        name='environment_test_yolo_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'show_window': False,
            'conf_threshold': 0.10,
            'log_detections': False,
        }],
    )

    checker = Node(
        package='robot_tasks',
        executable='check_dqn_environment',
        name='dqn_environment_checker',
        output='screen',
        arguments=[
            '--action-duration', action_duration,
            '--seed', seed,
            '--report', report,
        ],
        on_exit=[Shutdown(reason='DQN environment check finished')],
    )

    return LaunchDescription([
        DeclareLaunchArgument('action_duration', default_value='0.10'),
        DeclareLaunchArgument('seed', default_value='42'),
        DeclareLaunchArgument(
            'report',
            default_value='/root/ros2_ws/dqn_environment_check.json',
        ),
        yolo,
        checker,
    ])
