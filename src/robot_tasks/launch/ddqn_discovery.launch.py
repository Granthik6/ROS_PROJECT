"""Run DDQN, PPO, or random exploration until all objects or timeout."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    algorithm = LaunchConfiguration('algorithm')
    model = LaunchConfiguration('model')
    output_dir = LaunchConfiguration('output_dir')
    max_minutes = LaunchConfiguration('max_minutes')
    action_duration = LaunchConfiguration('action_duration')
    seed = LaunchConfiguration('seed')
    show_window = LaunchConfiguration('show_window')
    safety_recovery = LaunchConfiguration('safety_recovery')
    world = LaunchConfiguration('world')

    detector = Node(
        package='robot_tasks', executable='yolo_node',
        name='object_discovery_yolo', output='screen',
        parameters=[{
            'use_sim_time': True,
            'show_window': ParameterValue(show_window, value_type=bool),
            'conf_threshold': 0.10,
            'pcd_conf_threshold': 0.20,
            'log_detections': False,
        }],
    )
    evaluator = Node(
        package='robot_tasks', executable='evaluate_ddqn_discovery',
        name='object_discovery_evaluator', output='screen',
        arguments=[
            '--algorithm', algorithm,
            '--model', model,
            '--output-dir', output_dir,
            '--max-minutes', max_minutes,
            '--action-duration', action_duration,
            '--seed', seed,
            '--safety-recovery', safety_recovery,
            '--world', world,
        ],
        on_exit=[Shutdown(reason='Object discovery evaluation finished')],
    )
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='ddqn'),
        DeclareLaunchArgument('model', default_value=''),
        DeclareLaunchArgument(
            'output_dir', default_value='/root/ros2_ws/rl_evaluations/ddqn_all_objects'),
        DeclareLaunchArgument('max_minutes', default_value='30'),
        DeclareLaunchArgument('action_duration', default_value='0.15'),
        DeclareLaunchArgument('seed', default_value='42'),
        DeclareLaunchArgument('show_window', default_value='false'),
        DeclareLaunchArgument('safety_recovery', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value='/root/ros2_ws/src/robot_tasks/worlds/realistic_room.world'),
        detector, evaluator,
    ])
