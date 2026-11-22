"""Evaluate a saved DQN, DDQN, or PPO model against random actions."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    algorithm = LaunchConfiguration('algorithm')
    model = LaunchConfiguration('model')
    episodes = LaunchConfiguration('episodes')
    episode_steps = LaunchConfiguration('episode_steps')
    action_duration = LaunchConfiguration('action_duration')
    environment_profile = LaunchConfiguration('environment_profile')
    seed = LaunchConfiguration('seed')
    run_id = LaunchConfiguration('run_id')

    yolo = Node(
        package='robot_tasks',
        executable='yolo_node',
        name='exploration_evaluation_yolo_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'show_window': False,
            'conf_threshold': 0.10,
            'log_detections': False,
        }],
    )
    evaluator = Node(
        package='robot_tasks',
        executable='evaluate_exploration',
        name='exploration_evaluator',
        output='screen',
        arguments=[
            '--algorithm', algorithm,
            '--model', model,
            '--episodes', episodes,
            '--episode-steps', episode_steps,
            '--action-duration', action_duration,
            '--environment-profile', environment_profile,
            '--seed', seed,
            '--run-id', run_id,
        ],
        on_exit=[Shutdown(reason='Exploration evaluation finished')],
    )
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='ddqn'),
        DeclareLaunchArgument('model'),
        DeclareLaunchArgument('episodes', default_value='10'),
        DeclareLaunchArgument('episode_steps', default_value='250'),
        DeclareLaunchArgument('action_duration', default_value='0.05'),
        DeclareLaunchArgument('environment_profile', default_value='coverage_objects'),
        DeclareLaunchArgument('seed', default_value='1000'),
        DeclareLaunchArgument('run_id', default_value='exploration_vs_random'),
        yolo,
        evaluator,
    ])
