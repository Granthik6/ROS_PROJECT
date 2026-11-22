from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    timesteps = LaunchConfiguration('timesteps')
    run_id = LaunchConfiguration('run_id')
    seed = LaunchConfiguration('seed')
    action_duration = LaunchConfiguration('action_duration')
    episode_steps = LaunchConfiguration('episode_steps')
    environment_profile = LaunchConfiguration('environment_profile')
    resume = LaunchConfiguration('resume')
    show_window = LaunchConfiguration('show_window')

    yolo = Node(
        package='robot_tasks',
        executable='yolo_node',
        name='dqn_yolo_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'show_window': ParameterValue(show_window, value_type=bool),
            'conf_threshold': 0.10,
            'log_detections': False,
        }],
    )

    trainer = Node(
        package='robot_tasks',
        executable='train_dqn',
        name='dqn_trainer',
        output='screen',
        arguments=[
            '--timesteps', timesteps,
            '--run-id', run_id,
            '--seed', seed,
            '--action-duration', action_duration,
            '--episode-steps', episode_steps,
            '--environment-profile', environment_profile,
            '--resume', resume,
        ],
        on_exit=[Shutdown(reason='DQN training finished')],
    )

    return LaunchDescription([
        DeclareLaunchArgument('timesteps', default_value='10000'),
        DeclareLaunchArgument('run_id', default_value='dqn_10000'),
        DeclareLaunchArgument('seed', default_value='42'),
        DeclareLaunchArgument('action_duration', default_value='0.05'),
        DeclareLaunchArgument('episode_steps', default_value='250'),
        DeclareLaunchArgument('environment_profile', default_value='legacy'),
        DeclareLaunchArgument('resume', default_value=''),
        DeclareLaunchArgument('show_window', default_value='false'),
        yolo,
        trainer,
    ])
