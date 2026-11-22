"""Run DDQN, PPO, or random exploration until all objects or timeout."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            Shutdown)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    algorithm = LaunchConfiguration('algorithm')
    environment_profile = LaunchConfiguration('environment_profile')
    random_policy = LaunchConfiguration('random_policy')
    model = LaunchConfiguration('model')
    stochastic_policy = LaunchConfiguration('stochastic_policy')
    output_dir = LaunchConfiguration('output_dir')
    max_minutes = LaunchConfiguration('max_minutes')
    action_duration = LaunchConfiguration('action_duration')
    inspection_sweep_every_m = LaunchConfiguration('inspection_sweep_every_m')
    guided_completion_audit_report = LaunchConfiguration(
        'guided_completion_audit_report')
    guided_completion_after_stall_minutes = LaunchConfiguration(
        'guided_completion_after_stall_minutes')
    global_coverage = LaunchConfiguration('global_coverage')
    global_coverage_stall_seconds = LaunchConfiguration(
        'global_coverage_stall_seconds')
    breadcrumb_return = LaunchConfiguration('breadcrumb_return')
    seed = LaunchConfiguration('seed')
    show_window = LaunchConfiguration('show_window')
    detector_model = LaunchConfiguration('detector_model')
    inference_imgsz = LaunchConfiguration('inference_imgsz')
    hose_fallback_model = LaunchConfiguration('hose_fallback_model')
    safety_recovery = LaunchConfiguration('safety_recovery')
    world = LaunchConfiguration('world')

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('slam_toolbox'), 'launch',
            'online_async_launch.py')),
        condition=IfCondition(global_coverage),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': os.path.join(
                get_package_share_directory('robot_tasks'), 'config',
                'slam_toolbox_params.yaml'),
        }.items(),
    )

    detector = Node(
        package='robot_tasks', executable='yolo_node',
        name='object_discovery_yolo', output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_path': ParameterValue(detector_model, value_type=str),
            'hose_fallback_model_path': ParameterValue(
                hose_fallback_model, value_type=str),
            'hose_fallback_interval_frames': 2,
            'person_fallback_model_path': '/root/ros2_ws/yolo11n.pt',
            'person_fallback_interval_frames': 5,
            'show_window': ParameterValue(show_window, value_type=bool),
            'conf_threshold': 0.10,
            'pcd_conf_threshold': 0.20,
            'pcd_interval_frames': 3,
            'inference_imgsz': ParameterValue(inference_imgsz, value_type=int),
            'max_image_age_s': 0.75,
            'log_detections': False,
        }],
    )
    evaluator = Node(
        package='robot_tasks', executable='evaluate_ddqn_discovery',
        name='object_discovery_evaluator', output='screen',
        arguments=[
            '--algorithm', algorithm,
            '--random-policy', random_policy,
            '--environment-profile', environment_profile,
            '--detector-model', detector_model,
            '--model', model,
            '--stochastic-policy', stochastic_policy,
            '--output-dir', output_dir,
            '--max-minutes', max_minutes,
            '--action-duration', action_duration,
            '--inspection-sweep-every-m', inspection_sweep_every_m,
            '--guided-completion-audit-report', guided_completion_audit_report,
            '--guided-completion-after-stall-minutes',
            guided_completion_after_stall_minutes,
            '--global-coverage', global_coverage,
            '--global-coverage-stall-seconds', global_coverage_stall_seconds,
            '--breadcrumb-return', breadcrumb_return,
            '--seed', seed,
            '--safety-recovery', safety_recovery,
            '--world', world,
        ],
        on_exit=[Shutdown(reason='Object discovery evaluation finished')],
    )
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='ddqn'),
        DeclareLaunchArgument('random_policy', default_value='uniform'),
        DeclareLaunchArgument('environment_profile',
                              default_value='coverage_objects'),
        DeclareLaunchArgument('model', default_value=''),
        DeclareLaunchArgument('stochastic_policy', default_value='false'),
        DeclareLaunchArgument(
            'output_dir', default_value='/root/ros2_ws/rl_evaluations/ddqn_all_objects'),
        DeclareLaunchArgument('max_minutes', default_value='30'),
        DeclareLaunchArgument('action_duration', default_value='0.15'),
        DeclareLaunchArgument('inspection_sweep_every_m', default_value='0'),
        DeclareLaunchArgument('guided_completion_audit_report',
                              default_value=''),
        DeclareLaunchArgument('guided_completion_after_stall_minutes',
                              default_value='10'),
        DeclareLaunchArgument('global_coverage', default_value='false'),
        DeclareLaunchArgument('global_coverage_stall_seconds',
                              default_value='12'),
        DeclareLaunchArgument('breadcrumb_return', default_value='false'),
        DeclareLaunchArgument('seed', default_value='42'),
        DeclareLaunchArgument('show_window', default_value='false'),
        DeclareLaunchArgument(
            'detector_model',
            default_value='/root/ros2_ws/realistic_sim_yolo_runs/'
                          'realistic_sim_yolo11n/weights/best.pt'),
        DeclareLaunchArgument('inference_imgsz', default_value='512'),
        DeclareLaunchArgument('hose_fallback_model', default_value=''),
        DeclareLaunchArgument('safety_recovery', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value='/root/ros2_ws/src/robot_tasks/worlds/realistic_room.world'),
        slam, detector, evaluator,
    ])
