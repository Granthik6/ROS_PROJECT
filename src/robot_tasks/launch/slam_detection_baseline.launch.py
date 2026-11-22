import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_tasks_dir = get_package_share_directory('robot_tasks')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    duration_sec = LaunchConfiguration('duration_sec')
    run_id = LaunchConfiguration('run_id')
    rviz = LaunchConfiguration('rviz')
    show_window = LaunchConfiguration('show_window')
    model_path = LaunchConfiguration('model_path')
    hose_fallback_model = LaunchConfiguration('hose_fallback_model')
    person_fallback_model_path = LaunchConfiguration(
        'person_fallback_model_path'
    )
    person_fallback_interval_frames = LaunchConfiguration(
        'person_fallback_interval_frames'
    )
    drive_robot = LaunchConfiguration('drive_robot')
    min_confidence = LaunchConfiguration('min_confidence')
    class_confidence_overrides = LaunchConfiguration(
        'class_confidence_overrides'
    )
    yolo_conf_threshold = LaunchConfiguration('yolo_conf_threshold')
    association_radius_m = LaunchConfiguration('association_radius_m')
    min_confirmations = LaunchConfiguration('min_confirmations')
    random_seed = LaunchConfiguration('random_seed')
    exploration_policy = LaunchConfiguration('exploration_policy')
    use_nav2 = LaunchConfiguration('use_nav2')
    nav2_autostart = LaunchConfiguration('nav2_autostart')
    manual_nav2_activation = LaunchConfiguration('manual_nav2_activation')
    tile_size_m = LaunchConfiguration('tile_size_m')
    frontier_gain_weight_m = LaunchConfiguration('frontier_gain_weight_m')
    frontier_visit_penalty_m = LaunchConfiguration('frontier_visit_penalty_m')
    distant_goal_every = LaunchConfiguration('distant_goal_every')
    nav_scan_interval_s = LaunchConfiguration('nav_scan_interval_s')
    stop_when_all_world_objects = LaunchConfiguration(
        'stop_when_all_world_objects'
    )
    world_match_radius_m = LaunchConfiguration('world_match_radius_m')
    world_file = LaunchConfiguration('world_file')
    inference_imgsz = LaunchConfiguration('inference_imgsz')
    pcd_interval_frames = LaunchConfiguration('pcd_interval_frames')
    max_image_age_s = LaunchConfiguration('max_image_age_s')

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                slam_toolbox_dir,
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': os.path.join(
                robot_tasks_dir,
                'config',
                'slam_toolbox_params.yaml'
            ),
        }.items()
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        condition=IfCondition(use_nav2),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': os.path.join(
                robot_tasks_dir, 'config', 'coverage_nav2.yaml'
            ),
            'autostart': nav2_autostart,
            'use_composition': 'False',
            'use_respawn': 'True',
        }.items(),
    )

    yolo = Node(
        package='robot_tasks',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'show_window': ParameterValue(show_window, value_type=bool),
            'model_path': model_path,
            'hose_fallback_model_path': hose_fallback_model,
            'hose_fallback_interval_frames': 2,
            'person_fallback_model_path': person_fallback_model_path,
            'person_fallback_interval_frames': ParameterValue(
                person_fallback_interval_frames, value_type=int
            ),
            'conf_threshold': ParameterValue(
                yolo_conf_threshold, value_type=float
            ),
            'log_detections': False,
            'inference_imgsz': ParameterValue(inference_imgsz, value_type=int),
            'pcd_interval_frames': ParameterValue(
                pcd_interval_frames, value_type=int),
            'max_image_age_s': ParameterValue(max_image_age_s,
                                               value_type=float),
        }]
    )

    baseline = Node(
        package='robot_tasks',
        executable='baseline_experiment',
        name='baseline_experiment',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'duration_sec': ParameterValue(duration_sec, value_type=float),
            'run_id': run_id,
            'drive_robot': ParameterValue(drive_robot, value_type=bool),
            'min_confidence': ParameterValue(min_confidence, value_type=float),
            'class_confidence_overrides': class_confidence_overrides,
            'association_radius_m': ParameterValue(
                association_radius_m,
                value_type=float
            ),
            'min_confirmations': ParameterValue(
                min_confirmations,
                value_type=int
            ),
            'random_seed': ParameterValue(random_seed, value_type=int),
            'exploration_policy': exploration_policy,
            'manual_nav2_activation': ParameterValue(
                manual_nav2_activation, value_type=bool
            ),
            'tile_size_m': ParameterValue(tile_size_m, value_type=float),
            'frontier_gain_weight_m': ParameterValue(
                frontier_gain_weight_m, value_type=float
            ),
            'frontier_visit_penalty_m': ParameterValue(
                frontier_visit_penalty_m, value_type=float
            ),
            'distant_goal_every': ParameterValue(
                distant_goal_every, value_type=int
            ),
            'nav_scan_interval_s': ParameterValue(
                nav_scan_interval_s, value_type=float
            ),
            'stop_when_all_world_objects': ParameterValue(
                stop_when_all_world_objects, value_type=bool
            ),
            'world_match_radius_m': ParameterValue(
                world_match_radius_m, value_type=float
            ),
            'world_file': world_file,
        }],
        on_exit=[Shutdown(reason='Baseline experiment finished')]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='baseline_rviz',
        output='screen',
        arguments=[
            '-d',
            os.path.join(
                robot_tasks_dir,
                'config',
                'slam_detection.rviz'
            ),
        ],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz)
    )

    return LaunchDescription([
        DeclareLaunchArgument('duration_sec', default_value='180.0'),
        DeclareLaunchArgument('run_id', default_value=''),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('show_window', default_value='true'),
        DeclareLaunchArgument(
            'model_path',
            default_value=(
                '/root/ros2_ws/realistic_sim_yolo_runs/'
                'realistic_sim_yolo11n/weights/best.pt'
            ),
        ),
        DeclareLaunchArgument('hose_fallback_model', default_value=''),
        DeclareLaunchArgument('person_fallback_model_path', default_value=''),
        DeclareLaunchArgument(
            'person_fallback_interval_frames', default_value='5'
        ),
        DeclareLaunchArgument('drive_robot', default_value='true'),
        DeclareLaunchArgument('min_confidence', default_value='0.25'),
        DeclareLaunchArgument('class_confidence_overrides', default_value=''),
        DeclareLaunchArgument('yolo_conf_threshold', default_value='0.10'),
        DeclareLaunchArgument('association_radius_m', default_value='1.5'),
        DeclareLaunchArgument('min_confirmations', default_value='2'),
        DeclareLaunchArgument('random_seed', default_value='42'),
        DeclareLaunchArgument('exploration_policy', default_value='random'),
        DeclareLaunchArgument('use_nav2', default_value='false'),
        DeclareLaunchArgument('nav2_autostart', default_value='true'),
        DeclareLaunchArgument('manual_nav2_activation', default_value='false'),
        DeclareLaunchArgument('tile_size_m', default_value='0.5'),
        DeclareLaunchArgument('frontier_gain_weight_m', default_value='3.0'),
        DeclareLaunchArgument('frontier_visit_penalty_m', default_value='0.4'),
        DeclareLaunchArgument('distant_goal_every', default_value='6'),
        DeclareLaunchArgument('nav_scan_interval_s', default_value='45.0'),
        DeclareLaunchArgument('stop_when_all_world_objects',
                              default_value='false'),
        DeclareLaunchArgument('world_match_radius_m', default_value='1.25'),
        DeclareLaunchArgument('world_file', default_value=os.path.join(
            robot_tasks_dir, 'worlds', 'realistic_room.world')),
        DeclareLaunchArgument('inference_imgsz', default_value='512'),
        DeclareLaunchArgument('pcd_interval_frames', default_value='3'),
        DeclareLaunchArgument('max_image_age_s', default_value='0.75'),
        slam,
        navigation,
        yolo,
        baseline,
        rviz_node,
    ])
