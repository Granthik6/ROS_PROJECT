import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    gazebo_ros_dir = get_package_share_directory(
        'gazebo_ros'
    )

    robot_tasks_dir = get_package_share_directory(
        'robot_tasks'
    )

    model_path = os.path.join(robot_tasks_dir, 'models')
    workspace_model_path = '/root/ros2_ws/gazebo_models'
    turtlebot3_model_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'models')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    gazebo_model_path = os.pathsep.join(
        path for path in (
            model_path, workspace_model_path, turtlebot3_model_path,
            existing_model_path) if path
    )

    turtlebot3_description_dir = get_package_share_directory(
        'turtlebot3_description'
    )

    default_world_file = os.path.join(
        robot_tasks_dir,
        'worlds',
        'realistic_room.world'
    )
    world_file = LaunchConfiguration('world', default=default_world_file)

    robot_model_file = os.path.join(
        robot_tasks_dir,
        'models',
        'turtlebot3_burger_camera',
        'model.sdf'
    )

    urdf_file = os.path.join(
        turtlebot3_description_dir,
        'urdf',
        'turtlebot3_burger.urdf'
    )

    with open(urdf_file, 'r') as file:
        robot_description = file.read()

    robot_description = robot_description.replace(
        '${namespace}',
        ''
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True,
            }
        ]
    )

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.105',
            '0.0',
            '0.45',
            '-1.57079632679',
            '0.0',
            '-1.57079632679',
            'base_link',
            'camera_link',
        ],
        output='screen'
    )

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros_dir,
                'launch',
                'gzserver.launch.py'
            )
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'true',
        }.items()
    )

    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros_dir,
                'launch',
                'gzclient.launch.py'
            )
        ),
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    x_pose = LaunchConfiguration(
        'x_pose',
        default='0.0'
    )

    y_pose = LaunchConfiguration(
        'y_pose',
        default='-2.0'
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity',
            'burger',
            '-file',
            robot_model_file,
            '-x',
            x_pose,
            '-y',
            y_pose,
            '-z',
            '0.01',
            '-timeout',
            '120.0',
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('world', default_value=default_world_file),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        camera_tf,
        spawn_robot,
    ])
