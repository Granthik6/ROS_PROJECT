from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'robot_tasks'


setup(

    name=package_name,

    version='0.0.0',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[

        (
            'share/ament_index/resource_index/packages',

            [
                'resource/' + package_name
            ],
        ),

        (
            'share/' + package_name,

            [
                'package.xml'
            ],
        ),

        (
            os.path.join(
                'share',
                package_name,
                'launch'
            ),

            glob(
                'launch/*.launch.py'
            ),
        ),

        (
            os.path.join(
                'share',
                package_name,
                'worlds'
            ),

            glob(
                'worlds/*.world'
            ),
        ),

        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),

            glob(
                'config/*'
            ),
        ),

        (
            os.path.join(
                'share',
                package_name,
                'urdf'
            ),

            glob(
                'urdf/*'
            ),
        ),

        (
            os.path.join(
                'share',
                package_name,
                'models',
                'turtlebot3_burger_camera'
            ),

            glob(
                'models/turtlebot3_burger_camera/*'
            ),
        ),

        *[
            (
                os.path.join('share', package_name, 'models', model_name),
                glob(os.path.join('models', model_name, '*.*')),
            )
            for model_name in ('pcd_bottle', 'pcd_doll')
        ],
        *[
            (
                os.path.join('share', package_name, 'models', model_name, 'meshes'),
                glob(os.path.join('models', model_name, 'meshes', '*')),
            )
            for model_name in ('pcd_bottle', 'pcd_doll')
        ],
        *[
            (
                os.path.join('share', package_name, 'models', model_name, 'source'),
                glob(os.path.join('models', model_name, 'source', '*.pcd')),
            )
            for model_name in ('pcd_bottle', 'pcd_doll')
        ],

    ],

    install_requires=[
        'setuptools',
        'gymnasium',
        'stable-baselines3',
    ],

    zip_safe=True,

    maintainer='Thatipamula Granthik Ranadheer',

    maintainer_email='granthikranadheer.t24@iiits.in',

    description=(
        'ROS 2 robot simulation, perception, '
        '3D object detection, SLAM, navigation '
        'and exploration'
    ),

    license='',

    tests_require=[
        'pytest'
    ],

    entry_points={

        'console_scripts': [

            'camera_node = robot_tasks.camera_node:main',

            'camera = robot_tasks.camera:main',

            'yolo_node = robot_tasks.yolo_node:main',

            'confidence_logger = robot_tasks.confidence_logger:main',
            'evaluate_world_coverage = robot_tasks.world_coverage_report:main',
            'aggregate_world_coverage = robot_tasks.aggregate_world_coverage:main',

            'plot_distance_confidence = robot_tasks.plot_distance_confidence:main',

            'hog_svm_detector = robot_tasks.hog_svm_detector:main',

            'hog_svm_node = robot_tasks.hog_svm_node:main',

            'compare_vision = robot_tasks.compare_vision:main',

            'odom = robot_tasks.odom:main',

            'lidar = robot_tasks.lidar:main',

            'navigate_goal = robot_tasks.navigate_goal:main',

            'random_explorer = robot_tasks.random_explorer:main',

            'map_checker = robot_tasks.map_checker:main',

            'move_robot = robot_tasks.move_robot:main',

            'publisher = robot_tasks.publisher:main',

            'subscriber = robot_tasks.subscriber:main',
            'pointcloud_xyz = robot_tasks.pointcloud_xyz:main',

            'object_mapper = robot_tasks.object_mapper:main',

            'baseline_experiment = robot_tasks.baseline_experiment:main',

            'train_dqn = robot_tasks.train_dqn:main',

            'evaluate_dqn = robot_tasks.evaluate_dqn:main',

            'train_exploration = robot_tasks.train_exploration:main',

            'evaluate_exploration = robot_tasks.evaluate_exploration:main',

            'evaluate_ddqn_discovery = robot_tasks.evaluate_ddqn_discovery:main',
            'evaluate_object_discovery = robot_tasks.evaluate_ddqn_discovery:main',

            'check_dqn_environment = robot_tasks.check_dqn_environment:main',

        ],

    },

)
