import csv
import json
import math
import os
import random
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import rclpy

from action_msgs.msg import GoalStatus
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import BackUp, DriveOnHeading, NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from robot_tasks.coverage_planner import (
    plan_frontier, plan_sensor_probe, world_to_cell,
)
from robot_tasks.world_coverage_report import (
    WORLD_FILE,
    build_report,
    match_instances,
    read_world,
    write_discovery_curve,
    write_missing_object_audit,
)
from robot_tasks.world_frame import detection_in_world


plt.switch_backend('Agg')


DEFAULT_OUTPUT_ROOT = '/root/ros2_ws/baseline_runs'
GLOBAL_RESULTS = '/root/ros2_ws/baseline_results.csv'


class BaselineExperiment(Node):
    """Run and document a timed random-policy SLAM perception baseline."""

    SUMMARY_FIELDS = [
        'run_id',
        'status',
        'duration_sec',
        'distance_traveled_m',
        'total_detections',
        'accepted_observations',
        'unique_objects_found',
        'unique_classes_found',
        'classes',
        'time_to_first_object_s',
        'objects_per_minute',
        'mean_accepted_confidence',
        'map_coverage_percent',
        'random_seed',
    ]

    def __init__(self):
        super().__init__('baseline_experiment')

        self.declare_parameter('duration_sec', 180.0)
        self.declare_parameter('run_id', '')
        self.declare_parameter('output_root', DEFAULT_OUTPUT_ROOT)
        self.declare_parameter('drive_robot', True)
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('class_confidence_overrides', '')
        self.declare_parameter('association_radius_m', 1.5)
        self.declare_parameter('min_confirmations', 2)
        self.declare_parameter('random_seed', 42)
        self.declare_parameter('exploration_policy', 'random')
        self.declare_parameter('tile_size_m', 0.5)
        self.declare_parameter('frontier_gain_weight_m', 3.0)
        self.declare_parameter('frontier_visit_penalty_m', 0.4)
        self.declare_parameter('distant_goal_every', 6)
        self.declare_parameter('nav_scan_interval_s', 45.0)
        self.declare_parameter('manual_nav2_activation', False)
        self.declare_parameter('stop_when_all_world_objects', False)
        self.declare_parameter('world_file', WORLD_FILE)
        self.declare_parameter('world_match_radius_m', 1.25)

        self.duration_sec = max(
            1.0,
            float(self.get_parameter('duration_sec').value)
        )
        requested_run_id = str(self.get_parameter('run_id').value).strip()
        self.run_id = requested_run_id or time.strftime(
            'random_%Y%m%d_%H%M%S'
        )
        self.output_root = os.path.abspath(
            os.path.expanduser(
                str(self.get_parameter('output_root').value)
            )
        )
        self.drive_robot = bool(self.get_parameter('drive_robot').value)
        self.min_confidence = float(
            self.get_parameter('min_confidence').value
        )
        overrides = str(self.get_parameter('class_confidence_overrides').value)
        self.class_confidence_overrides = {}
        for entry in overrides.split(','):
            if not entry.strip():
                continue
            class_name, separator, threshold_text = entry.strip().partition(':')
            if not separator or not class_name or not threshold_text:
                raise ValueError(
                    'class_confidence_overrides must be class:threshold pairs'
                )
            threshold = float(threshold_text)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError('class confidence threshold must be 0 to 1')
            self.class_confidence_overrides[class_name] = threshold
        self.association_radius_m = max(
            0.05,
            float(self.get_parameter('association_radius_m').value)
        )
        self.min_confirmations = max(
            1,
            int(self.get_parameter('min_confirmations').value)
        )
        self.random_seed = int(
            self.get_parameter('random_seed').value
        )
        self.exploration_policy = str(
            self.get_parameter('exploration_policy').value
        ).lower()
        if self.exploration_policy not in ('random', 'coverage', 'nav2'):
            raise ValueError('exploration_policy must be random, coverage or nav2')
        self.tile_size_m = max(0.1, float(
            self.get_parameter('tile_size_m').value
        ))
        self.frontier_gain_weight_m = max(
            0.0, float(self.get_parameter('frontier_gain_weight_m').value)
        )
        self.frontier_visit_penalty_m = max(
            0.0, float(self.get_parameter('frontier_visit_penalty_m').value)
        )
        self.distant_goal_every = max(
            0, int(self.get_parameter('distant_goal_every').value)
        )
        self.nav_scan_interval_s = max(
            10.0, float(self.get_parameter('nav_scan_interval_s').value)
        )
        self.manual_nav2_activation = bool(
            self.get_parameter('manual_nav2_activation').value
        )
        self.stop_when_all_world_objects = bool(
            self.get_parameter('stop_when_all_world_objects').value
        )
        self.world_file = str(self.get_parameter('world_file').value)
        self.world_match_radius_m = float(
            self.get_parameter('world_match_radius_m').value
        )
        if self.world_match_radius_m <= 0:
            raise ValueError('world_match_radius_m must be positive')
        self.world_targets = (
            read_world(self.world_file) if self.stop_when_all_world_objects
            else []
        )
        self.random = random.Random(self.random_seed)

        self.run_dir = self._unique_run_directory(self.run_id)
        self.run_id = os.path.basename(self.run_dir)
        os.makedirs(self.run_dir, exist_ok=False)

        self.observations_file = open(
            os.path.join(self.run_dir, 'observations.csv'),
            'w',
            newline='',
            encoding='utf-8'
        )
        self.observations_writer = csv.DictWriter(
            self.observations_file,
            fieldnames=[
                'elapsed_s',
                'class',
                'confidence',
                'stamp_s',
                'map_x_m',
                'map_y_m',
                'map_z_m',
                'world_x_m',
                'world_y_m',
                'world_z_m',
                'is_new_object',
                'matched_object_id',
            ]
        )
        self.observations_writer.writeheader()
        self.observations_file.flush()

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            20
        )
        self.create_subscription(
            String,
            '/yolo/detections',
            self.detection_callback,
            20
        )
        # This simulator pose is stored solely for offline scoring. Frontier
        # and action selection use SLAM, odometry and LiDAR only.
        self.create_subscription(
            ModelStates, '/gazebo/model_states',
            self.ground_truth_callback, 1,
        )

        self.cmd_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/object_markers',
            marker_qos
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_map = None
        self.latest_scan = None
        self.latest_odom = None
        self.latest_odom_time = 0.0
        self.latest_world_robot_pose = None
        self.latest_world_robot_time = 0.0
        self.world_scoring_samples = 0
        self.world_objects = []
        self.world_path = []
        self.last_odom_xy = None
        self.distance_traveled = 0.0
        self.path = []
        self.objects = []
        self.total_detections = 0
        self.accepted_observations = 0
        self.accepted_confidences = []
        self.first_object_time = None

        self.started = False
        self.finished = False
        self.start_monotonic = None
        self.last_wait_log = 0.0
        self.last_path_sample = 0.0
        self.last_path_checkpoint = 0.0

        self.motion_state = 'forward'
        self.motion_state_until = 0.0
        self.turn_direction = 1.0
        self.stuck_check_xy = None
        self.stuck_check_time = time.monotonic()
        self.stuck_windows = 0
        self.current_map_pose = None
        self.visited_tiles = set()
        self.last_tile = None
        self.last_new_tile_time = time.monotonic()
        self.tile_revisits = 0
        self.coverage_path = []
        self.last_coverage_plan = 0.0
        self.coverage_goal_tile = None
        self.coverage_goal_best_distance = float('inf')
        self.coverage_goal_last_progress = time.monotonic()
        self.coverage_goal_penalties = {}
        self.last_command_linear = 0.0
        self.last_scan_tile_count = -2
        self.view_scans = 0
        self.nav_client = None
        self.nav_active_client = None
        self.nav_active_future = None
        self.nav_active_cached = False
        self.nav_last_active_query = 0.0
        self.nav_state_clients = {}
        self.nav_state_futures = {}
        self.nav_state_checked_at = {}
        self.nav_state_active = {}
        self.spin_client = None
        self.backup_client = None
        self.drive_heading_client = None
        self.action_send_future = None
        self.action_result_future = None
        self.action_goal_handle = None
        self.action_kind = None
        self.action_started = 0.0
        self.action_cancel_requested = False
        self.nav_goal_tile = None
        self.nav_goal_xy = None
        self.nav_goal_start_xy = None
        self.nav_goal_last_motion_xy = None
        self.nav_goal_last_motion_time = 0.0
        self.nav_scan_pending = self.exploration_policy == 'nav2'
        self.nav_last_scan_time = 0.0
        self.nav_goals_sent = 0
        self.nav_goals_succeeded = 0
        self.nav_goals_failed = 0
        self.nav_last_no_goal_scan = 0.0
        self.nav_no_goal_cycles = 0
        self.nav_backups = 0
        self.nav_last_backup_time = 0.0
        self.nav_failed_streak = 0
        self.nav_failed_waypoint_times = {}
        self.nav_probe_budget = 0
        self.nav_probe_goals = 0
        self.nav_goal_is_probe = False
        self.nav_goal_is_distant = False
        self.nav_direct_probe_pending = False
        self.nav_direct_drive_pending = False
        self.nav_direct_target = None
        self.nav_direct_probes = 0
        self.nav_direct_succeeded = 0
        self.nav_failed_waypoints = {}
        self.nav_last_direct_escape = 0.0
        if self.exploration_policy == 'nav2':
            self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
            self.nav_active_client = self.create_client(
                Trigger, '/lifecycle_manager_navigation/is_active'
            )
            if self.manual_nav2_activation:
                for name in ('bt_navigator', 'velocity_smoother'):
                    self.nav_state_clients[name] = self.create_client(
                        GetState, f'/{name}/get_state')
            self.spin_client = ActionClient(self, Spin, '/spin')
            self.backup_client = ActionClient(self, BackUp, '/backup')
            self.drive_heading_client = ActionClient(
                self, DriveOnHeading, '/drive_on_heading'
            )

        self.control_timer = self.create_timer(0.1, self.control_step)
        self.marker_timer = self.create_timer(0.5, self.publish_markers)

        self.get_logger().info(
            f'Baseline run {self.run_id!r} prepared in {self.run_dir}'
        )
        self.get_logger().info(
            'Waiting for /map, /scan, /odom, and the YOLO detection publisher...'
        )

    def _unique_run_directory(self, run_id):
        os.makedirs(self.output_root, exist_ok=True)
        candidate = os.path.join(self.output_root, run_id)
        suffix = 2
        while os.path.exists(candidate):
            candidate = os.path.join(self.output_root, f'{run_id}_{suffix}')
            suffix += 1
        return candidate

    def map_callback(self, msg):
        self.latest_map = msg

    def ground_truth_callback(self, msg):
        # The model list is read only to locate the robot. Other Gazebo model
        # poses are never stored or passed to navigation.
        try:
            index = msg.name.index('burger')
        except ValueError:
            return
        self.latest_world_robot_pose = msg.pose[index]
        self.latest_world_robot_time = time.monotonic()

    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        self.latest_odom = msg
        self.latest_odom_time = time.monotonic()
        position = msg.pose.pose.position
        current_xy = (position.x, position.y)

        if self.started and self.last_odom_xy is not None:
            step_distance = math.hypot(
                current_xy[0] - self.last_odom_xy[0],
                current_xy[1] - self.last_odom_xy[1]
            )
            if step_distance < 0.5:
                self.distance_traveled += step_distance

        self.last_odom_xy = current_xy

    def ready(self):
        sensors_ready = (
            self.latest_map is not None and
            self.latest_scan is not None and
            self.latest_odom is not None and
            self.count_publishers('/yolo/detections') > 0
        )
        if not sensors_ready or self.nav_client is None:
            return sensors_ready
        if not self.nav_client.server_is_ready():
            return False
        if self.manual_nav2_activation:
            return self._manual_nav2_ready()
        if self.nav_active_cached:
            return True
        if not self.nav_active_client.service_is_ready():
            return False
        if self.nav_active_future is not None:
            if not self.nav_active_future.done():
                return False
            try:
                self.nav_active_cached = bool(
                    self.nav_active_future.result().success
                )
            except Exception as error:
                self.get_logger().warning(
                    f'Nav2 active-state check failed: {error}'
                )
            self.nav_active_future = None
            if self.nav_active_cached:
                return True
        if time.monotonic() - self.nav_last_active_query >= 1.0:
            self.nav_last_active_query = time.monotonic()
            self.nav_active_future = self.nav_active_client.call_async(
                Trigger.Request()
            )
        return False

    def _manual_nav2_ready(self):
        """Wait for the navigator and command smoother to be active."""
        now = time.monotonic()
        for name, client in self.nav_state_clients.items():
            if self.nav_state_active.get(name):
                continue
            future = self.nav_state_futures.get(name)
            if future is not None and future.done():
                try:
                    self.nav_state_active[name] = (
                        future.result().current_state.id ==
                        State.PRIMARY_STATE_ACTIVE)
                except Exception as error:
                    self.get_logger().warning(
                        f'Nav2 state check failed for {name}: {error}')
                self.nav_state_futures.pop(name, None)
            if self.nav_state_active.get(name):
                continue
            if (client.service_is_ready() and
                    name not in self.nav_state_futures and
                    now - self.nav_state_checked_at.get(name, 0.0) >= 1.0):
                self.nav_state_checked_at[name] = now
                self.nav_state_futures[name] = client.call_async(
                    GetState.Request())
        self.nav_active_cached = all(
            self.nav_state_active.get(name, False)
            for name in self.nav_state_clients)
        return self.nav_active_cached

    def start_experiment(self):
        self.started = True
        self.start_monotonic = time.monotonic()
        self.stuck_check_time = self.start_monotonic
        self.stuck_check_xy = self.last_odom_xy
        self.get_logger().info(
            f'BASELINE STARTED: {self.exploration_policy} policy for '
            f'{self.duration_sec:.1f}s, '
            f'seed={self.random_seed}, min_confidence={self.min_confidence:.2f}'
        )

    def elapsed(self):
        if self.start_monotonic is None:
            return 0.0
        return time.monotonic() - self.start_monotonic

    def detection_callback(self, msg):
        if not self.started or self.finished:
            return

        try:
            detection = json.loads(msg.data)
            class_name = str(detection['class'])
            confidence = float(detection.get('confidence', 0.0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

        self.total_detections += 1
        minimum_confidence = self.class_confidence_overrides.get(
            class_name, self.min_confidence
        )
        if confidence < minimum_confidence:
            return

        map_xyz = detection.get('map_xyz')
        if map_xyz is None:
            map_xyz = self.odom_to_map(detection.get('odom_xyz'))
        if not isinstance(map_xyz, (list, tuple)) or len(map_xyz) != 3:
            return

        try:
            x, y, z = map(float, map_xyz)
        except (TypeError, ValueError):
            return

        world_xyz = None
        odom_xyz = detection.get('odom_xyz')
        if (isinstance(odom_xyz, (list, tuple)) and len(odom_xyz) == 3 and
                self.latest_odom is not None and
                self.latest_world_robot_pose is not None and
                time.monotonic() - self.latest_odom_time < 1.0 and
                time.monotonic() - self.latest_world_robot_time < 1.0):
            try:
                world_xyz = detection_in_world(
                    odom_xyz, self.latest_odom.pose.pose,
                    self.latest_world_robot_pose,
                )
            except (TypeError, ValueError):
                pass

        match = None
        match_distance = self.association_radius_m
        for obj in self.objects:
            if obj['class'] != class_name:
                continue
            distance = math.hypot(obj['x'] - x, obj['y'] - y)
            if distance < match_distance:
                match = obj
                match_distance = distance

        is_new = match is None
        newly_confirmed = False
        if match is None:
            match = {
                'id': len(self.objects) + 1,
                'class': class_name,
                'x': x,
                'y': y,
                'z': z,
                'best_confidence': confidence,
                'mean_confidence': confidence,
                'observations': 1,
                'first_seen_s': self.elapsed(),
                'last_seen_s': self.elapsed(),
                'confirmed_s': (
                    self.elapsed() if self.min_confirmations == 1 else None
                ),
            }
            self.objects.append(match)
            newly_confirmed = self.min_confirmations == 1
            if (
                self.min_confirmations == 1 and
                self.first_object_time is None
            ):
                self.first_object_time = match['confirmed_s']
            self.get_logger().info(
                f'NEW OBJECT #{match["id"]}: {class_name} at '
                f'({x:.2f}, {y:.2f}) confidence={confidence:.3f}'
            )
        else:
            count = match['observations']
            match['x'] = (match['x'] * count + x) / (count + 1)
            match['y'] = (match['y'] * count + y) / (count + 1)
            match['z'] = (match['z'] * count + z) / (count + 1)
            match['mean_confidence'] = (
                match['mean_confidence'] * count + confidence
            ) / (count + 1)
            match['observations'] += 1
            match['last_seen_s'] = self.elapsed()
            match['best_confidence'] = max(
                match['best_confidence'],
                confidence
            )
            if (
                match['confirmed_s'] is None and
                match['observations'] >= self.min_confirmations
            ):
                match['confirmed_s'] = self.elapsed()
                newly_confirmed = True
                if self.first_object_time is None:
                    self.first_object_time = match['confirmed_s']
                self.get_logger().info(
                    f'CONFIRMED OBJECT #{match["id"]}: {class_name} '
                    f'with {match["observations"]} observations'
                )

        self.accepted_observations += 1
        self.accepted_confidences.append(confidence)
        frame_stamp = detection.get('stamp')
        self.observations_writer.writerow({
            'elapsed_s': f'{self.elapsed():.3f}',
            'class': class_name,
            'confidence': f'{confidence:.6f}',
            'stamp_s': f'{float(frame_stamp):.9f}' if frame_stamp is not None else '',
            'map_x_m': f'{x:.6f}',
            'map_y_m': f'{y:.6f}',
            'map_z_m': f'{z:.6f}',
            'world_x_m': f'{world_xyz[0]:.6f}' if world_xyz else '',
            'world_y_m': f'{world_xyz[1]:.6f}' if world_xyz else '',
            'world_z_m': f'{world_xyz[2]:.6f}' if world_xyz else '',
            'is_new_object': is_new,
            'matched_object_id': match['id'],
        })
        self.observations_file.flush()
        if world_xyz is not None:
            self.world_scoring_samples += 1
            self.track_world_detection(
                class_name, confidence, world_xyz, frame_stamp,
            )

    def track_world_detection(self, class_name, confidence, xyz, frame_stamp=None):
        """Cluster detections for evaluation; never used by navigation."""
        x, y, z = xyz
        match = None
        best_distance = min(self.association_radius_m, 0.75)
        for obj in self.world_objects:
            if obj['class'] != class_name:
                continue
            distance = math.hypot(obj['x'] - x, obj['y'] - y)
            if distance < best_distance:
                match, best_distance = obj, distance
        now = self.elapsed()
        if match is None:
            match = {
                'id': len(self.world_objects) + 1,
                'class': class_name,
                'x': x, 'y': y, 'z': z,
                'best_confidence': confidence,
                'mean_confidence': confidence,
                'observations': 1,
                'first_seen_s': now,
                'last_seen_s': now,
                'confirmed_s': now if self.min_confirmations == 1 else None,
                'last_stamp': frame_stamp,
            }
            self.world_objects.append(match)
            newly_confirmed = self.min_confirmations == 1
        else:
            if frame_stamp is not None and match['last_stamp'] == frame_stamp:
                return
            count = match['observations']
            if match['confirmed_s'] is None:
                match['x'] = (match['x'] * count + x) / (count + 1)
                match['y'] = (match['y'] * count + y) / (count + 1)
                match['z'] = (match['z'] * count + z) / (count + 1)
            match['mean_confidence'] = (
                match['mean_confidence'] * count + confidence
            ) / (count + 1)
            match['best_confidence'] = max(match['best_confidence'], confidence)
            match['observations'] += 1
            match['last_seen_s'] = now
            match['last_stamp'] = frame_stamp
            newly_confirmed = match['confirmed_s'] is None and (
                match['observations'] >= self.min_confirmations
            )
            if newly_confirmed:
                match['confirmed_s'] = now
        if newly_confirmed and self.stop_when_all_world_objects:
            detections = [{
                'id': obj['id'],
                'class': obj['class'],
                'x': obj['x'],
                'y': obj['y'],
                'first_seen_s': obj['first_seen_s'],
                'confirmed_s': obj['confirmed_s'],
            } for obj in self.world_objects if obj['confirmed_s'] is not None]
            matches, _ = match_instances(
                self.world_targets, detections, self.world_match_radius_m
            )
            if matches and all(item['detected'] for item in matches):
                self.finish('all_world_objects_found')

    def odom_to_map(self, xyz):
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'odom',
                Time(),
                timeout=Duration(seconds=0.05)
            )
            point = PointStamped()
            point.header.frame_id = 'odom'
            point.point.x = float(xyz[0])
            point.point.y = float(xyz[1])
            point.point.z = float(xyz[2])
            transformed = do_transform_point(point, transform)
            return [
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            ]
        except Exception:
            return None

    def sector_clearance(self, center_angle, half_width):
        scan = self.latest_scan
        if scan is None:
            return float('inf')

        values = []
        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance) or distance <= scan.range_min:
                continue
            angle = scan.angle_min + index * scan.angle_increment
            difference = math.atan2(
                math.sin(angle - center_angle),
                math.cos(angle - center_angle)
            )
            if abs(difference) <= half_width:
                values.append(distance)
        return min(values) if values else float('inf')

    def enter_backup(self):
        self.motion_state = 'backup'
        self.motion_state_until = time.monotonic() + self.random.uniform(0.7, 1.1)
        self.stuck_windows = 0
        if self.exploration_policy == 'coverage':
            if self.coverage_goal_tile is not None:
                tile = self.coverage_goal_tile
                self.coverage_goal_penalties[tile] = 12.0
            self.coverage_path = []
            self.last_coverage_plan = 0.0

    def enter_turn(self, hard=False):
        self.motion_state = 'turn'
        if self.exploration_policy == 'coverage':
            left = self.sector_clearance(math.radians(70), math.radians(18))
            right = self.sector_clearance(math.radians(-70), math.radians(18))
            self.turn_direction = 1.0 if left >= right else -1.0
        else:
            self.turn_direction = self.random.choice((-1.0, 1.0))
        maximum = 2.8 if hard else 2.0
        self.motion_state_until = time.monotonic() + self.random.uniform(1.1, maximum)

    def update_path(self):
        now = time.monotonic()
        if now - self.last_path_sample < 0.25:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                Time(),
                timeout=Duration(seconds=0.02)
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            yaw = math.atan2(
                2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                1.0 - 2.0 * (rotation.y ** 2 + rotation.z ** 2)
            )
            point = (translation.x, translation.y)
            self.current_map_pose = (point[0], point[1], yaw)
            tile = (
                math.floor(point[0] / self.tile_size_m),
                math.floor(point[1] / self.tile_size_m),
            )
            if tile != self.last_tile:
                if tile in self.visited_tiles:
                    self.tile_revisits += 1
                else:
                    self.last_new_tile_time = now
                self.visited_tiles.add(tile)
                self.last_tile = tile
            if not self.path or math.hypot(
                point[0] - self.path[-1][0],
                point[1] - self.path[-1][1]
            ) > 0.02:
                self.path.append(point)
            if (self.latest_world_robot_pose is not None and
                    now - self.latest_world_robot_time < 1.0):
                world_point = (
                    self.latest_world_robot_pose.position.x,
                    self.latest_world_robot_pose.position.y,
                )
                if not self.world_path or math.hypot(
                    world_point[0] - self.world_path[-1][0],
                    world_point[1] - self.world_path[-1][1],
                ) > 0.02:
                    self.world_path.append(world_point)
            self.last_path_sample = now
        except Exception:
            pass

    def coverage_command(self, front, left, right):
        """Follow a frontier path; scan in place if no reachable frontier exists."""
        command = Twist()
        if self.current_map_pose is None or self.latest_map is None:
            return self.reactive_coverage_command()

        now = time.monotonic()
        if self.coverage_path and self.coverage_goal_tile is not None:
            goal_x, goal_y = self.coverage_path[-1]
            distance = math.hypot(
                goal_x - self.current_map_pose[0],
                goal_y - self.current_map_pose[1],
            )
            if distance < self.coverage_goal_best_distance - 0.15:
                self.coverage_goal_best_distance = distance
                self.coverage_goal_last_progress = now
            elif distance < 0.35 or now - self.coverage_goal_last_progress > 12.0:
                tile = self.coverage_goal_tile
                self.coverage_goal_penalties[tile] = (
                    self.coverage_goal_penalties.get(tile, 0.0) + 12.0
                )
                self.coverage_path = []
                self.last_coverage_plan = 0.0

        if now - self.last_coverage_plan >= 5.0 or not self.coverage_path:
            info = self.latest_map.info
            orientation = info.origin.orientation
            origin_yaw = math.atan2(
                2.0 * (orientation.w * orientation.z +
                       orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)
            )
            self.coverage_path = plan_frontier(
                self.latest_map.data,
                info.width,
                info.height,
                info.resolution,
                (info.origin.position.x, info.origin.position.y, origin_yaw),
                self.current_map_pose[:2],
                self.visited_tiles,
                tile_size=self.tile_size_m,
                goal_penalties=self.coverage_goal_penalties,
            )
            self.last_coverage_plan = now
            if self.coverage_path:
                goal_x, goal_y = self.coverage_path[-1]
                goal_tile = (
                    math.floor(goal_x / self.tile_size_m),
                    math.floor(goal_y / self.tile_size_m),
                )
                if goal_tile != self.coverage_goal_tile:
                    self.coverage_goal_tile = goal_tile
                    self.coverage_goal_best_distance = math.hypot(
                        goal_x - self.current_map_pose[0],
                        goal_y - self.current_map_pose[1],
                    )
                    self.coverage_goal_last_progress = now

        if not self.coverage_path:
            return self.reactive_coverage_command()

        x, y, yaw = self.current_map_pose
        waypoint = self.coverage_path[-1]
        for candidate in self.coverage_path:
            if math.hypot(candidate[0] - x, candidate[1] - y) >= 0.4:
                waypoint = candidate
                break
        error = math.atan2(waypoint[1] - y, waypoint[0] - x) - yaw
        error = math.atan2(math.sin(error), math.cos(error))
        command.angular.z = max(-0.65, min(0.65, 1.5 * error))
        if front < 0.45 or left < 0.27 or right < 0.27:
            command.angular.z = 0.55 if left >= right else -0.55
            return command
        if abs(error) < 0.45:
            command.linear.x = 0.14 if abs(error) < 0.22 else 0.08
        return command

    def reactive_coverage_command(self):
        """Move toward clear, little-visited space while SLAM has no route."""
        command = Twist()
        if self.current_map_pose is None:
            command.angular.z = 0.4
            return command
        x, y, yaw = self.current_map_pose
        options = []
        for angle in (0.0, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57, 3.14):
            clearance = self.sector_clearance(angle, 0.24)
            if clearance < 0.55:
                continue
            target_x = x + 0.9 * math.cos(yaw + angle)
            target_y = y + 0.9 * math.sin(yaw + angle)
            tile = (
                math.floor(target_x / self.tile_size_m),
                math.floor(target_y / self.tile_size_m),
            )
            novelty = 2.0 if tile not in self.visited_tiles else 0.0
            score = novelty + min(clearance, 2.5) - 0.1 * abs(angle)
            options.append((score, -abs(angle), angle))
        if not options:
            command.angular.z = 0.45
            return command
        angle = max(options)[2]
        if abs(angle) > 0.3:
            command.angular.z = 0.6 if angle > 0 else -0.6
        else:
            command.linear.x = 0.12
            command.angular.z = max(-0.4, min(0.4, angle))
        return command

    def _send_nav2_spin(self):
        if not self.spin_client.server_is_ready():
            return
        goal = Spin.Goal()
        goal.target_yaw = float(2.0 * math.pi)
        goal.time_allowance = Duration(seconds=35.0).to_msg()
        self.action_kind = 'spin'
        self.action_started = time.monotonic()
        self.action_send_future = self.spin_client.send_goal_async(goal)
        self.view_scans += 1
        self.nav_scan_pending = False
        self.nav_last_scan_time = time.monotonic()
        self.get_logger().info('Scanning surroundings with a full camera turn')

    def _send_nav2_backup(self):
        if not self.backup_client.server_is_ready():
            return
        goal = BackUp.Goal()
        goal.target.x = -0.40
        goal.speed = 0.10
        goal.time_allowance = Duration(seconds=12.0).to_msg()
        self.action_kind = 'backup'
        self.action_started = time.monotonic()
        self.action_send_future = self.backup_client.send_goal_async(goal)
        self.nav_backups += 1
        self.get_logger().info(
            f'No accessible frontier; backing away from obstruction '
            f'(recovery #{self.nav_backups})'
        )

    def _start_nav2_direct_probe(self, info, origin_yaw):
        """Bypass a stale global map only where current LiDAR permits it."""
        self.nav_direct_probe_pending = False
        if not (self.spin_client.server_is_ready() and
                self.drive_heading_client.server_is_ready()):
            return
        x, y, yaw = self.current_map_pose
        target = self.nav_direct_target
        if target is not None:
            candidate_angle = math.atan2(target[1] - y, target[0] - x) - yaw
            candidate_angle = math.atan2(
                math.sin(candidate_angle), math.cos(candidate_angle)
            )
            candidate_distance = math.hypot(target[0] - x, target[1] - y)
            clearance = self.sector_clearance(candidate_angle, 0.22)
            if (candidate_distance < 0.55 or
                    clearance < min(candidate_distance, 0.85) + 0.35):
                target = None
        if target is None:
            target = self._plan_nav2_probe(info, origin_yaw)
        if target is None:
            # A stale or locally boxed-in SLAM map can have no legal probe
            # cell even when LiDAR still sees open space. Use a short sensor-
            # clear escape in that case; this never consults world objects.
            choices = []
            for angle in (0.0, math.pi / 4, -math.pi / 4,
                          math.pi / 2, -math.pi / 2,
                          3 * math.pi / 4, -3 * math.pi / 4, math.pi):
                clearance = self.sector_clearance(angle, 0.22)
                if clearance < 1.0:
                    continue
                # A scan sector with no return is usable but less certain
                # than one with a measured, distant surface.
                useful_clearance = (
                    min(clearance, 2.0) if math.isfinite(clearance) else 1.8
                )
                choices.append((useful_clearance - 0.1 * abs(angle), angle))
            if not choices:
                return
            angle = max(choices)[1]
            target = (
                x + 0.7 * math.cos(yaw + angle),
                y + 0.7 * math.sin(yaw + angle),
            )
            self.get_logger().info(
                f'LiDAR-only escape toward {math.degrees(angle):.0f} deg'
            )
        error = math.atan2(target[1] - y, target[0] - x) - yaw
        error = math.atan2(math.sin(error), math.cos(error))
        self.nav_direct_target = target
        if abs(error) < 0.12:
            self.nav_direct_drive_pending = True
            return
        goal = Spin.Goal()
        goal.target_yaw = float(error)
        goal.time_allowance = Duration(seconds=18.0).to_msg()
        self.action_kind = 'direct_probe_spin'
        self.action_started = time.monotonic()
        self.action_send_future = self.spin_client.send_goal_async(goal)
        self.get_logger().info(
            f'Local-costmap probe: turn {error:.2f} rad toward '
            f'({target[0]:.2f}, {target[1]:.2f})'
        )

    def _send_nav2_direct_drive(self):
        self.nav_direct_drive_pending = False
        if not self.drive_heading_client.server_is_ready():
            return
        clearance = self.sector_clearance(0.0, 0.22)
        remaining = math.hypot(
            self.nav_direct_target[0] - self.current_map_pose[0],
            self.nav_direct_target[1] - self.current_map_pose[1],
        )
        distance = min(0.85, remaining, clearance - 0.35)
        if distance < 0.30:
            self.get_logger().info('Local-costmap probe skipped: front blocked')
            return
        goal = DriveOnHeading.Goal()
        goal.target.x = float(distance)
        goal.speed = 0.10
        goal.time_allowance = Duration(seconds=18.0).to_msg()
        self.action_kind = 'direct_probe_drive'
        self.action_started = time.monotonic()
        self.action_send_future = self.drive_heading_client.send_goal_async(goal)
        self.nav_direct_probes += 1
        self.get_logger().info(
            f'Local-costmap probe #{self.nav_direct_probes}: '
            f'advance {distance:.2f} m'
        )

    def _send_nav2_goal(self, path, probe=False, distant=False):
        # Short waypoints let Nav2 replan as SLAM reveals nearby obstacles.
        # Sending a distant map frontier directly often exhausts the 60 s
        # action timeout before the robot can negotiate a narrow passage.
        if not probe and not distant and len(path) > 35:
            path = path[:30]
        x, y = path[-1]
        if len(path) > 1:
            previous_x, previous_y = path[-2]
        else:
            previous_x, previous_y = self.current_map_pose[:2]
        yaw = math.atan2(y - previous_y, x - previous_x)
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.pose = pose
        self.nav_goal_tile = (
            math.floor(x / self.tile_size_m),
            math.floor(y / self.tile_size_m),
        )
        self.nav_goal_xy = (x, y)
        self.nav_goal_start_xy = self.current_map_pose[:2]
        self.nav_goal_last_motion_xy = self.current_map_pose[:2]
        self.nav_goal_last_motion_time = time.monotonic()
        self.nav_goal_is_probe = probe
        self.nav_goal_is_distant = distant
        self.action_kind = 'navigate'
        self.action_started = time.monotonic()
        self.action_cancel_requested = False
        self.action_send_future = self.nav_client.send_goal_async(goal)
        self.nav_goals_sent += 1
        if probe:
            self.nav_probe_goals += 1
        self.get_logger().info(
            f'Nav2 {"LiDAR probe" if probe else "distant frontier" if distant else "frontier"} '
            f'#{self.nav_goals_sent}: ({x:.2f}, {y:.2f}), '
            f'{len(path)} path cells, {len(self.visited_tiles)} tiles visited'
        )

    def _complete_nav2_action(self, succeeded):
        if self.action_kind == 'navigate':
            # Nav2 can succeed with a short partial path if a frontier is
            # blocked in its costmap. Verify the actual robot position.
            if succeeded and self.current_map_pose is not None:
                distance_to_goal = math.hypot(
                    self.nav_goal_xy[0] - self.current_map_pose[0],
                    self.nav_goal_xy[1] - self.current_map_pose[1],
                )
                succeeded = distance_to_goal <= 0.35
            if self.nav_goal_tile is not None:
                # Adjacent tiles can represent the same blocked frontier.
                # Exclude its small neighborhood after a failed goal.
                gx, gy = self.nav_goal_tile
                offsets = (0,) if succeeded else (-1, 0, 1)
                for dx in offsets:
                    for dy in offsets:
                        self.coverage_goal_penalties[(gx + dx, gy + dy)] = 12.0
            if succeeded:
                self.nav_goals_succeeded += 1
                self.nav_failed_streak = 0
                self.nav_probe_budget = 0
                if self.nav_goal_tile is not None:
                    self.nav_failed_waypoints.pop(self.nav_goal_tile, None)
                    self.nav_failed_waypoint_times.pop(self.nav_goal_tile, None)
                if len(self.visited_tiles) - self.last_scan_tile_count >= 3:
                    self.nav_scan_pending = True
            else:
                self.nav_goals_failed += 1
                self.nav_failed_streak += 1
                if self.nav_goal_tile is not None:
                    tile = self.nav_goal_tile
                    self.nav_failed_waypoints[tile] = (
                        self.nav_failed_waypoints.get(tile, 0) + 1
                    )
                    self.nav_failed_waypoint_times[tile] = time.monotonic()
                    if self.nav_failed_waypoints[tile] == 2:
                        self.get_logger().warning(
                            f'Excluding repeatedly blocked waypoint tile {tile} '
                            'from future frontier routes'
                        )
                if self.nav_failed_streak >= 2 and not self.nav_goal_is_probe:
                    self.nav_probe_budget = 3
                if self.nav_goal_is_probe:
                    self.nav_direct_target = self.nav_goal_xy
                    self.nav_direct_probe_pending = True
                self.get_logger().warning(
                    f'Nav2 frontier failed; choosing another tile '
                    f'({self.nav_goals_failed} failures)'
                )
        elif self.action_kind == 'spin':
            self.last_scan_tile_count = len(self.visited_tiles)
            if not succeeded:
                self.get_logger().warning('Nav2 camera sweep did not complete')
        elif self.action_kind == 'backup':
            if succeeded:
                self.nav_failed_streak = 0
                self.nav_probe_budget = 0
                self.get_logger().info('Nav2 backup succeeded; replanning')
            else:
                self.get_logger().warning('Nav2 backup failed')
        elif self.action_kind == 'direct_probe_spin':
            self.nav_direct_drive_pending = succeeded
        elif self.action_kind == 'direct_probe_drive':
            if succeeded:
                self.nav_direct_succeeded += 1
                self.nav_failed_streak = 0
                self.nav_probe_budget = 0
                self.nav_scan_pending = True
                self.get_logger().info('Local-costmap probe reached new space')
            else:
                if self.nav_direct_target is not None:
                    tx, ty = self.nav_direct_target
                    tile = (math.floor(tx / self.tile_size_m),
                            math.floor(ty / self.tile_size_m))
                    self.coverage_goal_penalties[tile] = 12.0
                self.get_logger().warning(
                    'Local-costmap probe stopped before reaching goal'
                )
        self.action_kind = None
        self.action_send_future = None
        self.action_result_future = None
        self.action_goal_handle = None
        self.action_cancel_requested = False
        self.nav_goal_is_probe = False
        self.nav_goal_is_distant = False

    def _plan_nav2_probe(self, info, origin_yaw):
        if self.latest_scan is None:
            return None
        return plan_sensor_probe(
            self.latest_map.data,
            info.width,
            info.height,
            info.resolution,
            (info.origin.position.x, info.origin.position.y, origin_yaw),
            self.current_map_pose,
            self.sector_clearance,
            self.visited_tiles,
            tile_size=self.tile_size_m,
            goal_penalties=self.coverage_goal_penalties,
        )

    def _dispatch_nav2_probe(self, target, info, origin_yaw):
        """Use local collision checking if live LiDAR clears a map obstacle."""
        row, col = world_to_cell(
            target[0], target[1], info.resolution,
            (info.origin.position.x, info.origin.position.y, origin_yaw),
        )
        if (0 <= row < info.height and 0 <= col < info.width and
                self.latest_map.data[row * info.width + col] >= 50):
            self.nav_direct_target = target
            self._start_nav2_direct_probe(info, origin_yaw)
        else:
            self._send_nav2_goal([self.current_map_pose[:2], target], probe=True)

    def nav2_control_step(self):
        """Send map-frontier goals to Nav2; world objects never guide motion."""
        if self.action_send_future is not None:
            if self.action_send_future.done():
                try:
                    self.action_goal_handle = self.action_send_future.result()
                except Exception as error:
                    self.get_logger().warning(f'Nav2 goal request failed: {error}')
                    self._complete_nav2_action(False)
                    return
                self.action_send_future = None
                if not self.action_goal_handle.accepted:
                    self._complete_nav2_action(False)
                    return
                self.action_result_future = self.action_goal_handle.get_result_async()
            return

        if self.action_result_future is not None:
            if (self.action_kind == 'navigate' and
                    self.current_map_pose is not None and
                    self.nav_goal_last_motion_xy is not None and
                    math.hypot(
                        self.current_map_pose[0] - self.nav_goal_last_motion_xy[0],
                        self.current_map_pose[1] - self.nav_goal_last_motion_xy[1],
                    ) >= 0.12):
                self.nav_goal_last_motion_xy = self.current_map_pose[:2]
                self.nav_goal_last_motion_time = time.monotonic()
            if self.action_result_future.done():
                try:
                    status = self.action_result_future.result().status
                    succeeded = status == GoalStatus.STATUS_SUCCEEDED
                except Exception as error:
                    self.get_logger().warning(f'Nav2 result failed: {error}')
                    succeeded = False
                self._complete_nav2_action(succeeded)
            elif (self.action_kind == 'navigate' and
                  not self.action_cancel_requested and
                  (time.monotonic() - self.action_started >
                   (25.0 if self.nav_goal_is_probe else
                    90.0 if self.nav_goal_is_distant else 40.0) or
                   time.monotonic() - self.nav_goal_last_motion_time > 20.0)):
                self.action_goal_handle.cancel_goal_async()
                self.action_cancel_requested = True
                self.get_logger().warning('Nav2 goal timed out; replanning')
            return

        if self.nav_direct_drive_pending:
            self._send_nav2_direct_drive()
            return
        if self.nav_direct_probe_pending:
            if self.current_map_pose is None or self.latest_map is None:
                return
            info = self.latest_map.info
            orientation = info.origin.orientation
            origin_yaw = math.atan2(
                2.0 * (orientation.w * orientation.z +
                       orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
            )
            self._start_nav2_direct_probe(info, origin_yaw)
            return

        if (self.nav_last_scan_time and
                time.monotonic() - self.nav_last_scan_time >=
                self.nav_scan_interval_s):
            self.nav_scan_pending = True
        if self.nav_scan_pending:
            self._send_nav2_spin()
            return

        if (self.nav_failed_streak >= 3 and
                time.monotonic() - self.last_new_tile_time > 45.0 and
                time.monotonic() - self.nav_last_backup_time > 45.0 and
                self.backup_client.server_is_ready()):
            self.nav_last_backup_time = time.monotonic()
            self._send_nav2_backup()
            return

        if (self.current_map_pose is not None and self.latest_map is not None and
                self.nav_probe_budget == 0 and
                time.monotonic() - self.last_new_tile_time > 90.0):
            info = self.latest_map.info
            orientation = info.origin.orientation
            origin_yaw = math.atan2(
                2.0 * (orientation.w * orientation.z +
                       orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
            )
            for candidate_radius in (0.25, 0.10, 0.0):
                path = plan_frontier(
                    self.latest_map.data, info.width, info.height,
                    info.resolution,
                    (info.origin.position.x, info.origin.position.y, origin_yaw),
                    self.current_map_pose[:2], self.visited_tiles,
                    tile_size=self.tile_size_m,
                    robot_radius=candidate_radius,
                    goal_penalties=self.coverage_goal_penalties,
                    distance_weight=-0.5,
                    prefer_open_area=True,
                )
                if path:
                    self.get_logger().info(
                        'No new floor tile for 90 s; stepping toward distant '
                        'unvisited SLAM free space'
                    )
                    self.nav_no_goal_cycles = 0
                    # Keep the long-range target but dispatch only the next
                    # short waypoint. Nav2 may reject a full SLAM path where
                    # its inflated costmap sees a blocked corridor.
                    self._send_nav2_goal(path)
                    return

        if (self.current_map_pose is not None and self.latest_map is not None and
                time.monotonic() - self.last_new_tile_time > 45.0 and
                time.monotonic() - self.nav_last_direct_escape > 20.0):
            self.nav_last_direct_escape = time.monotonic()
            self.nav_direct_target = None
            info = self.latest_map.info
            orientation = info.origin.orientation
            origin_yaw = math.atan2(
                2.0 * (orientation.w * orientation.z +
                       orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
            )
            self.get_logger().info(
                'No new floor tile for 45 s; probing sensor-clear space'
            )
            self._start_nav2_direct_probe(info, origin_yaw)
            return

        if self.current_map_pose is None or self.latest_map is None:
            return
        info = self.latest_map.info
        orientation = info.origin.orientation
        origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z +
                   orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
        )
        distance_weight = (
            -0.25 if self.distant_goal_every > 0 and
            self.nav_goals_sent > 0 and
            self.nav_goals_sent % self.distant_goal_every == 0
            else 1.0
        )
        if (self.nav_probe_budget > 0 or
                (self.nav_goals_sent > 0 and
                 self.nav_goals_sent % 5 == 0)):
            probe = self._plan_nav2_probe(info, origin_yaw)
            if self.nav_probe_budget > 0:
                self.nav_probe_budget -= 1
            if probe is not None:
                self._dispatch_nav2_probe(probe, info, origin_yaw)
                return
        path = []
        blocked_tiles = {
            (tile[0] + dx, tile[1] + dy)
            for tile, failures in self.nav_failed_waypoints.items()
            if failures >= 1 and
            time.monotonic() - self.nav_failed_waypoint_times.get(
                tile, 0.0
            ) < 600.0
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }
        # SLAM can paint the robot's map cell beside an obstacle even when
        # LiDAR shows a traversable corridor. A zero-inflation map route is a
        # last resort. If exclusion cuts off every route, let Nav2 plan to a
        # distant frontier without those exclusions rather than box us in.
        relaxed = False
        for relaxed in (False, True):
            for candidate_radius in (0.25, 0.10, 0.0):
                path = plan_frontier(
                    self.latest_map.data,
                    info.width,
                    info.height,
                    info.resolution,
                    (info.origin.position.x, info.origin.position.y, origin_yaw),
                    self.current_map_pose[:2],
                    self.visited_tiles,
                    tile_size=self.tile_size_m,
                    robot_radius=candidate_radius,
                    goal_penalties=self.coverage_goal_penalties,
                    information_gain_weight=self.frontier_gain_weight_m,
                    distance_weight=-0.25 if relaxed else distance_weight,
                    gain_radius_m=2.0,
                    nearby_visit_weight=self.frontier_visit_penalty_m,
                    avoid_tiles=() if relaxed else blocked_tiles,
                )
                if path:
                    if candidate_radius < 0.25:
                        self.get_logger().info(
                            'Using a narrow SLAM frontier; Nav2 will verify clearance'
                        )
                    break
            if path:
                break
        if path:
            self.nav_no_goal_cycles = 0
            self._send_nav2_goal(path, distant=relaxed)
        elif time.monotonic() - self.nav_last_no_goal_scan >= 15.0:
            probe = self._plan_nav2_probe(info, origin_yaw)
            if probe is not None:
                self._dispatch_nav2_probe(probe, info, origin_yaw)
                return
            self.nav_last_no_goal_scan = time.monotonic()
            self.nav_no_goal_cycles += 1
            if self.nav_no_goal_cycles >= 2:
                self.nav_no_goal_cycles = 0
                self._send_nav2_backup()
            else:
                self.get_logger().info(
                    'No reachable frontier; scanning for new map space'
                )
                self.nav_scan_pending = True

    def control_step(self):
        if self.finished:
            return

        if not self.started:
            if self.ready():
                self.start_experiment()
            elif time.monotonic() - self.last_wait_log >= 5.0:
                self.get_logger().info(
                    'Still waiting: '
                    f'map={self.latest_map is not None}, '
                    f'scan={self.latest_scan is not None}, '
                    f'odom={self.latest_odom is not None}, '
                    f'yolo_publishers={self.count_publishers("/yolo/detections")}, '
                    f'nav2_active={self.nav_client is None or self.nav_active_cached}'
                )
                self.last_wait_log = time.monotonic()
            return

        if self.elapsed() >= self.duration_sec:
            self.finish('completed')
            return

        self.update_path()
        # Preserve trajectory evidence during long runs, including runs
        # interrupted before finish() can write its final report.
        if time.monotonic() - self.last_path_checkpoint >= 30.0:
            self.save_robot_path()
            self.last_path_checkpoint = time.monotonic()
        if not self.drive_robot:
            return
        if self.exploration_policy == 'nav2':
            self.nav2_control_step()
            return

        now = time.monotonic()
        front = self.sector_clearance(0.0, math.radians(22.0))
        left = self.sector_clearance(math.radians(70.0), math.radians(18.0))
        right = self.sector_clearance(math.radians(-70.0), math.radians(18.0))

        if (front < 0.28 and
                self.motion_state not in ('backup', 'scan')):
            self.enter_backup()

        if now - self.stuck_check_time >= 4.0:
            if self.last_odom_xy is not None and self.stuck_check_xy is not None:
                moved = math.hypot(
                    self.last_odom_xy[0] - self.stuck_check_xy[0],
                    self.last_odom_xy[1] - self.stuck_check_xy[1]
                )
                if moved >= 0.12 or self.motion_state == 'scan':
                    self.stuck_windows = 0
                elif self.motion_state not in ('backup', 'turn'):
                    self.stuck_windows += 1
                threshold = 1 if self.exploration_policy == 'random' else 3
                if self.stuck_windows >= threshold:
                    self.enter_backup()
            self.stuck_check_xy = self.last_odom_xy
            self.stuck_check_time = now

        command = Twist()
        if self.motion_state == 'backup':
            command.linear.x = -0.10
            if now >= self.motion_state_until:
                self.enter_turn(hard=True)
        elif self.motion_state == 'scan':
            command.angular.z = 0.65
            if now >= self.motion_state_until:
                self.motion_state = 'forward'
                self.coverage_goal_last_progress = now
        elif self.motion_state == 'turn':
            command.angular.z = self.turn_direction * 0.65
            if now >= self.motion_state_until:
                if self.exploration_policy == 'coverage':
                    self.motion_state = 'escape'
                    self.motion_state_until = now + 3.0
                else:
                    self.motion_state = 'forward'
        elif self.motion_state == 'escape':
            if front < 0.4:
                self.enter_backup()
            elif now >= self.motion_state_until:
                self.motion_state = 'forward'
                self.last_coverage_plan = 0.0
            else:
                command.linear.x = 0.12
        elif (self.exploration_policy == 'coverage' and
              len(self.visited_tiles) - self.last_scan_tile_count >= 3):
            # A 360-degree camera sweep covers objects hidden beside the path.
            self.motion_state = 'scan'
            self.motion_state_until = now + 2.0 * math.pi / 0.65
            self.last_scan_tile_count = len(self.visited_tiles)
            self.view_scans += 1
            command.angular.z = 0.65
        elif self.exploration_policy == 'coverage':
            command = self.coverage_command(front, left, right)
        elif front < 0.55:
            self.enter_turn()
        elif left < 0.30 or right < 0.30:
            self.motion_state = 'turn'
            self.turn_direction = -1.0 if left < right else 1.0
            self.motion_state_until = now + self.random.uniform(0.7, 1.2)
        else:
            command.linear.x = 0.16

        self.last_command_linear = command.linear.x
        self.cmd_publisher.publish(command)

    def publish_markers(self):
        marker_array = MarkerArray()
        for index, obj in enumerate(self.confirmed_objects()):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'baseline_objects'
            marker.id = index * 2
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obj['x']
            marker.pose.position.y = obj['y']
            marker.pose.position.z = max(0.12, obj['z'])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.22
            marker.scale.y = 0.22
            marker.scale.z = 0.22
            color = self.class_color(obj['class'])
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.95
            marker_array.markers.append(marker)

            label = Marker()
            label.header = marker.header
            label.ns = 'baseline_labels'
            label.id = index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = obj['x']
            label.pose.position.y = obj['y']
            label.pose.position.z = max(0.35, obj['z'] + 0.25)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color.r = 0.05
            label.color.g = 0.05
            label.color.b = 0.05
            label.color.a = 1.0
            label.text = (
                f'#{obj["id"]} {obj["class"]} '
                f'({obj["best_confidence"]:.2f})'
            )
            marker_array.markers.append(label)

        self.marker_publisher.publish(marker_array)

    def confirmed_objects(self):
        return [
            obj for obj in self.objects
            if obj['observations'] >= self.min_confirmations
        ]

    @staticmethod
    def class_color(class_name):
        value = sum((index + 1) * ord(char) for index, char in enumerate(class_name))
        palette = [
            (0.84, 0.15, 0.16),
            (0.12, 0.47, 0.71),
            (0.17, 0.63, 0.17),
            (1.00, 0.50, 0.05),
            (0.58, 0.40, 0.74),
            (0.55, 0.34, 0.29),
            (0.89, 0.47, 0.76),
        ]
        return palette[value % len(palette)]

    def map_coverage_percent(self):
        if self.latest_map is None or not self.latest_map.data:
            return 0.0
        known = sum(value >= 0 for value in self.latest_map.data)
        return 100.0 * known / len(self.latest_map.data)

    def save_unique_objects(self):
        path = os.path.join(self.run_dir, 'unique_objects.csv')
        fieldnames = [
            'object_id',
            'class',
            'map_x_m',
            'map_y_m',
            'map_z_m',
            'best_confidence',
            'mean_confidence',
            'observations',
            'first_seen_s',
            'last_seen_s',
            'confirmed_s',
        ]
        with open(path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for obj in self.confirmed_objects():
                writer.writerow({
                    'object_id': obj['id'],
                    'class': obj['class'],
                    'map_x_m': f'{obj["x"]:.6f}',
                    'map_y_m': f'{obj["y"]:.6f}',
                    'map_z_m': f'{obj["z"]:.6f}',
                    'best_confidence': f'{obj["best_confidence"]:.6f}',
                    'mean_confidence': f'{obj["mean_confidence"]:.6f}',
                    'observations': obj['observations'],
                    'first_seen_s': f'{obj["first_seen_s"]:.3f}',
                    'last_seen_s': f'{obj["last_seen_s"]:.3f}',
                    'confirmed_s': f'{obj["confirmed_s"]:.3f}',
                })

        if self.world_scoring_samples:
            world_path = os.path.join(self.run_dir, 'world_unique_objects.csv')
            with open(world_path, 'w', newline='', encoding='utf-8') as output_file:
                writer = csv.DictWriter(output_file, fieldnames=[
                    'object_id', 'class', 'world_x_m', 'world_y_m', 'world_z_m',
                    'best_confidence', 'mean_confidence', 'observations',
                    'first_seen_s', 'last_seen_s', 'confirmed_s',
                ])
                writer.writeheader()
                for obj in self.world_objects:
                    if obj['confirmed_s'] is None:
                        continue
                    writer.writerow({
                        'object_id': obj['id'],
                        'class': obj['class'],
                        'world_x_m': f'{obj["x"]:.6f}',
                        'world_y_m': f'{obj["y"]:.6f}',
                        'world_z_m': f'{obj["z"]:.6f}',
                        'best_confidence': f'{obj["best_confidence"]:.6f}',
                        'mean_confidence': f'{obj["mean_confidence"]:.6f}',
                        'observations': obj['observations'],
                        'first_seen_s': f'{obj["first_seen_s"]:.3f}',
                        'last_seen_s': f'{obj["last_seen_s"]:.3f}',
                        'confirmed_s': f'{obj["confirmed_s"]:.3f}',
                    })

    def save_robot_path(self):
        path = os.path.join(self.run_dir, 'robot_path.csv')
        temporary_path = path + '.tmp'
        with open(temporary_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(('map_x_m', 'map_y_m'))
            for x, y in self.path:
                writer.writerow((f'{x:.6f}', f'{y:.6f}'))
        os.replace(temporary_path, path)
        if self.world_path:
            path = os.path.join(self.run_dir, 'world_robot_path.csv')
            temporary_path = path + '.tmp'
            with open(temporary_path, 'w', newline='', encoding='utf-8') as output_file:
                writer = csv.writer(output_file)
                writer.writerow(('world_x_m', 'world_y_m'))
                for x, y in self.world_path:
                    writer.writerow((f'{x:.6f}', f'{y:.6f}'))
            os.replace(temporary_path, path)

    def save_map_files(self):
        if self.latest_map is None:
            return

        info = self.latest_map.info
        grid = np.asarray(self.latest_map.data, dtype=np.int16).reshape(
            info.height,
            info.width
        )
        pixels = np.full(grid.shape, 205, dtype=np.uint8)
        pixels[grid == 0] = 254
        pixels[grid >= 65] = 0
        intermediate = (grid > 0) & (grid < 65)
        pixels[intermediate] = (
            254 - grid[intermediate].astype(np.float64) * 254.0 / 100.0
        ).astype(np.uint8)
        pixels = np.flipud(pixels)

        pgm_path = os.path.join(self.run_dir, 'slam_map.pgm')
        with open(pgm_path, 'wb') as output_file:
            output_file.write(
                f'P5\n{info.width} {info.height}\n255\n'.encode('ascii')
            )
            output_file.write(pixels.tobytes())

        orientation = info.origin.orientation
        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z +
                orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y ** 2 +
                orientation.z ** 2
            )
        )
        yaml_path = os.path.join(self.run_dir, 'slam_map.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as output_file:
            output_file.write('image: slam_map.pgm\n')
            output_file.write(f'resolution: {info.resolution}\n')
            output_file.write(
                'origin: '
                f'[{info.origin.position.x}, {info.origin.position.y}, {yaw}]\n'
            )
            output_file.write('negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')

    def save_map_plot(self):
        figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)

        if self.latest_map is not None:
            info = self.latest_map.info
            grid = np.asarray(self.latest_map.data).reshape(
                info.height,
                info.width
            )
            display = np.full(grid.shape, 0.72, dtype=float)
            display[grid == 0] = 1.0
            display[grid >= 50] = 0.05
            extent = [
                info.origin.position.x,
                info.origin.position.x + info.width * info.resolution,
                info.origin.position.y,
                info.origin.position.y + info.height * info.resolution,
            ]
            axis.imshow(
                display,
                cmap='gray',
                vmin=0.0,
                vmax=1.0,
                origin='lower',
                extent=extent,
                interpolation='nearest'
            )

        if self.path:
            path_x, path_y = zip(*self.path)
            axis.plot(
                path_x,
                path_y,
                color='#1976d2',
                linewidth=1.5,
                alpha=0.8,
                label='Robot path'
            )
            axis.scatter(
                [path_x[0]],
                [path_y[0]],
                marker='s',
                color='#2e7d32',
                s=55,
                label='Start'
            )

        labeled_classes = set()
        confirmed_objects = self.confirmed_objects()
        for obj in confirmed_objects:
            color = self.class_color(obj['class'])
            label = obj['class'] if obj['class'] not in labeled_classes else None
            labeled_classes.add(obj['class'])
            axis.scatter(
                [obj['x']],
                [obj['y']],
                s=90,
                color=color,
                edgecolors='black',
                linewidths=0.7,
                label=label,
                zorder=4
            )
            axis.annotate(
                f'#{obj["id"]} {obj["class"]}\n{obj["best_confidence"]:.2f}',
                (obj['x'], obj['y']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                zorder=5
            )

        axis.set(
            title=(
                f'{self.exploration_policy.title()} exploration: '
                f'{len(confirmed_objects)} grouped detection locations '
                f'in {self.elapsed() / 60.0:.2f} min'
            ),
            xlabel='Map X (m)',
            ylabel='Map Y (m)',
            aspect='equal',
        )
        axis.grid(True, alpha=0.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc='best', fontsize=8)

        figure.savefig(
            os.path.join(self.run_dir, 'map_with_objects.png'),
            dpi=200
        )
        plt.close(figure)

    def make_summary(self, status):
        duration = self.elapsed()
        confirmed_objects = self.confirmed_objects()
        classes = sorted({obj['class'] for obj in confirmed_objects})
        mean_confidence = (
            sum(self.accepted_confidences) / len(self.accepted_confidences)
            if self.accepted_confidences else 0.0
        )
        summary = {
            'run_id': self.run_id,
            'status': status,
            'duration_sec': round(duration, 3),
            'distance_traveled_m': round(self.distance_traveled, 3),
            'total_detections': self.total_detections,
            'accepted_observations': self.accepted_observations,
            'unique_objects_found': len(confirmed_objects),
            'unique_classes_found': len(classes),
            'classes': ';'.join(classes),
            'time_to_first_object_s': (
                round(self.first_object_time, 3)
                if self.first_object_time is not None else ''
            ),
            'objects_per_minute': (
                round(len(confirmed_objects) / (duration / 60.0), 3)
                if duration > 0.0 else 0.0
            ),
            'mean_accepted_confidence': round(mean_confidence, 6),
            'map_coverage_percent': round(self.map_coverage_percent(), 3),
            'random_seed': self.random_seed,
        }
        summary['exploration_policy'] = self.exploration_policy
        summary['visited_tiles'] = len(self.visited_tiles)
        summary['tile_revisits'] = self.tile_revisits
        summary['tile_size_m'] = self.tile_size_m
        summary['class_confidence_overrides'] = self.class_confidence_overrides
        summary['min_confirmations'] = self.min_confirmations
        summary['world_scoring_samples'] = self.world_scoring_samples
        if self.exploration_policy == 'nav2':
            summary['frontier_gain_weight_m'] = self.frontier_gain_weight_m
            summary['frontier_visit_penalty_m'] = self.frontier_visit_penalty_m
            summary['distant_goal_every'] = self.distant_goal_every
            summary['nav_scan_interval_s'] = self.nav_scan_interval_s
        summary['view_scans'] = self.view_scans
        if self.exploration_policy == 'nav2':
            summary['nav_goals_sent'] = self.nav_goals_sent
            summary['nav_goals_succeeded'] = self.nav_goals_succeeded
            summary['nav_goals_failed'] = self.nav_goals_failed
            summary['nav_backups'] = self.nav_backups
            summary['nav_probe_goals'] = self.nav_probe_goals
            summary['nav_direct_probes'] = self.nav_direct_probes
            summary['nav_direct_succeeded'] = self.nav_direct_succeeded
            summary['nav_blocked_waypoint_tiles'] = sum(
                failures >= 2 for failures in self.nav_failed_waypoints.values()
            )
        return summary

    def save_summary(self, summary):
        json_path = os.path.join(self.run_dir, 'summary.json')
        with open(json_path, 'w', encoding='utf-8') as output_file:
            json.dump(summary, output_file, indent=2)
            output_file.write('\n')

        markdown_path = os.path.join(self.run_dir, 'summary.md')
        with open(markdown_path, 'w', encoding='utf-8') as output_file:
            output_file.write(f'# Baseline run: {self.run_id}\n\n')
            output_file.write(
                f'{self.exploration_policy.title()} policy produced '
                f'**{summary["unique_objects_found"]} grouped detection '
                f'locations in {summary["duration_sec"] / 60.0:.2f} '
                f'minutes**.\n\n'
            )
            if 'world_objects_total' in summary:
                output_file.write(
                    f'Confirmed world match: **{summary["world_objects_matched"]}/'
                    f'{summary["world_objects_total"]} placed objects**.\n\n'
                )
            output_file.write('| Metric | Result |\n|---|---:|\n')
            for key in self.SUMMARY_FIELDS:
                output_file.write(f'| {key} | {summary[key]} |\n')
            for key in ('exploration_policy', 'visited_tiles',
                        'tile_revisits', 'tile_size_m', 'view_scans'):
                output_file.write(f'| {key} | {summary[key]} |\n')
            for key in ('nav_goals_sent', 'nav_goals_succeeded',
                        'nav_goals_failed', 'nav_backups',
                        'nav_probe_goals',
                        'nav_direct_probes', 'nav_direct_succeeded',
                        'frontier_gain_weight_m', 'distant_goal_every'):
                if key in summary:
                    output_file.write(f'| {key} | {summary[key]} |\n')
            for key in ('world_objects_total', 'world_objects_matched',
                        'all_world_objects_found',
                        'time_to_all_world_objects_s'):
                if key in summary:
                    output_file.write(f'| {key} | {summary[key]} |\n')
            output_file.write('\n## Output files\n\n')
            output_file.write('- `map_with_objects.png`\n')
            output_file.write('- `slam_map.pgm` and `slam_map.yaml`\n')
            output_file.write('- `unique_objects.csv`\n')
            output_file.write('- `robot_path.csv`\n')
            output_file.write('- `observations.csv`\n')
            output_file.write('- `world_discovery_curve.csv` and `.png`\n')
            output_file.write('- `missing_object_audit.csv`\n')

        new_file = not os.path.exists(GLOBAL_RESULTS)
        with open(GLOBAL_RESULTS, 'a', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=self.SUMMARY_FIELDS
            )
            if new_file:
                writer.writeheader()
            writer.writerow({key: summary[key] for key in self.SUMMARY_FIELDS})

    def finish(self, status):
        if self.finished:
            return
        self.finished = True
        # ROS may have invalidated the context before KeyboardInterrupt reaches
        # this method. Still write the partial report in that case.
        if rclpy.ok():
            if self.action_goal_handle is not None:
                self.action_goal_handle.cancel_goal_async()
            self.cmd_publisher.publish(Twist())
        self.observations_file.flush()
        self.observations_file.close()

        summary = self.make_summary(status)
        self.save_unique_objects()
        self.save_robot_path()
        self.save_map_files()
        self.save_map_plot()
        try:
            report = build_report(
                self.run_dir, self.world_file, self.world_match_radius_m,
                min_confirmations=self.min_confirmations,
            )
            report_path = os.path.join(
                self.run_dir, 'world_coverage_report.json'
            )
            with open(report_path, 'w', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
            write_discovery_curve(report, summary['duration_sec'], self.run_dir)
            write_missing_object_audit(report, self.run_dir)
            summary['world_objects_total'] = report['target_objects']
            summary['world_objects_matched'] = report['matched_objects']
            summary['world_scoring_frame'] = report['scoring_frame']
            summary['all_world_objects_found'] = report['all_objects_detected']
            summary['time_to_all_world_objects_s'] = report['time_to_all_s']
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            self.get_logger().warning(
                f'World coverage report could not be written: {error}'
            )
        self.save_summary(summary)

        self.get_logger().info(
            'BASELINE RESULT: '
            f'{self.exploration_policy.title()} policy found '
            f'{summary["unique_objects_found"]} grouped detection locations '
            f'in {summary["duration_sec"] / 60.0:.2f} minutes; '
            f'distance={summary["distance_traveled_m"]:.2f}m, '
            f'coverage={summary["map_coverage_percent"]:.1f}%, '
            f'world_matches={summary.get("world_objects_matched", "unknown")}/'
            f'{summary.get("world_objects_total", "unknown")}'
        )
        self.get_logger().info(f'Results saved to {self.run_dir}')

    def destroy_node(self):
        if rclpy.ok():
            self.cmd_publisher.publish(Twist())
        if not self.observations_file.closed:
            self.observations_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaselineExperiment()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if node.started and not node.finished:
            node.finish('interrupted')
    finally:
        if node.started and not node.finished:
            node.finish('interrupted')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
