"""Measure a DDQN, PPO, or random policy searching for Gazebo objects.

The policy receives only the ROS exploration environment observation. Gazebo
world poses are read by this evaluator solely to score camera detections.
"""

import argparse
import csv
from collections import Counter
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from stable_baselines3 import PPO

from robot_tasks.double_dqn import DoubleDQN
from robot_tasks.dqn_environment import (
    CoverageObjectExplorationEnv, CoverageSearchExplorationEnv,
)
from robot_tasks.global_coverage_assist import GlobalCoverageAssist
from robot_tasks.world_coverage_report import (
    WORLD_FILE, match_instances, read_world, search_bounds_from_world,
    write_discovery_curve,
)
from robot_tasks.world_frame import detection_in_world


DEFAULT_MODEL = '/root/ros2_ws/rl_runs/ddqn_50000/model_final.zip'
DEFAULT_OUTPUT = '/root/ros2_ws/rl_evaluations/ddqn_all_objects'
PPO_MODEL = '/root/ros2_ws/rl_runs/ppo_50000/model_final.zip'


def navigation_bounds(world_path):
    """Use optional room limits for navigation, never target positions."""
    world_bounds = search_bounds_from_world(world_path)
    setting = os.environ.get('ROBOT_TASKS_NAV_BOUNDS', '').strip()
    if not setting:
        return world_bounds
    try:
        bounds = tuple(float(value.strip()) for value in setting.split(','))
    except ValueError as error:
        raise ValueError('ROBOT_TASKS_NAV_BOUNDS must be xmin,xmax,ymin,ymax') from error
    if (len(bounds) != 4 or not all(map(math.isfinite, bounds)) or
            bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
        raise ValueError('ROBOT_TASKS_NAV_BOUNDS must be finite xmin,xmax,ymin,ymax')
    if world_bounds is not None and not (
            world_bounds[0] <= bounds[0] < bounds[1] <= world_bounds[1] and
            world_bounds[2] <= bounds[2] < bounds[3] <= world_bounds[3]):
        raise ValueError('ROBOT_TASKS_NAV_BOUNDS must stay inside the world walls')
    return bounds


class SafetyRecoveryMixin:
    """Keep a learned three-action policy moving when an obstacle traps it."""

    def __init__(self, *args, **kwargs):
        self.safety_interventions = 0
        self.recovery_events = 0
        self.turn_stall_recoveries = 0
        self.translation_stall_recoveries = 0
        super().__init__(*args, **kwargs)
        self._last_progress_xy = self.latest_odom_xy
        self._last_progress_time = time.monotonic()
        self._forward_attempts = 0
        self._last_forward_time = 0.0
        self._recovery_steps = 0
        self._recovery_direction = 1
        self._last_turn_yaw = (self.latest_odom_pose[2]
                               if self.latest_odom_pose is not None else None)
        self._last_turn_progress_time = time.monotonic()
        self._turn_attempts = 0

    def _side_clearance(self, center):
        if self.latest_scan is None:
            return float('inf')
        values = []
        for index, distance in enumerate(self.latest_scan.ranges):
            if not math.isfinite(distance) or distance <= 0:
                continue
            angle = (self.latest_scan.angle_min +
                     index * self.latest_scan.angle_increment)
            angle = math.atan2(math.sin(angle - center),
                               math.cos(angle - center))
            if abs(angle) < math.radians(25):
                values.append(float(distance))
        return min(values, default=float('inf'))

    def _publish_action(self, action):
        now = time.monotonic()
        pose = self.latest_odom_pose
        pose_xy = (self.latest_odom_pose[:2]
                   if self.latest_odom_pose is not None else None)
        pose_yaw = pose[2] if pose is not None else None
        if pose_yaw is not None:
            if (self._last_turn_yaw is None or
                    abs(math.atan2(
                        math.sin(pose_yaw - self._last_turn_yaw),
                        math.cos(pose_yaw - self._last_turn_yaw))) >= 0.08):
                self._last_turn_yaw = pose_yaw
                self._last_turn_progress_time = now
                self._turn_attempts = 0
        is_search_turn = (isinstance(self, CoverageSearchExplorationEnv)
                          and int(action) in (1, 2))
        if is_search_turn:
            self._turn_attempts += 1
        else:
            self._turn_attempts = 0
            self._last_turn_progress_time = now
            self._last_turn_yaw = pose_yaw
        if (pose_xy is not None and
                (self._last_progress_xy is None or
                 math.hypot(pose_xy[0] - self._last_progress_xy[0],
                            pose_xy[1] - self._last_progress_xy[1]) >= 0.15)):
            self._last_progress_xy = pose_xy
            self._last_progress_time = now
            self._forward_attempts = 0
            self._recovery_steps = 0
        if now - self._last_forward_time > 2.0:
            self._forward_attempts = 0
        if int(action) == 0:
            self._forward_attempts += 1
            self._last_forward_time = now
        turn_stalled = (self._recovery_steps == 0 and is_search_turn and
                        self._turn_attempts >= 8 and
                        now - self._last_turn_progress_time > 2.5)
        forward_stalled = (self._recovery_steps == 0 and
                           now - self._last_progress_time > 3.0 and
                           self._forward_attempts >= 8)
        translation_stalled = (
            self._recovery_steps == 0 and pose_xy is not None and
            is_search_turn and
            now - self._last_progress_time > 12.0)
        if turn_stalled or forward_stalled or translation_stalled:
            left = self._side_clearance(math.pi / 2)
            right = self._side_clearance(-math.pi / 2)
            self._recovery_direction = 1 if left >= right else -1
            self._recovery_steps = 30 if translation_stalled else 18
            self.recovery_events += 1
            if turn_stalled:
                self.turn_stall_recoveries += 1
            if translation_stalled:
                self.translation_stall_recoveries += 1
            self._forward_attempts = 0
            self._turn_attempts = 0
            self._last_turn_progress_time = now
        if self._recovery_steps:
            command = Twist()
            if self._recovery_steps > 8:
                if self._side_clearance(math.pi) > 0.25:
                    command.linear.x = -0.10
                command.angular.z = 0.65 * self._recovery_direction
            else:
                command.angular.z = 0.95 * self._recovery_direction
            self.cmd_publisher.publish(command)
            self.last_linear = command.linear.x
            self.last_angular = command.angular.z
            self._recovery_steps -= 1
            self.safety_interventions += 1
            if self._recovery_steps == 0:
                self._last_progress_xy = pose_xy
                self._last_progress_time = now
                self._last_turn_yaw = pose_yaw
                self._last_turn_progress_time = now
                self._turn_attempts = 0
            return
        bounds = getattr(self, 'search_bounds', None)
        pose = self.latest_odom_pose
        if bounds is not None and pose is not None:
            xmin, xmax, ymin, ymax = bounds
            x, y, yaw = pose
            if (x < xmin + 1.2 or x > xmax - 1.2 or
                    y < ymin + 1.2 or y > ymax - 1.2):
                center_heading = math.atan2(
                    (ymin + ymax) / 2 - y, (xmin + xmax) / 2 - x)
                error = math.atan2(math.sin(center_heading - yaw),
                                   math.cos(center_heading - yaw))
                if abs(error) > 0.22:
                    command = Twist()
                    command.angular.z = 0.9 if error > 0 else -0.9
                    self.cmd_publisher.publish(command)
                    self.last_linear = 0.0
                    self.last_angular = command.angular.z
                    self.safety_interventions += 1
                    return
        front = self._front_clearance()
        # In the three-action search profile, actions 1 and 2 are in-place
        # turns. They remain safe with a close obstacle in front and must not
        # be replaced by an unrelated turn chosen from noisy side ranges.
        if (isinstance(self, CoverageSearchExplorationEnv) and
                int(action) in (1, 2)):
            return super()._publish_action(action)
        if front >= 0.45:
            return super()._publish_action(action)
        command = Twist()
        left = self._side_clearance(math.pi / 2)
        right = self._side_clearance(-math.pi / 2)
        command.angular.z = 0.9 if left >= right else -0.9
        if front < 0.25:
            if self._side_clearance(math.pi) > 0.45:
                command.linear.x = -0.05
        self.cmd_publisher.publish(command)
        self.last_linear = command.linear.x
        self.last_angular = command.angular.z
        self.safety_interventions += 1


class SafeCoverageEnv(SafetyRecoveryMixin, CoverageObjectExplorationEnv):
    pass


class SafeCoverageSearchEnv(SafetyRecoveryMixin, CoverageSearchExplorationEnv):
    pass


class DiscoveryRecorder:
    """Observe and score detections without providing targets to the policy."""

    def __init__(self, node, output_dir, targets, min_confidence=0.25,
                 pcd_min_confidence=0.20, hose_min_confidence=0.18,
                 ball_min_confidence=0.12, small_min_confidence=0.15,
                 can_min_confidence=0.15,
                 match_radius_m=1.25,
                 cluster_radius_m=0.75, min_confirmations=2):
        self.node = node
        base_dir = Path(output_dir)
        self.output_dir = base_dir
        suffix = 2
        while self.output_dir.exists():
            self.output_dir = Path(f'{base_dir}_{suffix}')
            suffix += 1
        self.targets = targets
        self.min_confidence = min_confidence
        self.pcd_min_confidence = pcd_min_confidence
        self.hose_min_confidence = hose_min_confidence
        self.ball_min_confidence = ball_min_confidence
        self.small_min_confidence = small_min_confidence
        self.can_min_confidence = can_min_confidence
        self.match_radius_m = match_radius_m
        self.cluster_radius_m = cluster_radius_m
        self.min_confirmations = min_confirmations
        self.odom_pose = None
        self.world_pose = None
        self.odom_time = 0.0
        self.world_time = 0.0
        self.started = None
        self.accepting = True
        self.min_stamp = 0.0
        self.clusters = []
        self.matched_count = 0
        self.time_to_all_s = None
        self.accepted_observations = 0
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output_dir / 'observations.csv').open(
            'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.stream)
        self.writer.writerow((
            'elapsed_s', 'class', 'confidence', 'stamp_s',
            'world_x_m', 'world_y_m', 'world_z_m',
        ))
        node.create_subscription(Odometry, '/odom', self.on_odom, 10)
        node.create_subscription(ModelStates, '/gazebo/model_states',
                                 self.on_model_states, 1)
        node.create_subscription(String, '/yolo/detections',
                                 self.on_detection, 50)

    def on_odom(self, message):
        self.odom_pose = message.pose.pose
        self.odom_time = time.monotonic()

    def on_model_states(self, message):
        try:
            index = message.name.index('burger')
        except ValueError:
            return
        self.world_pose = message.pose[index]
        self.world_time = time.monotonic()

    def on_detection(self, message):
        if self.started is None or not self.accepting:
            return
        try:
            row = json.loads(message.data)
            name = str(row['class'])
            confidence = float(row['confidence'])
            xyz = row.get('odom_xyz')
            stamp_value = row.get('stamp')
            stamp = str(stamp_value) if stamp_value is not None else ''
            stamp_s = float(stamp_value) if stamp_value is not None else None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if stamp_s is not None and stamp_s < self.min_stamp:
            return
        threshold = (self.pcd_min_confidence if name in ('bottle', 'doll')
                     else self.hose_min_confidence if name == 'fire_hose_long'
                     else self.ball_min_confidence if name == 'ball_bearing'
                     else self.small_min_confidence
                     if name in ('cup', 'gear_part')
                     else self.can_min_confidence if name == 'coke_can'
                     else self.min_confidence)
        if (confidence < threshold or not isinstance(xyz, list)
                or len(xyz) != 3 or self.odom_pose is None
                or self.world_pose is None):
            return
        now = time.monotonic()
        if now - self.odom_time > 1.0 or now - self.world_time > 1.0:
            return
        try:
            world_xyz = detection_in_world(
                xyz, self.odom_pose, self.world_pose)
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(float(v)) for v in world_xyz):
            return
        elapsed = now - self.started
        self.writer.writerow((f'{elapsed:.3f}', name, f'{confidence:.5f}',
                              stamp, *(f'{v:.6f}' for v in world_xyz)))
        self.accepted_observations += 1
        self._add_cluster(name, confidence, world_xyz, stamp, elapsed)

    def _add_cluster(self, name, confidence, xyz, stamp, elapsed):
        match = None
        best = self.cluster_radius_m
        for cluster in self.clusters:
            if cluster['class'] != name:
                continue
            distance = math.hypot(cluster['x'] - xyz[0],
                                  cluster['y'] - xyz[1])
            if distance < best:
                match, best = cluster, distance
        if match is None:
            self.clusters.append({
                'id': str(len(self.clusters) + 1), 'class': name,
                'x': xyz[0], 'y': xyz[1], 'confidence': confidence,
                'observations': 1, 'last_stamp': stamp,
                'first_seen_s': elapsed,
                'confirmed_s': elapsed if self.min_confirmations == 1 else None,
            })
            if self.min_confirmations > 1:
                return
        else:
            if stamp and stamp == match['last_stamp']:
                return
            count = match['observations']
            if match['confirmed_s'] is None:
                match['x'] = (match['x'] * count + xyz[0]) / (count + 1)
                match['y'] = (match['y'] * count + xyz[1]) / (count + 1)
            match['observations'] += 1
            match['last_stamp'] = stamp
            match['confidence'] = max(match['confidence'], confidence)
            if match['confirmed_s'] is None and match['observations'] >= self.min_confirmations:
                match['confirmed_s'] = elapsed

        confirmed = [c for c in self.clusters if c['confirmed_s'] is not None]
        matches, _ = match_instances(self.targets, confirmed, self.match_radius_m)
        count = sum(row['detected'] for row in matches)
        if count > self.matched_count:
            self.matched_count = count
            self.node.get_logger().info(
                f'Confirmed world objects: {count}/{len(self.targets)} '
                f'at {elapsed / 60:.2f} minutes')
            if count == len(self.targets):
                self.time_to_all_s = elapsed

    def report(self, duration_s):
        self.stream.flush()
        confirmed = [c for c in self.clusters if c['confirmed_s'] is not None]
        matches, unmatched = match_instances(
            self.targets, confirmed, self.match_radius_m)
        count = sum(row['detected'] for row in matches)
        result = {
            'target_objects': len(self.targets),
            'matched_objects': count,
            'all_objects_detected': count == len(self.targets),
            'time_to_all_s': (max(row['confirmed_s'] for row in matches
                                  if row['detected']) if count == len(self.targets)
                              else None),
            'match_radius_m': self.match_radius_m,
            'min_confirmations': self.min_confirmations,
            'accepted_observations': self.accepted_observations,
            'confirmed_detection_clusters': len(confirmed),
            'unmatched_detections': unmatched,
            'objects': matches,
        }
        with (self.output_dir / 'world_coverage_report.json').open('w',
                encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        with (self.output_dir / 'world_objects.csv').open('w', newline='',
                encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                'name', 'class', 'x', 'y', 'detected', 'detection_id',
                'distance_error_m', 'first_seen_s', 'confirmed_s'))
            writer.writeheader()
            writer.writerows(matches)
        write_discovery_curve(result, duration_s, str(self.output_dir))
        return result

    def matching_rows(self):
        """Current verified matches, for evaluation and guided inspection only."""
        confirmed = [c for c in self.clusters if c['confirmed_s'] is not None]
        matches, _ = match_instances(self.targets, confirmed, self.match_radius_m)
        return matches

    def close(self):
        self.stream.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('ddqn', 'ppo', 'random'),
                        default='ddqn')
    parser.add_argument('--random-policy', choices=('uniform', 'novelty'),
                        default='uniform')
    parser.add_argument('--environment-profile',
                        choices=('coverage_objects', 'coverage_search'),
                        default='coverage_objects')
    parser.add_argument('--model', default='')
    parser.add_argument('--stochastic-policy',
                        type=lambda value: value.lower() in ('true', '1', 'yes'),
                        default=False)
    parser.add_argument('--detector-model', default='')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    parser.add_argument('--world', default=WORLD_FILE)
    parser.add_argument('--max-minutes', type=float, default=30.0)
    parser.add_argument('--action-duration', type=float, default=0.15)
    parser.add_argument('--inspection-sweep-every-m', type=float, default=0.0)
    parser.add_argument('--guided-completion-audit-report', default='')
    parser.add_argument('--guided-completion-after-stall-minutes', type=float,
                        default=10.0)
    parser.add_argument('--global-coverage',
                        type=lambda value: value.lower() in ('true', '1', 'yes'),
                        default=False)
    parser.add_argument('--global-coverage-stall-seconds', type=float,
                        default=12.0)
    parser.add_argument('--breadcrumb-return',
                        type=lambda value: value.lower() in ('true', '1', 'yes'),
                        default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-confidence', type=float, default=0.25)
    parser.add_argument('--pcd-min-confidence', type=float, default=0.20)
    parser.add_argument('--hose-min-confidence', type=float, default=0.18)
    parser.add_argument('--ball-min-confidence', type=float, default=0.12)
    parser.add_argument('--small-min-confidence', type=float, default=0.15)
    parser.add_argument('--can-min-confidence', type=float, default=0.15)
    parser.add_argument('--match-radius-m', type=float, default=1.25)
    parser.add_argument(
        '--safety-recovery',
        type=lambda value: value.lower() in ('true', '1', 'yes'),
        default=False,
    )
    args, _ = parser.parse_known_args()
    if (args.max_minutes < 0 or args.action_duration <= 0 or
            args.inspection_sweep_every_m < 0 or
            args.guided_completion_after_stall_minutes < 0 or
            args.global_coverage_stall_seconds <= 0):
        parser.error('time and inspection sweep distance must be nonnegative; action duration must be positive')
    if args.global_coverage and args.environment_profile != 'coverage_search':
        parser.error('global coverage requires environment-profile coverage_search')
    if not args.model and args.algorithm == 'ddqn':
        args.model = DEFAULT_MODEL
    elif not args.model and args.algorithm == 'ppo':
        args.model = PPO_MODEL
    if args.algorithm != 'random' and not Path(args.model).is_file():
        parser.error(f'Model not found: {args.model}')
    return args


def load_guided_viewpoints(path, world, targets):
    """Load previously camera-verified poses; never call this for policy control."""
    if not path:
        return {}
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if (Path(report.get('world_file', '')).resolve() != Path(world).resolve()
            or report.get('diagnostic') !=
            'known_position_planned_views_not_autonomous'):
        raise ValueError('Guided-view audit does not match this Gazebo world')
    by_name = {row['name']: row for row in report['objects']}
    if set(by_name) != {row['name'] for row in targets}:
        raise ValueError('Guided-view audit must include every world target')
    result = {}
    for target in targets:
        row = by_name[target['name']]
        if row['class'] != target['class']:
            raise ValueError(f'Class mismatch for {target["name"]}')
        views = [view for view in row['views']
                 if view.get('localized_frames', 0) >= 2]
        if not views:
            raise ValueError(f'No verified camera view for {target["name"]}')
        result[target['name']] = [tuple(float(value) for value in view['pose'])
                                  for view in views]
        if any(len(pose) != 3 or not all(map(math.isfinite, pose))
               for pose in result[target['name']]):
            raise ValueError(f'Invalid camera pose for {target["name"]}')
    return result


def guided_inspection(env, recorder, viewpoints, view_seconds=5.0,
                      stop_requested=lambda: False):
    """Teleport to known views and collect *fresh* YOLO detections.

    This is an oracle-assisted diagnostic phase, not policy exploration.
    The caller records the pre-guidance count before entering it.
    """
    rows = []
    if not env.set_state_client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError('/gazebo/set_entity_state unavailable for inspection')
    for target in recorder.targets:
        if stop_requested():
            break
        name = target['name']
        if next(row for row in recorder.matching_rows()
                if row['name'] == name)['detected']:
            continue
        for pose in viewpoints[name]:
            env._stop()
            recorder.accepting = False
            request = SetEntityState.Request()
            request.state.name = env.model_name
            request.state.reference_frame = 'world'
            request.state.pose.position.x = pose[0]
            request.state.pose.position.y = pose[1]
            request.state.pose.position.z = 0.02
            request.state.pose.orientation.z = math.sin(pose[2] / 2)
            request.state.pose.orientation.w = math.cos(pose[2] / 2)
            future = env.set_state_client.call_async(request)
            deadline = time.monotonic() + 5.0
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(env.node, timeout_sec=0.05)
            moved = (future.done() and future.result() is not None and
                     future.result().success)
            if moved:
                env._spin_for(0.8)
                recorder.min_stamp = (
                    env.node.get_clock().now().nanoseconds * 1e-9)
                recorder.accepting = True
                deadline = time.monotonic() + view_seconds
                while time.monotonic() < deadline and not stop_requested():
                    rclpy.spin_once(env.node, timeout_sec=0.05)
                    if next(row for row in recorder.matching_rows()
                            if row['name'] == name)['detected']:
                        break
            recorder.accepting = True
            detected = next(row for row in recorder.matching_rows()
                            if row['name'] == name)['detected']
            rows.append({
                'name': name, 'class': target['class'],
                'world_x_m': pose[0], 'world_y_m': pose[1],
                'yaw_rad': pose[2], 'service_success': bool(moved),
                'detected': bool(detected),
                'elapsed_s': round(time.monotonic() - recorder.started, 3),
            })
            env.node.get_logger().info(
                f'Guided camera inspection {name}: '
                f'{"confirmed" if detected else "not confirmed"}; '
                f'{sum(row["detected"] for row in recorder.matching_rows())}/'
                f'{len(recorder.targets)} total')
            if detected:
                break
    env._stop()
    return rows


def choose_random_action(env, observation, rng, policy):
    """Optionally bias random exploration toward unvisited nearby cells."""
    if policy != 'novelty':
        return env.action_space.sample()
    if not isinstance(env, CoverageSearchExplorationEnv):
        raise ValueError('novelty random policy requires coverage_search')
    bounds = getattr(env, 'search_bounds', None)
    pose = env.latest_odom_pose
    if bounds is not None and pose is not None:
        x, y, yaw = pose
        xmin, xmax, ymin, ymax = bounds
        if (x < xmin + 0.8 or x > xmax - 0.8 or
                y < ymin + 0.8 or y > ymax - 0.8):
            heading = math.atan2((ymin + ymax) / 2 - y,
                                 (xmin + xmax) / 2 - x)
            error = math.atan2(math.sin(heading - yaw),
                               math.cos(heading - yaw))
            if abs(error) > 0.25:
                return 1 if error > 0 else 2
    novelty = observation[-8:]
    clearance = env._front_clearance()
    left = 0.3 + float(sum(novelty[1:4]))
    right = 0.3 + float(sum(novelty[5:8]))
    if clearance < 0.55:
        if isinstance(env, SafetyRecoveryMixin):
            left *= max(0.05, min(2.0, env._side_clearance(math.pi / 2)))
            right *= max(0.05, min(2.0, env._side_clearance(-math.pi / 2)))
        return rng.choices((1, 2), weights=(left, right))[0]
    # Persist forward motion while the path is clear. Rare random turns are
    # weighted toward unseen cells, avoiding a turn-only loop near props.
    if rng.random() < (0.94 if novelty[0] >= 0.5 else 0.78):
        return 0
    return rng.choices((1, 2), weights=(left, right))[0]


class InspectionSweep:
    """Take an initial and periodic full camera view without target poses."""

    def __init__(self, every_m, action_duration):
        self.every_m = float(every_m)
        self.turn_steps = math.ceil(2 * math.pi / (0.9 * action_duration)) + 2
        self.last_xy = None
        self.remaining = 0
        self.events = 0
        self.actions = 0
        self.initial_pending = self.every_m > 0

    def action(self, pose, proposed):
        if self.every_m <= 0 or pose is None:
            return proposed, False
        xy = pose[:2]
        if self.last_xy is None:
            self.last_xy = xy
        if self.initial_pending:
            self.initial_pending = False
            self.remaining = self.turn_steps
            self.events += 1
        elif self.remaining == 0 and math.hypot(
                xy[0] - self.last_xy[0], xy[1] - self.last_xy[1]
        ) >= self.every_m:
            self.last_xy = xy
            self.remaining = self.turn_steps
            self.events += 1
        if self.remaining:
            self.remaining -= 1
            self.actions += 1
            return 1, True
        return proposed, False


class DetectionConfirmation:
    """Hold a camera view after a new YOLO candidate appears once.

    This uses only the detector's own recent clusters. It does not read world
    target positions or whether a candidate matches an object on the list.
    """

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.attempted = set()
        self.remaining = []
        self.events = 0
        self.actions = 0

    def action(self, clusters, elapsed, proposed):
        if not self.enabled:
            return proposed, False
        if not self.remaining and self.events < 80:
            fresh = [
                cluster for cluster in clusters
                if cluster['id'] not in self.attempted
                and cluster['confirmed_s'] is None
                and 0.0 <= elapsed - cluster['first_seen_s'] <= 0.75
            ]
            if fresh:
                self.attempted.update(cluster['id'] for cluster in fresh)
                # Six small left/right pairs return to the same heading and
                # give the camera several chances to observe the candidate.
                self.remaining = [2, 1] * 6
                self.events += 1
        if self.remaining:
            self.actions += 1
            return self.remaining.pop(0), True
        return proposed, False


def main():
    args = parse_args()
    targets = read_world(args.world)
    viewpoints = load_guided_viewpoints(
        args.guided_completion_audit_report, args.world, targets)
    environment_classes = {
        ('coverage_objects', False): CoverageObjectExplorationEnv,
        ('coverage_objects', True): SafeCoverageEnv,
        ('coverage_search', False): CoverageSearchExplorationEnv,
        ('coverage_search', True): SafeCoverageSearchEnv,
    }
    environment_class = environment_classes[(args.environment_profile,
                                             args.safety_recovery)]
    env = environment_class(
        action_duration=args.action_duration,
        max_episode_steps=(1000000000 if args.max_minutes == 0 else
                           max(1000000, int(args.max_minutes * 60 /
                                            args.action_duration) + 1)),
        min_confidence=args.min_confidence,
        hose_min_confidence=args.hose_min_confidence,
        ball_min_confidence=args.ball_min_confidence,
        small_min_confidence=args.small_min_confidence,
        can_min_confidence=args.can_min_confidence,
        seed=args.seed,
        class_quota=(Counter(obj['class'] for obj in targets)
                     if args.environment_profile == 'coverage_search'
                     else None),
    )
    env.search_bounds = navigation_bounds(args.world)
    recorder = None
    status = 'completed_time_limit'
    total_reward = 0.0
    steps = 0
    reward_points = []
    reward_sample_stride = 10
    steps_stream = None
    steps_writer = None
    stop = False
    random_generator = random.Random(args.seed)
    inspection_sweep = InspectionSweep(
        args.inspection_sweep_every_m, args.action_duration)
    confirmation = DetectionConfirmation(
        os.environ.get('ROBOT_TASKS_CONFIRM_NEW_DETECTIONS', 'false').lower()
        in ('1', 'true', 'yes'))
    planner_bounds_enabled = os.environ.get(
        'ROBOT_TASKS_PLANNER_BOUNDS_ENABLED', 'true').lower() not in (
            'false', '0', 'no')
    sector_balance_weight = float(os.environ.get(
        'ROBOT_TASKS_SECTOR_BALANCE_WEIGHT', '0'))
    coverage_assist = (GlobalCoverageAssist(
        env.node, tile_size=env.coverage_cell_size,
        stall_seconds=args.global_coverage_stall_seconds,
        search_bounds=env.search_bounds,
        enable_breadcrumb_return=args.breadcrumb_return,
        planner_bounds_enabled=planner_bounds_enabled,
        sector_balance_weight=sector_balance_weight)
        if args.global_coverage else None)
    visit_stream = None
    visit_writer = None
    visit_records = {}
    guided_rows = []
    pre_guidance_matched_objects = None
    last_match_count = 0
    last_match_time = None
    last_guided_time = None

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        model = None
        if args.algorithm == 'ddqn':
            model = DoubleDQN.load(args.model, env=env, device='cpu')
        elif args.algorithm == 'ppo':
            model = PPO.load(args.model, env=env, device='cpu')
        env.action_space.seed(args.seed)
        recorder = DiscoveryRecorder(
            env.node, args.output_dir, targets,
            min_confidence=args.min_confidence,
            pcd_min_confidence=args.pcd_min_confidence,
            hose_min_confidence=args.hose_min_confidence,
            ball_min_confidence=args.ball_min_confidence,
            small_min_confidence=args.small_min_confidence,
            can_min_confidence=args.can_min_confidence,
            match_radius_m=args.match_radius_m,
        )
        observation, _ = env.reset(seed=args.seed)
        steps_stream = (recorder.output_dir / 'steps.csv').open(
            'w', newline='', encoding='utf-8')
        steps_writer = csv.writer(steps_stream)
        steps_writer.writerow((
            'elapsed_s', 'step', 'action', 'reward', 'cumulative_reward',
            'matched_world_objects', 'environment_clusters', 'collision',
            'visited_cells', 'applied_linear_m_s',
            'applied_angular_rad_s', 'policy_action', 'action_source',
        ))
        visit_stream = (recorder.output_dir / 'visit_history.csv').open(
            'w', newline='', encoding='utf-8')
        visit_writer = csv.writer(visit_stream)
        visit_writer.writerow((
            'elapsed_s', 'step', 'odom_x_m', 'odom_y_m', 'odom_yaw_rad',
            'cell_x', 'cell_y', 'first_visit', 'policy_action',
            'applied_action', 'action_source',
        ))
        recorder.min_stamp = env.node.get_clock().now().nanoseconds * 1e-9
        recorder.started = time.monotonic()
        last_match_time = recorder.started
        deadline = (float('inf') if args.max_minutes == 0 else
                    recorder.started + args.max_minutes * 60)
        limit_text = ('no time limit' if args.max_minutes == 0 else
                      f'{args.max_minutes:g} minute limit')
        env.node.get_logger().info(
            f'{args.algorithm.upper()} discovery started: {len(targets)} targets, '
            f'{limit_text}, model={args.model}')
        next_log = recorder.started + 30
        while time.monotonic() < deadline and not stop:
            if model is None:
                action = choose_random_action(
                    env, observation, random_generator, args.random_policy)
            else:
                action, _ = model.predict(
                    observation, deterministic=not args.stochastic_policy)
            policy_action = int(action)
            action_source = 'policy'
            if coverage_assist is not None:
                action, action_source = coverage_assist.action(
                    env.latest_odom_pose, env.visited_cells, policy_action,
                    env._front_clearance(), scan=env.latest_scan,
                    detector_count=sum(obj['confirmed'] for obj in env.objects),
                    detector_classes={obj['class'] for obj in env.objects
                                      if obj['confirmed']})
            # A camera sweep may take several seconds. Finish a LiDAR escape
            # first so the recovery can actually leave the blocked area.
            confirming = False
            if (coverage_assist is not None and
                    coverage_assist.last_status in (
                        'escaping_blocked_path', 'returning_inside_bounds',
                        'returning_by_breadcrumbs')):
                sweeping = False
            else:
                action, confirming = confirmation.action(
                    recorder.clusters, time.monotonic() - recorder.started,
                    int(action))
                if not confirming:
                    action, sweeping = inspection_sweep.action(
                        env.latest_odom_pose, int(action))
                else:
                    sweeping = False
            if confirming:
                action_source = 'detection_confirmation'
            elif sweeping:
                action_source = 'inspection_sweep'
            observation, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            total_reward += reward
            elapsed = time.monotonic() - recorder.started
            pose = env.latest_odom_pose
            cell = env._coverage_cell()
            first_visit = cell is not None and cell not in visit_records
            if cell is not None:
                record = visit_records.setdefault(cell, {
                    'first_step': steps, 'first_elapsed_s': round(elapsed, 3),
                    'last_step': steps, 'visit_samples': 0,
                })
                record['last_step'] = steps
                record['visit_samples'] += 1
            visit_writer.writerow((
                f'{elapsed:.3f}', steps,
                f'{pose[0]:.4f}' if pose else '',
                f'{pose[1]:.4f}' if pose else '',
                f'{pose[2]:.4f}' if pose else '',
                cell[0] if cell else '', cell[1] if cell else '',
                int(first_visit), policy_action, int(action), action_source,
            ))
            if steps % 100 == 0:
                visit_stream.flush()
            steps_writer.writerow((
                f'{elapsed:.3f}', steps, int(action),
                f'{reward:.6f}', f'{total_reward:.6f}',
                recorder.matched_count, info.get('objects_found', 0),
                int(info.get('collision', False)),
                info.get('visited_cells', 0),
                f'{env.last_linear:.3f}', f'{env.last_angular:.3f}',
                policy_action, action_source,
            ))
            if steps % 100 == 0:
                steps_stream.flush()
            if steps % reward_sample_stride == 0:
                reward_points.append((elapsed / 60, total_reward))
                if len(reward_points) > 10000:
                    reward_points = reward_points[::2]
                    reward_sample_stride *= 2
            if recorder.time_to_all_s is not None:
                status = 'all_objects_found'
                break
            if recorder.matched_count > last_match_count:
                last_match_count = recorder.matched_count
                last_match_time = time.monotonic()
            if (viewpoints and (
                    (last_guided_time is None and
                     time.monotonic() - last_match_time >=
                     args.guided_completion_after_stall_minutes * 60)
                    or (last_guided_time is not None and
                        time.monotonic() - last_guided_time >= 3.0))):
                if pre_guidance_matched_objects is None:
                    pre_guidance_matched_objects = sum(
                        row['detected'] for row in recorder.matching_rows())
                env.node.get_logger().warning(
                    f'{args.algorithm.upper()} exploration stalled; starting known-viewpoint '
                    'camera inspection. Subsequent detections are guided, '
                    f'not autonomous {args.algorithm.upper()} discoveries.')
                guided_rows.extend(guided_inspection(
                    env, recorder, viewpoints,
                    stop_requested=lambda: stop))
                last_guided_time = time.monotonic()
                last_match_count = recorder.matched_count
                last_match_time = last_guided_time
                if recorder.time_to_all_s is not None:
                    status = 'all_objects_found_with_guided_completion'
                    break
                if stop:
                    status = 'interrupted'
                    break
            if terminated or truncated:
                status = 'environment_ended'
                break
            if time.monotonic() >= next_log:
                env.node.get_logger().info(
                    f'{args.algorithm.upper()} progress: '
                    f'{recorder.matched_count}/{len(targets)} '
                    f'world objects, reward={total_reward:.2f}, '
                    f'elapsed={elapsed / 60:.2f} min, steps={steps}, '
                    f'visited_cells={len(env.visited_cells)}, '
                    f'coverage={coverage_assist.last_status if coverage_assist else "off"}')
                next_log += 30
        if stop:
            status = 'interrupted'
    finally:
        if recorder is not None:
            duration = (time.monotonic() - recorder.started
                        if recorder.started is not None else 0.0)
            recorder.accepting = False
            env._stop()
            env.close()
            if steps_stream is not None:
                steps_stream.close()
            with (recorder.output_dir / 'visited_cells.csv').open(
                    'w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(('cell_x', 'cell_y', 'center_odom_x_m',
                                 'center_odom_y_m', 'first_step',
                                 'first_elapsed_s', 'last_step',
                                 'visit_samples'))
                for (cell_x, cell_y), record in sorted(visit_records.items()):
                    writer.writerow((
                        cell_x, cell_y,
                        round((cell_x + 0.5) * env.coverage_cell_size, 3),
                        round((cell_y + 0.5) * env.coverage_cell_size, 3),
                        record['first_step'], record['first_elapsed_s'],
                        record['last_step'], record['visit_samples'],
                    ))
            if visit_records:
                figure, axis = plt.subplots(figsize=(7, 7),
                                            constrained_layout=True)
                axis.scatter(
                    [(cell[0] + 0.5) * env.coverage_cell_size
                     for cell in visit_records],
                    [(cell[1] + 0.5) * env.coverage_cell_size
                     for cell in visit_records],
                    s=12, marker='s', color='#3274a1', alpha=0.65,
                )
                axis.set(xlabel='Odom x (m)', ylabel='Odom y (m)',
                         title=f'Visited cells: {len(visit_records)}',
                         aspect='equal')
                axis.grid(alpha=0.2)
                figure.savefig(recorder.output_dir / 'visited_cells.png',
                               dpi=180)
                plt.close(figure)
            map_saved = coverage_assist is not None and coverage_assist.map is not None
            if map_saved:
                grid = coverage_assist.map
                np.savez_compressed(
                    recorder.output_dir / 'slam_occupancy_map.npz',
                    data=np.asarray(grid.data, dtype=np.int8).reshape(
                        grid.info.height, grid.info.width),
                    resolution_m=grid.info.resolution,
                    origin_x_m=grid.info.origin.position.x,
                    origin_y_m=grid.info.origin.position.y,
                    frame_id=grid.header.frame_id,
                )
            if coverage_assist is not None:
                history = []
                for event in coverage_assist.goal_history:
                    event = dict(event)
                    event['elapsed_s'] = round(
                        event.pop('time') - recorder.started, 3)
                    history.append(event)
                with (recorder.output_dir / 'coverage_goal_history.json').open(
                        'w', encoding='utf-8') as stream:
                    json.dump(history, stream, indent=2)
                    stream.write('\n')
            report = recorder.report(duration)
            if viewpoints:
                with (recorder.output_dir / 'guided_inspections.csv').open(
                        'w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=(
                        'name', 'class', 'world_x_m', 'world_y_m', 'yaw_rad',
                        'service_success', 'detected', 'elapsed_s'))
                    writer.writeheader()
                    writer.writerows(guided_rows)
            summary = {
                'algorithm': args.algorithm,
                'environment_profile': args.environment_profile,
                'policy': (f'{args.random_policy}_random_actions'
                           if model is None else
                           'saved_model_stochastic' if args.stochastic_policy
                           else 'saved_model_deterministic'),
                'control_mode': (f'{args.algorithm}_plus_known_viewpoint_completion'
                                 if guided_rows else
                                 'policy_plus_global_coverage_assist'
                                 if args.global_coverage else 'policy_only'),
                'pre_guidance_matched_objects': (
                    pre_guidance_matched_objects if pre_guidance_matched_objects
                    is not None else report['matched_objects']),
                'ddqn_phase_matched_objects': (
                    (pre_guidance_matched_objects if pre_guidance_matched_objects
                     is not None else report['matched_objects'])
                    if args.algorithm == 'ddqn' else None),
                'autonomous_all_objects_detected': (
                    (pre_guidance_matched_objects if pre_guidance_matched_objects
                     is not None else report['matched_objects']) == len(targets)),
                'guided_completion_enabled': bool(viewpoints),
                'guided_completion_audit_report': (
                    os.path.abspath(args.guided_completion_audit_report)
                    if viewpoints else None),
                'guided_completion_after_stall_minutes': (
                    args.guided_completion_after_stall_minutes
                    if viewpoints else None),
                'guided_completion_inspections': len(guided_rows),
                'guided_completion_added_objects': (
                    report['matched_objects'] - pre_guidance_matched_objects
                    if pre_guidance_matched_objects is not None else 0),
                'model': os.path.abspath(args.model) if model is not None else None,
                'detector_model': (os.path.abspath(args.detector_model)
                                   if args.detector_model else None),
                'world_file': os.path.abspath(args.world),
                'navigation_bounds_m': env.search_bounds,
                'seed': args.seed, 'status': status,
                'elapsed_s': round(duration, 3),
                'elapsed_minutes': round(duration / 60, 3),
                'time_limit_minutes': (None if args.max_minutes == 0 else
                                       args.max_minutes),
                'steps': steps, 'total_reward': round(total_reward, 6),
                'mean_reward_per_step': round(total_reward / steps, 6)
                if steps else None,
                'action_duration_s': args.action_duration,
                'inspection_sweep_every_m': args.inspection_sweep_every_m,
                'inspection_sweep_events': inspection_sweep.events,
                'inspection_sweep_actions': inspection_sweep.actions,
                'detection_confirmation_enabled': confirmation.enabled,
                'detection_confirmation_events': confirmation.events,
                'detection_confirmation_actions': confirmation.actions,
                'global_coverage_enabled': args.global_coverage,
                'planner_bounds_enabled': (
                    planner_bounds_enabled if args.global_coverage else None),
                'sector_balance_weight': (
                    sector_balance_weight if args.global_coverage else None),
                'breadcrumb_return_enabled': (
                    args.breadcrumb_return if args.global_coverage else False),
                'global_coverage_stall_seconds': (
                    args.global_coverage_stall_seconds
                    if args.global_coverage else None),
                'global_coverage_status': (
                    coverage_assist.last_status if coverage_assist else None),
                'global_coverage_intervention_steps': (
                    coverage_assist.intervention_steps if coverage_assist else 0),
                'global_coverage_planned_goals': (
                    coverage_assist.planned_goals if coverage_assist else 0),
                'global_coverage_failed_plans': (
                    coverage_assist.failed_plans if coverage_assist else 0),
                'breadcrumb_return_events': (
                    coverage_assist.breadcrumb_return_events
                    if coverage_assist else 0),
                'unique_visited_cells': len(visit_records),
                'visit_cell_size_m': env.coverage_cell_size,
                'slam_map_saved': bool(map_saved),
                'min_confidence': args.min_confidence,
                'pcd_min_confidence': args.pcd_min_confidence,
                'hose_min_confidence': args.hose_min_confidence,
                'ball_min_confidence': args.ball_min_confidence,
                'small_min_confidence': args.small_min_confidence,
                'can_min_confidence': args.can_min_confidence,
                'match_radius_m': args.match_radius_m,
                'safety_recovery': args.safety_recovery,
                'safety_interventions': getattr(env, 'safety_interventions', 0),
                'recovery_events': getattr(env, 'recovery_events', 0),
                'turn_stall_recoveries': getattr(
                    env, 'turn_stall_recoveries', 0),
                'translation_stall_recoveries': getattr(
                    env, 'translation_stall_recoveries', 0),
                'reward_formula': {
                    'new_area': env.new_area_reward,
                    'confidence_scale': env.confidence_reward_scale,
                    'new_object': env.new_object_reward,
                    'collision': -env.collision_penalty,
                    'revisit': -env.revisit_penalty,
                    **({'time': -0.02, 'forward_motion': 0.05,
                        'turn_motion': -0.03}
                       if args.environment_profile == 'coverage_search' else {}),
                },
                'target_objects': report['target_objects'],
                'matched_objects': report['matched_objects'],
                'all_objects_detected': report['all_objects_detected'],
                'time_to_all_s': report['time_to_all_s'],
                'time_to_all_minutes': (
                    round(report['time_to_all_s'] / 60, 3)
                    if report['time_to_all_s'] is not None else None),
                'missing_objects': [row['name'] for row in report['objects']
                                    if not row['detected']],
            }
            with (recorder.output_dir / 'summary.json').open('w',
                    encoding='utf-8') as stream:
                json.dump(summary, stream, indent=2)
                stream.write('\n')
            if steps:
                duration_minutes = duration / 60
                if not reward_points or reward_points[-1][0] < duration_minutes:
                    reward_points.append((duration_minutes, total_reward))
                figure, axis = plt.subplots(figsize=(8, 4.5),
                                            constrained_layout=True)
                axis.plot([point[0] for point in reward_points],
                          [point[1] for point in reward_points])
                axis.set(xlabel='Elapsed minutes', ylabel='Cumulative reward',
                         title=f'{args.algorithm.upper()} reward during autonomous discovery')
                axis.grid(alpha=0.25)
                figure.savefig(recorder.output_dir / 'reward_curve.png',
                               dpi=180)
                plt.close(figure)
            recorder.close()
            print(json.dumps(summary, indent=2), flush=True)
        if visit_stream is not None:
            visit_stream.close()
        if recorder is None:
            env.close()


if __name__ == '__main__':
    main()
