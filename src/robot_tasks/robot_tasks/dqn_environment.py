import json
import math
import os
import random
import time

import gymnasium as gym
import numpy as np
import rclpy

from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class RosExplorationEnv(gym.Env):
    """Gymnasium environment for object-seeking TurtleBot exploration."""

    metadata = {'render_modes': []}

    ACTIONS = (
        (0.18, 0.00),
        (0.12, 0.65),
        (0.12, -0.65),
        (0.00, 0.80),
        (0.00, -0.80),
        (-0.10, 0.00),
    )

    def __init__(
        self,
        action_duration=0.15,
        max_episode_steps=250,
        min_confidence=0.25,
        association_radius_m=1.5,
        min_confirmations=2,
        model_name='burger',
        seed=42,
        startup_timeout=90.0,
        class_quota=None,
        hose_min_confidence=0.18,
        ball_min_confidence=0.12,
        small_min_confidence=0.15,
        can_min_confidence=0.25,
    ):
        super().__init__()
        self.action_duration = max(0.02, float(action_duration))
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.min_confidence = float(min_confidence)
        self.hose_min_confidence = float(hose_min_confidence)
        self.ball_min_confidence = float(ball_min_confidence)
        self.small_min_confidence = float(small_min_confidence)
        self.can_min_confidence = float(can_min_confidence)
        self.association_radius_m = max(0.05, float(association_radius_m))
        self.min_confirmations = max(1, int(min_confirmations))
        self.model_name = str(model_name)
        self.startup_timeout = max(1.0, float(startup_timeout))
        self.class_quota = dict(class_quota) if class_quota is not None else None

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        self._rng = random.Random(seed)
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=[])

        node_name = f'dqn_environment_{os.getpid()}'
        self.node = Node(node_name)
        self.node.set_parameters([
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ])
        self.cmd_publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.node.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            String,
            '/yolo/detections',
            self._detection_callback,
            20,
        )
        self.set_state_client = self.node.create_client(
            SetEntityState,
            '/gazebo/set_entity_state',
        )

        self.latest_scan = None
        self.latest_odom_xy = None
        self.latest_odom_pose = None
        self.previous_step_xy = None
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.latest_confidence = 0.0
        self.latest_confidence_time = 0.0
        self.objects = []
        self.pending_object_rewards = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        self.collision = False
        self.closed = False

        self._wait_until_ready()

    def _scan_callback(self, message):
        self.latest_scan = message

    def _odom_callback(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        x = float(position.x)
        y = float(position.y)
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
        )
        self.latest_odom_xy = (x, y)
        self.latest_odom_pose = (x, y, yaw)

    def _detection_callback(self, message):
        try:
            detection = json.loads(message.data)
            class_name = str(detection['class'])
            confidence = float(detection.get('confidence', 0.0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

        self.latest_confidence = max(self.latest_confidence, confidence)
        self.latest_confidence_time = time.monotonic()
        threshold = (self.hose_min_confidence if class_name == 'fire_hose_long'
                     else self.ball_min_confidence if class_name == 'ball_bearing'
                     else self.small_min_confidence
                     if class_name in ('cup', 'gear_part')
                     else self.can_min_confidence if class_name == 'coke_can'
                     else self.min_confidence)
        if confidence < threshold:
            return
        if self.class_quota is not None:
            quota = self.class_quota.get(class_name, 0)
            if not quota or sum(
                    obj['confirmed'] and obj['class'] == class_name
                    for obj in self.objects) >= quota:
                return

        xyz = detection.get('odom_xyz') or detection.get('map_xyz')
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            return
        try:
            x, y, z = map(float, xyz)
        except (TypeError, ValueError):
            return

        match = None
        match_distance = self.association_radius_m
        for obj in self.objects:
            if obj['class'] != class_name:
                continue
            distance = math.hypot(obj['x'] - x, obj['y'] - y)
            if distance < match_distance:
                match = obj
                match_distance = distance

        if match is None:
            match = {
                'class': class_name,
                'x': x,
                'y': y,
                'z': z,
                'observations': 1,
                'best_confidence': confidence,
                'confirmed': self.min_confirmations == 1,
            }
            self.objects.append(match)
            if match['confirmed']:
                self.pending_object_rewards += 1
            return

        count = match['observations']
        match['x'] = (match['x'] * count + x) / (count + 1)
        match['y'] = (match['y'] * count + y) / (count + 1)
        match['z'] = (match['z'] * count + z) / (count + 1)
        match['observations'] += 1
        match['best_confidence'] = max(match['best_confidence'], confidence)
        if not match['confirmed'] and match['observations'] >= self.min_confirmations:
            match['confirmed'] = True
            self.pending_object_rewards += 1

    def _spin_for(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            timeout = min(0.02, max(0.0, deadline - time.monotonic()))
            rclpy.spin_once(self.node, timeout_sec=timeout)

    def _wait_until_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            yolo_ready = self.node.count_publishers('/yolo/detections') > 0
            service_ready = self.set_state_client.service_is_ready()
            if self.latest_scan is not None and self.latest_odom_xy is not None:
                if yolo_ready and service_ready:
                    return
            if time.monotonic() - last_log > 5.0:
                self.node.get_logger().info(
                    'Waiting for RL inputs: '
                    f'scan={self.latest_scan is not None}, '
                    f'odom={self.latest_odom_xy is not None}, '
                    f'yolo={yolo_ready}, '
                    f'set_state_service={service_ready}'
                )
                last_log = time.monotonic()
        raise RuntimeError(
            'RL environment startup timed out. Start realistic_sim.launch.py '
            'and the YOLO node before training.'
        )

    def _confirmed_objects(self):
        return [obj for obj in self.objects if obj['confirmed']]

    def _lidar_values(self):
        if self.latest_scan is None:
            return np.ones(12, dtype=np.float32)
        maximum = min(float(self.latest_scan.range_max), 3.5)
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=maximum, posinf=maximum, neginf=0.0)
        ranges = np.clip(ranges, 0.0, maximum)
        sectors = [
            float(np.min(section)) if section.size else maximum
            for section in np.array_split(ranges, 12)
        ]
        return np.asarray(sectors, dtype=np.float32) / maximum

    def _front_clearance(self):
        if self.latest_scan is None:
            return float('inf')
        best = float('inf')
        for index, distance in enumerate(self.latest_scan.ranges):
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            angle = self.latest_scan.angle_min + index * self.latest_scan.angle_increment
            wrapped_angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(wrapped_angle) <= math.radians(25.0):
                best = min(best, float(distance))
        return best

    def _observation(self):
        confidence = self.latest_confidence
        if time.monotonic() - self.latest_confidence_time > 0.75:
            confidence = 0.0
        extra = np.asarray([
            np.clip(confidence, 0.0, 1.0),
            np.clip(len(self._confirmed_objects()) / 20.0, 0.0, 1.0),
            np.clip((self.last_linear + 0.10) / 0.28, 0.0, 1.0),
            np.clip((self.last_angular + 0.80) / 1.60, 0.0, 1.0),
        ], dtype=np.float32)
        return np.concatenate((self._lidar_values(), extra)).astype(np.float32)

    def _publish_action(self, action):
        linear, angular = self.ACTIONS[int(action)]
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self.cmd_publisher.publish(command)
        self.last_linear = linear
        self.last_angular = angular

    def _stop(self):
        self.cmd_publisher.publish(Twist())
        self.last_linear = 0.0
        self.last_angular = 0.0

    def _set_start_pose(self):
        if not self.set_state_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/gazebo/set_entity_state is unavailable')

        request = SetEntityState.Request()
        request.state.name = self.model_name
        request.state.reference_frame = 'world'
        request.state.pose.position.x = self._rng.uniform(-0.10, 0.10)
        request.state.pose.position.y = self._rng.uniform(-2.10, -1.90)
        request.state.pose.position.z = 0.02
        yaw = self._rng.uniform(-math.pi, math.pi)
        request.state.pose.orientation.z = math.sin(yaw / 2.0)
        request.state.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.set_state_client.call_async(request)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError(f'Failed to reset Gazebo entity {self.model_name!r}')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._stop()
        self._set_start_pose()
        self._spin_for(0.35)

        self.objects = []
        self.pending_object_rewards = 0
        self.latest_confidence = 0.0
        self.latest_confidence_time = 0.0
        self.episode_step = 0
        self.episode_reward = 0.0
        self.collision = False
        self.previous_step_xy = self.latest_odom_xy
        return self._observation(), {
            'objects_found': 0,
            'object_classes': [],
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid discrete action: {action}')

        self.pending_object_rewards = 0
        self.latest_confidence = 0.0
        self._publish_action(action)
        self._spin_for(self.action_duration)
        self.episode_step += 1

        reward = -0.01
        if self.last_linear > 0.0:
            reward += 0.03
        elif abs(self.last_angular) > 0.0:
            reward -= 0.005

        if self.latest_odom_xy is not None and self.previous_step_xy is not None:
            distance = math.hypot(
                self.latest_odom_xy[0] - self.previous_step_xy[0],
                self.latest_odom_xy[1] - self.previous_step_xy[1],
            )
            reward += min(distance, 0.10) * 0.5
        self.previous_step_xy = self.latest_odom_xy

        reward += 5.0 * self.pending_object_rewards
        if self.latest_confidence >= self.min_confidence:
            reward += 0.02 * self.latest_confidence

        self.collision = (
            self.last_linear > 0.0 and self._front_clearance() < 0.14
        )
        terminated = self.collision
        if self.collision:
            reward -= 2.0
            self._stop()

        truncated = self.episode_step >= self.max_episode_steps
        if truncated:
            self._stop()

        self.episode_reward += reward
        confirmed = self._confirmed_objects()
        info = {
            'objects_found': len(confirmed),
            'object_classes': sorted({obj['class'] for obj in confirmed}),
            'collision': self.collision,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        if self.closed:
            return
        self.closed = True
        if rclpy.ok(context=self.node.context):
            self._stop()
            self._spin_for(0.05)
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()


class CoverageExplorationEnv(RosExplorationEnv):
    """Exploration task with 360-beam state and explicit coverage reward."""

    ACTIONS = (
        (0.18, 0.00),
        (0.10, 0.80),
        (0.10, -0.80),
    )
    ACTION_NAMES = ('forward', 'left', 'right')
    LIDAR_SAMPLES = 360

    def __init__(
        self,
        *args,
        coverage_cell_size=0.50,
        new_area_reward=1.0,
        confidence_reward_scale=0.5,
        collision_penalty=2.0,
        revisit_penalty=0.05,
        new_object_reward=0.0,
        **kwargs,
    ):
        self.coverage_cell_size = max(0.05, float(coverage_cell_size))
        self.new_area_reward = float(new_area_reward)
        self.confidence_reward_scale = float(confidence_reward_scale)
        self.collision_penalty = abs(float(collision_penalty))
        self.revisit_penalty = abs(float(revisit_penalty))
        self.new_object_reward = float(new_object_reward)
        self.visited_cells = set()
        super().__init__(*args, **kwargs)

        low = np.concatenate((
            np.zeros(self.LIDAR_SAMPLES, dtype=np.float32),
            np.asarray([-np.inf, -np.inf, -math.pi, 0.0], dtype=np.float32),
        ))
        high = np.concatenate((
            np.ones(self.LIDAR_SAMPLES, dtype=np.float32),
            np.asarray([np.inf, np.inf, math.pi, 1.0], dtype=np.float32),
        ))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _lidar_360_values(self):
        if self.latest_scan is None or not self.latest_scan.ranges:
            return np.ones(self.LIDAR_SAMPLES, dtype=np.float32)

        maximum = min(float(self.latest_scan.range_max), 3.5)
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=maximum, posinf=maximum, neginf=0.0)
        ranges = np.clip(ranges, 0.0, maximum) / maximum
        if ranges.size == self.LIDAR_SAMPLES:
            return ranges.astype(np.float32)

        source = np.linspace(0.0, 1.0, ranges.size, endpoint=False)
        target = np.linspace(0.0, 1.0, self.LIDAR_SAMPLES, endpoint=False)
        periodic_source = np.append(source, 1.0)
        periodic_ranges = np.append(ranges, ranges[0])
        return np.interp(target, periodic_source, periodic_ranges).astype(np.float32)

    def _fresh_confidence(self):
        if time.monotonic() - self.latest_confidence_time > 0.75:
            return 0.0
        return float(np.clip(self.latest_confidence, 0.0, 1.0))

    def _observation(self):
        pose = self.latest_odom_pose or (0.0, 0.0, 0.0)
        extra = np.asarray([
            pose[0],
            pose[1],
            pose[2],
            self._fresh_confidence(),
        ], dtype=np.float32)
        return np.concatenate((self._lidar_360_values(), extra)).astype(np.float32)

    def _coverage_cell(self):
        if self.latest_odom_xy is None:
            return None
        return (
            math.floor(self.latest_odom_xy[0] / self.coverage_cell_size),
            math.floor(self.latest_odom_xy[1] / self.coverage_cell_size),
        )

    def _calculate_reward_components(
        self, new_area, confidence, collision, new_objects=0,
    ):
        components = {
            'new_area': self.new_area_reward if new_area else 0.0,
            'confidence': self.confidence_reward_scale * confidence,
            'collision': -self.collision_penalty if collision else 0.0,
            'revisit': 0.0 if new_area else -self.revisit_penalty,
        }
        if self.new_object_reward:
            components['new_object'] = self.new_object_reward * new_objects
        return components

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self.visited_cells = set()
        start_cell = self._coverage_cell()
        if start_cell is not None:
            self.visited_cells.add(start_cell)
        info.update({
            'visited_cells': len(self.visited_cells),
            'state_layout': 'lidar[360], pose[x,y,yaw], confidence',
            'action_names': self.ACTION_NAMES,
        })
        return self._observation(), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid discrete action: {action}')

        self.pending_object_rewards = 0
        self.latest_confidence = 0.0
        self._publish_action(action)
        self._spin_for(self.action_duration)
        self.episode_step += 1

        cell = self._coverage_cell()
        new_area = cell is not None and cell not in self.visited_cells
        if new_area:
            self.visited_cells.add(cell)

        confidence = self._fresh_confidence()
        self.collision = (
            self.last_linear > 0.0 and self._front_clearance() < 0.14
        )
        reward_components = self._calculate_reward_components(
            new_area,
            confidence,
            self.collision,
            self.pending_object_rewards,
        )
        terminated = False
        if self.collision:
            self._stop()

        truncated = self.episode_step >= self.max_episode_steps
        if truncated:
            self._stop()

        reward = float(sum(reward_components.values()))
        self.episode_reward += reward
        confirmed = self._confirmed_objects()
        info = {
            'objects_found': len(confirmed),
            'object_classes': sorted({obj['class'] for obj in confirmed}),
            'collision': self.collision,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'new_area': new_area,
            'visited_cells': len(self.visited_cells),
            'confidence': confidence,
            'reward_components': reward_components,
            'action_name': self.ACTION_NAMES[int(action)],
        }
        return self._observation(), reward, terminated, truncated, info


class CoverageObjectExplorationEnv(CoverageExplorationEnv):
    """Coverage exploration with a direct bonus for each new confirmed object."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('new_object_reward', 10.0)
        super().__init__(*args, **kwargs)


class CoverageSearchExplorationEnv(CoverageObjectExplorationEnv):
    """Long-horizon three-action search with nearby visit memory."""

    ACTIONS = (
        (0.25, 0.00),
        (0.00, 0.90),
        (0.00, -0.90),
    )

    def __init__(self, *args, **kwargs):
        self.turn_penalty = 0.0
        self.progress_reward_scale = 0.0
        self.blocked_forward_penalty = 0.0
        self.spin_penalty = 0.0
        self.stationary_turn_steps = 0
        kwargs.setdefault('new_area_reward', 5.0)
        kwargs.setdefault('confidence_reward_scale', 0.10)
        kwargs.setdefault('collision_penalty', 1.0)
        kwargs.setdefault('revisit_penalty', 0.20)
        kwargs.setdefault('new_object_reward', 10.0)
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((self.observation_space.low,
                                np.zeros(8, dtype=np.float32))),
            high=np.concatenate((self.observation_space.high,
                                 np.ones(8, dtype=np.float32))),
            dtype=np.float32,
        )

    def _observation(self):
        base = super()._observation()
        if self.latest_odom_pose is None:
            novelty = np.ones(8, dtype=np.float32)
        else:
            x, y, yaw = self.latest_odom_pose
            bounds = getattr(self, 'search_bounds', None)
            values = []
            for index in range(8):
                direction = yaw + index * math.pi / 4
                unvisited = 0
                for distance in (0.75, 1.5, 3.0, 5.0):
                    px = x + distance * math.cos(direction)
                    py = y + distance * math.sin(direction)
                    if bounds is not None and not (
                            bounds[0] + 0.35 < px < bounds[1] - 0.35 and
                            bounds[2] + 0.35 < py < bounds[3] - 0.35):
                        continue
                    cell = (
                        math.floor(px / self.coverage_cell_size),
                        math.floor(py / self.coverage_cell_size),
                    )
                    unvisited += cell not in self.visited_cells
                values.append(unvisited / 4)
            novelty = np.asarray(values, dtype=np.float32)
        return np.concatenate((base, novelty)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.stationary_turn_steps = 0
        info['state_layout'] = (
            'lidar[360], pose[x,y,yaw], confidence, '
            'relative_unvisited_direction[8]'
        )
        return observation, info

    def _movement_reward_components(self, action, distance, front_clearance):
        components = {}
        if action == 0:
            if self.progress_reward_scale:
                components['progress'] = (
                    self.progress_reward_scale * min(distance, 0.05))
            if (self.blocked_forward_penalty and front_clearance < 0.40):
                components['blocked_forward'] = -self.blocked_forward_penalty
        elif self.spin_penalty and self.stationary_turn_steps >= 30:
            components['prolonged_spin'] = -self.spin_penalty
        return components

    def step(self, action):
        before = self.latest_odom_xy
        observation, reward, terminated, truncated, info = super().step(action)
        after = self.latest_odom_xy
        distance = (math.hypot(after[0] - before[0], after[1] - before[1])
                    if before is not None and after is not None else 0.0)
        if int(action) in (1, 2) and distance < 0.005:
            self.stationary_turn_steps += 1
        else:
            self.stationary_turn_steps = 0
        additions = self._movement_reward_components(
            int(action), distance, self._front_clearance())
        extra = sum(additions.values())
        info['reward_components'].update(additions)
        reward += extra
        self.episode_reward += extra
        info['episode_reward'] = self.episode_reward
        info['stationary_turn_steps'] = self.stationary_turn_steps
        return observation, float(reward), terminated, truncated, info

    def _calculate_reward_components(
        self, new_area, confidence, collision, new_objects=0,
    ):
        components = super()._calculate_reward_components(
            new_area, confidence, collision, new_objects,
        )
        components['time'] = -0.02
        components['motion'] = 0.05 if self.last_linear > 0 else -0.03
        if self.last_linear <= 0 and self.turn_penalty:
            components['turn'] = -self.turn_penalty
        return components


def make_exploration_env(profile='legacy', **kwargs):
    """Construct a compatible legacy or coverage exploration environment."""
    profiles = {
        'legacy': RosExplorationEnv,
        'coverage': CoverageExplorationEnv,
        'coverage_objects': CoverageObjectExplorationEnv,
        'coverage_search': CoverageSearchExplorationEnv,
    }
    try:
        environment_class = profiles[profile]
    except KeyError as error:
        choices = ', '.join(sorted(profiles))
        raise ValueError(f'Unknown environment profile {profile!r}; choose {choices}') from error
    return environment_class(**kwargs)
