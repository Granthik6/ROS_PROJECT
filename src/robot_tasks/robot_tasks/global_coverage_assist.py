"""Sensor-map coverage assist for long autonomous discovery runs.

The assist reads SLAM's occupancy grid and the robot's complete visited-cell
set. It never reads Gazebo object positions or the evaluator's target list.
It temporarily steers toward a reachable unvisited frontier only after the
underlying policy has stopped visiting new cells.
"""

import math
import time

from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from robot_tasks.coverage_planner import plan_frontier, plan_sensor_probe


class GlobalCoverageAssist:
    def __init__(self, node, tile_size=0.5, stall_seconds=12.0,
                 search_bounds=None, enable_breadcrumb_return=False,
                 planner_bounds_enabled=True, sector_balance_weight=0.0):
        self.node = node
        self.tile_size = float(tile_size)
        self.stall_seconds = float(stall_seconds)
        self.search_bounds = search_bounds
        self.planner_bounds_enabled = bool(planner_bounds_enabled)
        self.sector_balance_weight = max(0.0, float(sector_balance_weight))
        self.map = None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node, spin_thread=False)
        node.create_subscription(OccupancyGrid, '/map', self._on_map, 10)
        self.path = []
        self.goal = None
        self.goal_started = 0.0
        self.goal_penalties = {}
        self.blocked_tiles = {}
        self.last_visited_count = 0
        self.last_new_cell_time = time.monotonic()
        self.intervention_steps = 0
        self.planned_goals = 0
        self.failed_plans = 0
        self.next_plan_time = 0.0
        self.last_status = 'waiting_for_map'
        self.blocked_steps = 0
        self.goal_history = []
        self.escape_heading = None
        self.escape_origin = None
        self.escape_started = 0.0
        self.escape_events = 0
        self.breadcrumbs = []
        self.returning_by_breadcrumbs = False
        self.last_detector_count = 0
        self.seen_detector_classes = set()
        self.last_detector_time = time.monotonic()
        self.return_cooldown_until = 0.0
        self.breadcrumb_return_events = 0
        self.enable_breadcrumb_return = bool(enable_breadcrumb_return)
        self.returning_inside_bounds = False
        self.bounds_transform = None
        self.odom_to_map_transform = None

    def _on_map(self, message):
        self.map = message

    def _map_pose(self, odom_pose):
        if odom_pose is None or self.map is None:
            return None
        frame = self.map.header.frame_id or 'map'
        try:
            transform = self.buffer.lookup_transform(frame, 'odom', Time())
        except TransformException:
            self.last_status = 'waiting_for_map_to_odom_transform'
            self.bounds_transform = None
            self.odom_to_map_transform = None
            return None
        q = transform.transform.rotation
        angle = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        x, y, yaw = odom_pose
        dx = transform.transform.translation.x
        dy = transform.transform.translation.y
        cosine, sine = math.cos(angle), math.sin(angle)
        self.bounds_transform = (-cosine * dx - sine * dy,
                                 sine * dx - cosine * dy, -angle)
        self.odom_to_map_transform = (dx, dy, angle)
        return (dx + math.cos(angle) * x - math.sin(angle) * y,
                dy + math.sin(angle) * x + math.cos(angle) * y,
                yaw + angle)

    def _map_visited_tiles(self, visited_cells):
        """Convert every odom-frame visit to the current map frame."""
        frame = self.map.header.frame_id or 'map'
        try:
            transform = self.buffer.lookup_transform(frame, 'odom', Time())
        except TransformException:
            return set()
        q = transform.transform.rotation
        angle = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cosine, sine = math.cos(angle), math.sin(angle)
        dx, dy = (transform.transform.translation.x,
                  transform.transform.translation.y)
        result = set()
        for col, row in visited_cells:
            x = (col + 0.5) * self.tile_size
            y = (row + 0.5) * self.tile_size
            mx = dx + cosine * x - sine * y
            my = dy + sine * x + cosine * y
            result.add((math.floor(mx / self.tile_size),
                        math.floor(my / self.tile_size)))
        return result

    @staticmethod
    def _scan_clearance(scan, center, half_width):
        if scan is None:
            return 0.0
        values = []
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            difference = math.atan2(math.sin(angle - center),
                                    math.cos(angle - center))
            if abs(difference) > half_width:
                continue
            if math.isinf(distance):
                values.append(float(scan.range_max))
            elif math.isfinite(distance) and distance > 0:
                values.append(float(distance))
        return min(values, default=0.0)

    def _begin_escape(self, map_pose, scan, visited_cells, now,
                      preferred_heading=None,
                      require_preferred_half_plane=False):
        """Choose a LiDAR-clear direction when the mapped path is blocked."""
        if scan is None:
            return False
        visited = self._map_visited_tiles(visited_cells)
        choices = []
        for angle in (math.pi, 2.35, -2.35, 1.55, -1.55,
                      0.8, -0.8, 0.0):
            clearance = self._scan_clearance(scan, angle, 0.22)
            if clearance < 1.0:
                continue
            difference = None
            if preferred_heading is not None:
                difference = math.atan2(
                    math.sin(map_pose[2] + angle - preferred_heading),
                    math.cos(map_pose[2] + angle - preferred_heading))
                if (require_preferred_half_plane and
                        math.cos(difference) < -0.20):
                    continue
            x = map_pose[0] + math.cos(map_pose[2] + angle)
            y = map_pose[1] + math.sin(map_pose[2] + angle)
            tile = (math.floor(x / self.tile_size),
                    math.floor(y / self.tile_size))
            score = min(clearance, 2.5) + (0.7 if tile not in visited else 0)
            score -= 0.04 * abs(angle)
            if difference is not None:
                score += 2.0 * max(0.0, math.cos(difference))
            choices.append((score, angle))
        if not choices:
            return False
        _, angle = max(choices)
        self.escape_heading = math.atan2(
            math.sin(map_pose[2] + angle),
            math.cos(map_pose[2] + angle))
        self.escape_origin = map_pose[:2]
        self.escape_started = now
        self.escape_events += 1
        self.path = []
        self.goal = None
        self.goal_history.append({
            'event': 'blocked_path_escape', 'time': now,
            'robot_xy': map_pose[:2], 'heading': self.escape_heading,
            'visited_cells': len(visited_cells),
        })
        return True

    def _begin_breadcrumb_return(self, map_pose, now):
        """Retrace the driven route when camera discovery has stalled."""
        if len(self.breadcrumbs) < 5:
            return False
        start = self.breadcrumbs[0]
        if math.hypot(map_pose[0] - start[0],
                      map_pose[1] - start[1]) < 2.0:
            return False
        self.path = list(reversed(self.breadcrumbs))
        self.goal = start
        self.goal_started = now
        self.returning_by_breadcrumbs = True
        self.breadcrumb_return_events += 1
        self.blocked_steps = 0
        self.last_status = 'returning_by_breadcrumbs'
        self.goal_history.append({
            'event': 'breadcrumb_return_started', 'time': now,
            'robot_xy': map_pose[:2], 'goal_xy': start,
            'path_points': len(self.path),
        })
        return True

    def _finish_breadcrumb_return(self, map_pose, scan, visited_cells,
                                  front_clearance, now):
        visited = self._map_visited_tiles(visited_cells)
        preferred = None
        if visited:
            center_x = sum((col + 0.5) * self.tile_size
                           for col, _ in visited) / len(visited)
            center_y = sum((row + 0.5) * self.tile_size
                           for _, row in visited) / len(visited)
            preferred = math.atan2(
                map_pose[1] - center_y, map_pose[0] - center_x)
        self.returning_by_breadcrumbs = False
        self.path = []
        self.goal = None
        self.breadcrumbs = [map_pose[:2]]
        self.last_detector_time = now
        self.return_cooldown_until = now + 180.0
        self.goal_history.append({
            'event': 'breadcrumb_return_completed', 'time': now,
            'robot_xy': map_pose[:2], 'preferred_heading': preferred,
        })
        if self._begin_escape(map_pose, scan, visited_cells, now,
                              preferred_heading=preferred):
            return self._escape_action(map_pose, front_clearance, now)
        return None

    def _escape_action(self, map_pose, front_clearance, now):
        if self.escape_heading is None:
            return None
        moved = math.hypot(map_pose[0] - self.escape_origin[0],
                           map_pose[1] - self.escape_origin[1])
        if moved >= 0.9 or now - self.escape_started > 20.0:
            self.escape_heading = None
            self.escape_origin = None
            self.next_plan_time = now
            self.last_new_cell_time = now - self.stall_seconds
            return None
        error = math.atan2(math.sin(self.escape_heading - map_pose[2]),
                           math.cos(self.escape_heading - map_pose[2]))
        self.last_status = 'escaping_blocked_path'
        self.intervention_steps += 1
        if abs(error) > 0.25:
            return 1 if error > 0 else 2
        if front_clearance >= 0.48:
            return 0
        self.escape_heading = None
        self.escape_origin = None
        self.next_plan_time = now + 1.0
        return 1

    def _plan(self, map_pose, visited_cells, scan, now):
        self.blocked_tiles = {
            tile: expiry for tile, expiry in self.blocked_tiles.items()
            if expiry > now
        }
        grid = self.map
        origin = grid.info.origin
        q = origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        visited = self._map_visited_tiles(visited_cells)
        planner_bounds = (self.search_bounds
                          if self.planner_bounds_enabled else None)
        sector_center = None
        if self.search_bounds is not None and self.odom_to_map_transform:
            xmin, xmax, ymin, ymax = self.search_bounds
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            dx, dy, angle = self.odom_to_map_transform
            sector_center = (dx + math.cos(angle) * cx - math.sin(angle) * cy,
                             dy + math.sin(angle) * cx + math.cos(angle) * cy)
        self.path = plan_frontier(
            grid.data, grid.info.width, grid.info.height,
            grid.info.resolution, (origin.position.x, origin.position.y, yaw),
            map_pose[:2], visited_tiles=visited,
            tile_size=self.tile_size, robot_radius=0.22,
            goal_penalties=self.goal_penalties,
            avoid_tiles=self.blocked_tiles,
            information_gain_weight=1.5,
            nearby_visit_weight=0.5,
            prefer_open_area=True,
            min_goal_distance_m=1.25,
            search_bounds=planner_bounds,
            bounds_transform=self.bounds_transform,
            sector_center=sector_center,
            sector_balance_weight=self.sector_balance_weight,
        )
        goal_kind = 'frontier'
        if not self.path and scan is not None:
            probe = plan_sensor_probe(
                grid.data, grid.info.width, grid.info.height,
                grid.info.resolution,
                (origin.position.x, origin.position.y, yaw), map_pose,
                lambda angle, width: self._scan_clearance(
                    scan, angle, width),
                visited_tiles=visited, tile_size=self.tile_size,
                goal_penalties={
                    **self.goal_penalties,
                    **{tile: 12.0 for tile in self.blocked_tiles},
                },
                search_bounds=planner_bounds,
                bounds_transform=self.bounds_transform,
            )
            if probe is not None:
                self.path = [probe]
                goal_kind = 'sensor_probe'
        if not self.path:
            self.failed_plans += 1
            self.next_plan_time = now + 3.0
            self.last_status = 'no_reachable_frontier'
            self.goal_history.append({
                'event': 'no_reachable_frontier', 'time': now,
                'robot_xy': map_pose[:2],
                'visited_cells': len(visited_cells),
            })
            return False
        self.goal = self.path[-1]
        self.goal_started = now
        self.planned_goals += 1
        self.blocked_steps = 0
        self.last_status = f'following_{goal_kind}'
        self.goal_history.append({
            'event': 'goal_selected', 'goal_kind': goal_kind, 'time': now,
            'robot_xy': map_pose[:2], 'goal_xy': self.goal,
            'path_points': len(self.path),
            'visited_cells': len(visited_cells),
        })
        return True

    def action(self, odom_pose, visited_cells, proposed, front_clearance,
               scan=None, now=None, detector_count=None,
               detector_classes=None):
        """Return (applied_action, source) using actions 0/1/2 only."""
        now = time.monotonic() if now is None else now
        if len(visited_cells) > self.last_visited_count:
            self.last_visited_count = len(visited_cells)
            self.last_new_cell_time = now
        if self.search_bounds is not None and odom_pose is not None:
            xmin, xmax, ymin, ymax = self.search_bounds
            x, y, yaw = odom_pose
            inside_trigger = (xmin + 1.2 <= x <= xmax - 1.2 and
                              ymin + 1.2 <= y <= ymax - 1.2)
            inside_release = (xmin + 1.8 <= x <= xmax - 1.8 and
                              ymin + 1.8 <= y <= ymax - 1.8)
            if not inside_trigger and not self.returning_inside_bounds:
                self.returning_inside_bounds = True
                self.path = []
                self.goal = None
                self.returning_by_breadcrumbs = False
                self.escape_heading = None
                self.escape_origin = None
            if self.returning_inside_bounds and inside_release:
                self.returning_inside_bounds = False
                self.escape_heading = None
                self.escape_origin = None
                self.next_plan_time = now
                self.last_new_cell_time = now - self.stall_seconds
            if self.returning_inside_bounds:
                center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                heading = math.atan2(center[1] - y, center[0] - x)
                error = math.atan2(math.sin(heading - yaw),
                                   math.cos(heading - yaw))
                self.last_status = 'returning_inside_bounds'
                self.intervention_steps += 1
                map_pose = self._map_pose(odom_pose) if self.map else None
                map_center_heading = (
                    heading + map_pose[2] - yaw
                    if map_pose is not None else None)
                if map_pose is not None and self.escape_heading is not None and math.cos(
                        self.escape_heading - map_center_heading) <= 0.0:
                    self.escape_heading = None
                    self.escape_origin = None
                if map_pose is not None and self.escape_heading is not None:
                    escape_action = self._escape_action(
                        map_pose, front_clearance, now)
                    if escape_action is not None:
                        return escape_action, 'global_coverage'
                if abs(error) > 0.25:
                    return (1 if error > 0 else 2), 'global_coverage'
                if front_clearance >= 0.48:
                    return 0, 'global_coverage'
                if map_pose is not None and self._begin_escape(
                        map_pose, scan, visited_cells, now,
                        preferred_heading=map_center_heading,
                        require_preferred_half_plane=True):
                    escape_action = self._escape_action(
                        map_pose, front_clearance, now)
                    if escape_action is not None:
                        return escape_action, 'global_coverage'
                return 1, 'global_coverage'
        if self.map is None:
            self.last_status = 'waiting_for_map'
            return proposed, 'policy'
        map_pose = self._map_pose(odom_pose)
        if map_pose is None:
            return proposed, 'policy'
        if not self.breadcrumbs:
            self.breadcrumbs.append(map_pose[:2])
        elif (not self.returning_by_breadcrumbs and
              math.hypot(map_pose[0] - self.breadcrumbs[-1][0],
                         map_pose[1] - self.breadcrumbs[-1][1]) >= 0.35):
            self.breadcrumbs.append(map_pose[:2])
        if detector_count is not None and detector_count > self.last_detector_count:
            self.last_detector_count = detector_count
            if detector_classes is None:
                self.last_detector_time = now
        if detector_classes is not None and (
                set(detector_classes) - self.seen_detector_classes):
            self.seen_detector_classes.update(detector_classes)
            self.last_detector_time = now
        escape_action = self._escape_action(map_pose, front_clearance, now)
        if escape_action is not None:
            return escape_action, 'global_coverage'
        if (self.enable_breadcrumb_return and
                not self.returning_by_breadcrumbs and
                now >= self.return_cooldown_until and
                now - self.last_detector_time >= 120.0):
            self._begin_breadcrumb_return(map_pose, now)
        if self.goal is not None:
            distance_to_goal = math.hypot(
                map_pose[0] - self.goal[0], map_pose[1] - self.goal[1])
            if distance_to_goal < (0.50 if self.returning_by_breadcrumbs
                                   else 0.28):
                if self.returning_by_breadcrumbs:
                    applied = self._finish_breadcrumb_return(
                        map_pose, scan, visited_cells, front_clearance, now)
                    if applied is not None:
                        return applied, 'global_coverage'
                self.path = []
                self.goal = None
                self.last_new_cell_time = now
                self.last_status = 'frontier_reached'
                return proposed, 'policy'
            if now - self.goal_started > (
                    180.0 if self.returning_by_breadcrumbs else 35.0):
                tile = (math.floor(self.goal[0] / self.tile_size),
                        math.floor(self.goal[1] / self.tile_size))
                self.goal_penalties[tile] = min(
                    12.0, self.goal_penalties.get(tile, 0.0) + 4.0)
                if self.path:
                    waypoint = self.path[0]
                    blocked = (math.floor(waypoint[0] / self.tile_size),
                               math.floor(waypoint[1] / self.tile_size))
                    self.blocked_tiles[blocked] = now + 120.0
                self.path = []
                self.goal = None
                if self.returning_by_breadcrumbs:
                    self.returning_by_breadcrumbs = False
                    self.return_cooldown_until = now + 60.0
                self.last_status = 'frontier_timeout'
        if not self.path:
            if now - self.last_new_cell_time < self.stall_seconds:
                return proposed, 'policy'
            if now < self.next_plan_time:
                return proposed, 'policy'
            if not self._plan(map_pose, visited_cells, scan, now):
                return proposed, 'policy'

        while len(self.path) > 1 and math.hypot(
                self.path[0][0] - map_pose[0],
                self.path[0][1] - map_pose[1]) < 0.30:
            self.path.pop(0)
        if not self.path:
            self.goal = None
            return proposed, 'policy'
        waypoint = self.path[0]
        if not self.returning_by_breadcrumbs:
            for point in self.path:
                waypoint = point
                if math.hypot(point[0] - map_pose[0],
                              point[1] - map_pose[1]) >= 0.60:
                    break
        heading = math.atan2(waypoint[1] - map_pose[1],
                             waypoint[0] - map_pose[0])
        error = math.atan2(math.sin(heading - map_pose[2]),
                           math.cos(heading - map_pose[2]))
        blocked_clearance = 0.40 if self.returning_by_breadcrumbs else 0.48
        if front_clearance < blocked_clearance and abs(error) < 0.35:
            self.blocked_steps += 1
            if self.blocked_steps >= 8:
                tile = (math.floor(self.goal[0] / self.tile_size),
                        math.floor(self.goal[1] / self.tile_size))
                self.goal_penalties[tile] = min(
                    12.0, self.goal_penalties.get(tile, 0.0) + 4.0)
                blocked = (math.floor(waypoint[0] / self.tile_size),
                           math.floor(waypoint[1] / self.tile_size))
                self.blocked_tiles[blocked] = now + 120.0
                self.goal_history.append({
                    'event': 'path_blocked', 'time': now,
                    'robot_xy': map_pose[:2], 'goal_xy': self.goal,
                    'blocked_tile': blocked,
                    'visited_cells': len(visited_cells),
                })
                self.path = []
                self.goal = None
                if self.returning_by_breadcrumbs:
                    self.returning_by_breadcrumbs = False
                    self.return_cooldown_until = now + 60.0
                self.next_plan_time = now + 1.0
                self.blocked_steps = 0
                if self._begin_escape(map_pose, scan, visited_cells, now):
                    applied = self._escape_action(
                        map_pose, front_clearance, now)
                    return applied, 'global_coverage'
            self.last_status = 'blocked_frontier_path'
            applied = 1 if error >= 0 else 2
        elif abs(error) > 0.27:
            self.blocked_steps = 0
            applied = 1 if error > 0 else 2
        else:
            self.blocked_steps = 0
            applied = 0
        self.intervention_steps += 1
        return applied, 'global_coverage'
