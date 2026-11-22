"""Blocked-path recovery should use open LiDAR space and stop after moving."""

import math
from types import SimpleNamespace

from robot_tasks.global_coverage_assist import GlobalCoverageAssist


def test_blocked_path_escape_uses_clear_rear_direction():
    ranges = []
    for index in range(360):
        angle = -math.pi + index * 2 * math.pi / 360
        ranges.append(2.0 if abs(angle) > 2.7 else 0.3)
    scan = SimpleNamespace(
        ranges=ranges, angle_min=-math.pi,
        angle_increment=2 * math.pi / 360, range_max=3.5,
    )
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.tile_size = 0.5
    assist._map_visited_tiles = lambda _cells: set()
    assist.goal_history = []
    assist.escape_events = 0
    assist.escape_heading = None
    assist.escape_origin = None
    assist.stall_seconds = 1.0
    assist.intervention_steps = 0
    assist.path = [(1.0, 0.0)]
    assist.goal = (1.0, 0.0)

    assert assist._begin_escape((0.0, 0.0, 0.0), scan, set(), 10.0)
    assert abs(abs(assist.escape_heading) - math.pi) < 0.01
    assert assist._escape_action((0.0, 0.0, 0.0), 0.3, 10.1) in (1, 2)
    assert assist._escape_action((0.0, 0.0, math.pi), 2.0, 11.0) == 0
    assert assist._escape_action((-1.0, 0.0, math.pi), 2.0, 12.0) is None
    assert assist.escape_heading is None
    assert assist.path == []
    assert assist.escape_events == 1


def test_wall_escape_rejects_clear_direction_away_from_room_center():
    ranges = []
    for index in range(360):
        angle = -math.pi + index * 2 * math.pi / 360
        if abs(angle + 1.55) < 0.23:
            ranges.append(3.0)  # Clear south, toward the wall.
        elif abs(angle - 1.55) < 0.23:
            ranges.append(1.1)  # Narrower but clear north, into the room.
        else:
            ranges.append(0.3)
    scan = SimpleNamespace(
        ranges=ranges, angle_min=-math.pi,
        angle_increment=2 * math.pi / 360, range_max=3.5,
    )
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.tile_size = 0.5
    assist._map_visited_tiles = lambda _cells: {(0, 2)}
    assist.goal_history = []
    assist.escape_events = 0
    assist.path = []
    assist.goal = None

    assert assist._begin_escape(
        (0.0, 0.0, 0.0), scan, set(), 10.0,
        preferred_heading=math.pi / 2,
        require_preferred_half_plane=True,
    )
    assert abs(assist.escape_heading - 1.55) < 0.1


def test_wall_escape_allows_lateral_route_around_obstacle():
    ranges = []
    for index in range(360):
        angle = -math.pi + index * 2 * math.pi / 360
        ranges.append(2.0 if abs(angle - math.pi / 2) < 0.24 else 0.3)
    scan = SimpleNamespace(
        ranges=ranges, angle_min=-math.pi,
        angle_increment=2 * math.pi / 360, range_max=3.5,
    )
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.tile_size = 0.5
    assist._map_visited_tiles = lambda _cells: set()
    assist.goal_history = []
    assist.escape_events = 0
    assist.path = []
    assist.goal = None
    # Robot faces east at the east wall. West (pi) is the room interior,
    # but an obstacle blocks it. North (pi/2) is clear and should be usable.
    assert assist._begin_escape(
        (8.7, -0.6, 0.0), scan, set(), 10.0,
        preferred_heading=math.pi,
        require_preferred_half_plane=True,
    )
    assert abs(assist.escape_heading - math.pi / 2) < 0.1


def test_boundary_recovery_clears_escape_heading_away_from_center():
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.search_bounds = (-9.0, 9.0, -11.0, 7.0)
    assist.last_visited_count = 0
    assist.last_new_cell_time = 0.0
    assist.path = []
    assist.goal = None
    assist.escape_heading = 0.0  # South in a map rotated 90 degrees.
    assist.escape_origin = (0.0, 0.0)
    assist.intervention_steps = 0
    assist.returning_inside_bounds = False
    assist.returning_by_breadcrumbs = False
    assist.map = object()
    assist.stall_seconds = 1.0
    assist._map_pose = lambda odom: (odom[0], odom[1], odom[2] + math.pi / 2)

    action, source = assist.action(
        (0.0, -10.5, math.pi / 2), set(), 2, 2.0, now=5.0)
    assert (action, source) == (0, 'global_coverage')
    assert assist.escape_heading is None


def test_boundary_recovery_returns_toward_arena_interior():
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.search_bounds = (-9.0, 9.0, -11.0, 7.0)
    assist.last_visited_count = 0
    assist.last_new_cell_time = 0.0
    assist.path = [(0.0, -10.0)]
    assist.goal = (0.0, -10.0)
    assist.escape_heading = None
    assist.escape_origin = None
    assist.intervention_steps = 0
    assist.returning_inside_bounds = False
    assist.map = None
    assist.stall_seconds = 1.0
    action, source = assist.action(
        (-1.0, -10.5, math.pi / 2), set(), 2, 2.0, now=5.0)
    assert (action, source) == (0, 'global_coverage')
    assert assist.last_status == 'returning_inside_bounds'
    assert assist.goal is None
    # Crossing just inside the trigger must not release control back to a
    # frontier that immediately drives the robot out again.
    action, source = assist.action(
        (-1.0, -9.7, math.pi / 2), set(), 2, 2.0, now=6.0)
    assert (action, source) == (0, 'global_coverage')
    assert assist.returning_inside_bounds
    action, source = assist.action(
        (-1.0, -9.1, math.pi / 2), set(), 2, 2.0, now=7.0)
    assert (action, source) == (2, 'policy')
    assert not assist.returning_inside_bounds


def test_boundary_recovery_uses_world_aligned_odometry_when_slam_drifts():
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.search_bounds = (-9.0, 9.0, -11.0, 7.0)
    assist.last_visited_count = 0
    assist.last_new_cell_time = 100.0
    assist.stall_seconds = 1.0
    assist.path = []
    assist.goal = None
    assist.escape_heading = None
    assist.escape_origin = None
    assist.intervention_steps = 0
    assist.returning_inside_bounds = False
    assist.map = object()
    assist.breadcrumbs = []
    assist.enable_breadcrumb_return = False
    assist.last_detector_count = 0
    assist.seen_detector_classes = set()
    assist.last_detector_time = 100.0
    assist._map_pose = lambda _odom: (0.0, 4.0, -math.pi / 2)

    # In this Gazebo setup odometry is aligned with the world, while the
    # SLAM map can drift. The world wall limits must use odometry.
    action, source = assist.action(
        (0.0, 6.5, -math.pi / 2), set(), 2, 2.0, now=100.0)
    assert (action, source) == (0, 'global_coverage')
    assert assist.returning_inside_bounds

    # A drifting map pose outside the wall must not keep recovery active
    # after odometry has returned to the room interior.
    assist._map_pose = lambda _odom: (0.0, 6.5, -math.pi / 2)
    assist._plan = lambda *_args: False
    action, source = assist.action(
        (0.0, 4.0, -math.pi / 2), set(), 2, 2.0, now=101.0)
    assert (action, source) == (2, 'policy')
    assert not assist.returning_inside_bounds


def test_map_pose_records_inverse_transform_for_world_bounds():
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.map = SimpleNamespace(header=SimpleNamespace(frame_id='map'))
    quaternion = SimpleNamespace(
        w=math.sqrt(0.5), x=0.0, y=0.0, z=math.sqrt(0.5))
    transform = SimpleNamespace(transform=SimpleNamespace(
        rotation=quaternion,
        translation=SimpleNamespace(x=2.0, y=1.0)))
    assist.buffer = SimpleNamespace(
        lookup_transform=lambda *_args: transform)
    map_pose = assist._map_pose((1.0, 0.0, 0.0))
    assert math.isclose(map_pose[0], 2.0, abs_tol=1e-6)
    assert math.isclose(map_pose[1], 2.0, abs_tol=1e-6)
    tx, ty, yaw = assist.bounds_transform
    odom_x = tx + math.cos(yaw) * map_pose[0] - math.sin(yaw) * map_pose[1]
    odom_y = ty + math.sin(yaw) * map_pose[0] + math.cos(yaw) * map_pose[1]
    assert math.isclose(odom_x, 1.0, abs_tol=1e-6)
    assert math.isclose(odom_y, 0.0, abs_tol=1e-6)


def test_stalled_camera_discovery_retraces_route_then_heads_to_unvisited_side():
    assist = GlobalCoverageAssist.__new__(GlobalCoverageAssist)
    assist.tile_size = 0.5
    assist.breadcrumbs = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                          (1.5, 0.0), (2.5, 0.0)]
    assist.goal_history = []
    assist.breadcrumb_return_events = 0
    assist.blocked_steps = 0
    assert assist._begin_breadcrumb_return((2.5, 0.0, 0.0), 200.0)
    assert assist.path == list(reversed(assist.breadcrumbs))
    assert assist.goal == (0.0, 0.0)
    assert assist.breadcrumb_return_events == 1

    assist._map_visited_tiles = lambda _cells: {(2, 0), (4, 0), (6, 0)}
    seen = {}

    def begin_escape(_pose, _scan, _visited, _now, preferred_heading=None):
        seen['heading'] = preferred_heading
        return True

    assist._begin_escape = begin_escape
    assist._escape_action = lambda *_args: 2
    action = assist._finish_breadcrumb_return(
        (0.0, 0.0, 0.0), None, set(), 1.0, 240.0)
    assert action == 2
    assert math.cos(seen['heading']) < -0.95
    assert not assist.returning_by_breadcrumbs
    assert assist.return_cooldown_until == 420.0


