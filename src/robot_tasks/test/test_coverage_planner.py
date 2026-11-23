"""Check frontier paths and tile preference independently of ROS/Gazebo."""

import math

import numpy as np

from robot_tasks.coverage_planner import (
    cell_to_world,
    plan_frontier,
    plan_sensor_probe,
    world_to_cell,
)


def test_rotated_grid_coordinates_round_trip():
    origin = (3.0, -2.0, 0.7)
    point = cell_to_world(8, 13, 0.2, origin)
    assert world_to_cell(*point, 0.2, origin) == (8, 13)


def test_planner_reaches_frontier_without_crossing_unknown_or_obstacles():
    grid = np.full((20, 20), -1, dtype=np.int16)
    grid[4:16, 4:16] = 0
    grid[8:13, 10] = 100
    start = cell_to_world(10, 6, 0.2, (0.0, 0.0, 0.0))
    path = plan_frontier(
        grid.ravel(), 20, 20, 0.2, (0.0, 0.0, 0.0), start,
        robot_radius=0.2,
    )
    assert path
    assert len(path) > 1
    for point in path:
        row, col = world_to_cell(*point, 0.2, (0.0, 0.0, 0.0))
        assert grid[row, col] == 0
    end_row, end_col = world_to_cell(*path[-1], 0.2, (0.0, 0.0, 0.0))
    assert end_row in (4, 15) or end_col in (4, 15)


def test_planner_prefers_new_tiles_but_may_route_through_visited_ones():
    grid = np.full((11, 11), -1, dtype=np.int16)
    grid[2:9, 2:9] = 0
    start = cell_to_world(5, 5, 0.25, (0.0, 0.0, 0.0))
    visited = {
        (int(x / 0.5), int(y / 0.5))
        for row in range(2, 6)
        for col in range(2, 9)
        for x, y in [cell_to_world(row, col, 0.25, (0.0, 0.0, 0.0))]
    }
    path = plan_frontier(
        grid.ravel(), 11, 11, 0.25, (0.0, 0.0, 0.0), start,
        visited_tiles=visited, tile_size=0.5, robot_radius=0.0,
    )
    assert path
    goal_tile = tuple(int(value / 0.5) for value in path[-1])
    assert goal_tile not in visited


def test_planner_repositions_on_known_free_space_when_no_frontier_remains():
    grid = np.zeros((11, 11), dtype=np.int16)
    start = cell_to_world(5, 5, 0.25, (0.0, 0.0, 0.0))
    current_tile = tuple(int(value / 0.5) for value in start)
    path = plan_frontier(
        grid.ravel(), 11, 11, 0.25, (0.0, 0.0, 0.0), start,
        visited_tiles={current_tile}, tile_size=0.5, robot_radius=0.0,
    )
    assert path
    assert tuple(int(value / 0.5) for value in path[-1]) != current_tile


def test_information_gain_prefers_a_larger_unexplored_frontier():
    grid = np.full((25, 25), 100, dtype=np.int16)
    grid[8:17, 5:20] = 0
    grid[8:17, 4] = -1
    grid[5:20, 20:25] = -1
    start = cell_to_world(12, 10, 0.2, (0.0, 0.0, 0.0))
    path = plan_frontier(
        grid.ravel(), 25, 25, 0.2, (0.0, 0.0, 0.0), start,
        robot_radius=0.0, information_gain_weight=5.0,
    )
    assert path
    _, goal_col = world_to_cell(*path[-1], 0.2, (0.0, 0.0, 0.0))
    assert goal_col == 19


def test_distant_mode_selects_a_remote_reachable_frontier():
    grid = np.full((21, 21), -1, dtype=np.int16)
    grid[5:16, 5:16] = 0
    start = cell_to_world(10, 6, 0.2, (0.0, 0.0, 0.0))
    path = plan_frontier(
        grid.ravel(), 21, 21, 0.2, (0.0, 0.0, 0.0), start,
        robot_radius=0.0, distance_weight=-0.25,
    )
    assert path
    goal_row, goal_col = world_to_cell(
        *path[-1], 0.2, (0.0, 0.0, 0.0)
    )
    assert goal_row in (5, 15) and goal_col >= 13


def test_open_area_recovery_selects_remote_free_tile_beyond_local_frontier():
    grid = np.full((21, 21), 100, dtype=np.int16)
    grid[8:13, 2:19] = 0
    grid[8:13, 1] = -1
    origin = (0.0, 0.0, 0.0)
    start = cell_to_world(10, 5, 0.2, origin)
    local = plan_frontier(
        grid.ravel(), 21, 21, 0.2, origin, start,
        robot_radius=0.0, distance_weight=-0.5,
    )
    wide = plan_frontier(
        grid.ravel(), 21, 21, 0.2, origin, start,
        robot_radius=0.0, distance_weight=-0.5,
        prefer_open_area=True,
    )
    assert local and wide
    assert world_to_cell(*local[-1], 0.2, origin)[1] == 2
    assert world_to_cell(*wide[-1], 0.2, origin)[1] == 18


def test_open_area_recovery_can_revisit_a_distant_frontier():
    grid = np.full((21, 21), 100, dtype=np.int16)
    grid[8:13, 2:19] = 0
    grid[8:13, 1] = -1
    grid[8:13, 19] = -1
    origin = (0.0, 0.0, 0.0)
    start = cell_to_world(10, 5, 0.2, origin)
    visited = {
        (col, row)
        for row in range(8, 13)
        for col in range(2, 19)
    }
    visited.remove((2, 10))
    default = plan_frontier(
        grid.ravel(), 21, 21, 0.2, origin, start,
        visited_tiles=visited, tile_size=0.2,
        robot_radius=0.0, distance_weight=-0.5,
    )
    recovery = plan_frontier(
        grid.ravel(), 21, 21, 0.2, origin, start,
        visited_tiles=visited, tile_size=0.2,
        robot_radius=0.0, distance_weight=-0.5,
        prefer_open_area=True,
    )
    assert world_to_cell(*default[-1], 0.2, origin)[1] == 2
    assert world_to_cell(*recovery[-1], 0.2, origin)[1] == 18


def test_sensor_probe_selects_clear_unknown_space_without_world_targets():
    grid = np.full((40, 40), -1, dtype=np.int16)
    grid[16:24, 16:24] = 0
    start = (4.05, 4.05, 0.0)

    def clearance(angle, _half_width):
        return 3.0 if abs(angle - 1.4) < 0.1 else 0.4

    goal = plan_sensor_probe(
        grid.ravel(), 40, 40, 0.2, (0.0, 0.0, 0.0),
        start, clearance,
    )
    assert goal is not None
    assert goal[1] > start[1] + 0.5


def test_sensor_probe_prefers_large_unknown_region_and_rejects_blocked_scan():
    grid = np.full((40, 40), 100, dtype=np.int16)
    start = (4.05, 4.05, 0.0)
    grid[:, :20] = -1
    goal = plan_sensor_probe(
        grid.ravel(), 40, 40, 0.2, (0.0, 0.0, 0.0),
        start, lambda angle, width: 3.0,
    )
    assert goal is not None
    assert goal[0] < start[0]
    assert plan_sensor_probe(
        grid.ravel(), 40, 40, 0.2, (0.0, 0.0, 0.0),
        start, lambda angle, width: 0.5,
    ) is None


def test_sensor_probe_avoids_neighborhood_already_explored():
    grid = np.full((40, 40), -1, dtype=np.int16)
    start = (4.05, 4.05, 0.0)
    visited = {(tx, ty) for tx in range(4, 13) for ty in range(2, 9)}

    def clearance(angle, _half_width):
        return 3.0 if abs(abs(angle) - 1.4) < 0.1 else 0.4

    goal = plan_sensor_probe(
        grid.ravel(), 40, 40, 0.2, (0.0, 0.0, 0.0),
        start, clearance, visited_tiles=visited,
    )
    assert goal is not None
    assert goal[1] > start[1]


def test_frontier_route_avoids_repeatedly_blocked_intermediate_tile():
    grid = np.full((13, 13), -1, dtype=np.int16)
    grid[2:11, 2:11] = 0
    origin = (0.0, 0.0, 0.0)
    start = cell_to_world(6, 3, 0.25, origin)
    blocked_xy = cell_to_world(6, 5, 0.25, origin)
    blocked_tile = tuple(math.floor(value / 0.5) for value in blocked_xy)
    path = plan_frontier(
        grid.ravel(), 13, 13, 0.25, origin, start,
        tile_size=0.5, robot_radius=0.0,
        avoid_tiles={blocked_tile},
    )
    assert path
    assert all(
        tuple(math.floor(value / 0.5) for value in point) != blocked_tile
        for point in path
    )


def test_frontier_prefers_a_less_visited_region():
    grid = np.full((25, 25), -1, dtype=np.int16)
    grid[5:20, 5:20] = 0
    origin = (0.0, 0.0, 0.0)
    start = cell_to_world(12, 12, 0.2, origin)
    visited = {(tx, ty) for tx in range(4, 11) for ty in range(1, 9)}
    path = plan_frontier(
        grid.ravel(), 25, 25, 0.2, origin, start,
        visited_tiles=visited, tile_size=0.5, robot_radius=0.0,
        nearby_visit_weight=0.5,
    )
    assert path
    assert path[-1][0] < start[0]


def test_sector_balance_moves_toward_underexplored_half_of_map():
    grid = np.full((25, 25), -1, dtype=np.int16)
    grid[3:22, 3:22] = 0
    origin = (0.0, 0.0, 0.0)
    start = cell_to_world(12, 12, 0.2, origin)
    center = (2.5, 2.5)
    visited = {(col, row) for col in range(5, 10)
               for row in range(1, 10)}
    path = plan_frontier(
        grid.ravel(), 25, 25, 0.2, origin, start,
        visited_tiles=visited, tile_size=0.5, robot_radius=0.0,
        prefer_open_area=True, sector_center=center,
        sector_balance_weight=1.0,
    )
    assert path
    assert path[-1][0] < center[0]


def test_planner_keeps_goals_inside_search_walls():
    grid = np.zeros((20, 20), dtype=np.int16)
    start = (5.0, 5.0)
    path = plan_frontier(
        grid.ravel(), 20, 20, 0.5, (0.0, 0.0, 0.0), start,
        visited_tiles={(10, 10)}, tile_size=0.5, robot_radius=0.0,
        search_bounds=(0.0, 10.0, 0.0, 10.0), wall_margin_m=1.2,
    )
    assert path
    assert all(1.2 <= x <= 8.8 and 1.2 <= y <= 8.8
               for x, y in path)

    probe = plan_sensor_probe(
        grid.ravel(), 20, 20, 0.5, (0.0, 0.0, 0.0),
        (8.3, 5.0, 0.0), lambda angle, width: 3.0,
        search_bounds=(0.0, 10.0, 0.0, 10.0), wall_margin_m=1.2,
    )
    assert probe is not None
    assert 1.2 <= probe[0] <= 8.8
    assert 1.2 <= probe[1] <= 8.8


def test_planner_transforms_map_points_before_world_wall_check():
    grid = np.zeros((20, 20), dtype=np.int16)
    map_to_world = (2.0, 0.0, 0.0)
    bounds = (0.0, 10.0, 0.0, 10.0)
    path = plan_frontier(
        grid.ravel(), 20, 20, 0.5, (0.0, 0.0, 0.0),
        (5.0, 5.0), visited_tiles={(10, 10)}, tile_size=0.5,
        robot_radius=0.0, search_bounds=bounds,
        bounds_transform=map_to_world,
    )
    assert path
    assert all(1.2 <= x + 2.0 <= 8.8 and 1.2 <= y <= 8.8
               for x, y in path)

    probe = plan_sensor_probe(
        grid.ravel(), 20, 20, 0.5, (0.0, 0.0, 0.0),
        (6.0, 5.0, 0.0), lambda angle, width: 3.0,
        search_bounds=bounds, bounds_transform=map_to_world,
    )
    assert probe is not None
    assert 1.2 <= probe[0] + 2.0 <= 8.8
    assert 1.2 <= probe[1] <= 8.8
