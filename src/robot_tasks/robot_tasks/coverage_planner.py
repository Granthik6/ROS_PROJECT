"""Plan a path to reachable unexplored space on a SLAM occupancy grid."""

from collections import deque
import math

import numpy as np


NEIGHBORS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _inside_search_bounds(x, y, bounds, margin, bounds_transform=None):
    """Check a map point against bounds in another frame when requested."""
    if bounds_transform is not None:
        tx, ty, yaw = bounds_transform
        cosine, sine = math.cos(yaw), math.sin(yaw)
        x, y = (tx + cosine * x - sine * y,
                ty + sine * x + cosine * y)
    xmin, xmax, ymin, ymax = bounds
    return (xmin + margin <= x <= xmax - margin and
            ymin + margin <= y <= ymax - margin)


def world_to_cell(x, y, resolution, origin):
    """Return (row, column), including a rotated occupancy-grid origin."""
    ox, oy, yaw = origin
    dx, dy = x - ox, y - oy
    c, s = math.cos(yaw), math.sin(yaw)
    return (math.floor((-s * dx + c * dy) / resolution),
            math.floor((c * dx + s * dy) / resolution))


def cell_to_world(row, col, resolution, origin):
    ox, oy, yaw = origin
    x, y = (col + 0.5) * resolution, (row + 0.5) * resolution
    c, s = math.cos(yaw), math.sin(yaw)
    return ox + c * x - s * y, oy + s * x + c * y


def _inflate(obstacles, radius_cells):
    if radius_cells <= 0:
        return obstacles
    height, width = obstacles.shape
    padded = np.pad(obstacles, radius_cells, constant_values=True)
    result = np.zeros_like(obstacles)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            result |= padded[
                radius_cells + dy:radius_cells + dy + height,
                radius_cells + dx:radius_cells + dx + width,
            ]
    return result


def _frontiers(grid, free):
    unknown = grid < 0
    adjacent = np.zeros_like(free)
    adjacent[1:, :] |= unknown[:-1, :]
    adjacent[:-1, :] |= unknown[1:, :]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    return free & adjacent


def plan_frontier(
    data, width, height, resolution, origin, robot_xy,
    visited_tiles=(), tile_size=0.5, robot_radius=0.18,
    goal_penalties=None, information_gain_weight=0.0,
    distance_weight=1.0, avoid_tiles=(),
    nearby_visit_weight=0.0,
    gain_radius_m=1.0,
    prefer_open_area=False,
    min_goal_distance_m=0.45,
    search_bounds=None,
    wall_margin_m=1.2,
    bounds_transform=None,
    sector_center=None,
    sector_balance_weight=0.0,
):
    """Return world-coordinate path to a reachable frontier, or [].

    Revisited cells remain traversable: ruling them out could block a doorway.
    Unvisited frontier tiles are preferred as goals, followed by unvisited
    known-free tiles. With prefer_open_area, any frontier or unvisited free
    tile may win, allowing recovery by revisiting a distant frontier when
    nearby unvisited patches lead nowhere.
    Paths may cross old tiles, but never obstacles or unknown cells.
    """
    if width < 1 or height < 1 or resolution <= 0 or len(data) != width * height:
        return []
    grid = np.asarray(data, dtype=np.int16).reshape(height, width)
    free = (grid >= 0) & (grid <= 25)
    inflated = _inflate(grid >= 65, math.ceil(robot_radius / resolution))
    free &= ~inflated
    if search_bounds is not None:
        for rr, cc in np.argwhere(free):
            x, y = cell_to_world(rr, cc, resolution, origin)
            if not _inside_search_bounds(
                    x, y, search_bounds, wall_margin_m, bounds_transform):
                free[rr, cc] = False
    if avoid_tiles:
        avoid_tiles = set(avoid_tiles)
        for rr, cc in np.argwhere(free):
            x, y = cell_to_world(rr, cc, resolution, origin)
            tile = (math.floor(x / tile_size), math.floor(y / tile_size))
            if tile in avoid_tiles:
                free[rr, cc] = False
    row, col = world_to_cell(*robot_xy, resolution, origin)
    if not (0 <= row < height and 0 <= col < width):
        return []
    if not free[row, col]:
        options = []
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                rr, cc = row + dy, col + dx
                if 0 <= rr < height and 0 <= cc < width and free[rr, cc]:
                    options.append((dx * dx + dy * dy, rr, cc))
        if not options:
            return []
        _, row, col = min(options)

    frontiers = _frontiers(grid, free)
    unknown_integral = None
    gain_radius_cells = 0
    if information_gain_weight > 0:
        unknown_integral = np.pad(
            (grid < 0).astype(np.int32), ((1, 0), (1, 0))
        ).cumsum(axis=0).cumsum(axis=1)
        gain_radius_cells = max(1, math.ceil(gain_radius_m / resolution))
    visited_tiles = set(visited_tiles)
    sector_visits = [0, 0, 0, 0]
    if sector_center is not None and sector_balance_weight > 0:
        center_x, center_y = sector_center
        for tx, ty in visited_tiles:
            tile_x = (tx + 0.5) * tile_size
            tile_y = (ty + 0.5) * tile_size
            sector = 2 * (tile_x >= center_x) + (tile_y >= center_y)
            sector_visits[sector] += 1
    visit_density = {}
    neighborhood_radius_tiles = max(1, math.ceil(2.0 / tile_size))
    goal_penalties = goal_penalties or {}
    start = row * width + col
    parent = {start: -1}
    queue = deque([start])
    distance = {start: 0}
    best = [None, None, None]
    best_score = [float('inf')] * 3
    while queue:
        index = queue.popleft()
        rr, cc = divmod(index, width)
        steps = distance[index]
        if free[rr, cc]:
            x, y = cell_to_world(rr, cc, resolution, origin)
            tile = (math.floor(x / tile_size), math.floor(y / tile_size))
            if math.hypot(x - robot_xy[0], y - robot_xy[1]) >= min_goal_distance_m:
                if prefer_open_area and (frontiers[rr, cc] or
                                         tile not in visited_tiles):
                    group = 0
                elif frontiers[rr, cc] and tile not in visited_tiles:
                    group = 0
                elif tile not in visited_tiles:
                    group = 1
                elif frontiers[rr, cc]:
                    group = 2
                else:
                    group = None
                if group is not None and goal_penalties.get(tile, 0.0) < 12.0:
                    score = (distance_weight * steps * resolution +
                             goal_penalties.get(tile, 0.0))
                    if sector_center is not None and sector_balance_weight > 0:
                        sector = (2 * (x >= sector_center[0]) +
                                  (y >= sector_center[1]))
                        score += sector_balance_weight * min(
                            sector_visits[sector], 60)
                    if nearby_visit_weight > 0:
                        if tile not in visit_density:
                            visit_density[tile] = sum(
                                abs(tx - tile[0]) <= neighborhood_radius_tiles and
                                abs(ty - tile[1]) <= neighborhood_radius_tiles
                                for tx, ty in visited_tiles
                            )
                        score += nearby_visit_weight * min(
                            visit_density[tile], 20
                        )
                    if unknown_integral is not None and frontiers[rr, cc]:
                        r0 = max(0, rr - gain_radius_cells)
                        r1 = min(height - 1, rr + gain_radius_cells)
                        c0 = max(0, cc - gain_radius_cells)
                        c1 = min(width - 1, cc + gain_radius_cells)
                        unknown_cells = (
                            unknown_integral[r1 + 1, c1 + 1]
                            - unknown_integral[r0, c1 + 1]
                            - unknown_integral[r1 + 1, c0]
                            + unknown_integral[r0, c0]
                        )
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        score -= information_gain_weight * unknown_cells / area
                    if score < best_score[group]:
                        best[group], best_score[group] = index, score
        for dr, dc in NEIGHBORS:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < height and 0 <= nc < width and free[nr, nc]):
                continue
            neighbor = nr * width + nc
            if neighbor not in parent:
                parent[neighbor] = index
                distance[neighbor] = steps + 1
                queue.append(neighbor)

    selected = next((item for item in best if item is not None), None)
    if selected is None:
        return []
    cells = []
    while selected != start:
        cells.append(divmod(selected, width))
        selected = parent[selected]
    cells.reverse()
    return [cell_to_world(r, c, resolution, origin) for r, c in cells]


def plan_sensor_probe(
    data, width, height, resolution, origin, robot_pose, scan_clearance,
    visited_tiles=(), tile_size=0.5, goal_penalties=None,
    search_bounds=None, wall_margin_m=1.2, bounds_transform=None,
):
    """Choose a short LiDAR-clear goal toward unvisited or unknown map space.

    This is a fallback when a conservative known-free frontier cannot be
    reached. It uses only the current scan, SLAM map, and robot visit history;
    ground-truth object coordinates are never considered.
    """
    if width < 1 or height < 1 or resolution <= 0 or len(data) != width * height:
        return None
    grid = np.asarray(data, dtype=np.int16).reshape(height, width)
    visited_tiles = set(visited_tiles)
    goal_penalties = goal_penalties or {}
    x, y, yaw = robot_pose
    gain_radius_cells = max(1, math.ceil(2.0 / resolution))
    unknown_integral = np.pad(
        (grid < 0).astype(np.int32), ((1, 0), (1, 0))
    ).cumsum(axis=0).cumsum(axis=1)
    options = []
    for angle in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05,
                  1.4, -1.4, 1.75, -1.75, 2.1, -2.1, 2.5, -2.5, math.pi):
        clearance = scan_clearance(angle, 0.22)
        if math.isnan(clearance) or clearance < 0.85:
            continue
        for distance in (1.4, 1.0, 0.7):
            if clearance < distance + 0.35:
                continue
            gx = x + distance * math.cos(yaw + angle)
            gy = y + distance * math.sin(yaw + angle)
            if search_bounds is not None:
                if not _inside_search_bounds(
                        gx, gy, search_bounds, wall_margin_m,
                        bounds_transform):
                    continue
            row, col = world_to_cell(gx, gy, resolution, origin)
            if not (0 <= row < height and 0 <= col < width):
                continue
            occupancy = grid[row, col]
            tile = (math.floor(gx / tile_size), math.floor(gy / tile_size))
            if goal_penalties.get(tile, 0.0) >= 12.0:
                continue
            neighborhood_radius_tiles = max(1, math.ceil(2.0 / tile_size))
            nearby_visited = sum(
                abs(tx - tile[0]) <= neighborhood_radius_tiles and
                abs(ty - tile[1]) <= neighborhood_radius_tiles
                for tx, ty in visited_tiles
            )
            r0 = max(0, row - gain_radius_cells)
            r1 = min(height - 1, row + gain_radius_cells)
            c0 = max(0, col - gain_radius_cells)
            c1 = min(width - 1, col + gain_radius_cells)
            unknown_count = (
                unknown_integral[r1 + 1, c1 + 1]
                - unknown_integral[r0, c1 + 1]
                - unknown_integral[r1 + 1, c0]
                + unknown_integral[r0, c0]
            )
            unknown_fraction = unknown_count / ((r1 - r0 + 1) * (c1 - c0 + 1))
            score = (
                6.0 * unknown_fraction +
                (2.0 if tile not in visited_tiles else 0.0) +
                0.8 * min(clearance, 3.0) +
                0.2 * distance -
                0.15 * abs(angle) -
                (1.0 if occupancy >= 50 else 0.0) -
                0.15 * min(nearby_visited, 20) -
                goal_penalties.get(tile, 0.0)
            )
            options.append((score, gx, gy))
    if not options:
        return None
    _, gx, gy = max(options)
    return gx, gy
