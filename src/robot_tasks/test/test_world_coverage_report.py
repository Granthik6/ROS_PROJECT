"""Ensure a high detection count cannot masquerade as complete coverage."""

from types import SimpleNamespace
import csv
import math

from robot_tasks.world_coverage_report import (
    build_report, cluster_world_observations, match_instances, read_world,
)
from robot_tasks.world_frame import detection_in_world


def test_world_target_list_counts_instances_and_aliases():
    world = read_world('src/robot_tasks/worlds/realistic_room.world')
    assert len(world) == 34
    assert sum(item['class'] == 'person' for item in world) == 2
    assert sum(item['class'] == 'cup' for item in world) == 1


def test_duplicate_and_wrong_class_detections_do_not_complete_world():
    objects = [
        {'name': 'can', 'class': 'coke_can', 'x': 1.0, 'y': 2.0},
        {'name': 'cup', 'class': 'cup', 'x': 4.0, 'y': 2.0},
    ]
    detections = [
        {'id': '1', 'class': 'coke_can', 'x': 1.1, 'y': 2.0,
         'first_seen_s': 2.0},
        {'id': '2', 'class': 'coke_can', 'x': 1.2, 'y': 2.0,
         'first_seen_s': 3.0},
        {'id': '3', 'class': 'bowl', 'x': 4.0, 'y': 2.0,
         'first_seen_s': 4.0},
    ]
    results, unmatched = match_instances(objects, detections, 0.5)
    assert [item['detected'] for item in results] == [True, False]
    assert len(unmatched) == 2


def test_odom_detection_is_aligned_using_robot_world_pose():
    def pose(x, y, yaw):
        return SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(
                w=math.cos(yaw / 2), x=0.0, y=0.0,
                z=math.sin(yaw / 2),
            ),
        )

    point = detection_in_world(
        (3.0, 2.0, 0.3), pose(2.0, 2.0, 0.0),
        pose(4.0, 5.0, math.pi / 2),
    )
    assert math.isclose(point[0], 4.0, abs_tol=1e-6)
    assert math.isclose(point[1], 6.0, abs_tol=1e-6)
    assert math.isclose(point[2], 0.3, abs_tol=1e-6)


def test_world_scoring_prefers_aligned_coordinates(tmp_path):
    columns = ('object_id', 'class', 'map_x_m', 'map_y_m',
               'world_x_m', 'world_y_m', 'best_confidence',
               'first_seen_s', 'confirmed_s')
    row = dict(zip(columns, ('1', 'cabinet', '6.0', '2.0',
                             '4.5', '3.8', '0.9', '5.0', '6.0')))
    for filename in ('unique_objects.csv', 'world_unique_objects.csv'):
        with (tmp_path / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow(row)
    report = build_report(str(tmp_path))
    assert report['scoring_frame'] == 'gazebo_world'
    assert any(obj['name'] == 'cabinet_main' and obj['detected']
               for obj in report['objects'])


def test_confirmed_world_location_does_not_drift_with_later_sightings(tmp_path):
    path = tmp_path / 'observations.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'elapsed_s', 'class', 'confidence', 'world_x_m', 'world_y_m',
        ))
        writer.writeheader()
        for index, x in enumerate((4.5, 4.6, 5.0, 5.3, 5.6, 5.9)):
            writer.writerow({
                'elapsed_s': index, 'class': 'cabinet',
                'confidence': 0.8, 'world_x_m': x, 'world_y_m': 3.8,
            })
    clusters = cluster_world_observations(path)
    assert any(abs(obj['x'] - 4.55) < 0.01 for obj in clusters)


def test_two_boxes_in_one_frame_do_not_confirm_an_object(tmp_path):
    path = tmp_path / 'observations.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'elapsed_s', 'class', 'confidence', 'world_x_m', 'world_y_m',
            'stamp_s',
        ))
        writer.writeheader()
        for stamp in ('1.0', '1.0', '2.0'):
            writer.writerow({
                'elapsed_s': stamp, 'class': 'saucepan',
                'confidence': 0.05, 'world_x_m': -4.83,
                'world_y_m': -0.45, 'stamp_s': stamp,
            })
    clusters = cluster_world_observations(path)
    assert len(clusters) == 1
    assert clusters[0]['observations'] == 2
    assert clusters[0]['confirmed_s'] == 2.0
