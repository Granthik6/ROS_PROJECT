"""Compare confirmed detections with known Gazebo world instances."""

import argparse
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET


WORLD_FILE = '/root/ros2_ws/src/robot_tasks/worlds/realistic_room.world'
CLASS_ALIASES = {
    'plastic_cup': 'cup',
    'person_standing': 'person',
    'person_walking': 'person',
    'pcd_bottle': 'bottle',
    'pcd_doll': 'doll',
}
IGNORED_MODELS = {'sun', 'ground_plane'}


def read_world(path):
    world = ET.parse(path).getroot().find('world')
    if world is None:
        raise ValueError(f'No world element in {path}')
    instances = []
    for include in world.findall('include'):
        uri = include.findtext('uri', '').strip()
        model_type = uri.rsplit('/', 1)[-1]
        if model_type in IGNORED_MODELS:
            continue
        pose = include.findtext('pose', '').split()
        if len(pose) < 2:
            raise ValueError(f'Missing pose for model {uri}')
        instances.append({
            'name': include.findtext('name', model_type),
            'class': CLASS_ALIASES.get(model_type, model_type),
            'x': float(pose[0]),
            'y': float(pose[1]),
        })
    return instances


def search_bounds_from_world(path):
    """Read the four search walls without exposing target locations."""
    world = ET.parse(path).getroot().find('world')
    walls = {}
    for model in world.findall('model'):
        name = model.get('name', '')
        if name.startswith('search_boundary_'):
            pose = [float(value) for value in model.findtext('pose').split()]
            walls[name.removeprefix('search_boundary_')] = pose
    if set(walls) != {'south', 'north', 'west', 'east'}:
        return None
    return (walls['west'][0], walls['east'][0],
            walls['south'][1], walls['north'][1])


def read_detections(path):
    prefix = 'world' if os.path.basename(path) == 'world_unique_objects.csv' else 'map'
    with open(path, newline='', encoding='utf-8') as stream:
        return [{
            'id': row['object_id'],
            'class': row['class'],
            'x': float(row[f'{prefix}_x_m']),
            'y': float(row[f'{prefix}_y_m']),
            'confidence': float(row['best_confidence']),
            'first_seen_s': float(row['first_seen_s']),
            'confirmed_s': float(row.get('confirmed_s') or row['first_seen_s']),
        } for row in csv.DictReader(stream)]


def cluster_world_observations(path, min_confirmations=2, radius_m=0.75):
    """Group aligned observations without letting old clusters drift away.

    The confirmed center is held at the first confirming observations. A
    tight radius permits separate clusters for a real object and a nearby
    false positive; one-to-one world matching handles duplicate clusters.
    """
    clusters = []
    with open(path, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if not row.get('world_x_m') or not row.get('world_y_m'):
                continue
            x, y = float(row['world_x_m']), float(row['world_y_m'])
            elapsed = float(row['elapsed_s'])
            confidence = float(row['confidence'])
            frame_stamp = row.get('stamp_s') or None
            nearest = None
            nearest_distance = radius_m
            for obj in clusters:
                if obj['class'] != row['class']:
                    continue
                distance = math.hypot(obj['x'] - x, obj['y'] - y)
                if distance < nearest_distance:
                    nearest, nearest_distance = obj, distance
            if nearest is None:
                clusters.append({
                    'id': str(len(clusters) + 1),
                    'class': row['class'],
                    'x': x, 'y': y,
                    'confidence': confidence,
                    'first_seen_s': elapsed,
                    'confirmed_s': elapsed if min_confirmations == 1 else None,
                    'observations': 1,
                    'last_stamp': frame_stamp,
                })
                continue
            if frame_stamp is not None and nearest['last_stamp'] == frame_stamp:
                continue
            count = nearest['observations']
            if nearest['confirmed_s'] is None:
                nearest['x'] = (nearest['x'] * count + x) / (count + 1)
                nearest['y'] = (nearest['y'] * count + y) / (count + 1)
            nearest['confidence'] = max(nearest['confidence'], confidence)
            nearest['observations'] += 1
            nearest['last_stamp'] = frame_stamp
            if (nearest['confirmed_s'] is None and
                    nearest['observations'] >= min_confirmations):
                nearest['confirmed_s'] = elapsed
    return [obj for obj in clusters if obj['confirmed_s'] is not None]


def write_world_clusters(detections, run_dir):
    path = os.path.join(run_dir, 'world_unique_objects.csv')
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow((
            'object_id', 'class', 'world_x_m', 'world_y_m',
            'best_confidence', 'observations', 'first_seen_s', 'confirmed_s',
        ))
        for obj in detections:
            writer.writerow((
                obj['id'], obj['class'], f'{obj["x"]:.6f}',
                f'{obj["y"]:.6f}', f'{obj["confidence"]:.6f}',
                obj['observations'], f'{obj["first_seen_s"]:.3f}',
                f'{obj["confirmed_s"]:.3f}',
            ))
    return path


def match_instances(instances, detections, radius_m):
    """One-to-one class-and-position matching; raw counts never imply recall."""
    candidates = []
    for object_index, obj in enumerate(instances):
        for detection_index, detection in enumerate(detections):
            if obj['class'] != detection['class']:
                continue
            distance = math.hypot(
                obj['x'] - detection['x'], obj['y'] - detection['y']
            )
            if distance <= radius_m:
                candidates.append((distance, object_index, detection_index))
    matches = {}
    used_detections = set()
    for distance, object_index, detection_index in sorted(candidates):
        if object_index in matches or detection_index in used_detections:
            continue
        matches[object_index] = (detection_index, distance)
        used_detections.add(detection_index)

    results = []
    for index, obj in enumerate(instances):
        row = dict(obj)
        matched = matches.get(index)
        row['detected'] = matched is not None
        if matched is not None:
            detection = detections[matched[0]]
            row['detection_id'] = detection['id']
            row['distance_error_m'] = round(matched[1], 3)
            row['first_seen_s'] = detection['first_seen_s']
            row['confirmed_s'] = detection.get('confirmed_s',
                                               detection['first_seen_s'])
        results.append(row)
    unmatched = [
        detection for index, detection in enumerate(detections)
        if index not in used_detections
    ]
    return results, unmatched


def build_report(run_dir, world_path=WORLD_FILE, radius_m=1.25,
                 min_confirmations=2):
    if radius_m <= 0:
        raise ValueError('match radius must be positive')
    instances = read_world(world_path)
    world_detections = os.path.join(run_dir, 'world_unique_objects.csv')
    observation_path = os.path.join(run_dir, 'observations.csv')
    if os.path.exists(observation_path):
        aligned = cluster_world_observations(
            observation_path, min_confirmations=min_confirmations
        )
        if aligned:
            write_world_clusters(aligned, run_dir)
    scoring_frame = ('gazebo_world' if os.path.exists(world_detections)
                     else 'map_assumed_world')
    detections = read_detections(
        world_detections if scoring_frame == 'gazebo_world'
        else os.path.join(run_dir, 'unique_objects.csv')
    )
    results, unmatched = match_instances(instances, detections, radius_m)
    found = [row for row in results if row['detected']]
    return {
        'world_file': world_path,
        'run_dir': run_dir,
        'match_radius_m': radius_m,
        'scoring_frame': scoring_frame,
        'target_objects': len(instances),
        'matched_objects': len(found),
        'recall': round(len(found) / len(instances), 4) if instances else 0.0,
        'all_objects_detected': len(found) == len(instances) and bool(instances),
        'time_to_all_s': max((row['confirmed_s'] for row in found), default=None)
        if len(found) == len(instances) and instances else None,
        'confirmed_detection_count': len(detections),
        'unmatched_detections': unmatched,
        'objects': results,
    }


def write_discovery_curve(report, duration_s, run_dir):
    """Write an evaluation-only cumulative count of matched world objects."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = sorted(
        float(obj['confirmed_s'])
        for obj in report['objects'] if obj['detected']
    )
    points = [(0.0, 0)]
    points.extend((time_s, index) for index, time_s in enumerate(times, 1))
    end_s = max(float(duration_s), times[-1] if times else 0.0, 1.0)
    points.append((end_s, len(times)))

    csv_path = os.path.join(run_dir, 'world_discovery_curve.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('elapsed_s', 'matched_world_objects'))
        for time_s, count in points:
            writer.writerow((f'{time_s:.3f}', count))

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x, y = zip(*points)
    axis.step(x, y, where='post', label='Confirmed world objects')
    axis.axhline(
        report['target_objects'], color='#d95f02', linestyle='--',
        label=f"Target: {report['target_objects']}",
    )
    axis.set(
        xlabel='Elapsed time (s)', ylabel='Matched world objects',
        title='Autonomous object discovery over time',
        xlim=(0, end_s), ylim=(0, report['target_objects'] + 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc='lower right')
    png_path = os.path.join(run_dir, 'world_discovery_curve.png')
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path


def write_missing_object_audit(report, run_dir):
    """Show whether each miss is linked to coverage or detection evidence."""
    world_frame = report.get('scoring_frame') == 'gazebo_world'
    prefix = 'world' if world_frame else 'map'
    path_file = os.path.join(
        run_dir, 'world_robot_path.csv' if world_frame else 'robot_path.csv'
    )
    observation_file = os.path.join(run_dir, 'observations.csv')
    if os.path.exists(path_file):
        with open(path_file, newline='', encoding='utf-8') as stream:
            robot_path = [
                (float(row[f'{prefix}_x_m']), float(row[f'{prefix}_y_m']))
                for row in csv.DictReader(stream)
            ]
    else:
        robot_path = []
    with open(observation_file, newline='', encoding='utf-8') as stream:
        observations = list(csv.DictReader(stream))

    output_path = os.path.join(run_dir, 'missing_object_audit.csv')
    fields = (
        'world_name', 'class', 'world_x_m', 'world_y_m',
        'closest_robot_distance_m', 'same_class_nearby_observations',
        'best_nearby_confidence', 'same_class_observations_anywhere',
    )
    with open(output_path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for obj in report['objects']:
            if obj['detected']:
                continue
            robot_distance = min((
                math.hypot(x - obj['x'], y - obj['y'])
                for x, y in robot_path
            ), default=None)
            same_class = [
                row for row in observations if row['class'] == obj['class']
            ]
            nearby = [
                row for row in same_class
                if row.get(f'{prefix}_x_m') and row.get(f'{prefix}_y_m')
                and math.hypot(
                    float(row[f'{prefix}_x_m']) - obj['x'],
                    float(row[f'{prefix}_y_m']) - obj['y'],
                ) <= report['match_radius_m']
            ]
            writer.writerow({
                'world_name': obj['name'],
                'class': obj['class'],
                'world_x_m': obj['x'],
                'world_y_m': obj['y'],
                'closest_robot_distance_m': (
                    f'{robot_distance:.3f}' if robot_distance is not None else ''
                ),
                'same_class_nearby_observations': len(nearby),
                'best_nearby_confidence': (
                    f'{max(float(row["confidence"]) for row in nearby):.3f}'
                    if nearby else ''
                ),
                'same_class_observations_anywhere': len(same_class),
            })
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--world', default=WORLD_FILE)
    parser.add_argument('--match-radius-m', type=float, default=1.25)
    args = parser.parse_args()
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    summary_path = os.path.join(run_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as stream:
            summary = json.load(stream)
        duration_s = float(summary['duration_sec'])
        min_confirmations = int(summary.get('min_confirmations', 2))
    else:
        summary = None
        min_confirmations = 2
    report = build_report(
        run_dir, args.world, args.match_radius_m,
        min_confirmations=min_confirmations,
    )
    if summary is None:
        duration_s = max(
            (obj['confirmed_s'] for obj in report['objects']
             if obj['detected']),
            default=0.0,
        )
    write_discovery_curve(report, duration_s, run_dir)
    write_missing_object_audit(report, run_dir)
    path = os.path.join(run_dir, 'world_coverage_report.json')
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    if summary is not None:
        summary.update({
            'world_objects_total': report['target_objects'],
            'world_objects_matched': report['matched_objects'],
            'all_world_objects_found': report['all_objects_detected'],
            'time_to_all_world_objects_s': report['time_to_all_s'],
            'world_scoring_frame': report['scoring_frame'],
        })
        with open(summary_path, 'w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2)
            stream.write('\n')
        markdown_path = os.path.join(run_dir, 'summary.md')
        if os.path.exists(markdown_path):
            with open(markdown_path, encoding='utf-8') as stream:
                markdown = stream.read()
            markdown = re.sub(
                r'Confirmed world match: \*\*\d+/\d+ placed objects\*\*\.',
                f'Confirmed world match: **{report["matched_objects"]}/'
                f'{report["target_objects"]} placed objects**.',
                markdown,
            )
            with open(markdown_path, 'w', encoding='utf-8') as stream:
                stream.write(markdown)
    print(
        f"Matched {report['matched_objects']}/{report['target_objects']} "
        f"world objects. All found: {report['all_objects_detected']}."
    )
    if not report['all_objects_detected']:
        print('Still missing: ' + ', '.join(
            row['name'] for row in report['objects'] if not row['detected']
        ))
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
