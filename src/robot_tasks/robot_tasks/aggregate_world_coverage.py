"""Audit object discoveries across independent autonomous Gazebo runs.

This is an evaluation tool. World coordinates are never given to the robot's
navigation policy; they are used only to check recorded camera observations.
"""

import argparse
import csv
import json
from pathlib import Path

from robot_tasks.world_coverage_report import (
    WORLD_FILE, cluster_world_observations, match_instances, read_world,
)


def build_aggregate(run_dirs, world_path=WORLD_FILE, radius_m=1.25,
                    min_confirmations=2):
    if not run_dirs:
        raise ValueError('At least one run is required')
    if radius_m <= 0:
        raise ValueError('Match radius must be positive')
    instances = read_world(world_path)
    first_matches = {}
    runs = []
    elapsed_before_run = 0.0
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        with (run_dir / 'summary.json').open(encoding='utf-8') as stream:
            summary = json.load(stream)
        duration = float(summary['duration_sec'])
        if duration < 0:
            raise ValueError(f'Negative run duration in {run_dir}')
        observations = cluster_world_observations(
            run_dir / 'observations.csv',
            min_confirmations=min_confirmations,
        )
        matched, _ = match_instances(instances, observations, radius_m)
        by_id = {item['id']: item for item in observations}
        for item in matched:
            if not item['detected'] or item['name'] in first_matches:
                continue
            detection = by_id[item['detection_id']]
            first_matches[item['name']] = {
                'world_name': item['name'],
                'class': item['class'],
                'source_run_id': summary['run_id'],
                'source_run_dir': str(run_dir.resolve()),
                'first_seen_s': item['first_seen_s'],
                'confirmed_s': item['confirmed_s'],
                'cumulative_confirmed_s': round(
                    elapsed_before_run + item['confirmed_s'], 3
                ),
                'best_confidence': detection['confidence'],
                'distance_error_m': item['distance_error_m'],
            }
        runs.append({
            'run_id': summary['run_id'],
            'run_dir': str(run_dir.resolve()),
            'status': summary['status'],
            'duration_sec': duration,
            'matched_world_objects': sum(item['detected'] for item in matched),
        })
        elapsed_before_run += duration

    objects = [
        first_matches.get(item['name'], {
            'world_name': item['name'],
            'class': item['class'],
            'source_run_id': '',
            'source_run_dir': '',
            'first_seen_s': '',
            'confirmed_s': '',
            'cumulative_confirmed_s': '',
            'best_confidence': '',
            'distance_error_m': '',
        }) for item in instances
    ]
    all_found = len(first_matches) == len(instances) and bool(instances)
    return {
        'scope': 'combined_independent_autonomous_runs',
        'world_file': str(Path(world_path).resolve()),
        'match_radius_m': radius_m,
        'min_confirmations': min_confirmations,
        'target_world_objects': len(instances),
        'matched_world_objects': len(first_matches),
        'all_objects_detected_across_runs': all_found,
        'all_objects_detected_in_one_run': any(
            run['matched_world_objects'] == len(instances) for run in runs
        ),
        'time_to_all_across_runs_s': (
            max(row['cumulative_confirmed_s'] for row in first_matches.values())
            if all_found else None
        ),
        'total_run_duration_s': round(elapsed_before_run, 3),
        'runs': runs,
        'objects': objects,
    }


def save_aggregate(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'combined_world_coverage.json'
    csv_path = output_dir / 'combined_world_objects.csv'
    with json_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    fields = (
        'world_name', 'class', 'source_run_id', 'source_run_dir',
        'first_seen_s', 'confirmed_s', 'cumulative_confirmed_s',
        'best_confidence', 'distance_error_m',
    )
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report['objects'])
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True,
                        help='Run directories in chronological order')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--world', default=WORLD_FILE)
    parser.add_argument('--match-radius-m', type=float, default=1.25)
    parser.add_argument('--min-confirmations', type=int, default=2)
    args = parser.parse_args()
    report = build_aggregate(
        args.runs, args.world, args.match_radius_m,
        args.min_confirmations,
    )
    json_path, csv_path = save_aggregate(report, args.output_dir)
    print(f"Matched {report['matched_world_objects']}/"
          f"{report['target_world_objects']} across {len(report['runs'])} runs")
    print(f"Single run complete: {report['all_objects_detected_in_one_run']}")
    print(f'JSON: {json_path}')
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
