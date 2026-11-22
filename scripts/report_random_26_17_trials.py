#!/usr/bin/env python3
"""Summarize actual uniform-random 26-object ROS/Gazebo trial summaries."""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    series_dir = Path(sys.argv[1]).resolve()
    evaluations = series_dir.parent
    workspace = evaluations.parent
    rows = []
    for trial in range(1, 18):
        pointer = series_dir / f'trial_{trial:02d}.run_id'
        if not pointer.exists():
            continue
        run_id = pointer.read_text(encoding='utf-8').strip()
        result = evaluations / run_id / 'summary.json'
        if not result.exists():
            continue
        data = json.loads(result.read_text(encoding='utf-8'))
        expected_seed = 41 + trial
        if (data.get('algorithm') != 'random' or
                data.get('target_objects') != 26 or
                data.get('seed') != expected_seed or
                data.get('policy') != 'uniform_random_actions' or
                data.get('control_mode') != 'policy_only' or
                data.get('guided_completion_enabled') is not False or
                data.get('global_coverage_enabled') is not False or
                data.get('safety_recovery') is not False or
                data.get('inspection_sweep_every_m') != 0 or
                data.get('time_limit_minutes') != 12.0 or
                data.get('model') is not None or
                data.get('detector_model') != str(workspace /
                    'realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt') or
                data.get('world_file') != str(workspace / 'src/robot_tasks/worlds' /
                    'spaced_objects_reliable_26_v4.world')):
            raise ValueError(f'Incompatible trial summary: {result}')
        rows.append({
            'trial': trial,
            'seed': expected_seed,
            'run_id': run_id,
            'matched_objects': data['matched_objects'],
            'target_objects': data['target_objects'],
            'all_objects_detected': data['all_objects_detected'],
            'elapsed_minutes': data['elapsed_minutes'],
            'time_to_all_minutes': data['time_to_all_minutes'],
            'total_reward': data['total_reward'],
            'status': data['status'],
            'missing_objects': '; '.join(data['missing_objects']),
        })

    fields = ['trial', 'seed', 'run_id', 'matched_objects', 'target_objects',
              'all_objects_detected', 'elapsed_minutes', 'time_to_all_minutes',
              'total_reward', 'status', 'missing_objects']
    with (series_dir / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = [row for row in rows if row['all_objects_detected']]
    report = {
        'requested_trials': 17,
        'completed_trials': len(rows),
        'all_26_detected_trials': len(complete),
        'mean_matched_objects': (statistics.mean(r['matched_objects'] for r in rows)
                                 if rows else None),
        'mean_elapsed_minutes': (statistics.mean(r['elapsed_minutes'] for r in rows)
                                 if rows else None),
        'mean_time_to_all_minutes': (
            statistics.mean(r['time_to_all_minutes'] for r in complete)
            if complete else None),
        'mean_total_reward': (statistics.mean(r['total_reward'] for r in rows)
                              if rows else None),
        'trials': rows,
    }
    (series_dir / 'summary.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')

    if rows:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar([r['trial'] for r in rows], [r['matched_objects'] for r in rows],
               color=['#187a69' if r['all_objects_detected'] else '#d59034'
                      for r in rows])
        ax.set(xlabel='Trial', ylabel='Objects detected',
               title='Uniform random exploration: 26-object discovery')
        ax.set_xticks(range(1, 18))
        ax.set_ylim(0, 27)
        ax.axhline(26, color='#263d5b', linewidth=1, linestyle='--')
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        fig.savefig(series_dir / 'objects_per_trial.png', dpi=160)
        plt.close(fig)
    print(f'Recorded {len(rows)}/17 trials; {len(complete)} detected 26/26')


if __name__ == '__main__':
    main()
