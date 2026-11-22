#!/usr/bin/env python3
"""Summarize only saved ROS/Gazebo DDQN trial results; never infer runs."""

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
    root = series_dir.parent
    rows = []
    for trial in range(1, 18):
        pointer = series_dir / f'trial_{trial:02d}.run_id'
        if not pointer.exists():
            continue
        run_id = pointer.read_text(encoding='utf-8').strip()
        result = root / run_id / 'summary.json'
        if not result.exists():
            continue
        data = json.loads(result.read_text(encoding='utf-8'))
        expected_seed = 41 + trial
        if (data.get('algorithm') != 'ddqn' or
                data.get('target_objects') != 26 or
                data.get('seed') != expected_seed or
                data.get('policy') != 'saved_model_deterministic' or
                data.get('control_mode') != 'policy_plus_global_coverage_assist' or
                data.get('guided_completion_enabled') is not False or
                data.get('model') != str(root.parent / 'rl_runs' /
                                         'ddqn_safe_50000/model_final.zip') or
                data.get('detector_model') != str(root.parent /
                    'realistic_sim_yolo_runs/all36_balanced_finetune_20261115/weights/best.pt') or
                not str(data.get('world_file', '')).endswith(
                    '/spaced_objects_reliable_26_v4.world')):
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
    completed = len(rows)
    all_found = [r for r in rows if r['all_objects_detected']]
    report = {
        'requested_trials': 17,
        'completed_trials': completed,
        'all_26_detected_trials': len(all_found),
        'mean_matched_objects': (statistics.mean(r['matched_objects'] for r in rows)
                                 if rows else None),
        'mean_elapsed_minutes': (statistics.mean(r['elapsed_minutes'] for r in rows)
                                 if rows else None),
        'mean_time_to_all_minutes': (
            statistics.mean(r['time_to_all_minutes'] for r in all_found)
            if all_found else None),
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
        ax.set(xlabel='Trial', ylabel='Objects detected', title='DDQN: 26-object discovery')
        ax.set_xticks(range(1, 18))
        ax.set_ylim(0, 27)
        ax.axhline(26, color='#263d5b', linewidth=1, linestyle='--')
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        fig.savefig(series_dir / 'objects_per_trial.png', dpi=160)
        plt.close(fig)
    print(f'Recorded {completed}/17 trials; {len(all_found)} detected 26/26')


if __name__ == '__main__':
    main()
