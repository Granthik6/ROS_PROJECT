"""Summarize paired, policy-only Gazebo object-discovery trials."""

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(sys.argv[1]).resolve()
    algorithms = ('ddqn', 'ppo', 'random')
    seeds = (42, 43, 44)
    rows = []
    for seed in seeds:
        for algorithm in algorithms:
            gazebo_log = root / f'seed{seed}_{algorithm}_gazebo.log'
            if ('Unable to start server' in gazebo_log.read_text(
                    encoding='utf-8', errors='replace')):
                raise ValueError(
                    f'Gazebo did not start a fresh server: {gazebo_log}')
            path = root / f'seed{seed}_{algorithm}' / 'summary.json'
            summary = json.loads(path.read_text(encoding='utf-8'))
            if (summary['algorithm'] != algorithm or summary['seed'] != seed
                    or summary['target_objects'] != 36
                    or summary['guided_completion_enabled']
                    or summary['global_coverage_enabled']
                    or summary['inspection_sweep_events'] != 0
                    or summary['safety_recovery']):
                raise ValueError(f'Incomparable trial: {path}')
            with (path.parent / 'steps.csv').open(newline='',
                                                  encoding='utf-8') as stream:
                steps = list(csv.DictReader(stream))
            if len(steps) != summary['steps'] or any(
                    step['action_source'] != 'policy' for step in steps):
                raise ValueError(f'Action log is incomplete or assisted: {path}')
            actions = Counter(int(step['action']) for step in steps)
            rows.append({
                'algorithm': algorithm,
                'seed': seed,
                'matched_objects': summary['matched_objects'],
                'elapsed_minutes': summary['elapsed_minutes'],
                'time_limit_minutes': summary['time_limit_minutes'],
                'total_reward': summary['total_reward'],
                'steps': summary['steps'],
                'forward_actions': actions[0],
                'left_actions': actions[1],
                'right_actions': actions[2],
                'visited_cells': summary['unique_visited_cells'],
                'status': summary['status'],
                'summary_file': str(path),
            })

    durations = {row['time_limit_minutes'] for row in rows}
    if len(durations) != 1:
        raise ValueError(f'Trials have unequal time limits: {durations}')
    duration = durations.pop()

    with (root / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for algorithm in algorithms:
        subset = [row for row in rows if row['algorithm'] == algorithm]
        counts = [row['matched_objects'] for row in subset]
        aggregate[algorithm] = {
            'runs': len(subset),
            'mean_objects': round(statistics.mean(counts), 3),
            'stdev_objects': round(statistics.stdev(counts), 3),
            'min_objects': min(counts),
            'max_objects': max(counts),
            'mean_reward': round(statistics.mean(
                row['total_reward'] for row in subset), 3),
            'mean_elapsed_minutes': round(statistics.mean(
                row['elapsed_minutes'] for row in subset), 3),
            'mean_visited_cells': round(statistics.mean(
                row['visited_cells'] for row in subset), 3),
        }

    report = {
        'design': {
            'algorithms': algorithms,
            'seeds': seeds,
            'fixed_duration_minutes': duration,
            'world': '/root/ros2_ws/src/robot_tasks/worlds/spaced_objects_completion.world',
            'random_policy': 'uniform',
            'guided_completion': False,
            'global_coverage_assist': False,
            'inspection_sweeps': False,
            'safety_action_overrides': False,
            'trained_policy_action_selection': 'deterministic',
        },
        'aggregate': aggregate,
        'trials': rows,
    }
    with (root / 'comparison_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')

    colors = {'ddqn': '#2474a6', 'ppo': '#e18b3c', 'random': '#709458'}
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for index, algorithm in enumerate(algorithms):
        subset = [row for row in rows if row['algorithm'] == algorithm]
        mean = aggregate[algorithm]['mean_objects']
        axis.bar(index, mean, color=colors[algorithm], alpha=0.7, width=0.6)
        for offset, row in zip((-0.16, 0, 0.16), subset):
            axis.scatter(index + offset, row['matched_objects'], color='black',
                         s=25, zorder=3)
        axis.text(index, mean + 3.0, f'{mean:.1f}', ha='center', va='bottom')
    axis.set_xticks(range(len(algorithms)), [name.upper() for name in algorithms])
    axis.set_ylabel(f'World objects detected after {duration:g} minutes (of 36)')
    axis.set_ylim(0, 38)
    axis.set_title('Object discovery without guidance or action overrides')
    axis.grid(axis='y', alpha=0.25)
    axis.set_axisbelow(True)
    figure.savefig(root / 'objects_found_comparison.png', dpi=180)
    plt.close(figure)

    for algorithm in algorithms:
        print(f"{algorithm.upper()}: {aggregate[algorithm]['mean_objects']:.2f}/36 "
              f"mean, range {aggregate[algorithm]['min_objects']}-"
              f"{aggregate[algorithm]['max_objects']} ({len(seeds)} seeds)")
    print(f'Results: {root}')


if __name__ == '__main__':
    main()
