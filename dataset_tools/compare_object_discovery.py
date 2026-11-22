"""Summarize equal-duration DDQN, PPO, and random 36-object runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('runs', nargs=3, type=Path)
    args = parser.parse_args()

    data = {}
    for path in args.runs:
        summary = json.loads((path / 'summary.json').read_text())
        name = summary['algorithm']
        if name in data:
            parser.error(f'Duplicate algorithm: {name}')
        data[name] = (path, summary)
    if set(data) != {'ddqn', 'ppo', 'random'}:
        parser.error('Provide one DDQN, one PPO, and one random run')

    summaries = [data[name][1] for name in ('ddqn', 'ppo', 'random')]
    for key in ('world_file', 'target_objects', 'seed', 'action_duration_s',
                'safety_recovery'):
        if len({str(row[key]) for row in summaries}) != 1:
            parser.error(f'Runs do not share {key}')
    if all('environment_profile' in row for row in summaries):
        if len({row['environment_profile'] for row in summaries}) != 1:
            parser.error('Runs use different environment profiles')
    if all('detector_model' in row for row in summaries):
        if len({row['detector_model'] for row in summaries}) != 1:
            parser.error('Runs use different detector models')
    if any(row['status'] == 'interrupted' for row in summaries):
        parser.error('At least one run was interrupted')
    if all('time_limit_minutes' in row for row in summaries):
        if len({row['time_limit_minutes'] for row in summaries}) != 1:
            parser.error('Runs have different time limits')
    else:
        unfinished = [row['elapsed_s'] for row in summaries
                      if not row['all_objects_detected']]
        if unfinished and max(unfinished) - min(unfinished) > 5:
            parser.error('Unfinished runs have different time budgets')

    args.output.mkdir(parents=True, exist_ok=True)
    fields = ('algorithm', 'target_objects', 'matched_objects',
              'all_objects_detected', 'elapsed_minutes', 'time_to_all_minutes',
              'steps', 'total_reward', 'mean_reward_per_step',
              'safety_interventions', 'status')
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields}
                         for row in summaries)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                                constrained_layout=True)
    count_axis, reward_axis = axes
    colors = {'ddqn': '#1f77b4', 'ppo': '#ff7f0e', 'random': '#2ca02c'}
    for name in ('ddqn', 'ppo', 'random'):
        path, _ = data[name]
        with (path / 'world_discovery_curve.csv').open(newline='') as stream:
            points = list(csv.DictReader(stream))
        count_axis.step([float(p['elapsed_s']) / 60 for p in points],
                        [int(p['matched_world_objects']) for p in points],
                        where='post', label=name.upper(), color=colors[name])
        with (path / 'steps.csv').open(newline='') as stream:
            steps = list(csv.DictReader(stream))
        reward_axis.plot([float(p['elapsed_s']) / 60 for p in steps],
                         [float(p['cumulative_reward']) for p in steps],
                         label=name.upper(), color=colors[name])
    count_axis.axhline(summaries[0]['target_objects'], color='black',
                       linestyle='--', label='All objects')
    count_axis.set(xlabel='Elapsed minutes', ylabel='Verified objects',
                   title='Confirmed object discovery',
                   ylim=(0, summaries[0]['target_objects'] + 1))
    reward_axis.set(xlabel='Elapsed minutes', ylabel='Cumulative reward',
                    title='Exploration reward')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(args.output / 'comparison.png', dpi=180)
    plt.close(figure)

    lines = [
        '# Timed object discovery comparison', '',
        f"World: `{summaries[0]['world_file']}`", '',
        '| Policy | Verified | Time run (min) | Time to all (min) | '
        'Total reward | Mean reward/step |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summaries:
        time_to_all = (f"{row['time_to_all_minutes']:.3f}"
                       if row['time_to_all_minutes'] is not None else
                       'not reached')
        lines.append(
            f"| {row['algorithm'].upper()} | "
            f"{row['matched_objects']}/{row['target_objects']} | "
            f"{row['elapsed_minutes']:.3f} | {time_to_all} | "
            f"{row['total_reward']:.3f} | "
            f"{row['mean_reward_per_step']:.4f} |"
        )
    lines.extend([
        '', 'Each row is one saved deterministic policy run or one seeded '
        'random run. A verified count requires two observations and a '
        'one-to-one class-and-position match to an object placed in the world.',
        f"LiDAR safety recovery was {'enabled' if summaries[0]['safety_recovery'] else 'disabled'} "
        'for every policy.',
        'The reward is the exploration environment reward; it is not the '
        'verified object count. A missing time-to-all value means the target '
        'was not reached within the run.',
    ])
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
