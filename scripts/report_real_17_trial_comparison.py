#!/usr/bin/env python3
"""Summarize completed policy-only ROS 2 object-search trials from one experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


METHODS = ('ppo', 'ddqn', 'random')
COLORS = {'ppo': '#087f86', 'ddqn': '#cf8b29', 'random': '#8252a3'}
FIELDS = ('seed', 'algorithm', 'status', 'matched_objects', 'missing_objects',
          'elapsed_minutes', 'time_to_all_minutes', 'steps', 'total_reward',
          'mean_reward_per_step', 'global_coverage_intervention_steps',
          'inspection_sweep_actions', 'safety_interventions', 'trial_directory')


def load_rows(root: Path, manifest: dict) -> list[dict]:
    rows = []
    first = manifest['first_seed']
    last = first + manifest['number_of_seeds']
    for seed in range(first, last):
        for method in METHODS:
            directory = root / f'seed_{seed}_{method}'
            summary_path = directory / 'summary.json'
            if not summary_path.exists():
                continue
            with summary_path.open(encoding='utf-8') as stream:
                summary = json.load(stream)
            if summary['algorithm'] != method or summary['seed'] != seed:
                raise ValueError(f'Algorithm or seed mismatch: {summary_path}')
            if summary['target_objects'] != 36:
                raise ValueError(f'Wrong object target: {summary_path}')
            if summary['status'] not in ('all_objects_found', 'completed_time_limit'):
                continue
            if manifest['minutes_per_trial'] == 0 and (
                    summary['status'] != 'all_objects_found'
                    or summary['matched_objects'] != 36
                    or not summary['all_objects_detected']):
                raise ValueError(f'Unlimited trial did not finish at 36/36: {summary_path}')
            if summary.get('guided_completion_enabled'):
                raise ValueError(f'Run used known-object viewpoints: {summary_path}')
            if summary.get('global_coverage_enabled') != manifest['global_coverage']:
                raise ValueError(f'Coverage mode changed: {summary_path}')
            if summary.get('safety_recovery') != manifest['safety_action_overrides']:
                raise ValueError(f'Safety mode changed: {summary_path}')
            if manifest['control_mode'] == 'autonomous_coverage':
                if summary.get('inspection_sweep_every_m') != 0.5:
                    raise ValueError(f'Camera sweep setting changed: {summary_path}')
            elif summary.get('inspection_sweep_events') != 0:
                raise ValueError(f'Policy-only trial had camera sweeps: {summary_path}')
            if Path(summary['world_file']).resolve() != Path(manifest['world']).resolve():
                raise ValueError(f'World changed: {summary_path}')
            if Path(summary['detector_model']).resolve() != Path(manifest['detector']).resolve():
                raise ValueError(f'Detector changed: {summary_path}')
            expected_model = manifest['models'][method]
            if summary.get('model') != expected_model:
                raise ValueError(f'Model changed: {summary_path}')
            if not (root / f'seed_{seed}_{method}_gazebo.log').exists():
                raise ValueError(f'Missing Gazebo log for {summary_path}')
            if not (root / f'seed_{seed}_{method}_discovery.log').exists():
                raise ValueError(f'Missing discovery log for {summary_path}')
            object_file = directory / 'world_objects.csv'
            with object_file.open(newline='', encoding='utf-8') as stream:
                objects = list(csv.DictReader(stream))
            if len(objects) != 36 or len({x['name'] for x in objects}) != 36:
                raise ValueError(f'Invalid object inventory: {object_file}')
            matched = sum(x['detected'].lower() == 'true' for x in objects)
            if matched != summary['matched_objects']:
                raise ValueError(f'Object count mismatch: {object_file}')
            if matched == 36 and not summary['all_objects_detected']:
                raise ValueError(f'36/36 status mismatch: {summary_path}')
            if matched < 36 and summary['all_objects_detected']:
                raise ValueError(f'Incomplete 36/36 status mismatch: {summary_path}')
            missing = [x['name'] for x in objects if x['detected'].lower() != 'true']
            if set(missing) != set(summary['missing_objects']):
                raise ValueError(f'Missing-object list mismatch: {object_file}')
            rows.append({
                'seed': seed,
                'algorithm': method,
                'status': summary['status'],
                'matched_objects': matched,
                'missing_objects': missing,
                'elapsed_minutes': float(summary['elapsed_minutes']),
                'time_to_all_minutes': summary.get('time_to_all_minutes'),
                'steps': int(summary['steps']),
                'total_reward': float(summary['total_reward']),
                'mean_reward_per_step': float(summary['mean_reward_per_step']),
                'global_coverage_intervention_steps': int(summary.get('global_coverage_intervention_steps', 0)),
                'inspection_sweep_actions': int(summary.get('inspection_sweep_actions', 0)),
                'safety_interventions': int(summary.get('safety_interventions', 0)),
                'trial_directory': str(directory),
            })
    return rows


def write_csv(root: Path, rows: list[dict]) -> None:
    with (root / 'all_trials.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'missing_objects': ';'.join(row['missing_objects']),
                             'time_to_all_minutes': row['time_to_all_minutes'] or ''})


def write_json(root: Path, rows: list[dict], manifest: dict) -> dict:
    by_method = {}
    for method in METHODS:
        subset = [r for r in rows if r['algorithm'] == method]
        finished = [r for r in subset if r['matched_objects'] == 36]
        by_method[method] = {
            'recorded_trials': len(subset),
            'requested_trials': manifest['number_of_seeds'],
            'completed_36_of_36': len(finished),
            'mean_matched_objects': statistics.mean(r['matched_objects'] for r in subset)
                                    if subset else None,
            'mean_reward_per_step': statistics.mean(r['mean_reward_per_step'] for r in subset)
                                    if subset else None,
            'mean_time_to_all_minutes': statistics.mean(float(r['time_to_all_minutes'])
                                                       for r in finished) if finished else None,
            'mean_global_coverage_fraction': statistics.mean(
                r['global_coverage_intervention_steps'] / r['steps']
                for r in subset) if subset else None,
        }
    report = {
        'experiment_id': manifest['experiment_id'],
        'recorded_trials': len(rows),
        'requested_trials': manifest['number_of_seeds'] * len(METHODS),
        'minutes_per_trial': manifest['minutes_per_trial'],
        'methods': by_method,
    }
    (root / 'report_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def make_pdf(root: Path, rows: list[dict], report: dict, manifest: dict) -> None:
    out = root / 'report.pdf'
    temporary_out = root / 'report.pdf.tmp'
    navy = '#17364c'
    with PdfPages(temporary_out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        fig.text(.06, .94, 'PPO, DDQN and Random object-search trials',
                 fontsize=19, weight='bold', color=navy)
        limit_text = ('Each trial ends at 36/36; no time limit'
                      if manifest['minutes_per_trial'] == 0 else
                      f"{manifest['minutes_per_trial']:g}-minute cap per trial")
        fig.text(.06, .895, f"{manifest['experiment_id']}  |  "
                 f"{len(rows)}/{report['requested_trials']} recorded trials  |  "
                 f"{limit_text}",
                 fontsize=11, color='#506878')
        fig.text(.06, .855,
                 'Control: autonomous SLAM map coverage with camera sweeps; no known-object viewpoints'
                 if manifest['control_mode'] == 'autonomous_coverage' else
                 'Control: saved policy or uniform random actions only; no coverage controller',
                 fontsize=9.5, color='#506878')
        table_rows = []
        for method in METHODS:
            m = report['methods'][method]
            avg_objects = f"{m['mean_matched_objects']:.1f}" if m['mean_matched_objects'] is not None else '—'
            avg_reward = f"{m['mean_reward_per_step']:.3f}" if m['mean_reward_per_step'] is not None else '—'
            avg_time = f"{m['mean_time_to_all_minutes']:.2f}" if m['mean_time_to_all_minutes'] is not None else '—'
            avg_coverage = (f"{100 * m['mean_global_coverage_fraction']:.1f}%"
                            if m['mean_global_coverage_fraction'] is not None else '—')
            table_rows.append([method.upper(), f"{m['recorded_trials']}/{m['requested_trials']}",
                               f"{m['completed_36_of_36']}", avg_objects, avg_reward,
                               avg_time, avg_coverage])
        ax = fig.add_axes([.06, .625, .88, .18]); ax.axis('off')
        tab = ax.table(cellText=table_rows,
                       colLabels=['Method', 'Runs', '36/36', 'Mean found',
                                  'Reward/step', '36/36 time', 'Map steps'],
                       cellLoc='center', loc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.7)
        for (r, _), cell in tab.get_celld().items():
            cell.set_edgecolor('#cbd6de')
            if r == 0:
                cell.set_facecolor(navy); cell.get_text().set_color('white')
        ax = fig.add_axes([.10, .22, .77, .32])
        for method in METHODS:
            subset = [r for r in rows if r['algorithm'] == method]
            ax.plot([r['seed'] for r in subset], [r['matched_objects'] for r in subset],
                    'o-', label=method.upper(), color=COLORS[method], lw=1.6)
        ax.axhline(36, color='#a74938', ls='--', lw=1.2)
        ax.set(xlabel='Seed', ylabel='Matched objects', ylim=(0, 38))
        ax.grid(alpha=.25); ax.legend(frameon=False, ncol=3)
        fig.text(.06, .10,
                 'All runs use the same world, detector, seed schedule, and controller settings.',
                 fontsize=9.5, color='#506878')
        fig.text(.06, .065, 'Mean 36/36 time uses completed runs only; incomplete runs have no completion time.',
                 fontsize=9.5, color='#506878')
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        fig.text(.06, .94, 'Run-by-run results', fontsize=19, weight='bold', color=navy)
        ax = fig.add_axes([.06, .10, .88, .78]); ax.axis('off')
        matrix = []
        first = manifest['first_seed']
        for seed in range(first, first + manifest['number_of_seeds']):
            cells = [str(seed)]
            for method in METHODS:
                row = next((r for r in rows if r['seed'] == seed and r['algorithm'] == method), None)
                if row is None:
                    cells.append('pending')
                else:
                    cells.append(f"{row['matched_objects']}/36; {row['elapsed_minutes']:.2f}m")
            matrix.append(cells)
        tab = ax.table(cellText=matrix,
                       colLabels=['Seed', 'PPO', 'DDQN', 'Random'],
                       cellLoc='center', loc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(9)
        tab.scale(1, 1.35 if manifest['number_of_seeds'] >= 17 else 1.6)
        for (r, _), cell in tab.get_celld().items():
            cell.set_edgecolor('#cbd6de')
            if r == 0:
                cell.set_facecolor(navy); cell.get_text().set_color('white')
            elif r % 2 == 0:
                cell.set_facecolor('#f2f6f8')
        fig.text(.06, .055, 'Each entry gives matched objects and elapsed time; incomplete times are stop times.',
                 fontsize=9.5, color='#506878')
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        fig.text(.06, .94, 'Reward and map-controller activity by seed',
                 fontsize=19, weight='bold', color=navy)
        reward_ax = fig.add_axes([.10, .55, .77, .28])
        map_ax = fig.add_axes([.10, .16, .77, .28])
        for method in METHODS:
            subset = [r for r in rows if r['algorithm'] == method]
            xs = [r['seed'] for r in subset]
            reward_ax.plot(xs, [r['mean_reward_per_step'] for r in subset],
                           'o-', label=method.upper(), color=COLORS[method], lw=1.5)
            map_ax.plot(xs, [100 * r['global_coverage_intervention_steps'] / r['steps']
                             for r in subset],
                        'o-', label=method.upper(), color=COLORS[method], lw=1.5)
        reward_ax.set(ylabel='Mean reward per step', xlabel='Seed')
        map_ax.set(ylabel='Map-controller actions (%)', xlabel='Seed', ylim=(-3, 103))
        reward_ax.grid(alpha=.25); map_ax.grid(alpha=.25)
        reward_ax.legend(frameon=False, ncol=3)
        fig.text(.06, .08,
                 'A high map-controller percentage means that navigation actions mostly came from the shared planner.',
                 fontsize=9.3, color='#506878')
        pdf.savefig(fig); plt.close(fig)

        if rows:
            with (Path(rows[0]['trial_directory']) / 'world_objects.csv').open(
                    newline='', encoding='utf-8') as stream:
                inventory = list(csv.DictReader(stream))
            object_numbers = {item['name']: index for index, item in enumerate(inventory, start=1)}

            fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
            fig.text(.06, .94, '36-object inventory', fontsize=19, weight='bold', color=navy)
            fig.text(.06, .90, 'Missing-object IDs on the next pages refer to this inventory.',
                     fontsize=10, color='#506878')
            for column in range(2):
                items = inventory[column * 18:(column + 1) * 18]
                ax = fig.add_axes([.06 + column * .455, .12, .42, .72]); ax.axis('off')
                table_items = [[f'{column * 18 + i + 1:02d}', item['name'], item['class']]
                               for i, item in enumerate(items)]
                tab = ax.table(cellText=table_items,
                               colLabels=['ID', 'World object', 'Detector class'],
                               colWidths=[.10, .58, .32], cellLoc='left', loc='center')
                tab.auto_set_font_size(False); tab.set_fontsize(7.6); tab.scale(1, 1.45)
                for (r, _), cell in tab.get_celld().items():
                    cell.set_edgecolor('#cbd6de')
                    if r == 0:
                        cell.set_facecolor(navy); cell.get_text().set_color('white')
                    elif r % 2 == 0:
                        cell.set_facecolor('#f2f6f8')
            pdf.savefig(fig); plt.close(fig)

            for method in METHODS:
                fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
                fig.text(.06, .94, f'{method.upper()} trial details and missing objects',
                         fontsize=19, weight='bold', color=navy)
                fig.text(.06, .90,
                         'Missing IDs correspond to the inventory page; no IDs means 36/36.',
                         fontsize=10, color='#506878')
                for x, label in ((.065, 'Seed'), (.15, 'Found'), (.245, 'Min'),
                                 (.325, 'Map %'), (.405, 'Missing object IDs')):
                    fig.text(x, .85, label, fontsize=9, weight='bold', color=navy)
                for index, seed in enumerate(range(first, first + manifest['number_of_seeds'])):
                    row = next((r for r in rows if r['seed'] == seed and r['algorithm'] == method), None)
                    y = .82 - index * .042
                    if index % 2 == 0:
                        fig.add_artist(plt.Rectangle((.06, y - .021), .88, .04,
                                                     transform=fig.transFigure,
                                                     facecolor='#f2f6f8', edgecolor='none'))
                    fig.text(.065, y, str(seed), fontsize=8, color=navy)
                    if row is None:
                        fig.text(.15, y, 'pending', fontsize=8, color='#71818c')
                        continue
                    fig.text(.15, y, f"{row['matched_objects']}/36", fontsize=8, color=navy)
                    fig.text(.245, y, f"{row['elapsed_minutes']:.2f}", fontsize=8, color=navy)
                    fig.text(.325, y,
                             f"{100 * row['global_coverage_intervention_steps'] / row['steps']:.1f}",
                             fontsize=8, color=navy)
                    ids = [object_numbers[name] for name in row['missing_objects']]
                    if ids:
                        for line_number in range(0, len(ids), 17):
                            label = ' '.join(f'{n:02d}' for n in ids[line_number:line_number + 17])
                            fig.text(.405, y + .005 - .012 * (line_number // 17),
                                     label, fontsize=7.2, color=navy, va='top')
                    else:
                        fig.text(.405, y, 'None — complete', fontsize=8, color='#087f86')
                fig.text(.06, .06,
                         'Time is time to 36/36 for completed runs; otherwise it is the trial stop time.',
                         fontsize=9, color='#506878')
                pdf.savefig(fig); plt.close(fig)

            for seed in range(first, first + manifest['number_of_seeds']):
                seed_rows = [next((r for r in rows
                                   if r['seed'] == seed and r['algorithm'] == method), None)
                             for method in METHODS]
                if any(row is None for row in seed_rows):
                    continue
                fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
                fig.text(.06, .94, f'Seed {seed}: detection and reward traces',
                         fontsize=19, weight='bold', color=navy)
                discovery_ax = fig.add_axes([.10, .55, .77, .28])
                reward_ax = fig.add_axes([.10, .16, .77, .28])
                for method, row in zip(METHODS, seed_rows):
                    directory = Path(row['trial_directory'])
                    with (directory / 'world_discovery_curve.csv').open(
                            newline='', encoding='utf-8') as stream:
                        curve = list(csv.DictReader(stream))
                    with (directory / 'steps.csv').open(
                            newline='', encoding='utf-8') as stream:
                        steps = list(csv.DictReader(stream))
                    discovery_ax.step([float(item['elapsed_s']) / 60 for item in curve],
                                      [int(item['matched_world_objects']) for item in curve],
                                      where='post', lw=1.6, color=COLORS[method],
                                      label=method.upper())
                    reward_ax.plot([float(item['elapsed_s']) / 60 for item in steps],
                                   [float(item['cumulative_reward']) for item in steps],
                                   lw=1.4, color=COLORS[method], label=method.upper())
                discovery_ax.axhline(36, ls='--', lw=1, color='#a74938')
                discovery_ax.set(xlabel='Elapsed minutes', ylabel='Matched objects', ylim=(0, 38))
                reward_ax.set(xlabel='Elapsed minutes', ylabel='Cumulative reward')
                discovery_ax.grid(alpha=.25); reward_ax.grid(alpha=.25)
                discovery_ax.legend(frameon=False, ncol=3)
                fig.text(.06, .07,
                         'These are evaluation traces from this seed; they are not training curves.',
                         fontsize=9, color='#506878')
                pdf.savefig(fig); plt.close(fig)
    temporary_out.replace(out)
    print(out)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 scripts/report_real_17_trial_comparison.py EXPERIMENT_DIR')
    root = Path(sys.argv[1]).resolve()
    with (root / 'experiment.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    rows = load_rows(root, manifest)
    write_csv(root, rows)
    report = write_json(root, rows, manifest)
    make_pdf(root, rows, report, manifest)
    print(f"Recorded: {len(rows)}/{report['requested_trials']} trials")


if __name__ == '__main__':
    main()
