#!/usr/bin/env python3
"""Show the latest saved progress of the 17-trial random exploration run."""

import argparse
import csv
import json
import time
from pathlib import Path


ROOT = Path('/root/ros2_ws/rl_evaluations')
SERIES = ROOT / 'random_26_17_20261122'


def status_line() -> str:
    pointers = sorted(SERIES.glob('trial_*.run_id'))
    if not pointers:
        return 'No random trial has started yet.'
    pointer = pointers[-1]
    trial = int(pointer.stem.split('_')[1])
    run_id = pointer.read_text(encoding='utf-8').strip()
    run_dir = ROOT / run_id
    final = run_dir / 'summary.json'
    if final.exists():
        result = json.loads(final.read_text(encoding='utf-8'))
        return (f"Trial {trial}/17 complete: {result['matched_objects']}/26 "
                f"objects in {result['elapsed_minutes']:.2f} min")

    steps = run_dir / 'steps.csv'
    if not steps.exists():
        return f'Trial {trial}/17 is starting; no steps recorded yet.'
    latest = None
    with steps.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row.get('matched_world_objects') and row.get('elapsed_s'):
                latest = row
    if latest is None:
        return f'Trial {trial}/17 is starting; no steps recorded yet.'
    elapsed_min = float(latest['elapsed_s']) / 60
    return (f"Trial {trial}/17: {latest['matched_world_objects']}/26 "
            f"objects detected after {elapsed_min:.2f} min "
            f"(step {latest['step']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Refresh every 2 seconds')
    args = parser.parse_args()
    if not args.watch:
        print(status_line())
        return
    try:
        while True:
            print('\r' + status_line().ljust(100), end='', flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
