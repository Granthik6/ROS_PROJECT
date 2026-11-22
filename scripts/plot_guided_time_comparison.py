"""Plot actual time to 36 detections in the three guided runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('/root/ros2_ws')
OUT = ROOT / 'rl_evaluations/guided_time_to_36_20261111.png'
RUNS = {
    'DDQN': ROOT / 'rl_evaluations/ddqn_guided_fresh_20261111/summary.json',
    'PPO': ROOT / 'rl_evaluations/ppo_guided_fresh_20261111/summary.json',
    'Random': ROOT / 'rl_evaluations/random_guided_fresh_20261111/summary.json',
}
rows = []
for name, path in RUNS.items():
    result = json.loads(path.read_text(encoding='utf-8'))
    if (result['status'] != 'all_objects_found_with_guided_completion'
            or result['target_objects'] != 36
            or result['matched_objects'] != 36
            or result['seed'] != 42
            or result['guided_completion_after_stall_minutes'] != 0.5):
        raise ValueError(f'Run does not match comparison design: {path}')
    rows.append((name, result['time_to_all_minutes'],
                 result['pre_guidance_matched_objects'],
                 result['guided_completion_added_objects']))
rows.sort(key=lambda row: row[1])

colors = {'PPO': '#e18b3c', 'DDQN': '#2474a6', 'Random': '#709458'}
figure, axis = plt.subplots(figsize=(8.6, 4.6))
figure.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.23)
for index, (name, minutes, before, guided) in enumerate(rows):
    axis.barh(index, minutes, color=colors[name], height=0.58)
    axis.text(minutes + 0.06, index,
              f'{minutes:.2f} min  |  {before} before guidance + {guided} guided',
              va='center', fontsize=10)
axis.set_yticks(range(len(rows)), [row[0] for row in rows])
axis.invert_yaxis()
axis.set_xlim(0, max(row[1] for row in rows) + 2.3)
axis.set_xlabel('Minutes to confirmed 36/36 detections')
axis.set_title('Time to 36 objects with guided completion (one run each)')
axis.grid(axis='x', alpha=0.25)
axis.set_axisbelow(True)
figure.text(0.5, 0.03,
            'After exploration stalls, Gazebo moves the robot to audited camera views of missing objects.',
            ha='center', fontsize=8, color='#555555')
figure.savefig(OUT, dpi=180)
plt.close(figure)
print(OUT)
