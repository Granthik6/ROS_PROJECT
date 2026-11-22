import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = os.path.expanduser('~/ros2_ws/confidence_distance.csv')
OUT_PATH = os.path.expanduser('~/ros2_ws/confidence_vs_distance.png')

data = defaultdict(list)

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['detected'] != 'True':
            continue
        data[row['class']].append((float(row['true_distance_m']), float(row['confidence'])))

fig, ax = plt.subplots(figsize=(8, 6))

colors = {
    'cup': '#d62728', 'bowl': '#1f77b4', 'coke_can': '#8c564b',
    'cordless_drill': '#ff7f0e', 'hammer': '#7f7f7f',
}

for cls, points in data.items():
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, 'o-', label=cls, color=colors.get(cls), markersize=6)

ax.set_xlabel('True distance from camera to object (m)')
ax.set_ylabel('YOLO detection confidence')
ax.set_title('Realistic_sim YOLO: Confidence vs Distance\n(realistic_sim_yolo11n, per-class)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print('Saved plot to', OUT_PATH)
