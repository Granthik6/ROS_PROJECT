import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = '/root/ros2_ws/results/distance_confidence.csv'
DEFAULT_OUTPUT = '/root/ros2_ws/results/distance_vs_confidence.png'


def read_samples(input_csv, target_class):
    distances = []
    confidences = []

    with open(input_csv, newline='', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        required = {'class', 'confidence', 'distance_m'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                'CSV is missing required columns: ' + ', '.join(sorted(missing))
            )

        for row in reader:
            if target_class and row['class'] != target_class:
                continue

            try:
                distance = float(row['distance_m'])
                confidence = float(row['confidence'])
            except (TypeError, ValueError):
                continue

            if np.isfinite(distance) and np.isfinite(confidence):
                distances.append(distance)
                confidences.append(confidence)

    return np.asarray(distances), np.asarray(confidences)


def binned_curve(distances, confidences, bin_width):
    start = np.floor(distances.min() / bin_width) * bin_width
    stop = np.ceil(distances.max() / bin_width) * bin_width + bin_width
    edges = np.arange(start, stop + bin_width * 0.5, bin_width)
    if len(edges) < 2:
        edges = np.asarray([start, start + bin_width])

    indices = np.digitize(distances, edges, right=False) - 1
    centers = []
    means = []
    deviations = []
    counts = []

    for index in range(len(edges) - 1):
        values = confidences[indices == index]
        if len(values) == 0:
            continue
        centers.append((edges[index] + edges[index + 1]) / 2.0)
        means.append(float(np.mean(values)))
        deviations.append(float(np.std(values)))
        counts.append(len(values))

    return (
        np.asarray(centers),
        np.asarray(means),
        np.asarray(deviations),
        np.asarray(counts),
    )


def write_summary(path, centers, means, deviations, counts):
    with open(path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            'distance_bin_center_m',
            'mean_confidence',
            'std_confidence',
            'sample_count',
        ])
        for row in zip(centers, means, deviations, counts):
            writer.writerow([
                f'{row[0]:.6f}',
                f'{row[1]:.6f}',
                f'{row[2]:.6f}',
                int(row[3]),
            ])


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Plot YOLO confidence against measured camera distance.'
    )
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--class-name', default='')
    parser.add_argument('--bin-width', type=float, default=0.5)
    parser.add_argument('--title', default='YOLO confidence vs distance')
    return parser.parse_args(args)


def main(args=None):
    options = parse_args(args)
    if options.bin_width <= 0:
        raise SystemExit('--bin-width must be greater than zero')

    input_csv = os.path.abspath(os.path.expanduser(options.input))
    output_plot = os.path.abspath(os.path.expanduser(options.output))
    distances, confidences = read_samples(input_csv, options.class_name)

    if len(distances) == 0:
        class_text = f' for class {options.class_name!r}' if options.class_name else ''
        raise SystemExit(f'No valid samples found in {input_csv}{class_text}')

    centers, means, deviations, counts = binned_curve(
        distances,
        confidences,
        options.bin_width
    )

    output_dir = os.path.dirname(output_plot)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.scatter(
        distances,
        confidences,
        s=18,
        alpha=0.30,
        color='#1976d2',
        label=f'Individual detections (n={len(distances)})'
    )
    axis.errorbar(
        centers,
        means,
        yerr=deviations,
        marker='o',
        linewidth=2,
        capsize=3,
        color='#d32f2f',
        label=f'{options.bin_width:g}m-bin mean ± 1 SD'
    )
    axis.set(
        title=options.title,
        xlabel='Camera-to-detection distance (m)',
        ylabel='YOLO confidence',
        ylim=(0.0, 1.0),
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(output_plot, dpi=200)
    plt.close(figure)

    summary_path = os.path.splitext(output_plot)[0] + '_binned.csv'
    write_summary(summary_path, centers, means, deviations, counts)

    print(f'Read {len(distances)} samples from {input_csv}')
    print(f'Wrote plot: {output_plot}')
    print(f'Wrote binned data: {summary_path}')


if __name__ == '__main__':
    main()
