"""Evaluate a saved DQN, Double DQN, or PPO policy against random actions."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO

from robot_tasks.double_dqn import DoubleDQN
from robot_tasks.dqn_environment import make_exploration_env
from robot_tasks.evaluate_dqn import run_policy
from robot_tasks.train_dqn import unique_run_directory


plt.switch_backend('Agg')
DEFAULT_OUTPUT_ROOT = '/root/ros2_ws/rl_evaluations'
ALGORITHMS = {'dqn': DQN, 'ddqn': DoubleDQN, 'ppo': PPO}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare an RL policy to random')
    parser.add_argument('--algorithm', choices=tuple(ALGORITHMS), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--episode-steps', type=int, default=250)
    parser.add_argument('--action-duration', type=float, default=0.05)
    parser.add_argument(
        '--environment-profile',
        choices=('legacy', 'coverage', 'coverage_objects'),
        default='coverage_objects',
    )
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--run-id', default='exploration_vs_random')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    if args.episodes < 1:
        raise ValueError('--episodes must be positive')
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'Model does not exist: {args.model}')

    run_dir = unique_run_directory(
        os.path.abspath(os.path.expanduser(args.output_root)), args.run_id
    )
    env = make_exploration_env(
        profile=args.environment_profile,
        action_duration=args.action_duration,
        max_episode_steps=args.episode_steps,
        seed=args.seed,
    )
    env.action_space.seed(args.seed)
    device = 'cpu' if args.algorithm == 'ppo' else 'auto'
    model = ALGORITHMS[args.algorithm].load(args.model, env=env, device=device)
    try:
        rows = run_policy(env, model, args.algorithm, args.episodes, args.seed)
        env.action_space.seed(args.seed)
        rows += run_policy(env, None, 'random', args.episodes, args.seed)
    finally:
        env.close()

    with open(
        os.path.join(run_dir, 'evaluation.csv'), 'w', newline='', encoding='utf-8'
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    names = (args.algorithm, 'random')
    means = {}
    deviations = {}
    reward_means = {}
    for name in names:
        selected = [row for row in rows if row['policy'] == name]
        counts = np.asarray([row['objects_found'] for row in selected], dtype=float)
        rewards = np.asarray([row['reward'] for row in selected], dtype=float)
        means[name] = float(np.mean(counts))
        deviations[name] = float(np.std(counts))
        reward_means[name] = float(np.mean(rewards))

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].bar(
        names,
        [means[name] for name in names],
        yerr=[deviations[name] for name in names],
        color=('#1976d2', '#757575'),
        capsize=5,
    )
    axes[0].set(title='Objects found', ylabel='Mean confirmed objects')
    axes[1].bar(
        names,
        [reward_means[name] for name in names],
        color=('#1976d2', '#757575'),
    )
    axes[1].set(title='Episode reward', ylabel='Mean reward')
    for axis in axes:
        axis.grid(True, axis='y', alpha=0.2)
    figure.savefig(os.path.join(run_dir, 'policy_vs_random.png'), dpi=180)
    plt.close(figure)

    summary = {
        'algorithm': args.algorithm,
        'model': os.path.abspath(args.model),
        'episodes_per_policy': args.episodes,
        'episode_steps': args.episode_steps,
        'environment_profile': args.environment_profile,
        'policy_mean_objects': round(means[args.algorithm], 3),
        'random_mean_objects': round(means['random'], 3),
        'policy_mean_reward': round(reward_means[args.algorithm], 3),
        'random_mean_reward': round(reward_means['random'], 3),
        'policy_finds_more_objects': means[args.algorithm] > means['random'],
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(
        f'{args.algorithm.upper()} mean objects={means[args.algorithm]:.2f}; '
        f'random mean objects={means["random"]:.2f}. Results: {run_dir}',
        flush=True,
    )


if __name__ == '__main__':
    main()
