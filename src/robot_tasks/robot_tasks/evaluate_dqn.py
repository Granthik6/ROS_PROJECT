import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN

from robot_tasks.dqn_environment import make_exploration_env


plt.switch_backend('Agg')
DEFAULT_OUTPUT_ROOT = '/root/ros2_ws/dqn_evaluations'


def run_policy(env, model, policy_name, episodes, base_seed):
    rows = []
    for episode_index in range(episodes):
        observation, _ = env.reset(seed=base_seed + episode_index)
        terminated = False
        truncated = False
        reward_sum = 0.0
        info = {}
        while not terminated and not truncated:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
        rows.append({
            'policy': policy_name,
            'episode': episode_index + 1,
            'seed': base_seed + episode_index,
            'reward': round(reward_sum, 6),
            'objects_found': int(info.get('objects_found', 0)),
            'classes': ';'.join(info.get('object_classes', [])),
            'collision': bool(info.get('collision', False)),
            'steps': int(info.get('episode_step', 0)),
        })
    return rows


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare a DQN policy with random')
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--episode-steps', type=int, default=250)
    parser.add_argument('--action-duration', type=float, default=0.15)
    parser.add_argument(
        '--environment-profile',
        choices=('legacy', 'coverage', 'coverage_objects'),
        default='legacy',
    )
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--run-id', default='dqn_vs_random')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'DQN model does not exist: {args.model}')
    if args.episodes < 1:
        raise ValueError('--episodes must be positive')

    run_dir = os.path.join(
        os.path.abspath(os.path.expanduser(args.output_root)),
        args.run_id,
    )
    suffix = 2
    base = run_dir
    while os.path.exists(run_dir):
        run_dir = f'{base}_{suffix}'
        suffix += 1
    os.makedirs(run_dir)

    env = make_exploration_env(
        profile=args.environment_profile,
        action_duration=args.action_duration,
        max_episode_steps=args.episode_steps,
        seed=args.seed,
    )
    env.action_space.seed(args.seed)
    model = DQN.load(args.model, env=env, device='auto')
    try:
        rows = run_policy(env, model, 'dqn', args.episodes, args.seed)
        env.action_space.seed(args.seed)
        rows += run_policy(env, None, 'random', args.episodes, args.seed)
    finally:
        env.close()

    csv_path = os.path.join(run_dir, 'evaluation.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    policy_names = ('dqn', 'random')
    means = {}
    standard_deviations = {}
    reward_means = {}
    for name in policy_names:
        selected = [row for row in rows if row['policy'] == name]
        counts = np.asarray([row['objects_found'] for row in selected], dtype=float)
        rewards = np.asarray([row['reward'] for row in selected], dtype=float)
        means[name] = float(np.mean(counts))
        standard_deviations[name] = float(np.std(counts))
        reward_means[name] = float(np.mean(rewards))

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    colors = ('#1976d2', '#757575')
    axes[0].bar(
        policy_names,
        [means[name] for name in policy_names],
        yerr=[standard_deviations[name] for name in policy_names],
        color=colors,
        capsize=5,
    )
    axes[0].set(title='Objects found', ylabel='Mean confirmed objects')
    axes[1].bar(
        policy_names,
        [reward_means[name] for name in policy_names],
        color=colors,
    )
    axes[1].set(title='Episode reward', ylabel='Mean reward')
    for axis in axes:
        axis.grid(True, axis='y', alpha=0.2)
    figure.savefig(os.path.join(run_dir, 'dqn_vs_random.png'), dpi=180)
    plt.close(figure)

    summary = {
        'model': os.path.abspath(args.model),
        'episodes_per_policy': args.episodes,
        'episode_steps': args.episode_steps,
        'environment_profile': args.environment_profile,
        'dqn_mean_objects': round(means['dqn'], 3),
        'random_mean_objects': round(means['random'], 3),
        'dqn_mean_reward': round(reward_means['dqn'], 3),
        'random_mean_reward': round(reward_means['random'], 3),
        'dqn_finds_more_objects': means['dqn'] > means['random'],
    }
    with open(
        os.path.join(run_dir, 'summary.json'),
        'w',
        encoding='utf-8',
    ) as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write('\n')

    print(
        f'DQN mean objects={means["dqn"]:.2f}; '
        f'random mean objects={means["random"]:.2f}. Results: {run_dir}',
        flush=True,
    )


if __name__ == '__main__':
    main()
