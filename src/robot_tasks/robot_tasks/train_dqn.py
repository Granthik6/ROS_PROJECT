import argparse
import csv
import json
import os
import re
import signal
import time

import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from robot_tasks.dqn_environment import make_exploration_env


plt.switch_backend('Agg')
DEFAULT_OUTPUT_ROOT = '/root/ros2_ws/dqn_runs'


class RewardCurveCallback(BaseCallback):
    """Save episode metrics and a rolling reward curve during training."""

    def __init__(
        self,
        run_dir,
        previous_csv='',
        previous_until=None,
        algorithm_name='DQN',
        verbose=0,
    ):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.algorithm_name = algorithm_name
        self.rows = []
        if previous_csv and os.path.exists(previous_csv):
            with open(previous_csv, newline='', encoding='utf-8') as input_file:
                for row in csv.DictReader(input_file):
                    row_timesteps = int(row['timesteps'])
                    if previous_until is not None and row_timesteps > previous_until:
                        continue
                    self.rows.append({
                        'episode': int(row['episode']),
                        'timesteps': row_timesteps,
                        'reward': float(row['reward']),
                        'length': int(row['length']),
                        'objects_found': int(row['objects_found']),
                        'collision': row['collision'] == 'True',
                    })

    def _on_step(self):
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        for done, info in zip(dones, infos):
            if not done or 'episode' not in info:
                continue
            episode = info['episode']
            self.rows.append({
                'episode': len(self.rows) + 1,
                'timesteps': self.num_timesteps,
                'reward': float(episode['r']),
                'length': int(episode['l']),
                'objects_found': int(info.get('objects_found', 0)),
                'collision': bool(info.get('collision', False)),
            })
            if len(self.rows) % 25 == 0:
                self._save(plot=len(self.rows) % 100 == 0)
        return True

    def _on_training_end(self):
        self._save()

    def _save(self, plot=True):
        if not self.rows:
            return
        csv_path = os.path.join(self.run_dir, 'episodes.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

        if not plot:
            return

        rewards = [row['reward'] for row in self.rows]
        window = min(20, len(rewards))
        rolling = []
        for index in range(len(rewards)):
            start = max(0, index - window + 1)
            rolling.append(sum(rewards[start:index + 1]) / (index - start + 1))

        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        episodes = list(range(1, len(rewards) + 1))
        axis.plot(episodes, rewards, alpha=0.30, label='Episode reward')
        axis.plot(episodes, rolling, linewidth=2.0, label=f'Rolling mean ({window})')
        axis.set(
            title=f'{self.algorithm_name} exploration training reward',
            xlabel='Episode',
            ylabel='Reward',
        )
        axis.grid(True, alpha=0.2)
        axis.legend()
        figure.savefig(os.path.join(self.run_dir, 'reward_curve.png'), dpi=180)
        plt.close(figure)


def unique_run_directory(root, requested):
    os.makedirs(root, exist_ok=True)
    candidate = os.path.join(root, requested)
    suffix = 2
    while os.path.exists(candidate):
        candidate = os.path.join(root, f'{requested}_{suffix}')
        suffix += 1
    os.makedirs(candidate)
    return candidate


def replay_buffer_for(model_path):
    """Return the replay buffer saved beside a final model or checkpoint."""
    model_stem = os.path.splitext(model_path)[0]
    candidates = [model_stem + '_replay_buffer.pkl']
    checkpoint = re.match(r'^(.*)_(\d+)_steps$', model_stem)
    if checkpoint:
        prefix, steps = checkpoint.groups()
        candidates.append(f'{prefix}_replay_buffer_{steps}_steps.pkl')
    return next((path for path in candidates if os.path.exists(path)), '')


def episode_history_for(model_path):
    """Find episode history for either a final model or a checkpoint."""
    model_directory = os.path.dirname(os.path.abspath(model_path))
    candidates = [
        os.path.join(model_directory, 'episodes.csv'),
        os.path.join(os.path.dirname(model_directory), 'episodes.csv'),
    ]
    return next((path for path in candidates if os.path.exists(path)), '')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train DQN in the ROS 2 simulation')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--run-id', default='dqn_10000')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action-duration', type=float, default=0.15)
    parser.add_argument('--episode-steps', type=int, default=250)
    parser.add_argument(
        '--environment-profile',
        choices=('legacy', 'coverage', 'coverage_objects'),
        default='legacy',
    )
    parser.add_argument('--resume', nargs='?', default='', const='')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    if args.timesteps < 1:
        raise ValueError('--timesteps must be positive')

    run_dir = unique_run_directory(
        os.path.abspath(os.path.expanduser(args.output_root)),
        args.run_id,
    )
    started = time.time()
    env = make_exploration_env(
        profile=args.environment_profile,
        action_duration=args.action_duration,
        max_episode_steps=args.episode_steps,
        seed=args.seed,
    )
    monitored_env = Monitor(
        env,
        filename=os.path.join(run_dir, 'monitor.csv'),
        info_keywords=('objects_found', 'collision'),
    )

    if args.resume:
        model = DQN.load(args.resume, env=monitored_env, device='auto')
        replay_path = replay_buffer_for(args.resume)
        if replay_path:
            model.load_replay_buffer(replay_path)
            print(f'Loaded replay buffer: {replay_path}', flush=True)
    else:
        model = DQN(
            'MlpPolicy',
            monitored_env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=min(1000, max(10, args.timesteps // 10)),
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.30,
            exploration_final_eps=0.05,
            policy_kwargs={'net_arch': [128, 128]},
            verbose=1,
            seed=args.seed,
            device='auto',
        )

    checkpoint = CheckpointCallback(
        save_freq=max(1000, args.timesteps // 10),
        save_path=os.path.join(run_dir, 'checkpoints'),
        name_prefix='dqn',
        save_replay_buffer=True,
    )
    previous_csv = episode_history_for(args.resume) if args.resume else ''
    reward_callback = RewardCurveCallback(
        run_dir,
        previous_csv=previous_csv,
        previous_until=int(model.num_timesteps) if args.resume else None,
    )
    status = 'completed'
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint, reward_callback],
            reset_num_timesteps=not bool(args.resume),
            progress_bar=False,
        )
    except KeyboardInterrupt:
        status = 'interrupted'
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        model_path = os.path.join(run_dir, 'dqn_final')
        model.save(model_path)
        model.save_replay_buffer(model_path + '_replay_buffer.pkl')
        reward_callback._save()
        monitored_env.close()

    metadata = {
        'run_id': os.path.basename(run_dir),
        'status': status,
        'requested_timesteps': args.timesteps,
        'model_total_timesteps': int(model.num_timesteps),
        'model_path': model_path + '.zip',
        'reward_curve': os.path.join(run_dir, 'reward_curve.png'),
        'duration_minutes': round((time.time() - started) / 60.0, 3),
        'seed': args.seed,
        'action_duration': args.action_duration,
        'episode_steps': args.episode_steps,
        'environment_profile': args.environment_profile,
        'resume': args.resume,
    }
    with open(
        os.path.join(run_dir, 'training_summary.json'),
        'w',
        encoding='utf-8',
    ) as output_file:
        json.dump(metadata, output_file, indent=2)
        output_file.write('\n')
    print(f'DQN training {status}. Results: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
