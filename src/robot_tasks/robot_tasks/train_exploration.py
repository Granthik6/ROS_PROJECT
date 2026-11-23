"""Train Double DQN or PPO on the ROS object-exploration task."""

import argparse
from collections import Counter
import json
import os
import signal
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from rclpy.executors import ExternalShutdownException

from robot_tasks.double_dqn import DoubleDQN
from robot_tasks.dqn_environment import make_exploration_env
from robot_tasks.scan_features import ScanSectorExtractor
from robot_tasks.train_dqn import (
    RewardCurveCallback,
    episode_history_for,
    replay_buffer_for,
    unique_run_directory,
)
from robot_tasks.world_coverage_report import (
    read_world, search_bounds_from_world,
)


DEFAULT_OUTPUT_ROOT = '/root/ros2_ws/rl_runs'
ALGORITHMS = {'ddqn': DoubleDQN, 'ppo': PPO}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train DDQN or PPO in ROS 2')
    parser.add_argument('--algorithm', choices=tuple(ALGORITHMS), required=True)
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--run-id', default='exploration')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action-duration', type=float)
    parser.add_argument('--episode-steps', type=int)
    parser.add_argument(
        '--environment-profile',
        choices=('legacy', 'coverage', 'coverage_objects', 'coverage_search'),
        default='coverage_objects',
    )
    parser.add_argument('--resume', default='')
    parser.add_argument('--world', default='')
    parser.add_argument('--detector-model', default='')
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    parser.add_argument('--new-area-reward', type=float)
    parser.add_argument('--new-object-reward', type=float)
    parser.add_argument('--collision-penalty', type=float)
    parser.add_argument('--revisit-penalty', type=float)
    parser.add_argument('--turn-penalty', type=float, default=0.0)
    parser.add_argument('--progress-reward-scale', type=float, default=0.0)
    parser.add_argument('--blocked-forward-penalty', type=float, default=0.0)
    parser.add_argument('--spin-penalty', type=float, default=0.0)
    parser.add_argument('--feature-extractor',
                        choices=('flat', 'scan_sectors'), default='flat')
    args, _ = parser.parse_known_args()
    if args.action_duration is None:
        args.action_duration = (0.25 if args.environment_profile ==
                                'coverage_search' else 0.05)
    if args.episode_steps is None:
        args.episode_steps = (1000 if args.environment_profile ==
                              'coverage_search' else 250)
    return args


def make_model(algorithm, env, timesteps, seed, entropy_coef=0.0,
               feature_extractor='flat'):
    feature_kwargs = ({'features_extractor_class': ScanSectorExtractor}
                      if feature_extractor == 'scan_sectors' else {})
    if algorithm == 'ddqn':
        return DoubleDQN(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=min(1000, max(10, timesteps // 10)),
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.30,
            exploration_final_eps=0.05,
            policy_kwargs={'net_arch': [128, 128], **feature_kwargs},
            verbose=1,
            seed=seed,
            device='auto',
        )
    return PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=250,
        batch_size=50,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=entropy_coef,
        policy_kwargs={'net_arch': {'pi': [128, 128], 'vf': [128, 128]},
                       **feature_kwargs},
        verbose=1,
        seed=seed,
        device='cpu',
    )


def validate_resume(path, algorithm, profile, feature_extractor):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Model to resume does not exist: {path}')
    summary_path = os.path.join(os.path.dirname(path), 'training_summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path, encoding='utf-8') as stream:
            metadata = json.load(stream)
        if metadata.get('algorithm') != algorithm:
            raise ValueError('Resume model algorithm does not match --algorithm')
        if metadata.get('environment_profile') != profile:
            raise ValueError('Resume model environment profile does not match')
        if metadata.get('feature_extractor', 'flat') != feature_extractor:
            raise ValueError('Resume model feature extractor does not match')


def main():
    args = parse_arguments()
    if args.timesteps < 1:
        raise ValueError('--timesteps must be positive')
    if args.resume:
        validate_resume(args.resume, args.algorithm, args.environment_profile,
                        args.feature_extractor)
    if (args.feature_extractor == 'scan_sectors' and
            args.environment_profile != 'coverage_search'):
        raise ValueError('scan_sectors requires coverage_search')

    run_dir = unique_run_directory(
        os.path.abspath(os.path.expanduser(args.output_root)), args.run_id
    )
    started = time.time()
    env = make_exploration_env(
        profile=args.environment_profile,
        action_duration=args.action_duration,
        max_episode_steps=args.episode_steps,
        seed=args.seed,
        class_quota=(Counter(obj['class'] for obj in read_world(args.world))
                     if args.environment_profile == 'coverage_search' and
                     args.world else None),
    )
    if args.environment_profile == 'coverage_search' and args.world:
        env.search_bounds = search_bounds_from_world(args.world)
    for option, attribute in (
        ('new_area_reward', 'new_area_reward'),
        ('new_object_reward', 'new_object_reward'),
        ('collision_penalty', 'collision_penalty'),
        ('revisit_penalty', 'revisit_penalty'),
    ):
        value = getattr(args, option)
        if value is not None:
            setattr(env, attribute, value)
    if args.turn_penalty:
        if args.environment_profile != 'coverage_search':
            raise ValueError('--turn-penalty requires coverage_search')
        env.turn_penalty = args.turn_penalty
    for option in ('progress_reward_scale', 'blocked_forward_penalty',
                   'spin_penalty'):
        value = getattr(args, option)
        if value:
            if args.environment_profile != 'coverage_search':
                raise ValueError(f'--{option.replace("_", "-")} requires coverage_search')
            setattr(env, option, value)
    monitored_env = Monitor(
        env,
        filename=os.path.join(run_dir, 'monitor.csv'),
        info_keywords=('objects_found', 'collision'),
    )
    algorithm_class = ALGORITHMS[args.algorithm]
    if args.resume:
        device = 'cpu' if args.algorithm == 'ppo' else 'auto'
        model = algorithm_class.load(args.resume, env=monitored_env, device=device)
        if args.algorithm == 'ppo':
            model.ent_coef = args.entropy_coef
        if args.algorithm == 'ddqn':
            replay_path = replay_buffer_for(args.resume)
            if replay_path:
                model.load_replay_buffer(replay_path)
                print(f'Loaded replay buffer: {replay_path}', flush=True)
    else:
        model = make_model(
            args.algorithm, monitored_env, args.timesteps, args.seed,
            entropy_coef=args.entropy_coef,
            feature_extractor=args.feature_extractor)

    checkpoint = CheckpointCallback(
        save_freq=max(250, args.timesteps // 10),
        save_path=os.path.join(run_dir, 'checkpoints'),
        name_prefix=args.algorithm,
        save_replay_buffer=False,
    )
    reward_callback = RewardCurveCallback(
        run_dir,
        previous_csv=episode_history_for(args.resume) if args.resume else '',
        previous_until=int(model.num_timesteps) if args.resume else None,
        algorithm_name=args.algorithm.upper(),
    )
    status = 'completed'
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint, reward_callback],
            reset_num_timesteps=not bool(args.resume),
            progress_bar=False,
        )
    except (KeyboardInterrupt, ExternalShutdownException):
        status = 'interrupted'
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        model_path = os.path.join(run_dir, 'model_final')
        model.save(model_path)
        if args.algorithm == 'ddqn':
            model.save_replay_buffer(model_path + '_replay_buffer.pkl')
        reward_callback._save()
        monitored_env.close()

    metadata = {
        'algorithm': args.algorithm,
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
        'world_file': os.path.abspath(args.world) if args.world else None,
        'detector_model': (os.path.abspath(args.detector_model)
                           if args.detector_model else None),
        'entropy_coef': args.entropy_coef,
        'new_area_reward': getattr(env, 'new_area_reward', None),
        'new_object_reward': getattr(env, 'new_object_reward', None),
        'collision_penalty': getattr(env, 'collision_penalty', None),
        'revisit_penalty': getattr(env, 'revisit_penalty', None),
        'turn_penalty': getattr(env, 'turn_penalty', 0.0),
        'feature_extractor': args.feature_extractor,
        'progress_reward_scale': getattr(env, 'progress_reward_scale', 0.0),
        'blocked_forward_penalty': getattr(env, 'blocked_forward_penalty', 0.0),
        'spin_penalty': getattr(env, 'spin_penalty', 0.0),
    }
    with open(
        os.path.join(run_dir, 'training_summary.json'), 'w', encoding='utf-8'
    ) as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print(f'{args.algorithm.upper()} training {status}: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
