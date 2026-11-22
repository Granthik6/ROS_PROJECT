import argparse
import json
import os

import numpy as np

from robot_tasks.dqn_environment import CoverageExplorationEnv


DEFAULT_REPORT = '/root/ros2_ws/dqn_environment_check.json'


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Exercise reset() and step() for the coverage DQN environment'
    )
    parser.add_argument('--action-duration', type=float, default=0.10)
    parser.add_argument('--episode-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', default=DEFAULT_REPORT)
    args, _ = parser.parse_known_args()
    return args


def observation_summary(observation):
    return {
        'shape': list(observation.shape),
        'dtype': str(observation.dtype),
        'lidar_min': round(float(np.min(observation[:360])), 6),
        'lidar_max': round(float(np.max(observation[:360])), 6),
        'pose_x': round(float(observation[360]), 6),
        'pose_y': round(float(observation[361]), 6),
        'pose_yaw': round(float(observation[362]), 6),
        'confidence': round(float(observation[363]), 6),
    }


def main():
    args = parse_arguments()
    environment = CoverageExplorationEnv(
        action_duration=args.action_duration,
        max_episode_steps=args.episode_steps,
        seed=args.seed,
    )
    report = {
        'passed': False,
        'state_layout': '360 lidar + x + y + yaw + detection confidence',
        'observation_size': int(environment.observation_space.shape[0]),
        'action_count': int(environment.action_space.n),
        'actions': list(environment.ACTION_NAMES),
        'reward_formula_checks': {},
        'reset': {},
        'steps': [],
    }
    try:
        observation, reset_info = environment.reset(seed=args.seed)
        if observation.shape != (364,):
            raise AssertionError(f'Expected observation shape (364,), got {observation.shape}')
        if not environment.observation_space.contains(observation):
            raise AssertionError('reset() observation is outside observation_space')
        if environment.action_space.n != 3:
            raise AssertionError('Expected exactly three discrete actions')

        positive = environment._calculate_reward_components(
            new_area=True,
            confidence=0.8,
            collision=False,
        )
        penalties = environment._calculate_reward_components(
            new_area=False,
            confidence=0.0,
            collision=True,
        )
        if not np.isclose(sum(positive.values()), 1.4):
            raise AssertionError('New-area and confidence reward formula failed')
        if not np.isclose(sum(penalties.values()), -2.05):
            raise AssertionError('Collision and revisit penalty formula failed')
        report['reward_formula_checks'] = {
            'new_area_plus_confidence': positive,
            'new_area_plus_confidence_total': sum(positive.values()),
            'collision_plus_revisit': penalties,
            'collision_plus_revisit_total': sum(penalties.values()),
        }

        report['reset'] = {
            'observation': observation_summary(observation),
            'info': reset_info,
        }

        for action in range(environment.action_space.n):
            observation, reward, terminated, truncated, info = environment.step(action)
            if observation.shape != (364,):
                raise AssertionError('step() returned an invalid observation shape')
            if not environment.observation_space.contains(observation):
                raise AssertionError('step() observation is outside observation_space')
            if not np.isfinite(reward):
                raise AssertionError('step() returned a non-finite reward')
            if set(info['reward_components']) != {
                'new_area', 'confidence', 'collision', 'revisit'
            }:
                raise AssertionError('step() reward components are incomplete')

            report['steps'].append({
                'action': action,
                'action_name': info['action_name'],
                'observation': observation_summary(observation),
                'reward': round(float(reward), 6),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'new_area': bool(info['new_area']),
                'visited_cells': int(info['visited_cells']),
                'reward_components': info['reward_components'],
            })
            if terminated or truncated:
                environment.reset(seed=args.seed + action + 1)

        report['passed'] = True
    finally:
        environment.close()

    report_path = os.path.abspath(os.path.expanduser(args.report))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write('\n')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Environment contract passed. Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
