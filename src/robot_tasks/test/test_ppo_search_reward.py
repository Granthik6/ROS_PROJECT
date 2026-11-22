"""PPO search reward should encourage movement over stationary spinning."""

from robot_tasks.dqn_environment import CoverageSearchExplorationEnv


def test_turn_penalty_applies_only_without_forward_motion():
    env = CoverageSearchExplorationEnv.__new__(CoverageSearchExplorationEnv)
    env.new_area_reward = 10.0
    env.confidence_reward_scale = 0.1
    env.collision_penalty = 3.0
    env.revisit_penalty = 0.5
    env.new_object_reward = 10.0
    env.turn_penalty = 0.3

    env.last_linear = 0.0
    turning = env._calculate_reward_components(False, 0.0, False)
    assert turning['turn'] == -0.3
    assert sum(turning.values()) < 0

    env.last_linear = 0.25
    forward = env._calculate_reward_components(True, 0.0, False)
    assert 'turn' not in forward
    assert forward['new_area'] == 10.0


def test_coverage_search_penalizes_blocked_forward_and_prolonged_spin():
    env = CoverageSearchExplorationEnv.__new__(CoverageSearchExplorationEnv)
    env.progress_reward_scale = 4.0
    env.blocked_forward_penalty = 2.0
    env.spin_penalty = 1.0

    clear = env._movement_reward_components(0, 0.025, 1.0)
    assert clear == {'progress': 0.1}

    blocked = env._movement_reward_components(0, 0.0, 0.20)
    assert blocked['blocked_forward'] == -2.0

    env.stationary_turn_steps = 29
    assert env._movement_reward_components(1, 0.0, 0.20) == {}
    env.stationary_turn_steps = 30
    assert env._movement_reward_components(1, 0.0, 0.20) == {
        'prolonged_spin': -1.0,
    }
