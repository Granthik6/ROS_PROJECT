"""Check that Double DQN separates action selection and evaluation."""

import torch

from robot_tasks.double_dqn import double_dqn_targets


def test_online_network_selects_and_target_network_evaluates():
    online_q = torch.tensor([[5.0, 1.0], [0.0, 4.0]])
    target_q = torch.tensor([[2.0, 9.0], [7.0, 3.0]])
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[0.0], [1.0]])

    targets = double_dqn_targets(
        online_q, target_q, rewards, dones, discounts=0.9
    )

    # The first online choice is action 0, worth 2 in the target network.
    # The second transition is terminal, so it cannot bootstrap.
    torch.testing.assert_close(targets, torch.tensor([[2.8], [2.0]]))
