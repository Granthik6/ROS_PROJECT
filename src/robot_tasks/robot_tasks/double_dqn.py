"""Double DQN for the discrete ROS exploration environment."""

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import DQN


def double_dqn_targets(online_q, target_q, rewards, dones, discounts):
    """Select actions online and evaluate those actions with the target net."""
    actions = online_q.argmax(dim=1, keepdim=True)
    next_values = target_q.gather(dim=1, index=actions)
    return rewards + (1.0 - dones) * discounts * next_values


class DoubleDQN(DQN):
    """Stable Baselines3 DQN with the Double DQN Bellman target."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None else self.gamma
            )

            with th.no_grad():
                online_q = self.q_net(replay_data.next_observations)
                target_q = self.q_net_target(replay_data.next_observations)
                targets = double_dqn_targets(
                    online_q,
                    target_q,
                    replay_data.rewards,
                    replay_data.dones,
                    discounts,
                )

            current_q = self.q_net(replay_data.observations).gather(
                dim=1, index=replay_data.actions.long()
            )
            loss = F.smooth_l1_loss(current_q, targets)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', np.mean(losses))
