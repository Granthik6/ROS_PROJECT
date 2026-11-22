"""The policy extractor must retain obstacle information from LiDAR."""

import gymnasium as gym
import numpy as np
import torch

from robot_tasks.scan_features import ScanSectorExtractor


def test_front_obstacle_changes_front_sector_features():
    space = gym.spaces.Box(-np.inf, np.inf, shape=(372,), dtype=np.float32)
    extractor = ScanSectorExtractor(space)
    observations = torch.ones((2, 372), dtype=torch.float32)
    observations[1, :30] = 0.1

    features = extractor(observations)

    assert features.shape == (2, 36)
    assert torch.isclose(features[0, 0], torch.tensor(1.0))
    assert torch.isclose(features[1, 0], torch.tensor(0.1))
    assert torch.isclose(features[1, 12], torch.tensor(0.1))
