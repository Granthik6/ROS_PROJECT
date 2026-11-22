"""Compact features derived from the 360-reading exploration observation."""

import math

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ScanSectorExtractor(BaseFeaturesExtractor):
    """Keep LiDAR input but expose nearby obstacles and novelty by sector."""

    def __init__(self, observation_space):
        if observation_space.shape != (372,):
            raise ValueError('ScanSectorExtractor requires the 372-value search state')
        super().__init__(observation_space, features_dim=36)

    def forward(self, observations):
        scan = observations[:, :360].reshape(-1, 12, 30)
        nearest = scan.amin(dim=2)
        average = scan.mean(dim=2)
        xy = torch.tanh(observations[:, 360:362] / 10.0)
        yaw = observations[:, 362:363] / math.pi
        confidence = observations[:, 363:364]
        novelty = observations[:, 364:372]
        return torch.cat((nearest, average, xy, yaw, confidence, novelty), dim=1)
