import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


MIN_LOG_STD = -20.0
MAX_LOG_STD = 2.0
LOG_2 = math.log(2.0)


def mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1)).squeeze(-1)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.act_dim = act_dim
        self.net = mlp(obs_dim, hidden_sizes, act_dim * 2)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.net(obs), 2, dim=-1)
        log_std = torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD)
        std = torch.exp(log_std)
        return mean, std

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self(obs)
        return torch.distributions.Normal(mean, std)

    def sample(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        return_raw: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()
        action = torch.tanh(z)
        logp = self.squashed_log_prob(dist, z)
        if return_raw:
            return action, logp, z
        return action, logp

    @staticmethod
    def squashed_log_prob(dist: torch.distributions.Normal, z: torch.Tensor) -> torch.Tensor:
        correction = 2.0 * (LOG_2 - z - F.softplus(-2.0 * z))
        return (dist.log_prob(z) - correction).sum(dim=-1)

    @staticmethod
    def raw_log_prob(dist: torch.distributions.Normal, z: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(z).sum(dim=-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)
