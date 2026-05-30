from typing import NamedTuple

import torch


class ExperienceBatch(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    cost: torch.Tensor
    done: torch.Tensor
    log_weight: torch.Tensor
    log_weight_dual: torch.Tensor


class TorchReplayBufferIS:
    """Torch replay buffer with the two cumulative FDPI importance weights."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        device: torch.device | str,
    ):
        self.device = torch.device(device)
        self.size = int(size)
        self.ptr = 0
        self.length = 0
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((size, act_dim), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.cost = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.done = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.log_weight = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.log_weight_dual = torch.zeros(size, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        return self.length

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        log_weight: torch.Tensor,
        log_weight_dual: torch.Tensor,
    ) -> None:
        batch_size = int(obs.shape[0])
        if batch_size > self.size:
            obs = obs[-self.size :]
            action = action[-self.size :]
            next_obs = next_obs[-self.size :]
            reward = reward[-self.size :]
            cost = cost[-self.size :]
            done = done[-self.size :]
            log_weight = log_weight[-self.size :]
            log_weight_dual = log_weight_dual[-self.size :]
            batch_size = self.size

        end = self.ptr + batch_size
        fields = (
            (self.obs, obs),
            (self.action, action),
            (self.next_obs, next_obs),
            (self.reward, reward),
            (self.cost, cost),
            (self.done, done),
            (self.log_weight, log_weight),
            (self.log_weight_dual, log_weight_dual),
        )
        if end <= self.size:
            for target, value in fields:
                target[self.ptr : end].copy_(value.to(self.device))
        else:
            split = self.size - self.ptr
            remain = end - self.size
            for target, value in fields:
                value = value.to(self.device)
                target[self.ptr :].copy_(value[:split])
                target[:remain].copy_(value[split:])

        self.ptr = end % self.size
        self.length = min(self.length + batch_size, self.size)

    def sample(self, batch_size: int) -> ExperienceBatch:
        if self.length <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, self.length, (batch_size,), device=self.device)
        return ExperienceBatch(
            self.obs[idx],
            self.action[idx],
            self.next_obs[idx],
            self.reward[idx],
            self.cost[idx],
            self.done[idx],
            self.log_weight[idx],
            self.log_weight_dual[idx],
        )
