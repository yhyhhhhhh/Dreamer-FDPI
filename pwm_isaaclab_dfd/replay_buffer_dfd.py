from __future__ import annotations

import torch

try:
    from pwm_isaaclab.replay_buffer import ProprioReplayBuffer
except ImportError:
    from replay_buffer import ProprioReplayBuffer

from .utils import SOURCE_DUAL, SOURCE_MAIN, as_column


class DFDReplayBuffer(ProprioReplayBuffer):
    """Dreamer proprio replay with DFD cost/source side buffers."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_envs,
        max_length=int(2e5),
        warmup_length=2500,
        device="cpu",
        include_force=False,
        force_dim=1,
        force_key="",
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_envs=num_envs,
            max_length=max_length,
            warmup_length=warmup_length,
            device=device,
            include_force=include_force,
            force_dim=force_dim,
            force_key=force_key,
        )
        default = (max_length // num_envs, num_envs)
        self.cost_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.source_buffer = torch.full((*default, 1), SOURCE_MAIN, dtype=torch.int64, device=device)

    def _window_has_source(self, env_idx, starts, horizon, source):
        if starts.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        length = torch.arange(horizon, device=self.device)
        indexes = starts[:, None] + length[None, :]
        source_windows = self.source_buffer[indexes, env_idx, 0]
        return (source_windows == int(source)).any(dim=-1)

    def _draw_starts(self, valid_starts, num_samples):
        if valid_starts.numel() == 0 or num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        ids = torch.randint(valid_starts.numel(), (num_samples,), device=self.device)
        return valid_starts[ids]

    def _sample_starts_with_dual_cap(self, env_idx, horizon, per_env_batch, max_dual_fraction):
        valid_starts = self._valid_starts(env_idx, horizon)
        if valid_starts.numel() == 0:
            raise ValueError(f"No valid replay sequences for env index {env_idx} and horizon {horizon}.")
        if max_dual_fraction is None or float(max_dual_fraction) >= 1.0:
            return self._draw_starts(valid_starts, per_env_batch)

        has_dual = self._window_has_source(env_idx, valid_starts, horizon, SOURCE_DUAL)
        dual_starts = valid_starts[has_dual]
        non_dual_starts = valid_starts[~has_dual]
        max_dual = int(per_env_batch * max(float(max_dual_fraction), 0.0))
        num_dual = min(max_dual, int(dual_starts.numel()), per_env_batch)
        num_non_dual = per_env_batch - num_dual

        starts = []
        if num_non_dual > 0 and non_dual_starts.numel() > 0:
            starts.append(self._draw_starts(non_dual_starts, num_non_dual))
        elif num_non_dual > 0:
            starts.append(self._draw_starts(valid_starts, num_non_dual))
        if num_dual > 0:
            starts.append(self._draw_starts(dual_starts, num_dual))
        starts = torch.cat(starts, dim=0)
        return starts[torch.randperm(starts.numel(), device=self.device)]

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        horizon,
        *,
        return_dict=False,
        max_dual_fraction=None,
        cost_positive_ratio=0.0,
    ):
        obs, action, reward, done, is_first = [], [], [], [], []
        force = []
        cost, source = [], []
        assert batch_size > 0
        assert batch_size >= self.num_envs and batch_size % self.num_envs == 0, (
            f"batch_size ({batch_size}) must be >= num_envs ({self.num_envs}) "
            "and divisible by num_envs."
        )
        length = torch.arange(horizon, device=self.device)
        per_env_batch = batch_size // self.num_envs
        for env_idx in range(self.num_envs):
            starts = self._sample_starts_with_dual_cap(
                env_idx,
                horizon,
                per_env_batch,
                max_dual_fraction,
            )
            if float(cost_positive_ratio) > 0.0:
                positive_starts = self._valid_cost_positive_starts(env_idx, horizon)
                if positive_starts.numel() > 0:
                    num_positive = max(1, int(round(per_env_batch * float(cost_positive_ratio))))
                    num_positive = min(num_positive, per_env_batch)
                    num_rest = per_env_batch - num_positive
                    pos = self._draw_starts(positive_starts, num_positive)
                    rest = self._draw_starts(self._valid_starts(env_idx, horizon), num_rest)
                    starts = torch.cat((pos, rest), dim=0)
                    starts = starts[torch.randperm(starts.numel(), device=self.device)]
            indexes = length[None, :] + starts[:, None]

            obs.append(self.obs_buffer[indexes, env_idx])
            action.append(self.action_buffer[indexes, env_idx])
            reward.append(self.reward_buffer[indexes, env_idx])
            done.append(self.done_buffer[indexes, env_idx])
            is_first.append(self.is_first_buffer[indexes, env_idx])
            cost.append(self.cost_buffer[indexes, env_idx])
            source.append(self.source_buffer[indexes, env_idx])
            if self.include_force:
                force.append(self.force_buffer[indexes, env_idx])

        batch = {
            "obs": torch.cat(obs, dim=0),
            "action": torch.cat(action, dim=0),
            "reward": torch.cat(reward, dim=0),
            "done": torch.cat(done, dim=0),
            "is_first": torch.cat(is_first, dim=0),
            "cost": torch.cat(cost, dim=0),
            "source": torch.cat(source, dim=0),
        }
        if self.include_force:
            batch["force"] = torch.cat(force, dim=0)
        if return_dict:
            return batch
        samples = (batch["obs"], batch["action"], batch["reward"], batch["done"], batch["is_first"])
        if self.include_force:
            samples = (*samples, batch["force"])
        return samples

    def _valid_cost_positive_starts(self, env_idx, horizon):
        valid_starts = self._valid_starts(env_idx, horizon)
        if valid_starts.numel() == 0:
            return valid_starts
        length = torch.arange(horizon, device=self.device)
        indexes = valid_starts[:, None] + length[None, :]
        cost_windows = self.cost_buffer[indexes, env_idx, 0]
        keep = (cost_windows > 0.0).any(dim=-1)
        return valid_starts[keep]

    def append(self, obs, action, reward, done, is_first, force=None, *, cost=None, source=SOURCE_MAIN):
        super().append(obs, action, reward, done, is_first, force=force)
        row = self.length
        self.cost_buffer[row] = as_column(cost, self.num_envs, self.device, fill_value=0.0)
        self.source_buffer[row] = as_column(
            source,
            self.num_envs,
            self.device,
            dtype=torch.int64,
            fill_value=int(SOURCE_MAIN),
        )

    def source_stats(self):
        if self.length < 0:
            return {"main": 0, "dual": 0, "random": 0}
        source = self.source_buffer[: self.length + 1, :, 0]
        return {
            "main": int((source == SOURCE_MAIN).sum().item()),
            "dual": int((source == SOURCE_DUAL).sum().item()),
            "random": int(((source != SOURCE_MAIN) & (source != SOURCE_DUAL)).sum().item()),
        }

    def cost_stats(self):
        if self.length < 0:
            return {"cost_rate": 0.0, "main_cost_rate": 0.0, "dual_cost_rate": 0.0}
        cost = self.cost_buffer[: self.length + 1, :, 0]
        source = self.source_buffer[: self.length + 1, :, 0]
        main_mask = source == SOURCE_MAIN
        dual_mask = source == SOURCE_DUAL
        return {
            "cost_rate": float(cost.float().mean().item()),
            "main_cost_rate": float(cost[main_mask].float().mean().item()) if main_mask.any() else 0.0,
            "dual_cost_rate": float(cost[dual_mask].float().mean().item()) if dual_mask.any() else 0.0,
        }
