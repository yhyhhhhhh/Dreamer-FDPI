from __future__ import annotations

import torch

try:
    from pwm_isaaclab.replay_buffer import ProprioReplayBuffer
except ImportError:
    from replay_buffer import ProprioReplayBuffer

from .cost_utils import SOURCE_DUAL, SOURCE_MAIN, as_column


class DFDV4ReplayBuffer(ProprioReplayBuffer):
    """Dreamer proprio replay with FDPI-Regime cost/source side buffers."""

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
        self.continuous_cost_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.binary_cost_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.extreme_cost_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.bottom_force_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.force_excess_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.source_buffer = torch.full((*default, 1), SOURCE_MAIN, dtype=torch.int64, device=device)
        self.num_appends = 0

    @property
    def cost_buffer(self):
        return self.continuous_cost_buffer

    def _valid_row_count(self):
        if self.length < 0:
            return 0
        return min(int(getattr(self, "num_appends", self.length + 1)), self.max_length // self.num_envs)

    def state_dict(self, *, cpu=True):
        rows = self._valid_row_count()

        def pack(tensor):
            value = tensor[:rows].detach().clone()
            return value.cpu() if cpu else value

        state = {
            "version": 1,
            "num_envs": int(self.num_envs),
            "max_length": int(self.max_length),
            "warmup_length": int(self.warmup_length),
            "include_force": bool(self.include_force),
            "force_dim": int(self.force_dim),
            "force_key": self.force_key,
            "length": int(self.length),
            "num_appends": int(getattr(self, "num_appends", rows)),
            "obs_buffer": pack(self.obs_buffer),
            "action_buffer": pack(self.action_buffer),
            "reward_buffer": pack(self.reward_buffer),
            "done_buffer": pack(self.done_buffer),
            "is_first_buffer": pack(self.is_first_buffer),
            "continuous_cost_buffer": pack(self.continuous_cost_buffer),
            "binary_cost_buffer": pack(self.binary_cost_buffer),
            "extreme_cost_buffer": pack(self.extreme_cost_buffer),
            "bottom_force_buffer": pack(self.bottom_force_buffer),
            "force_excess_buffer": pack(self.force_excess_buffer),
            "source_buffer": pack(self.source_buffer),
        }
        if self.include_force and self.force_buffer is not None:
            state["force_buffer"] = pack(self.force_buffer)
        return state

    def load_state_dict(self, state):
        if int(state.get("num_envs", self.num_envs)) != int(self.num_envs):
            raise ValueError(f"Replay num_envs mismatch: {state.get('num_envs')} != {self.num_envs}")
        if bool(state.get("include_force", self.include_force)) != bool(self.include_force):
            raise ValueError("Replay include_force mismatch.")
        self.warmup_length = int(state.get("warmup_length", self.warmup_length))
        self.force_key = state.get("force_key", self.force_key)

        def restore(name, required=True):
            if name not in state:
                if required:
                    raise KeyError(f"Replay checkpoint missing `{name}`.")
                return
            target = getattr(self, name)
            value = torch.as_tensor(state[name], dtype=target.dtype, device=self.device)
            if value.ndim != target.ndim or value.shape[1:] != target.shape[1:]:
                raise ValueError(f"Replay `{name}` shape mismatch: {tuple(value.shape)} vs {tuple(target.shape)}")
            if value.shape[0] > target.shape[0]:
                raise ValueError(f"Replay `{name}` has {value.shape[0]} rows, capacity is {target.shape[0]}.")
            if value.shape[0] > 0:
                target[: value.shape[0]].copy_(value)

        restore("obs_buffer")
        restore("action_buffer")
        restore("reward_buffer")
        restore("done_buffer")
        restore("is_first_buffer")
        restore("continuous_cost_buffer")
        restore("binary_cost_buffer")
        restore("extreme_cost_buffer")
        restore("bottom_force_buffer")
        restore("force_excess_buffer")
        restore("source_buffer")
        if self.include_force and self.force_buffer is not None:
            restore("force_buffer")

        self.length = int(state.get("length", -1))
        self.num_appends = int(state.get("num_appends", self.length + 1 if self.length >= 0 else 0))
        max_rows = self.max_length // self.num_envs
        if self.length >= max_rows:
            raise ValueError(f"Replay length index {self.length} exceeds capacity rows {max_rows}.")
        return self

    def _window_has_source(self, env_idx, starts, horizon, source):
        if starts.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        length = torch.arange(horizon, device=self.device)
        indexes = starts[:, None] + length[None, :]
        source_windows = self.source_buffer[indexes, env_idx, 0]
        return (source_windows == int(source)).any(dim=-1)

    def _window_has_high_cost(self, env_idx, starts, horizon, high_cost_threshold):
        if starts.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        length = torch.arange(horizon, device=self.device)
        indexes = starts[:, None] + length[None, :]
        cost_windows = self.continuous_cost_buffer[indexes, env_idx, 0]
        return (cost_windows > float(high_cost_threshold)).any(dim=-1)

    def _window_has_boundary_cost(self, env_idx, starts, horizon, boundary_low, boundary_high):
        if starts.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        length = torch.arange(horizon, device=self.device)
        indexes = starts[:, None] + length[None, :]
        cost_windows = self.continuous_cost_buffer[indexes, env_idx, 0]
        return ((cost_windows >= float(boundary_low)) & (cost_windows <= float(boundary_high))).any(dim=-1)

    def _draw_starts(self, valid_starts, num_samples):
        if valid_starts.numel() == 0 or num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        ids = torch.randint(valid_starts.numel(), (num_samples,), device=self.device)
        return valid_starts[ids]

    def _safety_critical_starts(self, env_idx, horizon, high_cost_threshold, boundary_low, boundary_high):
        valid_starts = self._valid_starts(env_idx, horizon)
        if valid_starts.numel() == 0:
            return valid_starts
        has_dual = self._window_has_source(env_idx, valid_starts, horizon, SOURCE_DUAL)
        has_high = self._window_has_high_cost(env_idx, valid_starts, horizon, high_cost_threshold)
        has_boundary = self._window_has_boundary_cost(env_idx, valid_starts, horizon, boundary_low, boundary_high)
        return valid_starts[has_dual | has_high | has_boundary]

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

    def _valid_cost_positive_starts(self, env_idx, horizon):
        valid_starts = self._valid_starts(env_idx, horizon)
        if valid_starts.numel() == 0:
            return valid_starts
        length = torch.arange(horizon, device=self.device)
        indexes = valid_starts[:, None] + length[None, :]
        cost_windows = self.continuous_cost_buffer[indexes, env_idx, 0]
        keep = (cost_windows > 0.0).any(dim=-1)
        return valid_starts[keep]

    def _sample_starts_safety_mixed(
        self,
        env_idx,
        horizon,
        per_env_batch,
        safety_critical_ratio,
        high_cost_threshold,
        boundary_low,
        boundary_high,
    ):
        valid_starts = self._valid_starts(env_idx, horizon)
        if valid_starts.numel() == 0:
            raise ValueError(f"No valid replay sequences for env index {env_idx} and horizon {horizon}.")
        safety_starts = self._safety_critical_starts(
            env_idx,
            horizon,
            high_cost_threshold,
            boundary_low,
            boundary_high,
        )
        num_safety = int(round(per_env_batch * max(float(safety_critical_ratio), 0.0)))
        num_safety = min(num_safety, per_env_batch)
        if safety_starts.numel() == 0:
            num_safety = 0
        num_uniform = per_env_batch - num_safety
        starts = []
        if num_uniform > 0:
            starts.append(self._draw_starts(valid_starts, num_uniform))
        if num_safety > 0:
            starts.append(self._draw_starts(safety_starts, num_safety))
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
        safety_critical_ratio=None,
        high_cost_threshold=0.1,
        boundary_low=0.05,
        boundary_high=0.4,
    ):
        obs, action, reward, done, is_first = [], [], [], [], []
        force = []
        continuous_cost, binary_cost, extreme_cost, bottom_force, force_excess, source = [], [], [], [], [], []
        assert batch_size > 0
        assert batch_size >= self.num_envs and batch_size % self.num_envs == 0, (
            f"batch_size ({batch_size}) must be >= num_envs ({self.num_envs}) "
            "and divisible by num_envs."
        )
        length = torch.arange(horizon, device=self.device)
        per_env_batch = batch_size // self.num_envs
        for env_idx in range(self.num_envs):
            if safety_critical_ratio is not None and float(safety_critical_ratio) > 0.0:
                starts = self._sample_starts_safety_mixed(
                    env_idx,
                    horizon,
                    per_env_batch,
                    safety_critical_ratio,
                    high_cost_threshold,
                    boundary_low,
                    boundary_high,
                )
            else:
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
            continuous_cost.append(self.continuous_cost_buffer[indexes, env_idx])
            binary_cost.append(self.binary_cost_buffer[indexes, env_idx])
            extreme_cost.append(self.extreme_cost_buffer[indexes, env_idx])
            bottom_force.append(self.bottom_force_buffer[indexes, env_idx])
            force_excess.append(self.force_excess_buffer[indexes, env_idx])
            source.append(self.source_buffer[indexes, env_idx])
            if self.include_force:
                force.append(self.force_buffer[indexes, env_idx])

        batch = {
            "obs": torch.cat(obs, dim=0),
            "action": torch.cat(action, dim=0),
            "reward": torch.cat(reward, dim=0),
            "done": torch.cat(done, dim=0),
            "is_first": torch.cat(is_first, dim=0),
            "continuous_cost": torch.cat(continuous_cost, dim=0),
            "binary_cost": torch.cat(binary_cost, dim=0),
            "extreme_cost": torch.cat(extreme_cost, dim=0),
            "bottom_force": torch.cat(bottom_force, dim=0),
            "force_excess": torch.cat(force_excess, dim=0),
            "source": torch.cat(source, dim=0),
        }
        batch["cost"] = batch["continuous_cost"]
        if self.include_force:
            batch["force"] = torch.cat(force, dim=0)
        if return_dict:
            return batch
        samples = (batch["obs"], batch["action"], batch["reward"], batch["done"], batch["is_first"])
        if self.include_force:
            samples = (*samples, batch["force"])
        return samples

    def append(
        self,
        obs,
        action,
        reward,
        done,
        is_first,
        force=None,
        *,
        continuous_cost=None,
        binary_cost=None,
        extreme_cost=None,
        bottom_force=None,
        force_excess=None,
        source=SOURCE_MAIN,
        cost=None,
    ):
        super().append(obs, action, reward, done, is_first, force=force)
        self.num_appends += 1
        row = self.length
        if continuous_cost is None:
            continuous_cost = cost
        self.continuous_cost_buffer[row] = as_column(continuous_cost, self.num_envs, self.device, fill_value=0.0)
        self.binary_cost_buffer[row] = as_column(binary_cost, self.num_envs, self.device, fill_value=0.0)
        self.extreme_cost_buffer[row] = as_column(extreme_cost, self.num_envs, self.device, fill_value=0.0)
        self.bottom_force_buffer[row] = as_column(bottom_force, self.num_envs, self.device, fill_value=0.0)
        self.force_excess_buffer[row] = as_column(force_excess, self.num_envs, self.device, fill_value=0.0)
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

    def cost_stats(self, *, high_cost_threshold=0.1, boundary_low=0.05, boundary_high=0.4):
        if self.length < 0:
            return {
                "cost_mean": 0.0,
                "main_cost_mean": 0.0,
                "dual_cost_mean": 0.0,
                "main_cost_rate": 0.0,
                "dual_cost_rate": 0.0,
                "extreme_cost_rate": 0.0,
                "main_extreme_rate": 0.0,
                "dual_extreme_rate": 0.0,
                "force_excess_mean": 0.0,
                "force_excess_max": 0.0,
                "high_cost_ratio": 0.0,
                "boundary_ratio": 0.0,
            }
        sl = slice(0, self.length + 1)
        cost = self.continuous_cost_buffer[sl, :, 0]
        binary = self.binary_cost_buffer[sl, :, 0]
        extreme = self.extreme_cost_buffer[sl, :, 0]
        force_excess = self.force_excess_buffer[sl, :, 0]
        source = self.source_buffer[sl, :, 0]
        main_mask = source == SOURCE_MAIN
        dual_mask = source == SOURCE_DUAL
        boundary = (cost >= float(boundary_low)) & (cost <= float(boundary_high))
        high = cost > float(high_cost_threshold)
        return {
            "cost_mean": float(cost.float().mean().item()),
            "main_cost_mean": float(cost[main_mask].float().mean().item()) if main_mask.any() else 0.0,
            "dual_cost_mean": float(cost[dual_mask].float().mean().item()) if dual_mask.any() else 0.0,
            "main_cost_rate": float(binary[main_mask].float().mean().item()) if main_mask.any() else 0.0,
            "dual_cost_rate": float(binary[dual_mask].float().mean().item()) if dual_mask.any() else 0.0,
            "extreme_cost_rate": float(extreme.float().mean().item()),
            "main_extreme_rate": float(extreme[main_mask].float().mean().item()) if main_mask.any() else 0.0,
            "dual_extreme_rate": float(extreme[dual_mask].float().mean().item()) if dual_mask.any() else 0.0,
            "force_excess_mean": float(force_excess.float().mean().item()),
            "force_excess_max": float(force_excess.float().max().item()),
            "high_cost_ratio": float(high.float().mean().item()),
            "boundary_ratio": float(boundary.float().mean().item()),
        }
