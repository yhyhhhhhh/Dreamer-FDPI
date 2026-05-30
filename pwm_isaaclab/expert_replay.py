from __future__ import annotations

import os
from collections import Counter

import torch

try:
    from pwm_isaaclab.expert_loader import SOURCE_DUAL_RESERVED, SOURCE_EXPERT, SOURCE_MAIN
    from pwm_isaaclab.replay_buffer import ProprioReplayBuffer
except ImportError:
    from expert_loader import SOURCE_DUAL_RESERVED, SOURCE_EXPERT, SOURCE_MAIN
    from replay_buffer import ProprioReplayBuffer


def _to_tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype, device=device)


def _as_column(value, length, device, dtype=torch.float32, fill_value=0.0):
    if value is None:
        return torch.full((length, 1), fill_value, dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    return tensor.reshape(length, -1)[:, :1]


def _resolve_source(source):
    if source is None:
        return None
    if source == "main":
        return SOURCE_MAIN
    if source == "expert":
        return SOURCE_EXPERT
    if source in ("dual", "dual_reserved"):
        return SOURCE_DUAL_RESERVED
    return int(source)


def _legacy_samples_to_dict(samples, source=SOURCE_MAIN):
    obs, action, reward, done, is_first = samples[:5]
    batch = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "is_first": is_first,
        "cost": torch.zeros_like(reward),
        "source": torch.full_like(done, int(source), dtype=torch.int64),
        "safety_margin": torch.full_like(reward, float("nan")),
        "uncertainty_score": torch.zeros_like(reward),
        "ood_score": torch.zeros_like(reward),
    }
    if len(samples) > 5:
        batch["force"] = samples[5]
    return batch


class SourceTaggedProprioReplayBuffer(ProprioReplayBuffer):
    """Proprio replay buffer with v0.1 expert metadata side buffers."""

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
        self.safety_margin_buffer = torch.full((*default, 1), float("nan"), dtype=torch.float32, device=device)
        self.uncertainty_score_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)
        self.ood_score_buffer = torch.zeros(*default, 1, dtype=torch.float32, device=device)

    def _valid_starts_for_source(self, env_idx, horizon, source=None):
        valid_starts = self._valid_starts(env_idx, horizon)
        source = _resolve_source(source)
        if source is None or valid_starts.numel() == 0:
            return valid_starts
        length = torch.arange(horizon, device=self.device)
        indexes = valid_starts[:, None] + length[None, :]
        source_windows = self.source_buffer[indexes, env_idx, 0]
        keep = (source_windows == source).all(dim=-1)
        return valid_starts[keep]

    def _valid_cost_positive_starts(self, env_idx, horizon, source=None):
        valid_starts = self._valid_starts_for_source(env_idx, horizon, source)
        if valid_starts.numel() == 0:
            return valid_starts
        length = torch.arange(horizon, device=self.device)
        indexes = valid_starts[:, None] + length[None, :]
        cost_windows = self.cost_buffer[indexes, env_idx, 0]
        keep = (cost_windows > 0).any(dim=-1)
        return valid_starts[keep]

    def can_sample(self, horizon, source=None):
        source = _resolve_source(source)
        if not self.ready() or self.length + 1 < horizon:
            return False
        for env_idx in range(self.num_envs):
            if self._valid_starts_for_source(env_idx, horizon, source).numel() == 0:
                return False
        return True

    @torch.no_grad()
    def sample(self, batch_size, horizon, source=None, return_dict=False, cost_positive_ratio=0.0):
        source = _resolve_source(source)
        obs, action, reward, done, is_first = [], [], [], [], []
        force = []
        cost, source_tag, safety_margin, uncertainty_score, ood_score = [], [], [], [], []
        assert batch_size > 0
        assert batch_size >= self.num_envs and batch_size % self.num_envs == 0, (
            f"batch_size ({batch_size}) must be >= num_envs ({self.num_envs}) "
            "and divisible by num_envs."
        )
        length = torch.arange(horizon, device=self.device)
        for env_idx in range(self.num_envs):
            valid_starts = self._valid_starts_for_source(env_idx, horizon, source)
            if valid_starts.numel() == 0:
                source_text = "" if source is None else f" and source={source}"
                raise ValueError(
                    f"No valid replay sequences for env index {env_idx}, horizon {horizon}{source_text}."
                )
            per_env_batch = batch_size // self.num_envs
            positive_starts = self._valid_cost_positive_starts(env_idx, horizon, source)
            if float(cost_positive_ratio) > 0.0 and positive_starts.numel() > 0:
                num_positive = max(1, int(round(per_env_batch * float(cost_positive_ratio))))
                num_positive = min(num_positive, per_env_batch)
                pos_ids = torch.randint(positive_starts.numel(), (num_positive,), device=self.device)
                starts = [positive_starts[pos_ids]]
                num_rest = per_env_batch - num_positive
                if num_rest > 0:
                    rest_ids = torch.randint(valid_starts.numel(), (num_rest,), device=self.device)
                    starts.append(valid_starts[rest_ids])
                starts = torch.cat(starts, dim=0)
                perm = torch.randperm(starts.numel(), device=self.device)
                starts = starts[perm]
            else:
                sample_ids = torch.randint(valid_starts.numel(), (per_env_batch,), device=self.device)
                starts = valid_starts[sample_ids]
            indexes = length[None, :] + starts[:, None]

            obs.append(self.obs_buffer[indexes, env_idx])
            action.append(self.action_buffer[indexes, env_idx])
            reward.append(self.reward_buffer[indexes, env_idx])
            done.append(self.done_buffer[indexes, env_idx])
            is_first.append(self.is_first_buffer[indexes, env_idx])
            cost.append(self.cost_buffer[indexes, env_idx])
            source_tag.append(self.source_buffer[indexes, env_idx])
            safety_margin.append(self.safety_margin_buffer[indexes, env_idx])
            uncertainty_score.append(self.uncertainty_score_buffer[indexes, env_idx])
            ood_score.append(self.ood_score_buffer[indexes, env_idx])
            if self.include_force:
                force.append(self.force_buffer[indexes, env_idx])

        batch = {
            "obs": torch.cat(obs, dim=0),
            "action": torch.cat(action, dim=0),
            "reward": torch.cat(reward, dim=0),
            "done": torch.cat(done, dim=0),
            "is_first": torch.cat(is_first, dim=0),
            "cost": torch.cat(cost, dim=0),
            "source": torch.cat(source_tag, dim=0),
            "safety_margin": torch.cat(safety_margin, dim=0),
            "uncertainty_score": torch.cat(uncertainty_score, dim=0),
            "ood_score": torch.cat(ood_score, dim=0),
        }
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
        cost=None,
        source=SOURCE_MAIN,
        safety_margin=None,
        uncertainty_score=None,
        ood_score=None,
    ):
        super().append(obs, action, reward, done, is_first, force=force)
        row = self.length
        self.cost_buffer[row] = _as_column(cost, self.num_envs, self.device, fill_value=0.0)
        self.source_buffer[row] = _as_column(
            None, self.num_envs, self.device, dtype=torch.int64, fill_value=int(source)
        )
        self.safety_margin_buffer[row] = _as_column(
            safety_margin, self.num_envs, self.device, fill_value=float("nan")
        )
        self.uncertainty_score_buffer[row] = _as_column(
            uncertainty_score, self.num_envs, self.device, fill_value=0.0
        )
        self.ood_score_buffer[row] = _as_column(ood_score, self.num_envs, self.device, fill_value=0.0)

    def _append_episode_fast(self, episode, source=SOURCE_EXPERT):
        if self.num_envs != 1:
            raise ValueError("Fast expert episode insertion expects a num_envs=1 replay buffer.")
        obs = _to_tensor(episode["obs"], self.device)
        action = _to_tensor(episode["action"], self.device)
        reward = _as_column(episode["reward"], obs.shape[0], self.device)
        done = _as_column(episode.get("done", episode.get("is_last")), obs.shape[0], self.device)
        is_first = _as_column(episode.get("is_first"), obs.shape[0], self.device)
        cost = _as_column(episode.get("cost"), obs.shape[0], self.device)
        safety_margin = _as_column(episode.get("safety_margin"), obs.shape[0], self.device, fill_value=float("nan"))
        uncertainty_score = _as_column(episode.get("uncertainty_score"), obs.shape[0], self.device)
        ood_score = _as_column(episode.get("ood_score"), obs.shape[0], self.device)

        start = self.length + 1
        stop = start + obs.shape[0]
        capacity = self.obs_buffer.shape[0]
        if stop > capacity:
            raise OverflowError(
                f"Expert replay capacity {capacity} is too small for insertion ending at row {stop}."
            )

        self.obs_buffer[start:stop, 0] = obs
        self.action_buffer[start:stop, 0] = action
        self.reward_buffer[start:stop, 0] = reward
        self.done_buffer[start:stop, 0] = done
        self.is_first_buffer[start:stop, 0] = is_first
        self.cost_buffer[start:stop, 0] = cost
        self.source_buffer[start:stop, 0] = int(source)
        self.safety_margin_buffer[start:stop, 0] = safety_margin
        self.uncertainty_score_buffer[start:stop, 0] = uncertainty_score
        self.ood_score_buffer[start:stop, 0] = ood_score
        if self.include_force:
            force = episode.get("force")
            if force is None:
                force = torch.zeros(obs.shape[0], self.force_dim, dtype=torch.float32, device=self.device)
            else:
                force = torch.as_tensor(force, dtype=torch.float32, device=self.device).reshape(
                    obs.shape[0], self.force_dim
                )
            self.force_buffer[start:stop, 0] = force
        self.length = stop - 1

    def add_expert_episode(self, episode):
        self._append_episode_fast(episode, source=SOURCE_EXPERT)

    def add_expert_dataset(self, episodes):
        for episode in episodes:
            self.add_expert_episode(episode)

    def num_expert_steps(self):
        return self._count_source(SOURCE_EXPERT)

    def num_online_steps(self):
        return self._count_source(SOURCE_MAIN)

    def _count_source(self, source):
        if self.length < 0:
            return 0
        return int((self.source_buffer[: self.length + 1, :, 0] == int(source)).sum().item())

    def source_stats(self):
        return {
            "main": self._count_source(SOURCE_MAIN),
            "expert": self._count_source(SOURCE_EXPERT),
            "dual_reserved": self._count_source(SOURCE_DUAL_RESERVED),
        }

    def save_replay(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "obs": self.obs_buffer[: self.length + 1].detach().cpu(),
                "action": self.action_buffer[: self.length + 1].detach().cpu(),
                "reward": self.reward_buffer[: self.length + 1].detach().cpu(),
                "done": self.done_buffer[: self.length + 1].detach().cpu(),
                "is_first": self.is_first_buffer[: self.length + 1].detach().cpu(),
                "cost": self.cost_buffer[: self.length + 1].detach().cpu(),
                "source": self.source_buffer[: self.length + 1].detach().cpu(),
                "safety_margin": self.safety_margin_buffer[: self.length + 1].detach().cpu(),
                "uncertainty_score": self.uncertainty_score_buffer[: self.length + 1].detach().cpu(),
                "ood_score": self.ood_score_buffer[: self.length + 1].detach().cpu(),
                "force": self.force_buffer[: self.length + 1].detach().cpu() if self.include_force else None,
                "source_stats": self.source_stats(),
            },
            path,
        )


def make_expert_replay(
    episodes,
    *,
    device="cpu",
    include_force=False,
    force_dim=1,
    force_key="",
    warmup_length=0,
):
    if not episodes:
        raise ValueError("Cannot build expert replay from an empty episode list.")
    obs_dim = int(episodes[0]["obs"].shape[-1])
    action_dim = int(episodes[0]["action"].shape[-1])
    total_steps = int(sum(len(ep["reward"]) for ep in episodes))
    replay = SourceTaggedProprioReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=1,
        max_length=max(total_steps, 1),
        warmup_length=warmup_length,
        device=device,
        include_force=include_force,
        force_dim=force_dim,
        force_key=force_key,
    )
    replay.add_expert_dataset(episodes)
    return replay


class HybridExpertReplay:
    """Online replay wrapper that falls back to expert samples before online warmup."""

    def __init__(
        self,
        online_replay,
        expert_replay=None,
        *,
        replace_random_prefill=True,
        expert_ratio_online=0.2,
    ):
        self.online_replay = online_replay
        self.expert_replay = expert_replay
        self.replace_random_prefill = bool(replace_random_prefill)
        self.expert_ratio_online = float(expert_ratio_online)
        self.num_envs = online_replay.num_envs
        self.max_length = online_replay.max_length
        self.warmup_length = online_replay.warmup_length
        self.device = online_replay.device
        self.include_force = getattr(online_replay, "include_force", False)
        self.force_key = getattr(online_replay, "force_key", "")

    @property
    def length(self):
        return self.online_replay.length

    def __len__(self):
        return len(self.online_replay) + self.num_expert_steps()

    def append(self, *args, **kwargs):
        return self.online_replay.append(*args, **kwargs)

    def ready(self):
        if self.online_replay.ready():
            return True
        return bool(
            self.replace_random_prefill
            and self.expert_replay is not None
            and len(self.expert_replay) > 0
        )

    def can_sample(self, horizon, source=None):
        source = _resolve_source(source)
        if source == SOURCE_EXPERT:
            return self.expert_replay is not None and self.expert_replay.can_sample(horizon, source=SOURCE_EXPERT)
        if source == SOURCE_MAIN:
            return self.online_replay.can_sample(horizon)
        return self.online_replay.can_sample(horizon) or (
            self.expert_replay is not None and self.expert_replay.can_sample(horizon, source=SOURCE_EXPERT)
        )

    @torch.no_grad()
    def sample(self, batch_size, horizon, source=None, return_dict=False, cost_positive_ratio=0.0):
        source = _resolve_source(source)
        if source == SOURCE_EXPERT:
            return self.expert_replay.sample(
                batch_size,
                horizon,
                source=SOURCE_EXPERT,
                return_dict=return_dict,
                cost_positive_ratio=cost_positive_ratio,
            )
        if source == SOURCE_MAIN:
            return self._sample_online(batch_size, horizon, return_dict=return_dict)
        if self.online_replay.can_sample(horizon):
            return self._sample_online(batch_size, horizon, return_dict=return_dict)
        if self.expert_replay is None:
            raise ValueError("Online replay cannot sample yet and no expert replay is available.")
        return self.expert_replay.sample(
            batch_size,
            horizon,
            source=SOURCE_EXPERT,
            return_dict=return_dict,
            cost_positive_ratio=cost_positive_ratio,
        )

    def _sample_online(self, batch_size, horizon, return_dict=False):
        try:
            return self.online_replay.sample(batch_size, horizon, return_dict=return_dict)
        except TypeError:
            samples = self.online_replay.sample(batch_size, horizon)
            return _legacy_samples_to_dict(samples, SOURCE_MAIN) if return_dict else samples

    def add_expert_episode(self, episode):
        if self.expert_replay is None:
            self.expert_replay = make_expert_replay(
                [episode],
                device=self.device,
                include_force=self.include_force,
                force_key=self.force_key,
            )
        else:
            self.expert_replay.add_expert_episode(episode)

    def add_expert_dataset(self, episodes):
        if self.expert_replay is None:
            self.expert_replay = make_expert_replay(
                episodes,
                device=self.device,
                include_force=self.include_force,
                force_key=self.force_key,
            )
        else:
            self.expert_replay.add_expert_dataset(episodes)

    def num_expert_steps(self):
        return 0 if self.expert_replay is None else self.expert_replay.num_expert_steps()

    def num_online_steps(self):
        if hasattr(self.online_replay, "num_online_steps"):
            return self.online_replay.num_online_steps()
        return len(self.online_replay)

    def source_stats(self):
        stats = Counter()
        if hasattr(self.online_replay, "source_stats"):
            stats.update(self.online_replay.source_stats())
        else:
            stats["main"] += len(self.online_replay)
        if self.expert_replay is not None:
            stats.update(self.expert_replay.source_stats())
        return dict(stats)
