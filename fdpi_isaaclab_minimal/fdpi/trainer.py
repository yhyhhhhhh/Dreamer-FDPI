from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .replay_buffer import TorchReplayBufferIS
from .sac_fpi_dual import TorchSACFPIDual

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    from tensorboardX import SummaryWriter


def policy_obs(obs_dict: dict[str, Any], device: torch.device) -> torch.Tensor:
    if not isinstance(obs_dict, dict) or "policy" not in obs_dict:
        raise KeyError("Expected IsaacLab observations to contain obs['policy'].")
    obs = obs_dict["policy"]
    if isinstance(obs, dict):
        tensors = [value.reshape(value.shape[0], -1) for value in obs.values() if torch.is_tensor(value)]
        if not tensors:
            raise TypeError("obs['policy'] is a dict but contains no torch tensors.")
        obs = torch.cat(tensors, dim=-1)
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs)
    return obs.to(device=device, dtype=torch.float32)


def _to_bool_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device).bool()
    return torch.as_tensor(value, device=device).bool()


def _to_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _reshape_env_vector(value: Any, device: torch.device, num_envs: int | None = None) -> torch.Tensor:
    tensor = _to_float_tensor(value, device).reshape(-1)
    if num_envs is not None and tensor.numel() == 1:
        tensor = tensor.expand(num_envs)
    return tensor


def _binary_or_continuous_cost(cost: torch.Tensor, continuous: bool) -> torch.Tensor:
    cost = torch.nan_to_num(cost, nan=0.0, posinf=1.0, neginf=0.0)
    if continuous:
        return cost.clamp(0.0, 1.0).to(torch.float32)
    return (cost > 0.0).to(torch.float32)


def _contact_force_cost(
    pipe_force: torch.Tensor,
    bottom_force: torch.Tensor,
    *,
    pipe_force_limit: float,
    bottom_force_limit: float,
    continuous: bool,
) -> torch.Tensor:
    pipe_limit = max(float(pipe_force_limit), 1e-6)
    bottom_limit = max(float(bottom_force_limit), 1e-6)
    pipe_excess = pipe_force / pipe_limit - 1.0
    bottom_excess = bottom_force / bottom_limit - 1.0
    excess = torch.maximum(pipe_excess, bottom_excess)
    if continuous:
        return torch.relu(excess).clamp(0.0, 1.0).to(torch.float32)
    return (excess > 0.0).to(torch.float32)


def _extract_contact_cost_from_obs(
    obs_dict: dict[str, Any] | None,
    device: torch.device,
    *,
    pipe_force_limit: float,
    bottom_force_limit: float,
    continuous: bool,
) -> torch.Tensor | None:
    if not isinstance(obs_dict, dict) or "force" not in obs_dict:
        return None
    force_obs = obs_dict["force"]
    if not torch.is_tensor(force_obs):
        force_obs = torch.as_tensor(force_obs)
    force_obs = force_obs.to(device=device, dtype=torch.float32)
    if force_obs.ndim < 2:
        return None
    force_obs = force_obs.reshape(force_obs.shape[0], -1)
    if force_obs.shape[-1] >= 6:
        pipe_force = torch.maximum(force_obs[:, 1], force_obs[:, 4])
        bottom_force = torch.maximum(force_obs[:, 2], force_obs[:, 5])
    elif force_obs.shape[-1] >= 2:
        pipe_force = force_obs[:, 0]
        bottom_force = force_obs[:, 1]
    else:
        return None
    return _contact_force_cost(
        pipe_force,
        bottom_force,
        pipe_force_limit=pipe_force_limit,
        bottom_force_limit=bottom_force_limit,
        continuous=continuous,
    )


def _extract_contact_cost_from_extras(
    extras: dict[str, Any],
    device: torch.device,
    *,
    pipe_force_limit: float,
    bottom_force_limit: float,
    continuous: bool,
    num_envs: int | None,
) -> torch.Tensor | None:
    diagnostics = extras.get("diagnostics", {}) if isinstance(extras, dict) else {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    for key in ("cost", "constraint"):
        value = extras.get(key) if isinstance(extras, dict) else None
        if value is None:
            value = diagnostics.get(key)
        if value is None:
            continue
        tensor = _to_float_tensor(value, device)
        if tensor.ndim >= 2:
            tensor = tensor.reshape(tensor.shape[0], -1).amax(dim=-1)
        else:
            tensor = tensor.reshape(-1)
            if num_envs is not None and tensor.numel() == 1:
                tensor = tensor.expand(num_envs)
        if continuous:
            return _binary_or_continuous_cost(tensor, continuous=True)
        break

    for key in ("constraint_violation", "violation"):
        if isinstance(extras, dict) and key in extras:
            return _reshape_env_vector(extras[key], device, num_envs).bool().to(torch.float32)
        if key in diagnostics:
            return _reshape_env_vector(diagnostics[key], device, num_envs).bool().to(torch.float32)

    for key in ("cost", "constraint"):
        value = extras.get(key) if isinstance(extras, dict) else None
        if value is None:
            value = diagnostics.get(key)
        if value is None:
            continue
        tensor = _to_float_tensor(value, device)
        if tensor.ndim >= 2:
            tensor = tensor.reshape(tensor.shape[0], -1).amax(dim=-1)
        else:
            tensor = tensor.reshape(-1)
            if num_envs is not None and tensor.numel() == 1:
                tensor = tensor.expand(num_envs)
        return _binary_or_continuous_cost(tensor, continuous=False)

    pipe_force = None
    bottom_force = None
    for mapping in (extras, diagnostics):
        if not isinstance(mapping, dict):
            continue
        pipe_force = pipe_force if pipe_force is not None else mapping.get("pipe_force")
        pipe_force = pipe_force if pipe_force is not None else mapping.get("wall_force")
        pipe_force = pipe_force if pipe_force is not None else mapping.get("wall_force_peak")
        bottom_force = bottom_force if bottom_force is not None else mapping.get("bottom_force")
        bottom_force = bottom_force if bottom_force is not None else mapping.get("bottom_force_peak")

    if pipe_force is None or bottom_force is None:
        return None
    pipe_tensor = _reshape_env_vector(pipe_force, device, num_envs)
    bottom_tensor = _reshape_env_vector(bottom_force, device, num_envs)
    return _contact_force_cost(
        pipe_tensor,
        bottom_tensor,
        pipe_force_limit=pipe_force_limit,
        bottom_force_limit=bottom_force_limit,
        continuous=continuous,
    )


def extract_safety_cost(
    extras: dict[str, Any],
    device: torch.device,
    obs_dict: dict[str, Any] | None,
    *,
    cost_source: str = "auto",
    pipe_force_limit: float = 1.0,
    bottom_force_limit: float = 1.0,
    continuous_contact_cost: bool = False,
    num_envs: int | None = None,
) -> torch.Tensor:
    diagnostics = extras.get("diagnostics", {}) if isinstance(extras, dict) else {}
    if cost_source in {"auto", "force_fail"}:
        if isinstance(diagnostics, dict) and "force_fail" in diagnostics:
            return _reshape_env_vector(diagnostics["force_fail"], device, num_envs).gt(0.0).to(torch.float32)
        if isinstance(extras, dict) and "force_fail" in extras:
            return _reshape_env_vector(extras["force_fail"], device, num_envs).gt(0.0).to(torch.float32)
        if cost_source == "force_fail":
            raise KeyError("Missing force_fail in extras or extras['diagnostics'].")

    if cost_source in {"auto", "contact_force"}:
        cost = _extract_contact_cost_from_extras(
            extras,
            device,
            pipe_force_limit=pipe_force_limit,
            bottom_force_limit=bottom_force_limit,
            continuous=continuous_contact_cost,
            num_envs=num_envs,
        )
        if cost is not None:
            return cost
        cost = _extract_contact_cost_from_obs(
            obs_dict,
            device,
            pipe_force_limit=pipe_force_limit,
            bottom_force_limit=bottom_force_limit,
            continuous=continuous_contact_cost,
        )
        if cost is not None:
            return cost

    raise KeyError("Missing safety cost. Expected force_fail, cost/constraint fields, or obs['force'].")


def terminal_next_obs(
    next_obs: torch.Tensor,
    done: torch.Tensor,
    extras: dict[str, Any],
    device: torch.device,
    require_terminal_observation: bool,
) -> torch.Tensor:
    if not bool(done.any().item()):
        return next_obs
    term_ids = extras.get("terminal_env_ids") if isinstance(extras, dict) else None
    term_obs = extras.get("terminal_observation") if isinstance(extras, dict) else None
    if torch.is_tensor(term_ids) and isinstance(term_obs, dict) and "policy" in term_obs and term_ids.numel() > 0:
        real_next_obs = next_obs.clone()
        ids = term_ids.to(device=device, dtype=torch.long)
        real_next_obs[ids] = policy_obs(term_obs, device)
        return real_next_obs
    if require_terminal_observation:
        raise RuntimeError("Episode ended but extras['terminal_observation']['policy'] was missing.")
    return next_obs


class FDPIIsaacLabTrainer:
    def __init__(
        self,
        *,
        env,
        algorithm: TorchSACFPIDual,
        buffer: TorchReplayBufferIS,
        log_dir: str | os.PathLike[str],
        total_steps: int,
        start_steps: int = 10000,
        batch_size: int = 512,
        beta: float = 0.5,
        dual_thresh: float = 0.8,
        feasible_window: int = 1000,
        sample_per_iteration: int = 1,
        updates_per_iteration: int = 1,
        log_every_steps: int = 10000,
        save_every_steps: int = 100000,
        max_checkpoints: int = 5,
        require_terminal_observation: bool = False,
        cost_source: str = "auto",
        pipe_force_cost_limit: float = 1.0,
        bottom_force_cost_limit: float = 1.0,
        continuous_contact_cost: bool = False,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        self.log_dir = Path(log_dir)
        self.total_steps = int(total_steps)
        self.start_steps = int(start_steps)
        self.batch_size = int(batch_size)
        self.beta = float(beta)
        self.dual_thresh = float(dual_thresh)
        self.feasible_window = int(feasible_window)
        self.sample_per_iteration = int(sample_per_iteration)
        self.updates_per_iteration = int(updates_per_iteration)
        self.log_every_steps = int(log_every_steps)
        self.save_every_steps = int(save_every_steps)
        self.max_checkpoints = int(max_checkpoints)
        self.require_terminal_observation = bool(require_terminal_observation)
        self.cost_source = str(cost_source)
        self.pipe_force_cost_limit = float(pipe_force_cost_limit)
        self.bottom_force_cost_limit = float(bottom_force_cost_limit)
        self.continuous_contact_cost = bool(continuous_contact_cost)
        self.device = algorithm.device
        self.num_envs = int(env.unwrapped.num_envs)
        if self.num_envs % 2 != 0:
            raise ValueError(f"FDPI dual sampling requires an even num_envs, got {self.num_envs}.")
        self.half_env_num = self.num_envs // 2
        self._saved: list[Path] = []

    def train(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(self.log_dir))

        reset_out = self.env.reset(seed=seed)
        obs_dict = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        obs = policy_obs(obs_dict, self.device)

        sample_steps = 0
        update_steps = 0
        next_log_step = self.log_every_steps
        next_save_step = self.save_every_steps
        feasible = deque(maxlen=self.feasible_window)
        stats: dict[str, list[float]] = {}
        ep_ret = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        ep_cost = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        ep_len = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        cumulative_log_weight = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        cumulative_log_weight_dual = torch.zeros_like(cumulative_log_weight)

        print(
            "[FDPI-MIN] train_start "
            f"total_steps={self.total_steps} num_envs={self.num_envs} start_steps={self.start_steps}",
            flush=True,
        )
        while sample_steps < self.total_steps:
            dual_active = len(feasible) > 0 and float(np.mean(feasible)) > self.dual_thresh
            self._append(stats, "dual_active", float(dual_active))

            for _ in range(self.sample_per_iteration):
                if sample_steps < self.start_steps:
                    action = torch.empty(
                        self.num_envs, self.algorithm.act_dim, dtype=torch.float32, device=self.device
                    ).uniform_(-1.0, 1.0)
                    log_weight = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                    log_weight_dual = torch.zeros_like(log_weight)
                else:
                    act, dual_act, log_weight, log_weight_dual = self.algorithm.act(obs, dual_active)
                    action = torch.cat((act[: self.half_env_num], dual_act[self.half_env_num :]), dim=0)

                next_obs_dict, reward, terminated, truncated, extras = self.env.step(action)
                reward = _to_float_tensor(reward, self.device).reshape(-1)
                terminated = _to_bool_tensor(terminated, self.device).reshape(-1)
                truncated = _to_bool_tensor(truncated, self.device).reshape(-1)
                episode_done = terminated | truncated
                bootstrap_done = terminated & ~truncated
                next_obs = policy_obs(next_obs_dict, self.device)
                cost = extract_safety_cost(
                    extras,
                    self.device,
                    next_obs_dict,
                    cost_source=self.cost_source,
                    pipe_force_limit=self.pipe_force_cost_limit,
                    bottom_force_limit=self.bottom_force_cost_limit,
                    continuous_contact_cost=self.continuous_contact_cost,
                    num_envs=self.num_envs,
                )
                real_next_obs = terminal_next_obs(
                    next_obs,
                    episode_done,
                    extras,
                    self.device,
                    self.require_terminal_observation,
                )

                self.buffer.add_batch(
                    obs.detach(),
                    action.detach(),
                    real_next_obs.detach(),
                    reward.detach(),
                    cost.detach(),
                    bootstrap_done.to(torch.float32).detach(),
                    cumulative_log_weight.detach(),
                    cumulative_log_weight_dual.detach(),
                )

                ep_ret += reward.to(torch.float64)
                ep_cost += cost.to(torch.float64)
                ep_len += 1
                self._append(stats, "step_cost_rate", float(cost.mean().detach().cpu()))

                done_idx = episode_done.nonzero(as_tuple=False).squeeze(-1)
                if done_idx.numel() > 0:
                    self._extend(stats, "episode_return", ep_ret[done_idx].detach().cpu().tolist())
                    self._extend(stats, "episode_cost", ep_cost[done_idx].detach().cpu().tolist())
                    self._extend(stats, "episode_length", ep_len[done_idx].detach().cpu().tolist())
                    ep_ret[done_idx] = 0.0
                    ep_cost[done_idx] = 0.0
                    ep_len[done_idx] = 0

                if sample_steps >= self.start_steps:
                    cumulative_log_weight[: self.half_env_num] = self.beta * (
                        cumulative_log_weight[: self.half_env_num] + log_weight[: self.half_env_num]
                    )
                    cumulative_log_weight_dual[self.half_env_num :] = self.beta * (
                        cumulative_log_weight_dual[self.half_env_num :]
                        + log_weight_dual[self.half_env_num :]
                    )
                    self._append(stats, "main_is_weight", float(torch.exp(log_weight[: self.half_env_num]).mean().cpu()))
                    self._append(
                        stats,
                        "dual_is_weight",
                        float(torch.exp(log_weight_dual[self.half_env_num :]).mean().cpu()),
                    )
                    if done_idx.numel() > 0:
                        cumulative_log_weight[done_idx] = 0.0
                        cumulative_log_weight_dual[done_idx] = 0.0

                obs = next_obs
                sample_steps += self.num_envs

            if sample_steps >= self.start_steps and len(self.buffer) >= self.batch_size:
                for _ in range(self.updates_per_iteration):
                    info = self.algorithm.update(self.buffer.sample(self.batch_size))
                    feasible.append(float(info["feasible_ratio"]))
                    for key, value in info.items():
                        if np.isfinite(value):
                            self._append(stats, f"update/{key}", float(value))
                    update_steps += 1

            if sample_steps >= next_log_step:
                self._flush(writer, stats, sample_steps, update_steps)
                while next_log_step <= sample_steps:
                    next_log_step += self.log_every_steps

            if self.save_every_steps > 0 and sample_steps >= next_save_step:
                self.save(sample_steps, update_steps)
                while next_save_step <= sample_steps:
                    next_save_step += self.save_every_steps

        self._flush(writer, stats, sample_steps, update_steps)
        self.save(sample_steps, update_steps)
        writer.close()
        print(
            "[FDPI-MIN] train_complete "
            f"sample_steps={sample_steps} update_steps={update_steps} log_dir={self.log_dir}",
            flush=True,
        )

    @staticmethod
    def _append(stats: dict[str, list[float]], key: str, value: float) -> None:
        stats.setdefault(key, []).append(float(value))

    @staticmethod
    def _extend(stats: dict[str, list[float]], key: str, values: list[float]) -> None:
        stats.setdefault(key, []).extend(float(value) for value in values)

    def _flush(
        self,
        writer: SummaryWriter,
        stats: dict[str, list[float]],
        sample_steps: int,
        update_steps: int,
    ) -> None:
        summary = {}
        for key, values in list(stats.items()):
            if not values:
                continue
            mean_value = float(np.mean(values))
            writer.add_scalar(key, mean_value, sample_steps)
            summary[key] = mean_value
            stats[key] = []
        writer.add_scalar("progress/sample_steps", sample_steps, sample_steps)
        writer.add_scalar("progress/update_steps", update_steps, sample_steps)
        writer.flush()
        print(
            "[FDPI-MIN] "
            f"sample_steps={sample_steps} update_steps={update_steps} "
            f"return={summary.get('episode_return', float('nan')):.3f} "
            f"episode_cost={summary.get('episode_cost', float('nan')):.3f} "
            f"step_cost_rate={summary.get('step_cost_rate', float('nan')):.4f}",
            flush=True,
        )

    def save(self, sample_steps: int, update_steps: int) -> None:
        path = self.log_dir / f"checkpoint_s{sample_steps}_u{update_steps}.pt"
        torch.save(self.algorithm.checkpoint(), path)
        self._saved.append(path)
        while len(self._saved) > self.max_checkpoints:
            old_path = self._saved.pop(0)
            if old_path.exists():
                old_path.unlink()
        print(f"[FDPI-MIN] saved_checkpoint={path}", flush=True)


def dump_json(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
