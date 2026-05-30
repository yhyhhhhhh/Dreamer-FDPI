from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import independent, normal


SOURCE_MAIN = 0
SOURCE_DUAL = 1
SOURCE_RANDOM = 2

SOURCE_NAMES = {
    SOURCE_MAIN: "main",
    SOURCE_DUAL: "dual",
    SOURCE_RANDOM: "random",
}


def cfg_get(node: Any, name: str, default: Any = None) -> Any:
    if node is None:
        return default
    if hasattr(node, name):
        return getattr(node, name)
    if isinstance(node, dict):
        return node.get(name, default)
    return default


def as_column(value, length: int, device, dtype=torch.float32, fill_value=0.0):
    if value is None:
        return torch.full((length, 1), fill_value, dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(length)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    return tensor.reshape(length, -1)[:, :1]


def linear_warmup(step: int | float, start: float, final: float, warmup_steps: int | float) -> float:
    warmup_steps = max(float(warmup_steps), 1.0)
    ratio = min(max(float(step) / warmup_steps, 0.0), 1.0)
    return float(start) + ratio * (float(final) - float(start))


def _to_float_vector(value, device, num_envs: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        if num_envs is None:
            return tensor.reshape(1)
        return tensor.reshape(1).expand(num_envs)
    if tensor.ndim >= 2 and num_envs is not None and tensor.shape[0] == num_envs:
        return tensor.reshape(num_envs, -1).amax(dim=-1)
    tensor = tensor.reshape(-1)
    if num_envs is not None and tensor.numel() == 1:
        tensor = tensor.expand(num_envs)
    return tensor


def _mapping_value(mapping: dict[str, Any] | None, keys: tuple[str, ...]):
    if not isinstance(mapping, dict):
        return None
    diagnostics = mapping.get("diagnostics", {})
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
        if isinstance(diagnostics, dict) and key in diagnostics and diagnostics[key] is not None:
            return diagnostics[key]
    return None


def bottom_force_from_obs(
    obs_dict: dict[str, Any] | None,
    *,
    device,
    num_envs: int,
    force_key: str = "",
    bottom_force_channels: tuple[int, ...] = (2, 5),
) -> torch.Tensor | None:
    if not isinstance(obs_dict, dict):
        return None
    candidate_keys = tuple(key for key in (force_key, "force") if key)
    for key in candidate_keys:
        value = obs_dict.get(key)
        if value is None:
            continue
        force = torch.as_tensor(value, dtype=torch.float32, device=device)
        if force.shape[0] != num_envs:
            continue
        force = force.reshape(num_envs, -1).abs()
        if bottom_force_channels and force.shape[-1] >= max(bottom_force_channels) + 1:
            channels = torch.as_tensor(bottom_force_channels, dtype=torch.long, device=device)
            return force.index_select(-1, channels).amax(dim=-1)
        if force.shape[-1] >= 2:
            return force[:, 1]
        return force[:, 0]
    return None


def extract_bottom_force(
    info: dict[str, Any] | None,
    obs_dict: dict[str, Any] | None,
    *,
    num_envs: int,
    device,
    force_key: str = "",
    bottom_force_channels: tuple[int, ...] = (2, 5),
) -> torch.Tensor:
    value = _mapping_value(info, ("bottom_force", "bottom_force_peak"))
    bottom_force = None
    if value is not None:
        bottom_force = _to_float_vector(value, device, num_envs)
    if bottom_force is None:
        bottom_force = bottom_force_from_obs(
            obs_dict,
            device=device,
            num_envs=num_envs,
            force_key=force_key,
            bottom_force_channels=bottom_force_channels,
        )
    if bottom_force is None:
        available_info = ", ".join(sorted(info.keys())) if isinstance(info, dict) else ""
        available_obs = ", ".join(sorted(obs_dict.keys())) if isinstance(obs_dict, dict) else ""
        raise KeyError(
            "DFD v2 bottom-force cost requested, but bottom force was not found. "
            "Expected info['bottom_force']/diagnostics['bottom_force'] or obs['force']; "
            f"info keys=[{available_info}], obs keys=[{available_obs}]."
        )
    return torch.nan_to_num(bottom_force.reshape(num_envs), nan=0.0, posinf=1.0e6).clamp_min(0.0)


def compute_continuous_cost(
    bottom_force,
    *,
    force_threshold: float,
    low_force_scale: float | None = None,
    cost_force_max: float | None = None,
    extreme_force_threshold: float = 5.0,
    force_scale: float | None = None,
    clip_cost: bool = True,
    cost_min: float = 0.0,
    cost_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    bottom_force = torch.as_tensor(bottom_force, dtype=torch.float32)
    force_excess = F.relu(bottom_force - float(force_threshold))
    if low_force_scale is None:
        low_force_scale = force_scale if force_scale is not None else 0.05
    if cost_force_max is None:
        cost_force_max = force_scale if force_scale is not None else 15.0
    low_scale = max(float(low_force_scale), 1.0e-6)
    force_max = max(float(cost_force_max), low_scale)
    normalizer = torch.log1p(bottom_force.new_tensor(force_max / low_scale)).clamp_min(1.0e-6)
    continuous_cost = torch.log1p(force_excess / low_scale) / normalizer
    if clip_cost:
        continuous_cost = continuous_cost.clamp(float(cost_min), float(cost_max))
    binary_cost = (bottom_force > float(force_threshold)).to(torch.float32)
    extreme_cost = (bottom_force > float(extreme_force_threshold)).to(torch.float32)
    return {
        "bottom_force": bottom_force.reshape(-1, 1),
        "force_excess": force_excess.reshape(-1, 1),
        "continuous_cost": continuous_cost.reshape(-1, 1),
        "binary_cost": binary_cost.reshape(-1, 1),
        "extreme_cost": extreme_cost.reshape(-1, 1),
    }


def extract_continuous_cost(
    info: dict[str, Any] | None,
    obs_dict: dict[str, Any] | None,
    *,
    num_envs: int,
    device,
    force_threshold: float,
    low_force_scale: float | None = None,
    cost_force_max: float | None = None,
    extreme_force_threshold: float = 5.0,
    force_scale: float | None = None,
    clip_cost: bool = True,
    cost_min: float = 0.0,
    cost_max: float = 1.0,
    force_key: str = "",
    bottom_force_channels: tuple[int, ...] = (2, 5),
) -> dict[str, torch.Tensor]:
    bottom_force = extract_bottom_force(
        info,
        obs_dict,
        num_envs=num_envs,
        device=device,
        force_key=force_key,
        bottom_force_channels=bottom_force_channels,
    )
    out = compute_continuous_cost(
        bottom_force,
        force_threshold=force_threshold,
        low_force_scale=low_force_scale,
        cost_force_max=cost_force_max,
        extreme_force_threshold=extreme_force_threshold,
        force_scale=force_scale,
        clip_cost=clip_cost,
        cost_min=cost_min,
        cost_max=cost_max,
    )
    return {key: value.to(device=device) for key, value in out.items()}


def continuous_cost_from_force_prediction(
    pred_bottom_force,
    *,
    force_threshold: float,
    low_force_scale: float | None = None,
    cost_force_max: float | None = None,
    force_scale: float | None = None,
    clip_cost: bool = True,
    cost_min: float = 0.0,
    cost_max: float = 1.0,
) -> torch.Tensor:
    shape = pred_bottom_force.shape
    out = compute_continuous_cost(
        pred_bottom_force.reshape(-1),
        force_threshold=force_threshold,
        low_force_scale=low_force_scale,
        cost_force_max=cost_force_max,
        force_scale=force_scale,
        clip_cost=clip_cost,
        cost_min=cost_min,
        cost_max=cost_max,
    )
    return out["continuous_cost"].reshape(*shape[:-1], 1)


def dreamer_agent_distribution(agent, feat):
    mean, std = agent.actor(feat).chunk(2, dim=-1)
    std = agent.std_scale * torch.sigmoid(std + 2) + agent.std_offset
    return independent.Independent(normal.Normal(torch.tanh(mean), std), 1)


@torch.no_grad()
def posterior_features(world_model, obs, action, is_first):
    with torch.autocast(
        device_type=world_model.device_type,
        dtype=world_model.tensor_dtype,
        enabled=world_model.use_amp,
    ):
        embed = world_model.encoder(obs)
        post, _, _, _ = world_model.dynamic.parallel_observe(embed, action, is_first)
        return world_model.dynamic.get_feat(post)


@torch.no_grad()
def posterior_states(world_model, obs, action, is_first):
    with torch.autocast(
        device_type=world_model.device_type,
        dtype=world_model.tensor_dtype,
        enabled=world_model.use_amp,
    ):
        embed = world_model.encoder(obs)
        post, _, _, _ = world_model.dynamic.parallel_observe(embed, action, is_first)
        return post


@contextmanager
def temporarily_disable_grads(module):
    params = list(module.parameters())
    old_values = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, old_value in zip(params, old_values):
            param.requires_grad_(old_value)


def disable_optimizer_dynamo_wrappers():
    if hasattr(torch.optim.Optimizer.add_param_group, "__wrapped__"):
        torch.optim.Optimizer.add_param_group = torch.optim.Optimizer.add_param_group.__wrapped__
    if hasattr(torch.optim.Optimizer.zero_grad, "__wrapped__"):
        torch.optim.Optimizer.zero_grad = torch.optim.Optimizer.zero_grad.__wrapped__
    if hasattr(torch.optim.Optimizer.state_dict, "__wrapped__"):
        torch.optim.Optimizer.state_dict = torch.optim.Optimizer.state_dict.__wrapped__


def unwrap_optimizer_step(optimizer):
    optimizer_cls = optimizer.__class__
    if hasattr(optimizer_cls.step, "__wrapped__"):
        optimizer_cls.step = optimizer_cls.step.__wrapped__
    if hasattr(optimizer.step, "__wrapped__"):
        optimizer.step = optimizer.step.__wrapped__.__get__(optimizer, optimizer_cls)


def ensure_optimizer_step_no_grad(optimizer):
    """Protect AdamW after wrapper unwrapping by running instance step in no_grad."""
    if getattr(optimizer, "_dfd_v2_step_no_grad", False):
        return optimizer
    step_fn = optimizer.step

    @functools.wraps(step_fn)
    def step_no_grad(*args, **kwargs):
        with torch.no_grad():
            return step_fn(*args, **kwargs)

    optimizer.step = step_no_grad
    optimizer._dfd_v2_step_no_grad = True
    return optimizer
