from __future__ import annotations

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
    if tensor.ndim >= 2:
        if num_envs is not None and tensor.shape[0] == num_envs:
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


def _bottom_force_from_obs(
    obs_dict: dict[str, Any] | None,
    *,
    device,
    num_envs: int,
    force_key: str = "",
    bottom_force_channels: tuple[int, int] = (2, 5),
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
        force = force.reshape(num_envs, -1)
        if force.shape[-1] >= max(bottom_force_channels) + 1:
            channels = torch.as_tensor(bottom_force_channels, dtype=torch.long, device=device)
            return force.index_select(-1, channels).amax(dim=-1)
        if force.shape[-1] >= 2:
            return force[:, 1]
    return None


def extract_bottom_force_cost(
    info: dict[str, Any] | None,
    obs_dict: dict[str, Any] | None,
    *,
    num_envs: int,
    device,
    threshold: float = 1.0,
    force_key: str = "",
    bottom_force_channels: tuple[int, int] = (2, 5),
) -> torch.Tensor:
    """Extract a binary violation label from bottom force only."""

    value = _mapping_value(info, ("bottom_force", "bottom_force_peak"))
    bottom_force = None
    if value is not None:
        bottom_force = _to_float_vector(value, device, num_envs)
    if bottom_force is None:
        bottom_force = _bottom_force_from_obs(
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
            "DFD bottom-force cost requested, but bottom force was not found. "
            f"Expected info['bottom_force']/diagnostics['bottom_force'] or obs['force']; "
            f"info keys=[{available_info}], obs keys=[{available_obs}]."
        )
    bottom_force = torch.nan_to_num(bottom_force.reshape(num_envs), nan=0.0, posinf=threshold + 1.0)
    return (bottom_force > float(threshold)).to(torch.float32).view(num_envs, 1)


def dreamer_agent_distribution(agent, feat):
    mean, std = agent.actor(feat).chunk(2, dim=-1)
    std = agent.std_scale * torch.sigmoid(std + 2) + agent.std_offset
    return independent.Independent(normal.Normal(torch.tanh(mean), std), 1)


def risk_advantage_modifier(
    *,
    feat: torch.Tensor,
    action: torch.Tensor,
    norm_adv: torch.Tensor,
    weight: torch.Tensor | None = None,
    feasibility,
    pf: float,
    cg: float,
    lambda_cri: float,
    lambda_inf: float,
    clip_safe_adv: bool = False,
    safe_adv_min: float = -5.0,
    safe_adv_max: float = 5.0,
    logger=None,
    step=None,
) -> torch.Tensor:
    with torch.no_grad():
        g = torch.maximum(
            feasibility.gp1(feat, action),
            feasibility.gp2(feat, action),
        ).clamp(0.0, 1.0)
        g_det = g.detach()
        pf = float(pf)
        cg = float(cg)
        lambda_cri = float(lambda_cri)
        lambda_inf = float(lambda_inf)

        feasible = g_det < (pf - cg)
        critical = (g_det >= (pf - cg)) & (g_det < pf)
        infeasible = g_det >= pf
        risk_margin = F.relu(g_det - (pf - cg)) / (cg + 1e-6)
        risk_excess = F.relu(g_det - pf) / (1.0 - pf + 1e-6)

        safe_adv = torch.zeros_like(norm_adv)
        safe_adv[feasible] = norm_adv[feasible]
        safe_adv[critical] = norm_adv[critical] - lambda_cri * risk_margin[critical]
        safe_adv[infeasible] = (
            torch.minimum(norm_adv[infeasible], torch.zeros_like(norm_adv[infeasible]))
            - lambda_inf * risk_excess[infeasible]
        )
        if clip_safe_adv:
            safe_adv = safe_adv.clamp(float(safe_adv_min), float(safe_adv_max))

    if logger is not None:
        logger.log("ActorCritic/risk_g_mean", g_det.float().mean().item(), step)
        logger.log("ActorCritic/risk_penalty", (norm_adv - safe_adv).float().mean().item(), step)
        logger.log("ActorCritic/feasible_ratio", feasible.float().mean().item(), step)
        logger.log("ActorCritic/critical_ratio", critical.float().mean().item(), step)
        logger.log("ActorCritic/infeasible_ratio", infeasible.float().mean().item(), step)
        logger.log("ActorCritic/safe_adv", safe_adv.float().mean().item(), step)
    return safe_adv


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
