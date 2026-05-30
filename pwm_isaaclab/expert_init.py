from __future__ import annotations

import os
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from tqdm import tqdm

try:
    from pwm_isaaclab.expert_loader import SOURCE_EXPERT
    from pwm_isaaclab.expert_world_model import cost_prediction_metrics
except ImportError:
    from expert_loader import SOURCE_EXPERT
    from expert_world_model import cost_prediction_metrics


def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    return tuple(value.to(device) if torch.is_tensor(value) else value for value in batch)


def _log_metrics(logger, prefix, metrics, step):
    if logger is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.log(f"{prefix}/{key}", float(value), step)


def _actor_dist(agent, feat):
    mean, raw_std = agent.actor(feat).chunk(2, dim=-1)
    std = agent.std_scale * torch.sigmoid(raw_std + 2) + agent.std_offset
    return Independent(Normal(torch.tanh(mean), std), 1), torch.tanh(mean)


def _unwrap_optimizer_step(optimizer):
    optimizer_cls = optimizer.__class__
    if hasattr(optimizer_cls.step, "__wrapped__"):
        optimizer_cls.step = optimizer_cls.step.__wrapped__
    if hasattr(optimizer.step, "__wrapped__"):
        optimizer.step = optimizer.step.__wrapped__.__get__(optimizer, optimizer_cls)


@torch.no_grad()
def _posterior_features_for_bc(world_model, obs, action, is_first):
    batch_size, horizon = obs.shape[:2]
    state = world_model.initial(batch_size)
    feats = []
    for step in range(horizon):
        feat, state = world_model.get_inference_feat(state, obs[:, step], is_first[:, step])
        feats.append(feat.detach())
        state = world_model.update_inference_state(state, action[:, step])
    return torch.stack(feats, dim=1)


def _bc_loss_and_metrics(world_model, agent, batch):
    obs = batch["obs"]
    action = batch["action"].clamp(-1.0, 1.0)
    is_first = batch["is_first"]
    with torch.no_grad():
        feat = _posterior_features_for_bc(world_model, obs, action, is_first)
    flat_feat = feat.reshape(-1, feat.shape[-1])
    flat_action = action.reshape(-1, action.shape[-1])

    with torch.autocast(device_type=agent.device_type, dtype=agent.tensor_dtype, enabled=agent.use_amp):
        dist, mean_action = _actor_dist(agent, flat_feat)
        log_prob = dist.log_prob(flat_action)
        bc_loss = -log_prob.mean()
        action_mse = F.mse_loss(mean_action, flat_action)
        entropy = dist.entropy().mean()

    metrics = {
        "bc_loss": float(bc_loss.detach().float().item()),
        "action_mse": float(action_mse.detach().float().item()),
        "action_log_prob": float(log_prob.detach().float().mean().item()),
        "actor_entropy": float(entropy.detach().float().item()),
    }
    return bc_loss, metrics


def pretrain_world_model_from_expert(
    replay,
    world_model,
    *,
    num_steps,
    batch_size,
    batch_length,
    logger=None,
    log_interval=100,
    cost_positive_ratio=0.0,
    progress=True,
):
    if int(num_steps) <= 0:
        return {}
    if not replay.can_sample(batch_length, source=SOURCE_EXPERT):
        raise ValueError(
            f"Expert replay cannot sample batch_length={batch_length}. "
            "Check expert episode lengths and replay warmup."
        )

    last_metrics = {}
    iterator = range(1, int(num_steps) + 1)
    for update_idx in tqdm(iterator, disable=not progress):
        batch = replay.sample(
            batch_size,
            batch_length,
            source=SOURCE_EXPERT,
            return_dict=True,
            cost_positive_ratio=cost_positive_ratio,
        )
        batch = batch_to_device(batch, world_model.device)
        metrics = world_model.update(
            None,
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["done"],
            batch["is_first"],
            force=batch.get("force"),
            cost=batch.get("cost"),
            logger=None,
            step=update_idx,
        )
        last_metrics = metrics
        if update_idx == 1 or update_idx % max(int(log_interval), 1) == 0 or update_idx == int(num_steps):
            _log_metrics(logger, "expert_init", metrics, update_idx)
    return last_metrics


def pretrain_actor_bc_from_expert(
    replay,
    world_model,
    agent,
    *,
    num_steps,
    batch_size,
    batch_length,
    logger=None,
    log_interval=100,
    progress=True,
):
    if int(num_steps) <= 0:
        return {}
    if not replay.can_sample(batch_length, source=SOURCE_EXPERT):
        raise ValueError(
            f"Expert replay cannot sample batch_length={batch_length}. "
            "Check expert episode lengths and replay warmup."
        )

    world_model.eval()
    _unwrap_optimizer_step(agent.optimizer)
    last_metrics = {}
    iterator = range(1, int(num_steps) + 1)
    for update_idx in tqdm(iterator, disable=not progress):
        batch = replay.sample(batch_size, batch_length, source=SOURCE_EXPERT, return_dict=True)
        batch = batch_to_device(batch, agent.device)
        agent.train()
        agent.optimizer.zero_grad(set_to_none=True)
        bc_loss, metrics = _bc_loss_and_metrics(world_model, agent, batch)
        agent.scaler.scale(bc_loss).backward()
        agent.scaler.unscale_(agent.optimizer)
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=100.0)
        with torch.no_grad():
            agent.scaler.step(agent.optimizer)
            agent.scaler.update()
        agent.optimizer.zero_grad(set_to_none=True)
        last_metrics = metrics
        if update_idx == 1 or update_idx % max(int(log_interval), 1) == 0 or update_idx == int(num_steps):
            _log_metrics(logger, "expert_init", metrics, update_idx)
    return last_metrics


@torch.no_grad()
def evaluate_world_model_on_expert(world_model, batch):
    world_model.eval()
    batch = batch_to_device(batch, world_model.device)
    obs = batch["obs"]
    action = batch["action"]
    reward = batch["reward"]
    done = batch["done"]
    is_first = batch["is_first"]
    cost = batch.get("cost")

    with torch.autocast(device_type=world_model.device_type, dtype=world_model.tensor_dtype, enabled=world_model.use_amp):
        post, prior, stoch, deter = world_model.dynamic.parallel_observe(world_model.encoder(obs), action, is_first)
        dyn_loss, rep_loss, real_kl, ent = world_model.dynamic.kl_loss(post, prior, world_model.kl_free)
        obs_hat = world_model.decoder(stoch)
        reward_logits = world_model.reward_head(deter)
        reward_pred = world_model.twohot_loss.decode(reward_logits)
        done_logits = world_model.done_head(deter)

        recon_loss = world_model.mse_loss(obs_hat, obs)
        reward_loss = world_model.twohot_loss(reward_logits, reward)
        done_loss = F.binary_cross_entropy_with_logits(done_logits, done)
        reward_abs_error = (reward_pred - reward).abs().mean()
        metrics = {
            "validation_recon_loss": float(recon_loss.detach().float().item()),
            "validation_reward_loss": float(reward_loss.detach().float().item()),
            "validation_reward_mae": float(reward_abs_error.detach().float().item()),
            "validation_discount_loss": float(done_loss.detach().float().item()),
            "validation_dyn_loss": float(dyn_loss.detach().float().item()),
            "validation_rep_loss": float(rep_loss.detach().float().item()),
            "validation_real_kl": float(real_kl.detach().float().item()),
            "validation_posterior_entropy": float(ent.detach().float().item()),
        }
        if cost is not None:
            posterior_feat = torch.cat((deter, stoch), dim=-1)
            cost_pred, p_violate, _ = world_model.predict_cost(posterior_feat)
            target = cost.reshape_as(cost_pred).to(cost_pred.dtype)
            metrics["validation_cost_mse"] = float(F.mse_loss(cost_pred, target).detach().float().item())
            metrics["validation_cost_mae"] = float((cost_pred - target).abs().mean().detach().float().item())
            metrics.update(cost_prediction_metrics(cost_pred, p_violate, target, prefix="validation_cost"))

            prior_stoch = world_model.dynamic.get_flatten_stoch(prior)
            prior_feat = torch.cat((prior["deter"], prior_stoch), dim=-1)
            prior_target = cost[:, 1 : 1 + prior_feat.shape[1]]
            if prior_feat.numel() > 0 and prior_target.numel() > 0:
                prior_pred, prior_prob, _ = world_model.predict_cost(prior_feat)
                prior_target = prior_target.reshape_as(prior_pred).to(prior_pred.dtype)
                metrics["validation_cost_prior_mse"] = float(F.mse_loss(prior_pred, prior_target).detach().float().item())
                metrics["validation_cost_prior_mae"] = float((prior_pred - prior_target).abs().mean().detach().float().item())
                metrics.update(cost_prediction_metrics(prior_pred, prior_prob, prior_target, prefix="validation_cost_prior"))
    return metrics


@torch.no_grad()
def evaluate_actor_bc_on_expert(world_model, agent, batch):
    world_model.eval()
    agent.eval()
    batch = batch_to_device(batch, agent.device)
    _, metrics = _bc_loss_and_metrics(world_model, agent, batch)
    return {f"validation_{key}": value for key, value in metrics.items()}


def save_expert_checkpoint(path, *, world_model=None, agent=None, config=None, expert_metadata=None, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, Any] = {
        "expert_metadata": expert_metadata or {},
        "extra": extra or {},
    }
    if config is not None:
        payload["config"] = config
    if world_model is not None:
        payload["world_model_state_dict"] = world_model.state_dict()
    if agent is not None:
        payload["agent_state_dict"] = agent.state_dict()
    torch.save(payload, path)
    return path


def load_expert_checkpoint(path, *, world_model=None, agent=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    if world_model is not None:
        state = checkpoint.get("world_model_state_dict", checkpoint)
        try:
            world_model.load_state_dict(state)
        except RuntimeError:
            result = world_model.load_state_dict(state, strict=False)
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            allowed_missing = all(key.startswith("cost_head.") for key in missing)
            allowed_unexpected = all(key.startswith("cost_head.") for key in unexpected)
            if not allowed_missing or not allowed_unexpected:
                raise
            warnings.warn(
                "Loaded a world model checkpoint with incompatible cost_head parameters; "
                "the new cost_head keeps its initialized parameters.",
                RuntimeWarning,
            )
    if agent is not None:
        state = checkpoint.get("agent_state_dict", checkpoint)
        agent.load_state_dict(state)
    return checkpoint
