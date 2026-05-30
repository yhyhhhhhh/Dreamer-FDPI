from __future__ import annotations

"""
Evaluate an expert-initialized policy/world model on offline datasets and online IsaacLab rollouts.

Typical offline dataset evaluation:

WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_before_online.pt \
  -device cuda:0 \
  --eval_dataset \
  --num_batches 200 \
  --batch_length 64 \
  --batch_size 64 \
  -save_dir eval_expert_policy_world_model

Quick offline smoke evaluation:

WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_before_online.pt \
  -device cuda:0 \
  --eval_dataset \
  -max_episodes 64 \
  -max_coverage_episodes 128 \
  --num_batches 20 \
  --batch_length 64 \
  --batch_size 64

Online environment world-model prediction evaluation:

WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /home/yhy/PaMoRL-main/ckpt/ur3-critic-warmup/20260524_223011_warmup/full_agent_after_critic_warmup.pt  \
  -env_name Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1 \
  -device cuda:0 \
  --online \
  -online_steps 8192 \
  -eval_num_envs 8 \
  -policy_mode greedy \
  --num_batches 100 \
  --batch_length 64 \
  --batch_size 64

Evaluate both offline datasets and online rollouts in one run by passing both --eval_dataset and --online.
"""

import argparse
import csv
import json
import math
import os
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import colorama
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pwm_isaaclab.expert_config import cfg_to_dict, load_expert_config
    from pwm_isaaclab.expert_init import load_expert_checkpoint
    from pwm_isaaclab.expert_loader import derive_cost_from_force_margin, load_expert_dataset
    from pwm_isaaclab.expert_pretrain import build_agent, build_world_model
    from pwm_isaaclab.expert_replay import SourceTaggedProprioReplayBuffer, make_expert_replay
    from pwm_isaaclab.expert_world_model import cost_prediction_metrics
    from pwm_isaaclab.modules.world_models import predict_force_from_outputs
    from pwm_isaaclab.trainer import _extract_force_obs, _is_first, _policy_obs, _reset_after_step
    from pwm_isaaclab.utils import seed_np_torch
except ImportError:
    from expert_config import cfg_to_dict, load_expert_config
    from expert_init import load_expert_checkpoint
    from expert_loader import derive_cost_from_force_margin, load_expert_dataset
    from expert_pretrain import build_agent, build_world_model
    from expert_replay import SourceTaggedProprioReplayBuffer, make_expert_replay
    from expert_world_model import cost_prediction_metrics
    from modules.world_models import predict_force_from_outputs
    from trainer import _extract_force_obs, _is_first, _policy_obs, _reset_after_step
    from utils import seed_np_torch


def _as_path_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item]


def _safe_float(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().item()
    return float(value)


def _safe_corrcoef(target, pred):
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if target.size == 0 or pred.size == 0:
        return 0.0
    target = target - target.mean()
    pred = pred - pred.mean()
    denom = float(np.linalg.norm(target) * np.linalg.norm(pred))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(target, pred) / denom)


def _get_world_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "world_model_state_dict" in checkpoint:
        return checkpoint["world_model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _infer_checkpoint_obs_dim(args):
    checkpoint_source = args.checkpoint_path or args.world_model_path
    if not checkpoint_source:
        return None
    checkpoint = torch.load(os.path.expanduser(checkpoint_source), map_location="cpu")
    state = _get_world_model_state_dict(checkpoint)
    if not isinstance(state, dict):
        return None
    for key in (
        "encoder.backbone.0.layer.0.weight",
        "decoder.backbone.4.bias",
        "decoder.backbone.4.weight",
    ):
        value = state.get(key)
        if torch.is_tensor(value):
            if key.endswith(".weight") and value.ndim >= 2:
                return int(value.shape[1] if key.startswith("encoder") else value.shape[0])
            if key.endswith(".bias") and value.ndim == 1:
                return int(value.shape[0])
    return None


def _match_obs_dim(obs, target_dim):
    current_dim = int(obs.shape[-1])
    target_dim = int(target_dim)
    if current_dim == target_dim:
        return obs
    if current_dim < target_dim:
        pad_shape = obs.shape[:-1] + (target_dim - current_dim,)
        padding = torch.zeros(pad_shape, dtype=obs.dtype, device=obs.device)
        return torch.cat((obs, padding), dim=-1)
    return obs[..., :target_dim]


def _is_binary_target(target):
    finite = target[torch.isfinite(target)]
    return (
        finite.numel() > 0
        and finite.min() >= 0
        and finite.max() <= 1
        and torch.allclose(finite, finite.round(), atol=1e-5)
    )


def _predict_force(world_model, feat):
    outputs = world_model.force_head(feat.reshape(-1, feat.shape[-1]))
    pred_force, nonzero_prob = predict_force_from_outputs(
        outputs,
        force_scale=world_model.force_scale,
        threshold=world_model.force_threshold,
        signed_force=world_model.force_signed_force,
    )
    view_shape = feat.shape[:-1] + (1,)
    return pred_force.view(*view_shape), nonzero_prob.view(*view_shape)


def _cost_loader_kwargs(conf):
    return {
        "cost_target_source": conf.expert.cost_target_source,
        "cost_pipe_force_limit": conf.expert.cost_pipe_force_limit,
        "cost_bottom_force_limit": conf.expert.cost_bottom_force_limit,
        "cost_pipe_force_channels": conf.expert.cost_pipe_force_channels,
        "cost_bottom_force_channels": conf.expert.cost_bottom_force_channels,
    }


def _cost_derivation_kwargs(conf):
    return {
        "cost_target_source": conf.expert.cost_target_source,
        "pipe_force_limit": conf.expert.cost_pipe_force_limit,
        "bottom_force_limit": conf.expert.cost_bottom_force_limit,
        "pipe_force_channels": conf.expert.cost_pipe_force_channels,
        "bottom_force_channels": conf.expert.cost_bottom_force_channels,
    }


def _force_candidate_keys(conf):
    force_key = getattr(getattr(conf, "ForceHead", object()), "Key", "")
    return (
        force_key,
        "force",
        "forceObs",
        "pipe_force_curr",
        "pipe_force",
        "contact_force",
        "pipe_contact_force",
        "ft_force",
    )


def _extract_raw_force_from_obs(obs_dict, conf, num_envs, device):
    for key in _force_candidate_keys(conf):
        if not key:
            continue
        value = obs_dict.get(key)
        if value is None:
            continue
        force = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(num_envs, -1)
        return force, key
    return None, ""


def _extract_safety_margin_from_obs(obs_dict, num_envs, device):
    for key in ("safety_margin", "constraint_margin", "margin"):
        value = obs_dict.get(key)
        if value is not None:
            return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(num_envs, 1), key
    return None, ""


def _derive_online_cost_components_from_obs(obs_dict, conf, num_envs, device):
    raw_force, force_key = _extract_raw_force_from_obs(obs_dict, conf, num_envs, device)
    source = str(getattr(conf.expert, "cost_target_source", "raw") or "raw").lower()
    nan = torch.full((num_envs, 1), float("nan"), dtype=torch.float32, device=device)
    out = {
        "cost": None,
        "pipe_cost": nan.clone(),
        "bottom_cost": nan.clone(),
        "pipe_force": nan.clone(),
        "bottom_force": nan.clone(),
        "source": "none",
        "force_key": force_key,
    }

    if source in ("force_margin", "force", "derived") and raw_force is not None:
        pipe_channels = [int(idx) for idx in conf.expert.cost_pipe_force_channels]
        bottom_channels = [int(idx) for idx in conf.expert.cost_bottom_force_channels]
        all_channels = pipe_channels + bottom_channels
        if all_channels and raw_force.shape[-1] > max(all_channels):
            pipe_force = raw_force[:, pipe_channels].amax(dim=-1, keepdim=True)
            bottom_force = raw_force[:, bottom_channels].amax(dim=-1, keepdim=True)
            pipe_cost = (pipe_force - float(conf.expert.cost_pipe_force_limit)).clamp_min(0.0)
            bottom_cost = (bottom_force - float(conf.expert.cost_bottom_force_limit)).clamp_min(0.0)
            out.update(
                {
                    "cost": torch.maximum(pipe_cost, bottom_cost),
                    "pipe_cost": pipe_cost,
                    "bottom_cost": bottom_cost,
                    "pipe_force": pipe_force,
                    "bottom_force": bottom_force,
                    "source": "force",
                }
            )
            return out

    safety_margin, _ = _extract_safety_margin_from_obs(obs_dict, num_envs, device)
    if source in ("force_margin", "margin", "derived") and safety_margin is not None:
        out["cost"] = (-safety_margin).clamp_min(0.0)
        out["source"] = "margin"
    if out["cost"] is None and (raw_force is not None or safety_margin is not None):
        cost_np, _ = derive_cost_from_force_margin(
            np.zeros(num_envs, dtype=np.float32),
            raw_force.detach().cpu().numpy() if raw_force is not None else None,
            safety_margin.detach().cpu().numpy() if safety_margin is not None else None,
            **_cost_derivation_kwargs(conf),
        )
        out["cost"] = torch.as_tensor(cost_np, dtype=torch.float32, device=device).view(num_envs, 1)
    return out


def _derive_online_cost_from_obs(obs_dict, conf, num_envs, device):
    components = _derive_online_cost_components_from_obs(obs_dict, conf, num_envs, device)
    if components["cost"] is not None:
        return components["cost"]

    raw_force, _ = _extract_raw_force_from_obs(obs_dict, conf, num_envs, device)
    safety_margin, _ = _extract_safety_margin_from_obs(obs_dict, num_envs, device)
    if raw_force is None and safety_margin is None:
        return None
    cost_np, _ = derive_cost_from_force_margin(
        np.zeros(num_envs, dtype=np.float32),
        raw_force.detach().cpu().numpy() if raw_force is not None else None,
        safety_margin.detach().cpu().numpy() if safety_margin is not None else None,
        **_cost_derivation_kwargs(conf),
    )
    return torch.as_tensor(cost_np, dtype=torch.float32, device=device).view(num_envs, 1)


@torch.no_grad()
def evaluate_world_model_batch(world_model, batch):
    batch = {
        key: value.to(world_model.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    obs = batch["obs"]
    action = batch["action"]
    reward = batch["reward"]
    done = batch["done"]
    is_first = batch["is_first"]
    cost = batch.get("cost")
    force = batch.get("force")
    horizon = obs.shape[1] - 1

    world_model.eval()
    with torch.autocast(
        device_type=world_model.device_type,
        dtype=world_model.tensor_dtype,
        enabled=world_model.use_amp,
    ):
        post, prior, stoch, deter = world_model.dynamic.parallel_observe(
            world_model.encoder(obs),
            action,
            is_first,
        )
        dyn_loss, rep_loss, real_kl, posterior_ent = world_model.dynamic.kl_loss(
            post,
            prior,
            world_model.kl_free,
        )

        obs_recon = world_model.decoder(stoch)
        reward_logits = world_model.reward_head(deter)
        reward_recon = world_model.twohot_loss.decode(reward_logits)
        done_logits = world_model.done_head(deter)
        done_prob = torch.sigmoid(done_logits)
        prior_obs = world_model.decoder(world_model.dynamic.get_flatten_stoch(prior))

        cost_metrics = {}
        if cost is not None and hasattr(world_model, "cost_head"):
            posterior_cost_feat = torch.cat((deter, stoch), dim=-1)
            cost_recon, cost_prob, _ = world_model.predict_cost(posterior_cost_feat)
            cost_target = cost.reshape_as(cost_recon).to(cost_recon.dtype)
            cost_loss = F.mse_loss(cost_recon, cost_target)
            cost_err = cost_recon - cost_target
            cost_metrics.update(
                {
                    "posterior_cost_loss": _safe_float(cost_loss),
                    "posterior_cost_mae": _safe_float(cost_err.abs().mean()),
                    "posterior_cost_rmse": _safe_float(torch.sqrt(cost_err.pow(2).mean())),
                    "posterior_cost_pred_mean": _safe_float(cost_recon.mean()),
                    "posterior_cost_target_mean": _safe_float(cost_target.mean()),
                }
            )
            cost_metrics.update(cost_prediction_metrics(cost_recon, cost_prob, cost_target, prefix="posterior_cost"))

        force_metrics = {}
        if force is not None and getattr(world_model, "force_enabled", False):
            force_target = force.to(dtype=obs.dtype, device=obs.device)
            posterior_force_feat = torch.cat((deter, stoch), dim=-1)
            posterior_force_pred, posterior_force_prob = _predict_force(world_model, posterior_force_feat)
            posterior_force_err = posterior_force_pred - force_target
            force_nonzero = force_target.abs() > world_model.force_criterion.eps
            force_metrics.update(
                {
                    "posterior_force_mae": _safe_float(posterior_force_err.abs().mean()),
                    "posterior_force_rmse": _safe_float(torch.sqrt(posterior_force_err.pow(2).mean())),
                    "posterior_force_pred_mean": _safe_float(posterior_force_pred.mean()),
                    "posterior_force_target_mean": _safe_float(force_target.mean()),
                    "posterior_force_target_nonzero_rate": _safe_float(force_nonzero.float().mean()),
                    "posterior_force_prob_mean": _safe_float(posterior_force_prob.mean()),
                }
            )

        rollout_state = world_model.initial(obs.shape[0])
        _, rollout_state = world_model.get_inference_feat(rollout_state, obs[:, 0], is_first[:, 0])
        rollout_obs = []
        rollout_reward = []
        rollout_done_prob = []
        rollout_cost = []
        rollout_cost_prob = []
        rollout_force = []
        for step in range(horizon):
            rollout_state = world_model.update_inference_state(rollout_state, action[:, step])
            flat_stoch = world_model.dynamic.get_flatten_stoch(rollout_state)
            rollout_obs.append(world_model.decoder(flat_stoch))
            rollout_reward.append(world_model.twohot_loss.decode(world_model.reward_head(rollout_state["deter"])))
            rollout_done_prob.append(torch.sigmoid(world_model.done_head(rollout_state["deter"])))
            if cost is not None and hasattr(world_model, "cost_head"):
                cost_pred, cost_prob, _ = world_model.predict_cost(world_model.dynamic.get_feat(rollout_state))
                rollout_cost.append(cost_pred)
                rollout_cost_prob.append(cost_prob)
            if force is not None and getattr(world_model, "force_enabled", False):
                pred_force, _ = _predict_force(world_model, world_model.dynamic.get_feat(rollout_state))
                rollout_force.append(pred_force)

    rollout_obs = torch.stack(rollout_obs, dim=1)
    rollout_reward = torch.stack(rollout_reward, dim=1)
    rollout_done_prob = torch.stack(rollout_done_prob, dim=1)

    recon_err = obs_recon - obs
    reward_recon_err = reward_recon - reward
    done_pred = (done_prob >= 0.5).float()
    one_step_err = prior_obs - obs[:, 1:]
    rollout_obs_err = rollout_obs - obs[:, 1:]
    rollout_reward_target = reward[:, :horizon]
    rollout_reward_err = rollout_reward - rollout_reward_target
    rollout_done_target = done[:, :horizon]
    rollout_done_acc = ((rollout_done_prob >= 0.5).float() == rollout_done_target).float()

    metrics = {
        "num_samples": int(obs.shape[0]),
        "horizon": int(horizon),
        "posterior_recon_loss": _safe_float(world_model.mse_loss(obs_recon, obs)),
        "posterior_obs_rmse": _safe_float(torch.sqrt(recon_err.pow(2).mean())),
        "posterior_reward_loss": _safe_float(world_model.twohot_loss(reward_logits, reward)),
        "posterior_reward_mae": _safe_float(reward_recon_err.abs().mean()),
        "posterior_reward_rmse": _safe_float(torch.sqrt(reward_recon_err.pow(2).mean())),
        "posterior_done_bce": _safe_float(F.binary_cross_entropy_with_logits(done_logits, done)),
        "posterior_done_acc": _safe_float((done_pred == done).float().mean()),
        "dyn_loss": _safe_float(dyn_loss),
        "rep_loss": _safe_float(rep_loss),
        "real_kl": _safe_float(real_kl),
        "posterior_entropy": _safe_float(posterior_ent),
        "one_step_obs_rmse": _safe_float(torch.sqrt(one_step_err.pow(2).mean())),
        "rollout_obs_rmse": _safe_float(torch.sqrt(rollout_obs_err.pow(2).mean())),
        "rollout_obs_mae": _safe_float(rollout_obs_err.abs().mean()),
        "rollout_reward_mae": _safe_float(rollout_reward_err.abs().mean()),
        "rollout_reward_rmse": _safe_float(torch.sqrt(rollout_reward_err.pow(2).mean())),
        "rollout_reward_bias": _safe_float(rollout_reward_err.mean()),
        "rollout_done_acc": _safe_float(rollout_done_acc.mean()),
        "rollout_done_prob_mean": _safe_float(rollout_done_prob.mean()),
        "rollout_done_target_mean": _safe_float(rollout_done_target.mean()),
    }
    metrics["rollout_reward_corr"] = _safe_corrcoef(
        rollout_reward_target.detach().float().cpu().numpy(),
        rollout_reward.detach().float().cpu().numpy(),
    )
    metrics.update(cost_metrics)
    metrics.update(force_metrics)

    curves = {
        "rollout_obs_rmse_by_step": torch.sqrt(rollout_obs_err.pow(2).mean(dim=(0, 2))).detach().float().cpu().numpy(),
        "rollout_obs_mae_by_step": rollout_obs_err.abs().mean(dim=(0, 2)).detach().float().cpu().numpy(),
        "rollout_reward_mae_by_step": rollout_reward_err.abs().mean(dim=(0, 2)).detach().float().cpu().numpy(),
        "rollout_reward_rmse_by_step": torch.sqrt(rollout_reward_err.pow(2).mean(dim=(0, 2))).detach().float().cpu().numpy(),
        "rollout_reward_bias_by_step": rollout_reward_err.mean(dim=(0, 2)).detach().float().cpu().numpy(),
        "rollout_done_acc_by_step": rollout_done_acc.mean(dim=(0, 2)).detach().float().cpu().numpy(),
        "rollout_done_prob_by_step": rollout_done_prob.mean(dim=(0, 2)).detach().float().cpu().numpy(),
        "rollout_done_target_by_step": rollout_done_target.mean(dim=(0, 2)).detach().float().cpu().numpy(),
    }

    if cost is not None and hasattr(world_model, "cost_head") and rollout_cost:
        rollout_cost = torch.stack(rollout_cost, dim=1)
        cost_target = cost[:, 1:].reshape_as(rollout_cost).to(rollout_cost.dtype)
        cost_err = rollout_cost - cost_target
        metrics.update(
            {
                "rollout_cost_mae": _safe_float(cost_err.abs().mean()),
                "rollout_cost_rmse": _safe_float(torch.sqrt(cost_err.pow(2).mean())),
                "rollout_cost_pred_mean": _safe_float(rollout_cost.mean()),
                "rollout_cost_target_mean": _safe_float(cost_target.mean()),
            }
        )
        rollout_cost_prob = torch.stack(rollout_cost_prob, dim=1) if rollout_cost_prob else None
        if rollout_cost_prob is not None:
            metrics.update(cost_prediction_metrics(rollout_cost, rollout_cost_prob, cost_target, prefix="rollout_cost"))
        curves["rollout_cost_mae_by_step"] = cost_err.abs().mean(dim=(0, 2)).detach().float().cpu().numpy()
        curves["rollout_cost_target_by_step"] = cost_target.mean(dim=(0, 2)).detach().float().cpu().numpy()
        curves["rollout_cost_pred_by_step"] = rollout_cost.mean(dim=(0, 2)).detach().float().cpu().numpy()

    if force is not None and getattr(world_model, "force_enabled", False) and rollout_force:
        rollout_force = torch.stack(rollout_force, dim=1)
        force_target = force[:, 1:].reshape_as(rollout_force).to(rollout_force.dtype)
        force_err = rollout_force - force_target
        metrics.update(
            {
                "rollout_force_mae": _safe_float(force_err.abs().mean()),
                "rollout_force_rmse": _safe_float(torch.sqrt(force_err.pow(2).mean())),
                "rollout_force_pred_mean": _safe_float(rollout_force.mean()),
                "rollout_force_target_mean": _safe_float(force_target.mean()),
            }
        )
        curves["rollout_force_mae_by_step"] = force_err.abs().mean(dim=(0, 2)).detach().float().cpu().numpy()
        curves["rollout_force_rmse_by_step"] = torch.sqrt(force_err.pow(2).mean(dim=(0, 2))).detach().float().cpu().numpy()
        curves["rollout_force_target_by_step"] = force_target.mean(dim=(0, 2)).detach().float().cpu().numpy()
        curves["rollout_force_pred_by_step"] = rollout_force.mean(dim=(0, 2)).detach().float().cpu().numpy()

    return metrics, curves


def _trace_column(value, num_envs, device="cpu", fill_value=float("nan")):
    if value is None:
        return torch.full((num_envs, 1), fill_value, dtype=torch.float32, device=device)
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(num_envs, -1)[:, :1]


def _init_online_trace(num_envs):
    return {
        "num_envs": int(num_envs),
        "obs": [],
        "action": [],
        "reward": [],
        "done": [],
        "is_first": [],
        "force": [],
        "cost": [],
        "pipe_cost": [],
        "bottom_cost": [],
        "pipe_force": [],
        "bottom_force": [],
        "cost_source": [],
    }


def _append_online_trace(trace, current_obs, action, reward, done, is_first, force, cost_components):
    if trace is None:
        return
    num_envs = int(trace["num_envs"])
    trace["obs"].append(torch.as_tensor(current_obs, dtype=torch.float32).detach().cpu())
    trace["action"].append(torch.as_tensor(action, dtype=torch.float32).detach().cpu())
    trace["reward"].append(_trace_column(reward, num_envs).detach().cpu())
    trace["done"].append(_trace_column(done, num_envs, fill_value=0.0).detach().cpu())
    trace["is_first"].append(_trace_column(is_first, num_envs, fill_value=0.0).detach().cpu())
    trace["force"].append(_trace_column(force, num_envs).detach().cpu())
    for key in ("cost", "pipe_cost", "bottom_cost", "pipe_force", "bottom_force"):
        trace[key].append(_trace_column(cost_components.get(key), num_envs).detach().cpu())
    trace["cost_source"].append(str(cost_components.get("source", "none")))


def _stack_online_trace(trace):
    if not trace or not trace.get("obs"):
        return {}
    tensor_keys = (
        "obs",
        "action",
        "reward",
        "done",
        "is_first",
        "force",
        "cost",
        "pipe_cost",
        "bottom_cost",
        "pipe_force",
        "bottom_force",
    )
    return {key: torch.stack(trace[key], dim=0) for key in tensor_keys if trace.get(key)}


def _parse_plot_env_ids(value):
    if value is None or str(value).strip() == "":
        return []
    env_ids = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            env_ids.append(int(item))
    return env_ids


def _selected_plot_env_ids(args, num_envs):
    requested = _parse_plot_env_ids(args.plot_env_ids)
    if not requested:
        requested = list(range(min(max(int(args.plot_num_envs), 0), int(num_envs))))
    return [env_id for env_id in requested if 0 <= int(env_id) < int(num_envs)]


def _resolve_lift_height_obs_index(args, obs_dim):
    index = int(args.lift_height_obs_index)
    if index >= 0:
        return index if index < int(obs_dim) else None
    if int(obs_dim) >= 47:
        return 37
    if int(obs_dim) == 46:
        return 39
    return None


def _resolve_lift_height_scale(args, obs_dim):
    scale = float(args.lift_height_scale)
    if scale > 0.0:
        return scale
    return 1000.0 if int(obs_dim) >= 47 else 1.0


def _extract_lift_height_curve(obs, index, scale):
    if index is None:
        return None
    value = torch.as_tensor(obs, dtype=torch.float32)
    if value.shape[-1] <= int(index):
        return None
    scale = float(scale) if float(scale) != 0.0 else 1.0
    return _as_np_1d(value[..., int(index)] / scale)


def _find_trace_window(done, env_id, desired_horizon):
    num_rows = int(done.shape[0])
    max_horizon = min(max(int(desired_horizon), 1), num_rows - 1)
    for horizon in range(max_horizon, 0, -1):
        num_starts = num_rows - horizon
        for start in range(num_starts):
            if not bool(done[start : start + horizon, env_id, 0].bool().any().item()):
                return start, horizon
    return None, 0


@torch.no_grad()
def _imagine_plot_batch(world_model, batch):
    batch = {
        key: value.to(world_model.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    obs = batch["obs"]
    action = batch["action"]
    is_first = batch["is_first"]
    horizon = obs.shape[1] - 1

    world_model.eval()
    with torch.autocast(
        device_type=world_model.device_type,
        dtype=world_model.tensor_dtype,
        enabled=world_model.use_amp,
    ):
        state = world_model.initial(obs.shape[0])
        _, state = world_model.get_inference_feat(state, obs[:, 0], is_first[:, 0])
        obs_pred = []
        reward_pred = []
        cost_pred = []
        force_pred = []
        for step in range(horizon):
            state = world_model.update_inference_state(state, action[:, step])
            feat = world_model.dynamic.get_feat(state)
            obs_pred.append(world_model.decoder(world_model.dynamic.get_flatten_stoch(state)))
            reward_pred.append(world_model.twohot_loss.decode(world_model.reward_head(state["deter"])))
            if hasattr(world_model, "cost_head"):
                pred_cost, _, _ = world_model.predict_cost(feat)
                cost_pred.append(pred_cost)
            if getattr(world_model, "force_enabled", False):
                pred_force, _ = _predict_force(world_model, feat)
                force_pred.append(pred_force)

    out = {
        "obs": torch.stack(obs_pred, dim=1).detach().float().cpu(),
        "reward": torch.stack(reward_pred, dim=1).detach().float().cpu(),
    }
    if cost_pred:
        out["cost"] = torch.stack(cost_pred, dim=1).detach().float().cpu()
    if force_pred:
        out["force"] = torch.stack(force_pred, dim=1).detach().float().cpu()
    return out


def _as_np_1d(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _has_finite(value):
    return value is not None and np.isfinite(value).any()


def _plot_line(ax, x, y, label, **kwargs):
    if _has_finite(y):
        ax.plot(x, y, label=label, **kwargs)


def _maybe_legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")


def _write_rollout_diag_csv(path, x, data):
    keys = [
        "actual_lift_height",
        "imagined_lift_height",
        "lift_height_error",
        "actual_reward",
        "imagined_reward",
        "reward_error",
        "actual_force",
        "imagined_force",
        "force_error",
        "actual_pipe_cost",
        "actual_bottom_cost",
        "actual_combined_cost",
        "imagined_cost",
        "cost_error_combined",
        "cost_error_pipe",
        "cost_error_bottom",
        "pipe_force",
        "bottom_force",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["step", *keys])
        for idx, step in enumerate(x):
            row = [int(step)]
            for key in keys:
                value = data.get(key)
                row.append(float(value[idx]) if value is not None and idx < len(value) else float("nan"))
            writer.writerow(row)


def _svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_line_segments(x, y, x0, y0, width, height, ymin, ymax):
    if y is None:
        return []
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return []
    x = np.asarray(x, dtype=np.float32).reshape(-1)[: y.size]
    if y.size == 1:
        x_min, x_max = float(x[0]), float(x[0] + 1.0)
    else:
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        if abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1.0
    y_den = max(float(ymax - ymin), 1e-6)
    segments = []
    current = []
    for xv, yv in zip(x, y):
        if not np.isfinite(yv):
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        px = x0 + (float(xv) - x_min) / (x_max - x_min) * width
        py = y0 + height - (float(yv) - ymin) / y_den * height
        current.append((px, py))
    if len(current) >= 2:
        segments.append(current)
    return segments


def _svg_panel(svg, x, lines, title, x0, y0, width, height, ylabel="", zero_line=False):
    finite = []
    for _, values, _, _ in lines:
        if _has_finite(values):
            finite.extend(np.asarray(values, dtype=np.float32)[np.isfinite(values)].reshape(-1).tolist())
    if zero_line:
        finite.append(0.0)
    if finite:
        ymin = float(min(finite))
        ymax = float(max(finite))
        if abs(ymax - ymin) < 1e-6:
            ymin -= 1.0
            ymax += 1.0
        else:
            pad = 0.08 * (ymax - ymin)
            ymin -= pad
            ymax += pad
    else:
        ymin, ymax = -1.0, 1.0

    svg.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" stroke="#333" stroke-width="1"/>')
    svg.append(f'<text x="{x0}" y="{y0 - 8}" font-size="16" font-weight="600">{_svg_escape(title)}</text>')
    if ylabel:
        svg.append(f'<text x="{x0}" y="{y0 + 18}" font-size="11" fill="#555">{_svg_escape(ylabel)}</text>')

    for frac in (0.25, 0.5, 0.75):
        gy = y0 + height * frac
        svg.append(f'<line x1="{x0}" y1="{gy}" x2="{x0 + width}" y2="{gy}" stroke="#ddd" stroke-width="1"/>')

    if zero_line and ymin <= 0.0 <= ymax:
        zy = y0 + height - (0.0 - ymin) / max(ymax - ymin, 1e-6) * height
        svg.append(f'<line x1="{x0}" y1="{zy}" x2="{x0 + width}" y2="{zy}" stroke="#444" stroke-width="1" opacity="0.65"/>')

    legend_y = y0 + 18
    legend_x = x0 + 10
    plotted = 0
    for label, values, color, dash in lines:
        if not _has_finite(values):
            continue
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        for segment in _svg_line_segments(x, values, x0, y0, width, height, ymin, ymax):
            points = " ".join(f"{px:.1f},{py:.1f}" for px, py in segment)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')
        ly = legend_y + plotted * 16
        svg.append(f'<line x1="{legend_x}" y1="{ly - 4}" x2="{legend_x + 24}" y2="{ly - 4}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        svg.append(f'<text x="{legend_x + 30}" y="{ly}" font-size="11">{_svg_escape(label)}</text>')
        plotted += 1

    if plotted == 0:
        svg.append(f'<text x="{x0 + width / 2}" y="{y0 + height / 2}" font-size="16" text-anchor="middle" fill="#777">not available</text>')

    svg.append(f'<text x="{x0}" y="{y0 + height + 18}" font-size="11" fill="#555">step {int(x[0]) if len(x) else 0}</text>')
    svg.append(f'<text x="{x0 + width}" y="{y0 + height + 18}" font-size="11" text-anchor="end" fill="#555">step {int(x[-1]) if len(x) else 0}</text>')


def _write_rollout_diag_svg(path, title, x, data):
    width, height = 1400, 1280
    panel_w, panel_h = 610, 200
    left_x, right_x = 70, 750
    row_y = [95, 380, 665, 950]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        f'<text x="{width / 2}" y="40" font-size="22" font-weight="700" text-anchor="middle">{_svg_escape(title)}</text>',
        '<text x="70" y="66" font-size="12" fill="#555">Lift height is decoded from policy obs; model cost is scalar while actual cost is split into pipe_cost and bottom_cost.</text>',
    ]
    _svg_panel(
        svg,
        x,
        [
            ("actual next lift height", data.get("actual_lift_height"), "#1f77b4", ""),
            ("imagined next lift height", data.get("imagined_lift_height"), "#ff7f0e", ""),
        ],
        "Object Lift Height",
        left_x,
        row_y[0],
        panel_w,
        panel_h,
        "height (m)",
    )
    _svg_panel(
        svg,
        x,
        [("imagined - actual", data.get("lift_height_error"), "#d62728", "")],
        "Object Lift Height Error",
        right_x,
        row_y[0],
        panel_w,
        panel_h,
        "height error (m)",
        zero_line=True,
    )
    _svg_panel(
        svg,
        x,
        [
            ("actual reward", data.get("actual_reward"), "#1f77b4", ""),
            ("imagined reward", data.get("imagined_reward"), "#ff7f0e", ""),
        ],
        "Reward",
        left_x,
        row_y[1],
        panel_w,
        panel_h,
        "reward",
    )
    _svg_panel(
        svg,
        x,
        [("imagined - actual", data.get("reward_error"), "#d62728", "")],
        "Reward Error",
        right_x,
        row_y[1],
        panel_w,
        panel_h,
        "",
        zero_line=True,
    )
    _svg_panel(
        svg,
        x,
        [
            ("actual next force", data.get("actual_force"), "#1f77b4", ""),
            ("imagined next force", data.get("imagined_force"), "#ff7f0e", ""),
        ],
        "Force",
        left_x,
        row_y[2],
        panel_w,
        panel_h,
        "force",
    )
    _svg_panel(
        svg,
        x,
        [("imagined - actual", data.get("force_error"), "#d62728", "")],
        "Force Error",
        right_x,
        row_y[2],
        panel_w,
        panel_h,
        "",
        zero_line=True,
    )
    _svg_panel(
        svg,
        x,
        [
            ("actual pipe_cost", data.get("actual_pipe_cost"), "#2ca02c", ""),
            ("actual bottom_cost", data.get("actual_bottom_cost"), "#9467bd", ""),
            ("actual max(pipe,bottom)", data.get("actual_combined_cost"), "#1f77b4", "6 4"),
            ("imagined scalar cost", data.get("imagined_cost"), "#ff7f0e", ""),
        ],
        "Cost: pipe_cost / bottom_cost",
        left_x,
        row_y[3],
        panel_w,
        panel_h,
        "cost",
    )
    _svg_panel(
        svg,
        x,
        [
            ("pred - max(pipe,bottom)", data.get("cost_error_combined"), "#d62728", ""),
            ("pred - pipe_cost", data.get("cost_error_pipe"), "#2ca02c", ""),
            ("pred - bottom_cost", data.get("cost_error_bottom"), "#9467bd", ""),
        ],
        "Cost Error",
        right_x,
        row_y[3],
        panel_w,
        panel_h,
        "",
        zero_line=True,
    )
    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(svg) + "\n")


def _convert_svg_to_png(svg_path, png_path):
    import shutil
    import subprocess

    convert_cmd = shutil.which("convert") or shutil.which("magick")
    if not convert_cmd:
        return False
    cmd = [convert_cmd, svg_path, png_path]
    if os.path.basename(convert_cmd) == "magick":
        cmd = [convert_cmd, "convert", svg_path, png_path]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        return False
    return os.path.exists(png_path)


def _save_rollout_diag_figure(path, title, x, data):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        svg_path = os.path.splitext(path)[0] + ".svg"
        _write_rollout_diag_svg(svg_path, title, x, data)
        return path if _convert_svg_to_png(svg_path, path) else svg_path

    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    _plot_line(ax, x, data.get("actual_lift_height"), "actual next lift height", color="tab:blue")
    _plot_line(ax, x, data.get("imagined_lift_height"), "imagined next lift height", color="tab:orange")
    ax.set_title("Object Lift Height")
    ax.set_ylabel("height (m)")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[0, 1]
    _plot_line(ax, x, data.get("lift_height_error"), "imagined - actual", color="tab:red")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Object Lift Height Error")
    ax.set_ylabel("height error (m)")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[1, 0]
    _plot_line(ax, x, data.get("actual_reward"), "actual reward", color="tab:blue")
    _plot_line(ax, x, data.get("imagined_reward"), "imagined reward", color="tab:orange")
    ax.set_title("Reward")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[1, 1]
    _plot_line(ax, x, data.get("reward_error"), "imagined - actual", color="tab:red")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Reward Error")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[2, 0]
    _plot_line(ax, x, data.get("actual_force"), "actual next force", color="tab:blue")
    _plot_line(ax, x, data.get("imagined_force"), "imagined next force", color="tab:orange")
    ax.set_title("Force")
    ax.set_ylabel("force")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[2, 1]
    _plot_line(ax, x, data.get("force_error"), "imagined - actual", color="tab:red")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Force Error")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[3, 0]
    _plot_line(ax, x, data.get("actual_pipe_cost"), "actual pipe_cost", color="tab:green")
    _plot_line(ax, x, data.get("actual_bottom_cost"), "actual bottom_cost", color="tab:purple")
    _plot_line(ax, x, data.get("actual_combined_cost"), "actual max(pipe,bottom)", color="tab:blue", linestyle="--")
    _plot_line(ax, x, data.get("imagined_cost"), "imagined scalar cost", color="tab:orange")
    ax.set_title("Cost: pipe_cost / bottom_cost")
    ax.set_xlabel("open-loop step")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    ax = axes[3, 1]
    _plot_line(ax, x, data.get("cost_error_combined"), "pred - max(pipe,bottom)", color="tab:red")
    _plot_line(ax, x, data.get("cost_error_pipe"), "pred - pipe_cost", color="tab:green", alpha=0.8)
    _plot_line(ax, x, data.get("cost_error_bottom"), "pred - bottom_cost", color="tab:purple", alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Cost Error")
    ax.set_xlabel("open-loop step")
    ax.grid(True, alpha=0.25)
    _maybe_legend(ax)

    for ax in axes.flat:
        if not ax.get_legend_handles_labels()[0]:
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_rollout_diagnostic_plots(output_dir, name, trace, world_model, args):
    if not args.plot_rollouts:
        return []
    stacked = _stack_online_trace(trace)
    if not stacked:
        return []

    num_envs = int(trace["num_envs"])
    env_ids = _selected_plot_env_ids(args, num_envs)
    desired_horizon = int(args.plot_horizon) if int(args.plot_horizon) > 0 else max(int(args.batch_length) - 1, 1)
    obs_dim = int(stacked["obs"].shape[-1])
    lift_height_index = _resolve_lift_height_obs_index(args, obs_dim)
    lift_height_scale = _resolve_lift_height_scale(args, obs_dim)
    group_dir = os.path.join(output_dir, name)
    plot_dir = os.path.join(group_dir, "rollout_diagnostics")
    os.makedirs(plot_dir, exist_ok=True)

    saved = []
    done = stacked["done"]
    for env_id in env_ids:
        start, horizon = _find_trace_window(done, env_id, desired_horizon)
        if horizon <= 0:
            print(
                colorama.Fore.YELLOW
                + f"Skipping rollout diagnostic env {env_id}: no contiguous window found."
                + colorama.Style.RESET_ALL
            )
            continue

        stop = start + horizon + 1
        batch = {
            key: stacked[key][start:stop, env_id].unsqueeze(0)
            for key in ("obs", "action", "reward", "done", "is_first", "cost", "force")
            if key in stacked
        }
        pred = _imagine_plot_batch(world_model, batch)
        x = np.arange(1, horizon + 1, dtype=np.int32)

        actual_lift_height = _extract_lift_height_curve(
            stacked["obs"][start + 1 : stop, env_id],
            lift_height_index,
            lift_height_scale,
        )
        imagined_lift_height = _extract_lift_height_curve(
            pred["obs"][0],
            lift_height_index,
            lift_height_scale,
        )
        actual_reward = _as_np_1d(stacked["reward"][start : start + horizon, env_id])
        imagined_reward = _as_np_1d(pred["reward"][0])
        actual_force = _as_np_1d(stacked["force"][start + 1 : stop, env_id])
        imagined_force = _as_np_1d(pred.get("force", torch.full((1, horizon, 1), float("nan")))[0])
        actual_pipe_cost = _as_np_1d(stacked["pipe_cost"][start + 1 : stop, env_id])
        actual_bottom_cost = _as_np_1d(stacked["bottom_cost"][start + 1 : stop, env_id])
        actual_combined_cost = _as_np_1d(stacked["cost"][start + 1 : stop, env_id])
        imagined_cost = _as_np_1d(pred.get("cost", torch.full((1, horizon, 1), float("nan")))[0])
        pipe_force = _as_np_1d(stacked["pipe_force"][start + 1 : stop, env_id])
        bottom_force = _as_np_1d(stacked["bottom_force"][start + 1 : stop, env_id])

        data = {
            "actual_lift_height": actual_lift_height,
            "imagined_lift_height": imagined_lift_height,
            "lift_height_error": (
                imagined_lift_height - actual_lift_height
                if imagined_lift_height is not None and actual_lift_height is not None
                else None
            ),
            "actual_reward": actual_reward,
            "imagined_reward": imagined_reward,
            "reward_error": imagined_reward - actual_reward,
            "actual_force": actual_force,
            "imagined_force": imagined_force,
            "force_error": imagined_force - actual_force,
            "actual_pipe_cost": actual_pipe_cost,
            "actual_bottom_cost": actual_bottom_cost,
            "actual_combined_cost": actual_combined_cost,
            "imagined_cost": imagined_cost,
            "cost_error_combined": imagined_cost - actual_combined_cost,
            "cost_error_pipe": imagined_cost - actual_pipe_cost,
            "cost_error_bottom": imagined_cost - actual_bottom_cost,
            "pipe_force": pipe_force,
            "bottom_force": bottom_force,
        }

        stem = f"env_{int(env_id):03d}_start_{int(start):06d}_h{int(horizon):03d}"
        png_path = os.path.join(plot_dir, f"{stem}.png")
        csv_path = os.path.join(plot_dir, f"{stem}.csv")
        title = f"{name} env={env_id} start={start} horizon={horizon}"
        image_path = _save_rollout_diag_figure(png_path, title, x, data)
        _write_rollout_diag_csv(csv_path, x, data)
        saved.append(image_path)

    if saved:
        print(
            colorama.Fore.GREEN
            + f"Saved {len(saved)} rollout diagnostic plots to {plot_dir}"
            + colorama.Style.RESET_ALL
        )
    return saved


def _mean_dicts(dicts):
    if not dicts:
        return {}
    keys = sorted({key for item in dicts for key in item.keys()})
    out = {}
    for key in keys:
        values = [item[key] for item in dicts if key in item and isinstance(item[key], (int, float))]
        if values:
            out[key] = float(np.mean(values))
    return out


def _mean_curves(curves):
    if not curves:
        return {}
    keys = sorted({key for item in curves for key in item.keys()})
    out = {}
    for key in keys:
        values = [np.asarray(item[key], dtype=np.float64) for item in curves if key in item]
        if values:
            min_len = min(value.shape[0] for value in values)
            out[key] = np.stack([value[:min_len] for value in values], axis=0).mean(axis=0)
    return out


class MixedReplaySampler:
    def __init__(self, replays):
        self.replays = [replay for replay in replays if replay is not None and len(replay) > 0]
        if not self.replays:
            raise ValueError("MixedReplaySampler needs at least one replay.")
        weights = np.asarray([len(replay) for replay in self.replays], dtype=np.float64)
        self.probs = weights / weights.sum()

    def can_sample(self, horizon, source="expert"):
        return any(replay.can_sample(horizon, source=source) for replay in self.replays)

    def sample(self, batch_size, horizon, source="expert", return_dict=True):
        valid_ids = [
            idx for idx, replay in enumerate(self.replays)
            if replay.can_sample(horizon, source=source)
        ]
        probs = self.probs[valid_ids]
        probs = probs / probs.sum()
        replay_id = int(np.random.choice(valid_ids, p=probs))
        return self.replays[replay_id].sample(batch_size, horizon, source=source, return_dict=return_dict)


def evaluate_replay(name, replay, world_model, args, source="expert"):
    if not replay.can_sample(args.batch_length, source=source):
        print(
            colorama.Fore.YELLOW
            + f"Skipping {name}: replay cannot sample batch_length={args.batch_length}."
            + colorama.Style.RESET_ALL
        )
        return {}, {}

    batch_metrics = []
    batch_curves = []
    for batch_idx in range(int(args.num_batches)):
        batch = replay.sample(args.batch_size, args.batch_length, source=source, return_dict=True)
        metrics, curves = evaluate_world_model_batch(world_model, batch)
        batch_metrics.append(metrics)
        batch_curves.append(curves)
        if batch_idx == 0 or (batch_idx + 1) % max(int(args.log_every), 1) == 0:
            print(
                colorama.Fore.CYAN
                + (
                    f"[{name}] batch {batch_idx + 1}/{args.num_batches}: "
                    f"rollout_obs_rmse={metrics['rollout_obs_rmse']:.4f}, "
                    f"rollout_reward_mae={metrics['rollout_reward_mae']:.4f}"
                )
                + colorama.Style.RESET_ALL
            )

    metrics = _mean_dicts(batch_metrics)
    curves = _mean_curves(batch_curves)
    metrics["num_eval_batches"] = int(args.num_batches)
    metrics["replay_steps"] = int(len(replay)) if hasattr(replay, "__len__") else 0
    return metrics, curves


def _write_metrics(output_dir, name, metrics, curves):
    group_dir = os.path.join(output_dir, name)
    os.makedirs(group_dir, exist_ok=True)
    with open(os.path.join(group_dir, "metrics.json"), "w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=False)

    curve_keys = sorted(curves.keys())
    if curve_keys:
        horizon = min(len(curves[key]) for key in curve_keys)
        with open(os.path.join(group_dir, "curves.csv"), "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["step", *curve_keys])
            for idx in range(horizon):
                writer.writerow([idx + 1, *[float(curves[key][idx]) for key in curve_keys]])
    print(colorama.Fore.GREEN + f"Saved {name} metrics to {group_dir}" + colorama.Style.RESET_ALL)


def _load_dataset_replay(path, conf, args, *, max_episodes, label):
    dataset = load_expert_dataset(
        path,
        format=args.dataset_format,
        action_tolerance=args.action_tolerance,
        max_episodes=max_episodes,
        **_cost_loader_kwargs(conf),
    )
    print(
        colorama.Fore.CYAN
        + (
            f"Loaded {label}: {dataset.metadata['num_episodes']} episodes / "
            f"{dataset.metadata['num_transitions']} transitions from {dataset.metadata['dataset_path']}"
        )
        + colorama.Style.RESET_ALL
    )
    replay = make_expert_replay(
        dataset,
        device=args.buffer_device or args.device,
        include_force=bool(args.include_force),
        force_dim=1,
    )
    return dataset, replay


def _build_output_dir(args):
    checkpoint_stub = "checkpoint"
    checkpoint_source = args.checkpoint_path or args.world_model_path or ""
    if checkpoint_source:
        checkpoint_stub = os.path.splitext(os.path.basename(os.path.abspath(checkpoint_source)))[0]
    output_dir = os.path.abspath(os.path.expanduser(os.path.join(args.save_dir, checkpoint_stub)))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _load_checkpoint(world_model, agent, args):
    if args.checkpoint_path:
        checkpoint = load_expert_checkpoint(
            args.checkpoint_path,
            world_model=world_model,
            agent=None,
            map_location=args.device,
        )
        if "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        elif args.online and not args.agent_path:
            raise ValueError("Online evaluation needs an agent checkpoint. Pass -agent_path or a full checkpoint.")
        return checkpoint

    if not args.world_model_path:
        raise ValueError("Pass -checkpoint_path or -world_model_path.")
    load_expert_checkpoint(
        args.world_model_path,
        world_model=world_model,
        agent=None,
        map_location=args.device,
    )

    if args.agent_path:
        agent_state = torch.load(args.agent_path, map_location=args.device)
        if isinstance(agent_state, dict) and "agent_state_dict" in agent_state:
            agent.load_state_dict(agent_state["agent_state_dict"])
        else:
            agent.load_state_dict(agent_state)
    elif args.online:
        raise ValueError("Online evaluation needs -agent_path when -checkpoint_path is not a full checkpoint.")
    return {}


def _prepare_offline_groups(conf, args):
    groups = {}
    expert_path = args.dataset_path or conf.expert.path
    if expert_path:
        max_episodes = args.max_episodes if args.max_episodes > 0 else None
        expert_dataset, expert_replay = _load_dataset_replay(
            expert_path,
            conf,
            args,
            max_episodes=max_episodes,
            label="expert_dataset",
        )
        groups["expert_dataset"] = (expert_dataset, expert_replay)

    coverage_paths = _as_path_list(conf.expert.wm_coverage_paths)
    coverage_paths.extend(_as_path_list(args.coverage_path))
    coverage_replays = []
    coverage_datasets = []
    if not args.no_coverage_dataset:
        max_coverage = args.max_coverage_episodes if args.max_coverage_episodes > 0 else None
        remaining = max_coverage
        for path_idx, path in enumerate(coverage_paths):
            load_limit = remaining if remaining is not None else None
            if load_limit is not None and load_limit <= 0:
                break
            dataset, replay = _load_dataset_replay(
                path,
                conf,
                args,
                max_episodes=load_limit,
                label=f"coverage_dataset_{path_idx}",
            )
            coverage_datasets.append(dataset)
            coverage_replays.append(replay)
            if remaining is not None:
                remaining -= len(dataset)

    if coverage_replays:
        groups["coverage_dataset"] = (coverage_datasets, MixedReplaySampler(coverage_replays))
    if "expert_dataset" in groups and coverage_replays:
        groups["wm_mix_dataset"] = (
            [groups["expert_dataset"][0], *coverage_datasets],
            MixedReplaySampler([groups["expert_dataset"][1], *coverage_replays]),
        )
    return groups


def _build_models_for_dims(conf, obs_dim, action_dim, args):
    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)
    _load_checkpoint(world_model, agent, args)
    world_model.eval()
    agent.eval()
    return world_model, agent


def _launch_isaac(headless=True):
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    import gymnasium
    import ur3_lite.tasks  # noqa: F401
    try:
        from pwm_isaaclab.env_wrapper import DreamerVecEnvWrapper
    except ImportError:
        from env_wrapper import DreamerVecEnvWrapper
    return simulation_app, parse_env_cfg, gymnasium, DreamerVecEnvWrapper


def _build_online_env(args, conf, parse_env_cfg, gymnasium, DreamerVecEnvWrapper):
    make_kwargs = {}
    if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
        make_kwargs = cfg_to_dict(conf.Env.MakeKwargs)
    num_envs = int(args.eval_num_envs or make_kwargs.get("num_envs", conf.JointTrainAgent.NumEnvs))
    use_fabric = bool(make_kwargs.get("use_fabric", True))
    env_seed = int(make_kwargs.get("seed", args.seed))
    env_cfg = parse_env_cfg(
        args.env_name,
        device=args.device,
        num_envs=num_envs,
        use_fabric=use_fabric,
    )
    env_cfg.seed = env_seed
    env = gymnasium.make(args.env_name, cfg=env_cfg)
    return DreamerVecEnvWrapper(env, device=args.device)


def _sample_policy_action(vec_env, world_model, agent, state, current_obs, is_first, policy_mode, device):
    if policy_mode == "random":
        env_action = np.asarray(vec_env.action_space.sample(), dtype=np.float32)
        action = torch.as_tensor(env_action, dtype=torch.float32, device=device)
        return env_action, action, state
    with torch.no_grad():
        feat, state = world_model.get_inference_feat(state, current_obs, is_first)
        env_action, action = agent.sample_as_env_action(feat, greedy=(policy_mode == "greedy"))
        state = world_model.update_inference_state(state, action)
    return env_action, action, state


def collect_online_replay(vec_env, world_model, agent, conf, args):
    num_envs = vec_env.num_envs
    env_obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
    obs_dim = int(getattr(world_model, "eval_obs_dim", env_obs_dim))
    action_dim = int(vec_env.single_action_space.shape[0])
    include_force = bool(args.include_force and getattr(world_model, "force_enabled", False))
    replay = SourceTaggedProprioReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        max_length=max(int(args.online_steps) + num_envs, num_envs * (args.batch_length + 2)),
        warmup_length=0,
        device=args.buffer_device or args.device,
        include_force=include_force,
        force_dim=1,
    )

    world_model.eval()
    agent.eval()
    state = world_model.initial(num_envs)
    current_obs_dict = vec_env.reset()
    current_obs = _match_obs_dim(_policy_obs(current_obs_dict).to(args.device), obs_dim)
    is_first = _is_first(current_obs_dict, num_envs, args.device)
    sum_reward = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
    episode_returns = []
    episode_successes = []
    force_disabled_reason = ""
    trace = _init_online_trace(num_envs) if args.plot_rollouts else None

    total_iters = math.ceil(int(args.online_steps) / num_envs)
    for iter_idx in range(total_iters):
        env_action, action, state = _sample_policy_action(
            vec_env,
            world_model,
            agent,
            state,
            current_obs,
            is_first,
            args.policy_mode,
            args.device,
        )
        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=args.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=args.device)

        force = None
        if include_force and not force_disabled_reason:
            try:
                force = _extract_force_obs(current_obs_dict, num_envs, args.device, "")
            except Exception as exc:
                force_disabled_reason = str(exc)
                include_force = False
                replay.include_force = False
                print(
                    colorama.Fore.YELLOW
                    + f"Online force collection disabled: {force_disabled_reason}"
                    + colorama.Style.RESET_ALL
                )

        cost_components = _derive_online_cost_components_from_obs(current_obs_dict, conf, num_envs, args.device)
        cost = cost_components["cost"]
        replay.append(current_obs, action, reward, done, is_first, force=force, cost=cost)
        _append_online_trace(trace, current_obs, action, reward, done, is_first, force, cost_components)
        sum_reward += reward

        episode_success = info.get("episode_success")
        if episode_success is not None:
            episode_success = torch.as_tensor(episode_success, dtype=torch.bool, device=args.device).view(-1)

        if done.any():
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            for idx in done_ids.tolist():
                episode_returns.append(float(sum_reward[idx].item()))
                if episode_success is not None:
                    episode_successes.append(float(episode_success[idx].item()))
                sum_reward[idx] = 0.0

        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, args.device)
        current_obs = _match_obs_dim(current_obs.to(args.device), obs_dim)
        if (iter_idx + 1) % max(int(args.online_log_every), 1) == 0:
            print(
                colorama.Fore.CYAN
                + f"Collected {(iter_idx + 1) * num_envs}/{args.online_steps} online steps"
                + colorama.Style.RESET_ALL
            )

    rollout_summary = {
        "online_steps_collected": int(len(replay)),
        "online_num_envs": int(num_envs),
        "online_policy_mode": args.policy_mode,
        "online_env_obs_dim": int(env_obs_dim),
        "online_model_obs_dim": int(obs_dim),
        "online_obs_dim_adapter": (
            "identity"
            if env_obs_dim == obs_dim
            else ("zero_pad_tail" if env_obs_dim < obs_dim else "truncate_tail")
        ),
        "online_num_finished_episodes": int(len(episode_returns)),
        "online_mean_episode_return": float(np.mean(episode_returns)) if episode_returns else None,
        "online_std_episode_return": float(np.std(episode_returns)) if episode_returns else None,
        "online_success_rate": float(np.mean(episode_successes)) if episode_successes else None,
        "online_force_collection_disabled_reason": force_disabled_reason,
    }
    return replay, rollout_summary, trace


def _print_topline(name, metrics):
    keys = (
        "rollout_obs_rmse",
        "one_step_obs_rmse",
        "rollout_reward_mae",
        "rollout_done_acc",
        "rollout_cost_mae",
        "rollout_force_mae",
    )
    parts = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
    print(colorama.Fore.GREEN + f"{name}: " + ", ".join(parts) + colorama.Style.RESET_ALL)


def _fmt_metric(metrics, key, digits=4):
    value = metrics.get(key)
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_markdown_report(output_dir, summary):
    config = summary.get("config", {})
    expert_conf = config.get("expert", {}) if isinstance(config, dict) else {}
    results = summary.get("results", {})
    lines = [
        "# Expert Policy / World Model Evaluation Report",
        "",
        f"- Config: `{summary.get('config_path')}`",
        f"- Checkpoint: `{summary.get('checkpoint_path') or summary.get('world_model_path')}`",
        f"- Cost target source: `{expert_conf.get('cost_target_source', 'unknown')}`",
        f"- Cost head mode: `{expert_conf.get('cost_head_mode', 'unknown')}`",
        "",
        "## World Model Metrics",
        "",
        "| Data | one-step obs RMSE | open-loop obs RMSE | reward MAE | reward corr | done acc | dyn/KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in results.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt_metric(metrics, "one_step_obs_rmse"),
                    _fmt_metric(metrics, "rollout_obs_rmse"),
                    _fmt_metric(metrics, "rollout_reward_mae"),
                    _fmt_metric(metrics, "rollout_reward_corr"),
                    _fmt_metric(metrics, "rollout_done_acc"),
                    _fmt_metric(metrics, "real_kl"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cost Head Metrics",
            "",
            "Cost metrics use the derived force/margin target. Focus on AUPRC, recall, and positive-only MAE; all-sample MAE can look good when violations are rare.",
            "",
            "| Data | cost positive ratio | AUPRC | random baseline | precision@0.5 | recall@0.5 | F1@0.5 | MAE all | MAE positive only |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in results.items():
        prefix = "rollout_cost"
        if f"{prefix}/positive_ratio" not in metrics:
            prefix = "posterior_cost"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt_metric(metrics, f"{prefix}/positive_ratio", 6),
                    _fmt_metric(metrics, f"{prefix}/auprc"),
                    _fmt_metric(metrics, f"{prefix}/random_auprc_baseline", 6),
                    _fmt_metric(metrics, f"{prefix}/precision@0.5"),
                    _fmt_metric(metrics, f"{prefix}/recall@0.5"),
                    _fmt_metric(metrics, f"{prefix}/f1@0.5"),
                    _fmt_metric(metrics, f"{prefix}/mae_all"),
                    _fmt_metric(metrics, f"{prefix}/mae_positive_only"),
                ]
            )
            + " |"
        )

    if "online_rollout" in results:
        metrics = results["online_rollout"]
        lines.extend(
            [
                "",
                "## Online Rollout Summary",
                "",
                f"- Env obs dim: `{metrics.get('online_env_obs_dim')}`",
                f"- Model obs dim: `{metrics.get('online_model_obs_dim')}`",
                f"- Obs adapter: `{metrics.get('online_obs_dim_adapter')}`",
                f"- Finished episodes: `{metrics.get('online_num_finished_episodes')}`",
                f"- Mean episode return: `{_fmt_metric(metrics, 'online_mean_episode_return')}`",
                f"- Success rate: `{_fmt_metric(metrics, 'online_success_rate')}`",
                f"- Rollout diagnostic plots: `{metrics.get('rollout_diagnostic_plot_count', 0)}`",
                f"- Rollout diagnostic directory: `{metrics.get('rollout_diagnostic_plot_dir', 'n/a')}`",
                f"- Lift height obs index: `{metrics.get('rollout_diagnostic_lift_height_obs_index', 'auto')}`",
                f"- Lift height scale: `{metrics.get('rollout_diagnostic_lift_height_scale', 'auto')}`",
            ]
        )

    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")
    print(colorama.Fore.GREEN + f"Saved markdown evaluation report to {report_path}" + colorama.Style.RESET_ALL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-checkpoint_path", type=str, default=None)
    parser.add_argument("-world_model_path", type=str, default=None)
    parser.add_argument("-agent_path", type=str, default=None)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-buffer_device", type=str, default=None)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-save_dir", type=str, default="eval_expert_policy_world_model")
    parser.add_argument("--eval_dataset", action="store_true")
    parser.add_argument("-dataset_path", type=str, default=None)
    parser.add_argument("-coverage_path", action="append", default=None)
    parser.add_argument("--no_coverage_dataset", action="store_true")
    parser.add_argument("-dataset_format", type=str, default="npz")
    parser.add_argument("-max_episodes", type=int, default=0)
    parser.add_argument("-max_coverage_episodes", type=int, default=0)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("-env_name", type=str, default=None)
    parser.add_argument("-online_steps", type=int, default=8192)
    parser.add_argument("-eval_num_envs", type=int, default=8)
    parser.add_argument("-policy_mode", type=str, choices=("greedy", "stochastic", "random"), default="greedy")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show_window", action="store_true")
    parser.add_argument("--include_force", dest="include_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="include_force", action="store_false")
    parser.add_argument("--plot_rollouts", dest="plot_rollouts", action="store_true", default=True)
    parser.add_argument("--no_plot_rollouts", dest="plot_rollouts", action="store_false")
    parser.add_argument("--plot_num_envs", type=int, default=4)
    parser.add_argument("--plot_env_ids", type=str, default="")
    parser.add_argument("--plot_horizon", type=int, default=0)
    parser.add_argument("--lift_height_obs_index", type=int, default=-1)
    parser.add_argument("--lift_height_scale", type=float, default=0.0)
    parser.add_argument("-action_tolerance", type=float, default=1e-4)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--online_log_every", type=int, default=20)
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    if not args.eval_dataset and not args.online:
        args.eval_dataset = True
    if args.online and not args.env_name:
        raise ValueError("Online evaluation requires -env_name.")
    if args.show_window:
        args.headless = False

    seed_np_torch(args.seed)
    conf = load_expert_config(args.config_path)
    if "cuda" not in str(args.device).lower() and bool(conf.BasicSettings.UseAmp):
        conf.defrost()
        conf.BasicSettings.UseAmp = False
        conf.freeze()
    output_dir = _build_output_dir(args)

    all_results = {}
    simulation_app = None

    if args.online:
        simulation_app, parse_env_cfg, gymnasium, DreamerVecEnvWrapper = _launch_isaac(headless=args.headless)
        vec_env = _build_online_env(args, conf, parse_env_cfg, gymnasium, DreamerVecEnvWrapper)
        env_obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
        checkpoint_obs_dim = _infer_checkpoint_obs_dim(args)
        obs_dim = int(checkpoint_obs_dim or env_obs_dim)
        action_dim = int(vec_env.single_action_space.shape[0])
        if env_obs_dim != obs_dim:
            print(
                colorama.Fore.YELLOW
                + (
                    f"Online env obs_dim={env_obs_dim} but checkpoint/model obs_dim={obs_dim}; "
                    "using a tail zero-pad/truncate adapter for evaluation only."
                )
                + colorama.Style.RESET_ALL
            )
        world_model, agent = _build_models_for_dims(conf, obs_dim, action_dim, args)
        world_model.eval_obs_dim = obs_dim
        world_model.env_obs_dim = env_obs_dim

        if args.eval_dataset:
            groups = _prepare_offline_groups(conf, args)
            for name, (_, replay) in groups.items():
                metrics, curves = evaluate_replay(name, replay, world_model, args)
                if metrics:
                    _write_metrics(output_dir, name, metrics, curves)
                    _print_topline(name, metrics)
                    all_results[name] = metrics

        online_replay, rollout_summary, online_trace = collect_online_replay(vec_env, world_model, agent, conf, args)
        metrics, curves = evaluate_replay("online_rollout", online_replay, world_model, args, source="main")
        metrics = {**rollout_summary, **metrics}
        plot_paths = save_rollout_diagnostic_plots(output_dir, "online_rollout", online_trace, world_model, args)
        if plot_paths:
            metrics["rollout_diagnostic_plot_count"] = int(len(plot_paths))
            metrics["rollout_diagnostic_plot_dir"] = os.path.relpath(os.path.dirname(plot_paths[0]), output_dir)
            lift_index = _resolve_lift_height_obs_index(args, int(metrics.get("online_model_obs_dim", obs_dim)))
            metrics["rollout_diagnostic_lift_height_obs_index"] = lift_index
            metrics["rollout_diagnostic_lift_height_scale"] = _resolve_lift_height_scale(
                args,
                int(metrics.get("online_model_obs_dim", obs_dim)),
            )
        _write_metrics(output_dir, "online_rollout", metrics, curves)
        _print_topline("online_rollout", metrics)
        all_results["online_rollout"] = metrics
    else:
        groups = _prepare_offline_groups(conf, args)
        if not groups:
            raise ValueError("No offline dataset was selected. Pass -dataset_path or set expert.path.")
        first_dataset = next(iter(groups.values()))[0]
        if not hasattr(first_dataset, "metadata") and isinstance(first_dataset, list):
            first_dataset = first_dataset[0]
        obs_dim = int(first_dataset.metadata["obs_dim"])
        action_dim = int(first_dataset.metadata["action_dim"])
        world_model, _ = _build_models_for_dims(conf, obs_dim, action_dim, args)
        for name, (_, replay) in groups.items():
            metrics, curves = evaluate_replay(name, replay, world_model, args)
            if metrics:
                _write_metrics(output_dir, name, metrics, curves)
                _print_topline(name, metrics)
                all_results[name] = metrics

    summary = {
        "config_path": os.path.abspath(os.path.expanduser(args.config_path)),
        "checkpoint_path": os.path.abspath(os.path.expanduser(args.checkpoint_path)) if args.checkpoint_path else None,
        "world_model_path": os.path.abspath(os.path.expanduser(args.world_model_path)) if args.world_model_path else None,
        "agent_path": os.path.abspath(os.path.expanduser(args.agent_path)) if args.agent_path else None,
        "args": vars(args),
        "config": cfg_to_dict(conf),
        "results": all_results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)
    _write_markdown_report(output_dir, summary)
    print(colorama.Fore.GREEN + f"Saved evaluation summary to {output_dir}" + colorama.Style.RESET_ALL)

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
