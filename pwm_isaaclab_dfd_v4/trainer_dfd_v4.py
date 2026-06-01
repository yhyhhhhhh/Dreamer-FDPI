from __future__ import annotations

import os
import random

import numpy as np
import torch
from tqdm import tqdm

try:
    import colorama
except ImportError:  # pragma: no cover
    class _EmptyColors:
        CYAN = GREEN = YELLOW = RESET_ALL = ""

    class _ColoramaFallback:
        Fore = _EmptyColors()
        Style = _EmptyColors()

    colorama = _ColoramaFallback()

try:
    from pwm_isaaclab.trainer import (
        OfflineEpisodeWriter,
        _extract_force_obs,
        _is_first,
        _log_info_value,
        _policy_obs,
        _reset_after_step,
    )
except ImportError:
    from trainer import (
        OfflineEpisodeWriter,
        _extract_force_obs,
        _is_first,
        _log_info_value,
        _policy_obs,
        _reset_after_step,
    )

try:
    from pwm_isaaclab.modules.world_models import predict_force_from_outputs
except ImportError:
    from modules.world_models import predict_force_from_outputs

from .cost_utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    cfg_get,
    continuous_cost_from_force_prediction,
    extract_continuous_cost,
)
from .dual_update_v4 import update_dual_v4
from .sampling_utils import FDPIRegimeStatsWindow, batch_composition, dual_ratio_from_fdpi_stats


def _cfg_float(node, name, default):
    return float(cfg_get(node, name, default))


def _cfg_int(node, name, default):
    return int(cfg_get(node, name, default))


def _cfg_bool(node, name, default=False):
    return bool(cfg_get(node, name, default))


def _node(fdpi_cfg, name):
    return cfg_get(fdpi_cfg, name, None)


def train_world_model_step_dfd_v4(batch, world_model, agent, logger, step):
    if agent is not None:
        agent.eval()
    metrics = world_model.update(
        agent,
        batch["obs"],
        batch["action"],
        batch["reward"],
        batch["done"],
        batch["is_first"],
        force=batch.get("force"),
        cost=batch.get("continuous_cost", batch.get("cost")),
        bottom_force=batch.get("bottom_force"),
        extreme_cost=batch.get("extreme_cost"),
        logger=logger,
        step=step,
    )
    if logger is not None and isinstance(metrics, dict):
        if "dyn_loss" in metrics:
            logger.log("WorldModel/dynamics_loss", metrics["dyn_loss"], step)
        if "cost_loss" in metrics:
            logger.log("WorldModel/cost_loss", metrics["cost_loss"], step)
        if "force_loss" in metrics:
            logger.log("WorldModel/force_loss", metrics["force_loss"], step)
        if "force_pred_mean" in metrics:
            logger.log("WorldModel/pred_bottom_force_mean", metrics["force_pred_mean"], step)
        pred_cost = metrics.get("cost/pred_mean", metrics.get("cost/predicted_cost_mean"))
        if isinstance(pred_cost, (int, float)):
            logger.log("WorldModel/pred_cost_mean", pred_cost, step)
    return metrics


def _predict_imagined_cost(world_model, feat, cost_cfg):
    if hasattr(world_model, "predict_cost"):
        pred_cost, _, _ = world_model.predict_cost(feat)
        return pred_cost.clamp(
            _cfg_float(cost_cfg, "CostMin", 0.0),
            _cfg_float(cost_cfg, "CostMax", 1.0),
        )
    if getattr(world_model, "force_enabled", False) and getattr(world_model, "force_head", None) is not None:
        flat_feat = feat.flatten(0, 1)
        with torch.autocast(device_type=world_model.device_type, dtype=world_model.tensor_dtype, enabled=world_model.use_amp):
            force_outputs = world_model.force_head(flat_feat)
            pred_force, _ = predict_force_from_outputs(
                force_outputs,
                force_scale=world_model.force_scale,
                threshold=world_model.force_threshold,
                signed_force=world_model.force_signed_force,
            )
        pred_force = pred_force.reshape(*feat.shape[:-1], 1)
        return continuous_cost_from_force_prediction(
            pred_force,
            force_threshold=_cfg_float(cost_cfg, "ForceThreshold", 0.1),
            low_force_scale=_cfg_float(cost_cfg, "LowForceScale", 0.05),
            cost_force_max=_cfg_float(cost_cfg, "CostForceMax", 15.0),
            force_scale=_cfg_float(cost_cfg, "ForceScale", 5.0),
            clip_cost=_cfg_bool(cost_cfg, "ClipCost", True),
            cost_min=_cfg_float(cost_cfg, "CostMin", 0.0),
            cost_max=_cfg_float(cost_cfg, "CostMax", 1.0),
        ).to(feat.device)
    return torch.zeros(*feat.shape[:-1], 1, dtype=feat.dtype, device=feat.device)


def _main_fdpi_cfg(fdpi_cfg):
    risk_cfg = _node(fdpi_cfg, "RiskCritic")
    main_cfg = _node(fdpi_cfg, "MainFDPIRegime")
    return {
        "Pf": _cfg_float(risk_cfg, "Pf", 0.40),
        "Cg": _cfg_float(risk_cfg, "Cg", 0.10),
        "RiskMax": _cfg_float(risk_cfg, "RiskMax", 1.0),
        "LambdaCri": _cfg_float(main_cfg, "LambdaCri", 0.001),
        "LambdaInf": _cfg_float(main_cfg, "LambdaInf", 0.002),
        "MinRewardWeightCri": _cfg_float(main_cfg, "MinRewardWeightCri", 0.80),
        "MinRewardWeightInf": _cfg_float(main_cfg, "MinRewardWeightInf", 0.80),
        "EntropyCoef": _cfg_float(main_cfg, "EntropyCoef", 1.0e-4),
        "ActionAnchorCoef": _cfg_float(main_cfg, "ActionAnchorCoef", 0.0),
        "TailRiskCoef": _cfg_float(main_cfg, "TailRiskCoef", 0.0),
        "TailRiskThreshold": _cfg_float(main_cfg, "TailRiskThreshold", _cfg_float(risk_cfg, "Pf", 0.40)),
        "DetachActionForLogProb": _cfg_bool(main_cfg, "DetachActionForLogProb", False),
    }


def train_agent_step_dfd_v4(
    samples,
    world_model,
    agent,
    gp_critic,
    imagine_horizon,
    logger,
    step,
    *,
    fdpi_cfg=None,
):
    world_model.eval()
    feat, action, discount, reward, weight = world_model.imagine_data(
        agent,
        *samples[:5],
        imagine_horizon,
        logger,
        step,
    )
    main_cfg = _node(fdpi_cfg, "MainFDPIRegime")
    if (
        _cfg_bool(main_cfg, "Enable", True)
        and int(step) >= _cfg_int(main_cfg, "StartStep", 1500000)
        and gp_critic is not None
    ):
        info = agent.update_fdpi_regime(
            feat,
            action,
            discount,
            reward,
            weight,
            gp_critic,
            _main_fdpi_cfg(fdpi_cfg),
            logger=logger,
            step=step,
        )
        info["used_fdpi_regime"] = True
        return info
    agent.update(feat, action, discount, reward, weight, logger, step)
    if logger is not None:
        logger.log("MainFDPI/enabled", 0.0, step)
    return {
        "used_fdpi_regime": False,
        "task_reward_mean": float(reward.detach().float().mean().item()),
    }


def _sample_policy_action(
    *,
    feat,
    agent,
    gp_critic,
    dual_policy,
    world_model,
    state,
    use_dual_sampling,
    dual_ratio,
    num_envs,
    device,
):
    main_action = agent.sample(feat, greedy=False)
    action = main_action
    source = torch.full((num_envs, 1), SOURCE_MAIN, dtype=torch.int64, device=device)
    g_main = None
    if gp_critic is not None:
        g_main = gp_critic.risk_no_grad(feat, main_action).detach()
    if use_dual_sampling and dual_policy is not None and float(dual_ratio) > 0.0:
        dual_mask = torch.rand(num_envs, device=device) < float(dual_ratio)
        if dual_mask.any():
            dual_action = dual_policy.sample(feat, greedy=False)
            dual_action = dual_action.to(device=main_action.device, dtype=main_action.dtype)
            action = main_action.clone()
            action[dual_mask] = dual_action[dual_mask]
            source[dual_mask] = SOURCE_DUAL
    env_action = action.detach().cpu().numpy()
    state = world_model.update_inference_state(state, action)
    return env_action, action, source, state, g_main


def _action_bounds(vec_env, device, dtype):
    space = getattr(vec_env, "single_action_space", getattr(vec_env, "action_space", None))
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return None, None
    low_t = torch.as_tensor(low, dtype=dtype, device=device).reshape(1, -1)
    high_t = torch.as_tensor(high, dtype=dtype, device=device).reshape(1, -1)
    return low_t, high_t


def _sample_warmup_policy_noise_action(
    *,
    current_obs,
    is_first,
    agent,
    world_model,
    state,
    vec_env,
    num_envs,
    device,
    noise_std,
    greedy_base=False,
):
    with torch.no_grad():
        world_model.eval()
        agent.eval()
        feat, state = world_model.get_inference_feat(state, current_obs, is_first)
        base_action = agent.sample(feat, greedy=bool(greedy_base))
        if float(noise_std) > 0.0:
            noise = torch.randn_like(base_action) * float(noise_std)
            action = base_action + noise
        else:
            action = base_action
        low, high = _action_bounds(vec_env, device, action.dtype)
        if low is not None and high is not None:
            action = torch.max(torch.min(action, high), low)
        else:
            action = action.clamp(-1.0, 1.0)
        source = torch.full((num_envs, 1), SOURCE_MAIN, dtype=torch.int64, device=device)
        state = world_model.update_inference_state(state, action)
        env_action = action.detach().cpu().numpy()
    return env_action, action, source, state


def _log_replay_stats(replay_buffer, logger, step, *, high_cost_threshold=0.1, boundary_low=0.05, boundary_high=0.4):
    if not hasattr(replay_buffer, "source_stats"):
        return
    stats = replay_buffer.source_stats()
    total = max(sum(stats.values()), 1)
    logger.log("Replay/source_main_ratio", stats.get("main", 0) / total, step)
    logger.log("Replay/source_dual_ratio", stats.get("dual", 0) / total, step)
    logger.log("Replay/source_random_ratio", stats.get("random", 0) / total, step)
    if hasattr(replay_buffer, "cost_stats"):
        cost_stats = replay_buffer.cost_stats(
            high_cost_threshold=high_cost_threshold,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
        )
        for key, value in cost_stats.items():
            logger.log(f"Replay/{key}", value, step)


def _log_batch_composition(logger, prefix, batch, step, *, high_cost_threshold, boundary_low, boundary_high):
    stats = batch_composition(
        batch,
        high_cost_threshold=high_cost_threshold,
        boundary_low=boundary_low,
        boundary_high=boundary_high,
    )
    for key, value in stats.items():
        logger.log(f"{prefix}/{key}", value, step)


_INFO_LOG_KEYWORDS = ("reward", "force")
_INFO_SKIP_KEYS = {"terminal_observation"}


def _should_log_info_key(path):
    lower_path = str(path).lower()
    return any(keyword in lower_path for keyword in _INFO_LOG_KEYWORDS)


def _numeric_numpy_array(value):
    if not isinstance(value, np.ndarray):
        return False
    return np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)


def _mask_info_value(value, mask):
    if value is None:
        return None

    if isinstance(value, dict):
        masked = {}
        for key, sub_value in value.items():
            sub_masked = _mask_info_value(sub_value, mask)
            if sub_masked is not None:
                masked[key] = sub_masked
        return masked or None

    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[0] != mask.numel():
            return None
        selected = mask.to(device=value.device)
        if not bool(selected.any().item()):
            return None
        return value.detach()[selected]

    if isinstance(value, (list, tuple)):
        try:
            value = np.asarray(value)
        except Exception:
            return None

    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != mask.numel() or not _numeric_numpy_array(value):
            return None
        selected = mask.detach().cpu().numpy().astype(bool)
        if not bool(selected.any()):
            return None
        return value[selected]

    return None


def _log_reward_force_info_value(logger, tag, value, step):
    if value is None:
        return

    if isinstance(value, dict):
        for key, sub_value in value.items():
            if key in _INFO_SKIP_KEYS:
                continue
            _log_reward_force_info_value(logger, f"{tag}/{key}", sub_value, step)
        return

    if _should_log_info_key(tag):
        _log_info_value(logger, tag, value, step)


def _log_reward_force_info_value_by_source(logger, tag, value, source_masks, step):
    if value is None:
        return

    if isinstance(value, dict):
        for key, sub_value in value.items():
            if key in _INFO_SKIP_KEYS:
                continue
            _log_reward_force_info_value_by_source(logger, f"{tag}/{key}", sub_value, source_masks, step)
        return

    if not _should_log_info_key(tag):
        return

    for prefix, mask in source_masks:
        masked_value = _mask_info_value(value, mask)
        if masked_value is not None:
            _log_info_value(logger, f"{prefix}/{tag}", masked_value, step)


def _log_info_dict_reward_force_by_source(logger, info, source, step):
    if not isinstance(info, dict):
        return
    source_mask = source.detach().reshape(-1)
    source_masks = (
        ("InfoMain", source_mask == SOURCE_MAIN),
        ("InfoDual", source_mask == SOURCE_DUAL),
    )
    for key, value in info.items():
        if key in _INFO_SKIP_KEYS:
            continue
        _log_reward_force_info_value(logger, f"Info/{key}", value, step)
        _log_reward_force_info_value_by_source(logger, key, value, source_masks, step)


def _extract_left_right_bottom_force(obs_dict, *, num_envs, device, force_key="", bottom_force_channels=(2, 5)):
    if not isinstance(obs_dict, dict):
        return None, None
    candidate_keys = tuple(key for key in (force_key, "force") if key)
    for key in candidate_keys:
        value = obs_dict.get(key)
        if value is None:
            continue
        force = torch.as_tensor(value, dtype=torch.float32, device=device)
        if force.ndim == 0 or force.shape[0] != num_envs:
            continue
        force = torch.nan_to_num(force.reshape(num_envs, -1).abs(), nan=0.0, posinf=1.0e6)
        if len(bottom_force_channels) >= 2 and force.shape[-1] > max(bottom_force_channels[:2]):
            left_idx = int(bottom_force_channels[0])
            right_idx = int(bottom_force_channels[1])
            return force[:, left_idx], force[:, right_idx]
    return None, None


def _log_bottom_side_info_by_source(logger, source, leftbottom, rightbottom, step):
    source_mask = source.detach().reshape(-1)
    for prefix, mask in (("InfoMain", source_mask == SOURCE_MAIN), ("InfoDual", source_mask == SOURCE_DUAL)):
        mask = mask.to(device=source.device)
        if not bool(mask.any().item()):
            continue
        if leftbottom is not None:
            logger.log(f"{prefix}/leftbottom", leftbottom.to(device=source.device)[mask].float().mean().item(), step)
        if rightbottom is not None:
            logger.log(f"{prefix}/rightbottom", rightbottom.to(device=source.device)[mask].float().mean().item(), step)


def _module_optimizer_state(module):
    optimizer = getattr(module, "optimizer", None)
    return optimizer.state_dict() if optimizer is not None else None


def _module_scaler_state(module):
    scaler = getattr(module, "scaler", None)
    return scaler.state_dict() if scaler is not None else None


def _agent_ema_state(agent):
    state = {}
    for name in ("lower_ema", "upper_ema"):
        ema = getattr(agent, name, None)
        if ema is not None:
            state[name] = {
                "scalar": float(getattr(ema, "scalar", 0.0)),
                "decay": float(getattr(ema, "decay", 0.0)),
            }
    return state


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _save_full_state_v4(
    path,
    *,
    env_steps,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    replay_buffer,
    save_replay_buffer=True,
    save_optimizer=True,
):
    payload = {
        "format": "fdpi_regime_dreamer_v4_full_state",
        "version": 1,
        "env_steps": int(env_steps),
        "world_model_state_dict": world_model.state_dict(),
        "agent_state_dict": agent.state_dict(),
        "gp_state_dict": gp_critic.state_dict(),
        "gd_state_dict": gd_critic.state_dict(),
        "dual_policy_state_dict": dual_policy.state_dict(),
        "agent_ema_state": _agent_ema_state(agent),
        "rng_state": _rng_state(),
    }
    if save_optimizer:
        payload["optimizer_state_dicts"] = {
            "world_model": _module_optimizer_state(world_model),
            "agent": _module_optimizer_state(agent),
            "gp": _module_optimizer_state(gp_critic),
            "gd": _module_optimizer_state(gd_critic),
            "dual_policy": _module_optimizer_state(dual_policy),
        }
        payload["scaler_state_dicts"] = {
            "world_model": _module_scaler_state(world_model),
            "agent": _module_scaler_state(agent),
        }
    if save_replay_buffer:
        payload["replay_buffer_state_dict"] = replay_buffer.state_dict(cpu=True)
    torch.save(payload, path)


def joint_train_dfd_v4(
    env_name,
    run_name,
    vec_env,
    max_steps,
    replay_buffer,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    fdpi_cfg,
    train_model_every_steps,
    train_agent_every_steps,
    model_update,
    agent_update,
    batch_size,
    batch_length,
    imagine_batch_size,
    imagine_context,
    imagine_horizon,
    save_every_steps,
    logger,
    device,
    offline_dataset_dir=None,
    checkpoint_dir=None,
    initial_env_steps=0,
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or f"ckpt/{run_name}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(colorama.Fore.CYAN + f"Saving DFD v4 checkpoints to {checkpoint_dir}" + colorama.Style.RESET_ALL)

    num_envs = vec_env.num_envs
    offline_episode_writer = None
    model_update = max(int(model_update), 1)
    agent_update = max(int(agent_update), 1)
    batch_size = int(batch_size)
    batch_length = int(batch_length)
    imagine_batch_size = int(imagine_batch_size)
    imagine_context = int(imagine_context)
    if imagine_batch_size <= 0:
        imagine_batch_size = batch_size
    if imagine_context <= 0:
        imagine_context = batch_length

    replay_cfg = _node(fdpi_cfg, "Replay")
    warmup_sampling_cfg = _node(fdpi_cfg, "WarmupSampling")
    cost_cfg = _node(fdpi_cfg, "ContinuousCost")
    risk_cfg = _node(fdpi_cfg, "RiskCritic")
    gp_cfg = _node(fdpi_cfg, "Gp")
    gd_cfg = _node(fdpi_cfg, "Gd")
    dual_update_cfg = _node(fdpi_cfg, "DualUpdate")
    dual_sampling_cfg = _node(fdpi_cfg, "DualSampling")
    wm_sampling_cfg = _node(fdpi_cfg, "WorldModelSampling")
    checkpoint_cfg = _node(fdpi_cfg, "Checkpoint")

    high_cost_threshold = _cfg_float(wm_sampling_cfg, "HighCostThreshold", _cfg_float(gp_cfg, "HighCostThreshold", 0.1))
    boundary_low = _cfg_float(wm_sampling_cfg, "BoundaryLow", _cfg_float(gp_cfg, "BoundaryLow", 0.05))
    boundary_high = _cfg_float(wm_sampling_cfg, "BoundaryHigh", _cfg_float(gp_cfg, "BoundaryHigh", 0.4))
    world_model_safety_ratio = (
        _cfg_float(wm_sampling_cfg, "SafetyCriticalRatio", 0.20)
        if _cfg_bool(wm_sampling_cfg, "EnableSafetyCriticalSampling", True)
        else 0.0
    )
    bottom_channels = tuple(int(v) for v in cfg_get(cost_cfg, "BottomForceChannels", [2, 5]))
    warmup_noise_std = _cfg_float(warmup_sampling_cfg, "NoiseStd", 0.50)
    warmup_greedy_base = _cfg_bool(warmup_sampling_cfg, "GreedyBase", False)
    save_full_state = _cfg_bool(checkpoint_cfg, "SaveFullState", True)
    save_replay_buffer = _cfg_bool(checkpoint_cfg, "SaveReplayBuffer", True)
    save_optimizer = _cfg_bool(checkpoint_cfg, "SaveOptimizer", True)
    full_state_prefix = str(cfg_get(checkpoint_cfg, "FullStatePrefix", "full_state_v4"))
    gp_update_steps = max(_cfg_int(gp_cfg, "UpdateSteps", 1), 1)
    gd_update_steps = max(_cfg_int(gd_cfg, "UpdateSteps", 1), 1)
    dual_update_steps = max(_cfg_int(dual_update_cfg, "UpdateSteps", 1), 1)
    pf = _cfg_float(risk_cfg, "Pf", 0.10)
    cg = _cfg_float(risk_cfg, "Cg", 0.03)
    feasible_window = FDPIRegimeStatsWindow(_cfg_int(dual_sampling_cfg, "FeasibleRatioWindow", 10000))

    model_update_count = 0
    agent_update_count = 0
    gp_update_count = 0
    gd_update_count = 0
    dual_update_count = 0
    last_dual_kl = 0.0

    episode_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_cost = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_bottom_force = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_bottom_force_peak = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_len = torch.zeros(num_envs, dtype=torch.float32, device=device)

    if offline_dataset_dir:
        offline_episode_writer = OfflineEpisodeWriter(offline_dataset_dir, num_envs)
        print(colorama.Fore.CYAN + f"Saving offline episodes to {offline_episode_writer.output_dir}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()
    gp_critic.eval()
    gd_critic.eval()
    dual_policy.eval()
    state = world_model.initial(num_envs)
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(device)
    is_first = _is_first(current_obs_dict, num_envs, device)
    episodes_completed = 0
    episode_successes = 0
    episode_failures = 0
    episode_timeouts = 0

    initial_env_steps = max(int(initial_env_steps), 0)
    logger.log(f"Rollout/IsaacLab/{env_name}_reward", 0, initial_env_steps)
    logger.log("Rollout/buffer_length", 0, initial_env_steps)
    remaining_steps = max(int(max_steps) - initial_env_steps, 0)
    total_iters = remaining_steps // num_envs
    train_model_every_iters = max(train_model_every_steps // num_envs, 1)
    train_agent_every_iters = max(train_agent_every_steps // num_envs, 1)
    save_every_iters = max(save_every_steps // num_envs, 1)

    for iter_idx in tqdm(range(total_iters)):
        env_steps = initial_env_steps + iter_idx * num_envs
        g_main_for_window = None

        if replay_buffer.ready():
            with torch.no_grad():
                world_model.eval()
                agent.eval()
                feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                stats = feasible_window.stats()
                dual_ratio, ratio_info = dual_ratio_from_fdpi_stats(
                    step=env_steps,
                    cfg=dual_sampling_cfg,
                    stats=stats,
                    last_dual_kl=last_dual_kl,
                )
                env_action, action, source, state, g_main_for_window = _sample_policy_action(
                    feat=feat,
                    agent=agent,
                    gp_critic=gp_critic,
                    dual_policy=dual_policy,
                    world_model=world_model,
                    state=state,
                    use_dual_sampling=dual_ratio > 0.0,
                    dual_ratio=dual_ratio,
                    num_envs=num_envs,
                    device=device,
                )
                logger.log("Dual/ratio", dual_ratio, env_steps)
                logger.log("Dual/active", float(dual_ratio > 0.0), env_steps)
                logger.log("Dual/kl_to_main", float(last_dual_kl), env_steps)
                for key, value in ratio_info.items():
                    logger.log(f"DualSampling/{key}", value, env_steps)
        else:
            env_action, action, source, state = _sample_warmup_policy_noise_action(
                current_obs=current_obs,
                is_first=is_first,
                agent=agent,
                world_model=world_model,
                state=state,
                vec_env=vec_env,
                num_envs=num_envs,
                device=device,
                noise_std=warmup_noise_std,
                greedy_base=warmup_greedy_base,
            )
            logger.log("Warmup/policy_noise", 1.0, env_steps)
            logger.log("Warmup/noise_std", warmup_noise_std, env_steps)

        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        info_for_log = dict(info) if isinstance(info, dict) else {}
        info_for_log.setdefault("reward", reward)
        _log_info_dict_reward_force_by_source(logger, info_for_log, source, env_steps)

        cost_parts = extract_continuous_cost(
            info,
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            force_threshold=_cfg_float(cost_cfg, "ForceThreshold", 0.1),
            low_force_scale=_cfg_float(cost_cfg, "LowForceScale", 0.05),
            cost_force_max=_cfg_float(cost_cfg, "CostForceMax", 15.0),
            force_scale=_cfg_float(cost_cfg, "ForceScale", 5.0),
            extreme_force_threshold=_cfg_float(cost_cfg, "ExtremeForceThreshold", 5.0),
            clip_cost=_cfg_bool(cost_cfg, "ClipCost", True),
            cost_min=_cfg_float(cost_cfg, "CostMin", 0.0),
            cost_max=_cfg_float(cost_cfg, "CostMax", 1.0),
            force_key=getattr(replay_buffer, "force_key", ""),
            bottom_force_channels=bottom_channels,
        )
        continuous_cost = cost_parts["continuous_cost"]
        binary_cost = cost_parts["binary_cost"]
        extreme_cost = cost_parts["extreme_cost"]
        bottom_force = cost_parts["bottom_force"]
        force_excess = cost_parts["force_excess"]
        leftbottom, rightbottom = _extract_left_right_bottom_force(
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            force_key=getattr(replay_buffer, "force_key", ""),
            bottom_force_channels=bottom_channels,
        )
        _log_bottom_side_info_by_source(logger, source, leftbottom, rightbottom, env_steps)

        if g_main_for_window is not None:
            feasible_window.append(
                g_main=g_main_for_window,
                source=source,
                continuous_cost=continuous_cost,
                pf=pf,
                cg=cg,
            )

        terminal = torch.as_tensor(
            next_obs_dict.get("is_terminal", torch.zeros_like(done, dtype=torch.int32)),
            dtype=torch.bool,
            device=device,
        ).view(-1)
        failure = torch.as_tensor(
            next_obs_dict.get("failure", torch.zeros_like(done, dtype=torch.int32)),
            dtype=torch.bool,
            device=device,
        ).view(-1)
        episode_success = info.get("episode_success")
        if episode_success is None:
            episode_success = terminal & ~failure
        else:
            episode_success = torch.as_tensor(episode_success, dtype=torch.bool, device=device).view(-1)
        episode_failure = info.get("episode_failure")
        if episode_failure is None:
            episode_failure = terminal & failure
        else:
            episode_failure = torch.as_tensor(episode_failure, dtype=torch.bool, device=device).view(-1)
        episode_timeout = info.get("episode_timeout")
        if episode_timeout is None:
            episode_timeout = done & ~terminal
        else:
            episode_timeout = torch.as_tensor(episode_timeout, dtype=torch.bool, device=device).view(-1)

        force = None
        if getattr(replay_buffer, "include_force", False):
            force = _extract_force_obs(
                current_obs_dict,
                num_envs,
                device,
                getattr(replay_buffer, "force_key", ""),
            )

        replay_buffer.append(
            current_obs,
            action,
            reward,
            done,
            is_first,
            force=force,
            continuous_cost=continuous_cost,
            binary_cost=binary_cost,
            extreme_cost=extreme_cost,
            bottom_force=bottom_force,
            force_excess=force_excess,
            source=source,
        )
        if offline_episode_writer is not None:
            offline_episode_writer.append_step(current_obs_dict, action, reward, done, is_first, env_steps + num_envs)

        episode_reward += reward
        episode_cost += continuous_cost.view(-1)
        episode_bottom_force += bottom_force.view(-1)
        episode_bottom_force_peak = torch.maximum(episode_bottom_force_peak, bottom_force.view(-1))
        episode_len += 1.0
        logger.log("Main/continuous_cost_mean", continuous_cost.float().mean().item(), env_steps)
        logger.log("Dual/source_count", float((source == SOURCE_DUAL).sum().item()), env_steps)
        if (source == SOURCE_DUAL).any():
            logger.log("Dual/real_cost_mean", continuous_cost[source.view(-1) == SOURCE_DUAL].float().mean().item(), env_steps)

        if done.any():
            done_indices = torch.nonzero(done, as_tuple=False).flatten()
            completed_now = int(done_indices.numel())
            success_now = int(episode_success[done_indices].sum().item())
            failure_now = int(episode_failure[done_indices].sum().item())
            timeout_now = int(episode_timeout[done_indices].sum().item())

            episodes_completed += completed_now
            episode_successes += success_now
            episode_failures += failure_now
            episode_timeouts += timeout_now

            for idx in done_indices.tolist():
                ep_len = max(float(episode_len[idx].item()), 1.0)
                if replay_buffer.ready():
                    logger.log(f"Rollout/IsaacLab/{env_name}_reward", episode_reward[idx].item(), env_steps)
                    logger.log("Rollout/episode_cost", episode_cost[idx].item(), env_steps)
                    logger.log("Rollout/buffer_length", len(replay_buffer), env_steps)
                    logger.log("Main/task_return", episode_reward[idx].item(), env_steps)
                    logger.log("Main/episode_cost_mean", episode_cost[idx].item() / ep_len, env_steps)
                    logger.log("Main/bottom_force_mean", episode_bottom_force[idx].item() / ep_len, env_steps)
                    logger.log("Main/bottom_force_peak", episode_bottom_force_peak[idx].item(), env_steps)
                    logger.log("Main/success_rate", episode_successes / max(episodes_completed, 1), env_steps)
                episode_reward[idx] = 0.0
                episode_cost[idx] = 0.0
                episode_bottom_force[idx] = 0.0
                episode_bottom_force_peak[idx] = 0.0
                episode_len[idx] = 0.0

            logger.log("Rollout/episodes_completed", episodes_completed, env_steps)
            logger.log("Rollout/episode_successes", episode_successes, env_steps)
            logger.log("Rollout/episode_failures", episode_failures, env_steps)
            logger.log("Rollout/episode_timeouts", episode_timeouts, env_steps)
            logger.log("Rollout/episode_success_rate", episode_successes / max(episodes_completed, 1), env_steps)
            logger.log("Rollout/episode_failure_rate", episode_failures / max(episodes_completed, 1), env_steps)
            logger.log("Rollout/episode_timeout_rate", episode_timeouts / max(episodes_completed, 1), env_steps)

        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, device)

        if replay_buffer.ready():
            if iter_idx % train_model_every_iters == 0 and replay_buffer.can_sample(batch_length):
                for _ in range(model_update):
                    batch = replay_buffer.sample(
                        batch_size,
                        batch_length,
                        return_dict=True,
                        safety_critical_ratio=world_model_safety_ratio,
                        high_cost_threshold=high_cost_threshold,
                        boundary_low=boundary_low,
                        boundary_high=boundary_high,
                    )
                    _log_batch_composition(
                        logger,
                        "WorldModelBatch",
                        batch,
                        env_steps,
                        high_cost_threshold=high_cost_threshold,
                        boundary_low=boundary_low,
                        boundary_high=boundary_high,
                    )
                    train_world_model_step_dfd_v4(batch, world_model, agent, logger, env_steps)
                    model_update_count += 1

            if _cfg_bool(gp_cfg, "Enable", True) and iter_idx % train_agent_every_iters == 0 and replay_buffer.can_sample(batch_length):
                for _ in range(gp_update_steps):
                    batch = replay_buffer.sample(
                        batch_size,
                        batch_length,
                        return_dict=True,
                        safety_critical_ratio=_cfg_float(gp_cfg, "SafetyCriticalRatio", 0.20),
                        high_cost_threshold=high_cost_threshold,
                        boundary_low=boundary_low,
                        boundary_high=boundary_high,
                    )
                    gp_critic.update(batch, world_model, agent, dual_policy, logger=logger, step=env_steps)
                    gp_update_count += 1

            if _cfg_bool(gd_cfg, "Enable", True) and iter_idx % train_agent_every_iters == 0 and replay_buffer.can_sample(batch_length):
                for _ in range(gd_update_steps):
                    batch = replay_buffer.sample(
                        batch_size,
                        batch_length,
                        return_dict=True,
                        safety_critical_ratio=_cfg_float(gd_cfg, "SafetyCriticalRatio", 0.40),
                        high_cost_threshold=high_cost_threshold,
                        boundary_low=boundary_low,
                        boundary_high=boundary_high,
                    )
                    gd_critic.update(batch, world_model, dual_policy, logger=logger, step=env_steps)
                    gd_update_count += 1

            if (
                _cfg_bool(dual_update_cfg, "Enable", True)
                and env_steps >= _cfg_int(dual_update_cfg, "StartStep", 100000)
                and iter_idx % train_agent_every_iters == 0
                and replay_buffer.can_sample(batch_length)
            ):
                for _ in range(dual_update_steps):
                    batch = replay_buffer.sample(batch_size, batch_length, return_dict=True)
                    info_dual = update_dual_v4(
                        batch,
                        world_model,
                        agent,
                        gd_critic,
                        dual_policy,
                        dual_update_cfg,
                        cost_cfg=cost_cfg,
                        logger=logger,
                        step=env_steps,
                    )
                    last_dual_kl = abs(float(info_dual.get("kl_to_main", 0.0))) if info_dual else last_dual_kl
                    dual_update_count += 1

            if iter_idx % train_agent_every_iters == 0 and replay_buffer.can_sample(imagine_context):
                for _ in range(agent_update):
                    imagine_samples = replay_buffer.sample(imagine_batch_size, imagine_context)
                    train_agent_step_dfd_v4(
                        imagine_samples,
                        world_model,
                        agent,
                        gp_critic,
                        imagine_horizon,
                        logger,
                        env_steps,
                        fdpi_cfg=fdpi_cfg,
                    )
                    agent_update_count += 1

            collected_steps = env_steps + num_envs
            logger.log("Train/model_updates", model_update_count, env_steps)
            logger.log("Train/agent_updates", agent_update_count, env_steps)
            logger.log("Train/gp_updates", gp_update_count, env_steps)
            logger.log("Train/gd_updates", gd_update_count, env_steps)
            logger.log("Train/dual_updates", dual_update_count, env_steps)
            logger.log("Train/model_update_ratio", model_update_count / collected_steps, env_steps)
            logger.log("Train/agent_update_ratio", agent_update_count / collected_steps, env_steps)
            _log_replay_stats(
                replay_buffer,
                logger,
                env_steps,
                high_cost_threshold=high_cost_threshold,
                boundary_low=boundary_low,
                boundary_high=boundary_high,
            )

        if iter_idx % save_every_iters == 0:
            print(colorama.Fore.GREEN + f"Saving DFD v4 model at total steps {env_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_v4_{env_steps}.pth"))
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_v4_{env_steps}.pth"))
            torch.save(gp_critic.state_dict(), os.path.join(checkpoint_dir, f"gp_v4_{env_steps}.pth"))
            torch.save(gd_critic.state_dict(), os.path.join(checkpoint_dir, f"gd_v4_{env_steps}.pth"))
            torch.save(dual_policy.state_dict(), os.path.join(checkpoint_dir, f"dual_policy_v4_{env_steps}.pth"))
            if save_full_state:
                full_state_path = os.path.join(checkpoint_dir, f"{full_state_prefix}_{env_steps}.pth")
                _save_full_state_v4(
                    full_state_path,
                    env_steps=env_steps,
                    world_model=world_model,
                    agent=agent,
                    gp_critic=gp_critic,
                    gd_critic=gd_critic,
                    dual_policy=dual_policy,
                    replay_buffer=replay_buffer,
                    save_replay_buffer=save_replay_buffer,
                    save_optimizer=save_optimizer,
                )
                print(colorama.Fore.GREEN + f"Saved DFD v4 full state to {full_state_path}" + colorama.Style.RESET_ALL)

    if offline_episode_writer is not None:
        offline_episode_writer.flush_pending(max_steps)
        print(
            colorama.Fore.CYAN
            + (
                f"Saved {offline_episode_writer.num_saved_episodes} offline episodes "
                f"({offline_episode_writer.num_saved_steps} steps) to {offline_episode_writer.output_dir}"
            )
            + colorama.Style.RESET_ALL
        )
