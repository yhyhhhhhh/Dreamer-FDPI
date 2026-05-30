from __future__ import annotations

import os

import colorama
import numpy as np
import torch
from tqdm import tqdm

try:
    from pwm_isaaclab.trainer import (
        OfflineEpisodeWriter,
        _extract_force_obs,
        _is_first,
        _log_info_dict,
        _policy_obs,
        _reset_after_step,
    )
except ImportError:
    from trainer import (
        OfflineEpisodeWriter,
        _extract_force_obs,
        _is_first,
        _log_info_dict,
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
    SOURCE_RANDOM,
    cfg_get,
    continuous_cost_from_force_prediction,
    extract_continuous_cost,
    linear_warmup,
)
from .dual_imagination import update_dual_in_imagination


def _cfg_float(node, name, default):
    return float(cfg_get(node, name, default))


def _cfg_int(node, name, default):
    return int(cfg_get(node, name, default))


def _cfg_bool(node, name, default=False):
    return bool(cfg_get(node, name, default))


def _node(dfd_cfg, name):
    return cfg_get(dfd_cfg, name, None)


def train_world_model_step_dfd_v2(batch, world_model, agent, logger, step):
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
        pred_cost_max = metrics.get("cost/pred_max")
        if isinstance(pred_cost_max, (int, float)):
            logger.log("WorldModel/pred_cost_max", pred_cost_max, step)
        extreme_ratio = metrics.get("cost/extreme_ratio")
        if isinstance(extreme_ratio, (int, float)):
            logger.log("WorldModel/extreme_force_rate", extreme_ratio, step)
        extreme_prob = metrics.get("cost/extreme_prob_mean")
        if isinstance(extreme_prob, (int, float)):
            logger.log("WorldModel/extreme_prob_mean", extreme_prob, step)
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


def train_agent_step_dfd_v2(
    samples,
    world_model,
    agent,
    imagine_horizon,
    logger,
    step,
    *,
    dfd_cfg=None,
):
    world_model.eval()
    feat, action, discount, reward, weight = world_model.imagine_data(
        agent,
        *samples[:5],
        imagine_horizon,
        logger,
        step,
    )
    main_cfg = _node(dfd_cfg, "MainCostAwareReward")
    cost_cfg = _node(dfd_cfg, "ContinuousCost")
    lambda_cost = 0.0
    used_safe_reward = False
    train_reward = reward
    if (
        _cfg_bool(main_cfg, "Enable", False)
        and int(step) >= _cfg_int(main_cfg, "StartStep", 150000)
    ):
        lambda_cost = _cfg_float(main_cfg, "LambdaCost", 0.03)
        with torch.no_grad():
            pred_cost = _predict_imagined_cost(world_model, feat[:, 1:], cost_cfg)
            train_reward = reward - float(lambda_cost) * pred_cost
            used_safe_reward = True
        if logger is not None:
            logger.log("Main/lambda_cost", lambda_cost, step)
            logger.log("Main/predicted_cost_mean", pred_cost.detach().float().mean().item(), step)
            logger.log("Main/task_imagined_reward", reward.detach().float().mean().item(), step)
            logger.log("Main/safe_imagined_reward", train_reward.detach().float().mean().item(), step)
    elif logger is not None:
        logger.log("Main/lambda_cost", 0.0, step)
    agent.update(feat, action, discount, train_reward, weight, logger, step)
    return {
        "used_safe_reward": used_safe_reward,
        "lambda_cost": lambda_cost,
        "task_reward_mean": float(reward.detach().float().mean().item()),
        "train_reward_mean": float(train_reward.detach().float().mean().item()),
    }


def _sample_policy_action(
    *,
    feat,
    agent,
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
    return env_action, action, source, state


def _log_replay_stats(replay_buffer, logger, step):
    if not hasattr(replay_buffer, "source_stats"):
        return
    stats = replay_buffer.source_stats()
    total = max(sum(stats.values()), 1)
    logger.log("Replay/source_main_ratio", stats.get("main", 0) / total, step)
    logger.log("Replay/source_dual_ratio", stats.get("dual", 0) / total, step)
    logger.log("Replay/source_random_ratio", stats.get("random", 0) / total, step)
    if hasattr(replay_buffer, "cost_stats"):
        cost_stats = replay_buffer.cost_stats()
        logger.log("Replay/cost_mean", cost_stats["cost_mean"], step)
        logger.log("Replay/main_cost_mean", cost_stats["main_cost_mean"], step)
        logger.log("Replay/dual_cost_mean", cost_stats["dual_cost_mean"], step)
        logger.log("Replay/main_cost_rate", cost_stats["main_cost_rate"], step)
        logger.log("Replay/dual_cost_rate", cost_stats["dual_cost_rate"], step)
        logger.log("Replay/extreme_cost_rate", cost_stats["extreme_cost_rate"], step)
        logger.log("Replay/main_extreme_rate", cost_stats["main_extreme_rate"], step)
        logger.log("Replay/dual_extreme_rate", cost_stats["dual_extreme_rate"], step)
        logger.log("Replay/force_excess_mean", cost_stats["force_excess_mean"], step)
        logger.log("Replay/force_excess_max", cost_stats["force_excess_max"], step)


def _current_lambda_cost(dfd_cfg, step):
    main_cfg = _node(dfd_cfg, "MainCostAwareReward")
    if _cfg_bool(main_cfg, "Enable", False) and int(step) >= _cfg_int(main_cfg, "StartStep", 150000):
        return _cfg_float(main_cfg, "LambdaCost", 0.03)
    return 0.0


def joint_train_dfd_v2(
    env_name,
    run_name,
    vec_env,
    max_steps,
    replay_buffer,
    world_model,
    agent,
    gd_critic,
    dual_policy,
    dfd_cfg,
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
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or f"ckpt/{run_name}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(colorama.Fore.CYAN + f"Saving DFD v2 checkpoints to {checkpoint_dir}" + colorama.Style.RESET_ALL)

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

    replay_cfg = _node(dfd_cfg, "Replay")
    cost_cfg = _node(dfd_cfg, "ContinuousCost")
    gd_cfg = _node(dfd_cfg, "Gd")
    dual_imag_cfg = _node(dfd_cfg, "DualImagination")
    dual_sampling_cfg = _node(dfd_cfg, "DualSampling")

    world_model_max_dual_fraction = _cfg_float(replay_cfg, "world_model_max_dual_fraction", 0.10)
    cost_positive_ratio = _cfg_float(replay_cfg, "cost_positive_ratio", 0.0)
    bottom_channels = tuple(int(v) for v in cfg_get(cost_cfg, "BottomForceChannels", [2, 5]))
    gd_update_steps = max(_cfg_int(gd_cfg, "UpdateSteps", 1), 1)
    dual_update_steps = max(_cfg_int(dual_imag_cfg, "UpdateSteps", 1), 1)

    dual_enabled = _cfg_bool(dual_sampling_cfg, "Enable", True)
    dual_start_step = _cfg_int(dual_sampling_cfg, "StartStep", 120000)
    dual_ratio_start = _cfg_float(dual_sampling_cfg, "RatioStart", 0.01)
    dual_ratio_final = _cfg_float(dual_sampling_cfg, "RatioFinal", 0.03)
    dual_ratio_warmup_steps = _cfg_int(dual_sampling_cfg, "RatioWarmupSteps", 100000)
    require_kl_healthy = _cfg_bool(dual_sampling_cfg, "RequireKLHealthy", True)
    dual_max_kl_for_sampling = _cfg_float(dual_imag_cfg, "MaxKLForSampling", 2.0)

    model_update_count = 0
    agent_update_count = 0
    gd_update_count = 0
    dual_update_count = 0
    last_dual_kl = 0.0

    episode_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_safe_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_cost = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_bottom_force = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_bottom_force_peak = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_len = torch.zeros(num_envs, dtype=torch.float32, device=device)

    if offline_dataset_dir:
        offline_episode_writer = OfflineEpisodeWriter(offline_dataset_dir, num_envs)
        print(colorama.Fore.CYAN + f"Saving offline episodes to {offline_episode_writer.output_dir}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()
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

    logger.log(f"Rollout/IsaacLab/{env_name}_reward", 0, 0)
    logger.log("Rollout/buffer_length", 0, 0)
    total_iters = max_steps // num_envs
    train_model_every_iters = max(train_model_every_steps // num_envs, 1)
    train_agent_every_iters = max(train_agent_every_steps // num_envs, 1)
    save_every_iters = max(save_every_steps // num_envs, 1)

    for iter_idx in tqdm(range(total_iters)):
        env_steps = iter_idx * num_envs

        if replay_buffer.ready():
            with torch.no_grad():
                world_model.eval()
                agent.eval()
                feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                dual_healthy = (not require_kl_healthy) or abs(float(last_dual_kl)) <= dual_max_kl_for_sampling
                dual_ratio = 0.0
                if dual_enabled and env_steps >= dual_start_step and dual_healthy:
                    dual_ratio = linear_warmup(
                        env_steps - dual_start_step,
                        dual_ratio_start,
                        dual_ratio_final,
                        dual_ratio_warmup_steps,
                    )
                env_action, action, source, state = _sample_policy_action(
                    feat=feat,
                    agent=agent,
                    dual_policy=dual_policy,
                    world_model=world_model,
                    state=state,
                    use_dual_sampling=dual_ratio > 0.0,
                    dual_ratio=dual_ratio,
                    num_envs=num_envs,
                    device=device,
                )
                logger.log("DualSampling/ratio", dual_ratio, env_steps)
                logger.log("DualSampling/healthy", float(dual_healthy), env_steps)
                logger.log("DualSampling/kl_to_main", float(last_dual_kl), env_steps)
        else:
            sampled = vec_env.action_space.sample()
            env_action = np.asarray(sampled, dtype=np.float32)
            action = torch.as_tensor(env_action, dtype=torch.float32, device=device)
            source = torch.full((num_envs, 1), SOURCE_RANDOM, dtype=torch.int64, device=device)

        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        _log_info_dict(logger, info, env_steps)

        cost_parts = extract_continuous_cost(
            info,
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            force_threshold=_cfg_float(cost_cfg, "ForceThreshold", 0.1),
            low_force_scale=_cfg_float(cost_cfg, "LowForceScale", 0.05),
            cost_force_max=_cfg_float(cost_cfg, "CostForceMax", 15.0),
            extreme_force_threshold=_cfg_float(cost_cfg, "ExtremeForceThreshold", 5.0),
            force_scale=_cfg_float(cost_cfg, "ForceScale", 5.0),
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

        lambda_cost = _current_lambda_cost(dfd_cfg, env_steps)
        episode_reward += reward
        episode_safe_reward += reward - lambda_cost * continuous_cost.view(-1)
        episode_cost += continuous_cost.view(-1)
        episode_bottom_force += bottom_force.view(-1)
        episode_bottom_force_peak = torch.maximum(episode_bottom_force_peak, bottom_force.view(-1))
        episode_len += 1.0

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
                    logger.log("Main/safe_return", episode_safe_reward[idx].item(), env_steps)
                    logger.log("Main/episode_cost_mean", episode_cost[idx].item() / ep_len, env_steps)
                    logger.log("Main/bottom_force_mean", episode_bottom_force[idx].item() / ep_len, env_steps)
                    logger.log("Main/bottom_force_peak", episode_bottom_force_peak[idx].item(), env_steps)
                    logger.log("Main/success_rate", episode_successes / max(episodes_completed, 1), env_steps)
                    logger.log("Main/lambda_cost", lambda_cost, env_steps)
                episode_reward[idx] = 0.0
                episode_safe_reward[idx] = 0.0
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
                        max_dual_fraction=world_model_max_dual_fraction,
                    )
                    train_world_model_step_dfd_v2(batch, world_model, agent, logger, env_steps)
                    model_update_count += 1

            if (
                _cfg_bool(gd_cfg, "Enable", True)
                and iter_idx % train_agent_every_iters == 0
                and replay_buffer.can_sample(batch_length)
            ):
                for _ in range(gd_update_steps):
                    batch = replay_buffer.sample(
                        batch_size,
                        batch_length,
                        return_dict=True,
                        cost_positive_ratio=cost_positive_ratio,
                    )
                    gd_critic.update(batch, world_model, dual_policy, logger=logger, step=env_steps)
                    gd_update_count += 1

            if (
                _cfg_bool(dual_imag_cfg, "Enable", True)
                and env_steps >= _cfg_int(dual_imag_cfg, "StartStep", 100000)
                and iter_idx % train_agent_every_iters == 0
                and replay_buffer.can_sample(batch_length)
            ):
                for _ in range(dual_update_steps):
                    batch = replay_buffer.sample(batch_size, batch_length, return_dict=True)
                    info_dual = update_dual_in_imagination(
                        batch,
                        world_model,
                        agent,
                        gd_critic,
                        dual_policy,
                        dual_imag_cfg,
                        logger=logger,
                        step=env_steps,
                    )
                    last_dual_kl = abs(float(info_dual.get("kl_to_main", 0.0))) if info_dual else last_dual_kl
                    dual_update_count += 1

            if iter_idx % train_agent_every_iters == 0 and replay_buffer.can_sample(imagine_context):
                for _ in range(agent_update):
                    imagine_samples = replay_buffer.sample(imagine_batch_size, imagine_context)
                    train_agent_step_dfd_v2(
                        imagine_samples,
                        world_model,
                        agent,
                        imagine_horizon,
                        logger,
                        env_steps,
                        dfd_cfg=dfd_cfg,
                    )
                    agent_update_count += 1

            collected_steps = env_steps + num_envs
            logger.log("Train/model_updates", model_update_count, env_steps)
            logger.log("Train/agent_updates", agent_update_count, env_steps)
            logger.log("Train/gd_updates", gd_update_count, env_steps)
            logger.log("Train/dual_imagination_updates", dual_update_count, env_steps)
            logger.log("Train/model_update_ratio", model_update_count / collected_steps, env_steps)
            logger.log("Train/agent_update_ratio", agent_update_count / collected_steps, env_steps)
            _log_replay_stats(replay_buffer, logger, env_steps)

        if iter_idx % save_every_iters == 0:
            print(colorama.Fore.GREEN + f"Saving DFD v2 model at total steps {env_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_v2_{env_steps}.pth"))
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_v2_{env_steps}.pth"))
            torch.save(gd_critic.state_dict(), os.path.join(checkpoint_dir, f"gd_v2_{env_steps}.pth"))
            torch.save(dual_policy.state_dict(), os.path.join(checkpoint_dir, f"dual_policy_v2_{env_steps}.pth"))

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
