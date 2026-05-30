from __future__ import annotations

import os
from functools import partial

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
        train_world_model_step,
    )
except ImportError:
    from trainer import (
        OfflineEpisodeWriter,
        _extract_force_obs,
        _is_first,
        _log_info_dict,
        _policy_obs,
        _reset_after_step,
        train_world_model_step,
    )

from .utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    SOURCE_RANDOM,
    cfg_get,
    extract_bottom_force_cost,
    linear_warmup,
    posterior_features,
    risk_advantage_modifier,
)


def _cfg_float(node, name, default):
    return float(cfg_get(node, name, default))


def _cfg_int(node, name, default):
    return int(cfg_get(node, name, default))


def _cfg_bool(node, name, default=False):
    return bool(cfg_get(node, name, default))


def train_feasibility_step(samples, world_model, agent, feasibility, dual_policy, logger, step):
    return feasibility.update(samples, world_model, agent, dual_policy, logger=logger, step=step)


def train_dual_policy_step(samples, world_model, agent, feasibility, dual_policy, target_kl, logger, step):
    world_model.eval()
    with torch.no_grad():
        feat = posterior_features(
            world_model,
            samples["obs"].to(world_model.device),
            samples["action"].to(world_model.device),
            samples["is_first"].to(world_model.device),
        )
    return dual_policy.update(feat, agent, feasibility, target_kl=target_kl, logger=logger, step=step)


def train_agent_step_dfd(
    samples,
    world_model,
    agent,
    imagine_horizon,
    logger,
    step,
    *,
    dfd_cfg=None,
    feasibility=None,
):
    world_model.eval()
    imagine_outputs = world_model.imagine_data(agent, *samples[:5], imagine_horizon, logger, step)

    modifier = None
    risk_cfg = cfg_get(dfd_cfg, "MainActorRisk", None)
    feas_cfg = cfg_get(dfd_cfg, "Feasibility", None)
    use_risk = _cfg_bool(dfd_cfg, "use_risk_conditioned_advantage", False)
    risk_start = _cfg_int(risk_cfg, "start_step", 150000)
    if use_risk and feasibility is not None and step >= risk_start:
        warmup_step = max(int(step) - risk_start, 0)
        lambda_cri = linear_warmup(
            warmup_step,
            _cfg_float(risk_cfg, "lambda_cri_start", 0.0),
            _cfg_float(risk_cfg, "lambda_cri_final", 0.02),
            _cfg_int(risk_cfg, "lambda_warmup_steps", 50000),
        )
        lambda_inf = linear_warmup(
            warmup_step,
            _cfg_float(risk_cfg, "lambda_inf_start", 0.0),
            _cfg_float(risk_cfg, "lambda_inf_final", 0.05),
            _cfg_int(risk_cfg, "lambda_warmup_steps", 50000),
        )
        modifier = partial(
            risk_advantage_modifier,
            feasibility=feasibility,
            pf=_cfg_float(feas_cfg, "pf", 0.10),
            cg=_cfg_float(feas_cfg, "cg", 0.03),
            lambda_cri=lambda_cri,
            lambda_inf=lambda_inf,
            clip_safe_adv=_cfg_bool(risk_cfg, "clip_safe_adv", True),
            safe_adv_min=_cfg_float(risk_cfg, "safe_adv_min", -5.0),
            safe_adv_max=_cfg_float(risk_cfg, "safe_adv_max", 5.0),
        )
        logger.log("ActorCritic/risk_lambda_cri", lambda_cri, step)
        logger.log("ActorCritic/risk_lambda_inf", lambda_inf, step)

    agent.update(*imagine_outputs, logger, step, advantage_modifier_fn=modifier)


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
        logger.log("Replay/cost_rate", cost_stats["cost_rate"], step)
        logger.log("Replay/main_cost_rate", cost_stats["main_cost_rate"], step)
        logger.log("Replay/dual_cost_rate", cost_stats["dual_cost_rate"], step)


def joint_train_dfd(
    env_name,
    run_name,
    vec_env,
    max_steps,
    replay_buffer,
    world_model,
    agent,
    feasibility,
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
    print(colorama.Fore.CYAN + f"Saving DFD checkpoints to {checkpoint_dir}" + colorama.Style.RESET_ALL)

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

    replay_cfg = cfg_get(dfd_cfg, "Replay", None)
    cost_cfg = cfg_get(dfd_cfg, "Cost", None)
    feas_cfg = cfg_get(dfd_cfg, "Feasibility", None)
    dual_cfg = cfg_get(dfd_cfg, "DualPolicy", None)

    use_dual = _cfg_bool(dfd_cfg, "use_dual", False)
    train_feasibility = _cfg_bool(dfd_cfg, "train_feasibility", False)
    use_dual_runtime = use_dual and train_feasibility and feasibility is not None and dual_policy is not None
    world_model_max_dual_fraction = _cfg_float(replay_cfg, "world_model_max_dual_fraction", 0.10)
    cost_threshold = _cfg_float(cost_cfg, "bottom_force_threshold", 1.0)
    bottom_channels = tuple(int(v) for v in cfg_get(cost_cfg, "bottom_force_channels", [2, 5]))
    dual_start_step = _cfg_int(dual_cfg, "start_step", 100000)
    dual_ratio_start = _cfg_float(dual_cfg, "ratio_start", 0.01)
    dual_ratio_final = _cfg_float(dual_cfg, "ratio_final", 0.02)
    dual_ratio_warmup_steps = _cfg_int(dual_cfg, "ratio_warmup_steps", 100000)
    dual_target_kl = _cfg_float(dual_cfg, "target_kl", 0.5)
    dual_max_kl_for_sampling = _cfg_float(dual_cfg, "max_kl_for_sampling", 2.0)
    feasibility_update = max(_cfg_int(feas_cfg, "update_steps", 1), 1)
    dual_update = max(_cfg_int(dual_cfg, "update_steps", 1), 1)

    model_update_count = 0
    agent_update_count = 0
    feasibility_update_count = 0
    dual_update_count = 0
    last_dual_kl = 0.0
    episode_cost = torch.zeros(num_envs, dtype=torch.float32, device=device)

    if offline_dataset_dir:
        offline_episode_writer = OfflineEpisodeWriter(offline_dataset_dir, num_envs)
        print(colorama.Fore.CYAN + f"Saving offline episodes to {offline_episode_writer.output_dir}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()
    if feasibility is not None:
        feasibility.eval()
    if dual_policy is not None:
        dual_policy.eval()
    state = world_model.initial(num_envs)
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(device)
    is_first = _is_first(current_obs_dict, num_envs, device)
    sum_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
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
                dual_healthy = abs(float(last_dual_kl)) <= dual_max_kl_for_sampling
                dual_ratio = 0.0
                if use_dual_runtime and env_steps >= dual_start_step and dual_healthy:
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
                logger.log("Dual/sampling_ratio", dual_ratio, env_steps)
                logger.log("Dual/sampling_healthy", float(dual_healthy), env_steps)
        else:
            sampled = vec_env.action_space.sample()
            env_action = np.asarray(sampled, dtype=np.float32)
            action = torch.as_tensor(env_action, dtype=torch.float32, device=device)
            source = torch.full((num_envs, 1), SOURCE_RANDOM, dtype=torch.int64, device=device)

        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        _log_info_dict(logger, info, env_steps)

        cost = extract_bottom_force_cost(
            info,
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            threshold=cost_threshold,
            force_key=getattr(replay_buffer, "force_key", ""),
            bottom_force_channels=bottom_channels,
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

        replay_buffer.append(current_obs, action, reward, done, is_first, force=force, cost=cost, source=source)
        if offline_episode_writer is not None:
            offline_episode_writer.append_step(current_obs_dict, action, reward, done, is_first, env_steps + num_envs)
        sum_reward += reward
        episode_cost += cost.view(-1)

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
                if replay_buffer.ready():
                    logger.log(f"Rollout/IsaacLab/{env_name}_reward", sum_reward[idx].item(), env_steps)
                    logger.log("Rollout/episode_cost", episode_cost[idx].item(), env_steps)
                    logger.log("Rollout/buffer_length", len(replay_buffer), env_steps)
                sum_reward[idx] = 0.0
                episode_cost[idx] = 0.0

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
                    samples = replay_buffer.sample(
                        batch_size,
                        batch_length,
                        max_dual_fraction=world_model_max_dual_fraction,
                    )
                    train_world_model_step(samples, world_model, agent, logger, env_steps)
                    model_update_count += 1

            if (
                train_feasibility
                and iter_idx % train_agent_every_iters == 0
                and replay_buffer.can_sample(batch_length)
            ):
                for _ in range(feasibility_update):
                    batch = replay_buffer.sample(batch_size, batch_length, return_dict=True)
                    train_feasibility_step(batch, world_model, agent, feasibility, dual_policy, logger, env_steps)
                    feasibility_update_count += 1

            if (
                use_dual_runtime
                and iter_idx % train_agent_every_iters == 0
                and replay_buffer.can_sample(batch_length)
            ):
                for _ in range(dual_update):
                    batch = replay_buffer.sample(batch_size, batch_length, return_dict=True)
                    info_dual = train_dual_policy_step(
                        batch,
                        world_model,
                        agent,
                        feasibility,
                        dual_policy,
                        dual_target_kl,
                        logger,
                        env_steps,
                    )
                    last_dual_kl = max(
                        abs(float(info_dual.get("kl_dual_main", 0.0))),
                        abs(float(info_dual.get("kl_main_dual", 0.0))),
                    )
                    dual_update_count += 1

            if iter_idx % train_agent_every_iters == 0 and replay_buffer.can_sample(imagine_context):
                for _ in range(agent_update):
                    imagine_samples = replay_buffer.sample(imagine_batch_size, imagine_context)
                    train_agent_step_dfd(
                        imagine_samples,
                        world_model,
                        agent,
                        imagine_horizon,
                        logger,
                        env_steps,
                        dfd_cfg=dfd_cfg,
                        feasibility=feasibility,
                    )
                    agent_update_count += 1

            collected_steps = env_steps + num_envs
            logger.log("Train/model_updates", model_update_count, env_steps)
            logger.log("Train/agent_updates", agent_update_count, env_steps)
            logger.log("Train/feasibility_updates", feasibility_update_count, env_steps)
            logger.log("Train/dual_updates", dual_update_count, env_steps)
            logger.log("Train/model_update_ratio", model_update_count / collected_steps, env_steps)
            logger.log("Train/agent_update_ratio", agent_update_count / collected_steps, env_steps)
            _log_replay_stats(replay_buffer, logger, env_steps)

        if iter_idx % save_every_iters == 0:
            print(colorama.Fore.GREEN + f"Saving DFD model at total steps {env_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_{env_steps}.pth"))
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_{env_steps}.pth"))
            if feasibility is not None:
                torch.save(feasibility.state_dict(), os.path.join(checkpoint_dir, f"feasibility_{env_steps}.pth"))
            if dual_policy is not None:
                torch.save(dual_policy.state_dict(), os.path.join(checkpoint_dir, f"dual_policy_{env_steps}.pth"))

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
