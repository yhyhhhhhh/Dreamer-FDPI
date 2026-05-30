from __future__ import annotations

"""
Verify that a pretrained PaMoRL/Dreamer world model, expert actor, and warmup
critic checkpoint can continue normal Dreamer online training.

Example:
  source /home/yhy/anaconda3/etc/profile.d/conda.sh
  conda activate isaaclab_14

  TERM=xterm WANDB_MODE=offline \
  PYTHONPATH=/home/yhy/surgical_robot_pro1/exts/ur3_lite:/home/yhy/PaMoRL-main:$PYTHONPATH \
  /home/yhy/IsaacLab-1.4.0/isaaclab.sh -p \
    /home/yhy/PaMoRL-main/pwm_isaaclab/verify_dreamer_continue.py \
    -n dreamer-continue-verify \
    -seed 0 \
    -config_path /home/yhy/PaMoRL-main/pwm_isaaclab/config_files/PWM_dreamer_continue_verify.yaml \
    -env_name Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1 \
    -device cuda:0 \
    -checkpoint_path /home/yhy/PaMoRL-main/ckpt/ur3-critic-warmup/20260524_223011_warmup/full_agent_after_critic_warmup.pt \
    --max_steps 5000000 \
    --buffer_warmup 4096 \
    --no_run_info_prompt
"""

import argparse
import json
import math
import os
import sys
import traceback
import warnings
from collections import deque
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import colorama
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import wandb

try:
    from pwm_isaaclab.env_wrapper import DreamerVecEnvWrapper
    from pwm_isaaclab.expert_config import cfg_to_dict, load_expert_config
    from pwm_isaaclab.expert_init import load_expert_checkpoint, save_expert_checkpoint
    from pwm_isaaclab.expert_pretrain import build_agent, build_world_model
    from pwm_isaaclab.expert_replay import SourceTaggedProprioReplayBuffer
    from pwm_isaaclab.trainer import _log_info_dict, train_agent_step, train_world_model_step
    from pwm_isaaclab.utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )
except ImportError:
    from env_wrapper import DreamerVecEnvWrapper
    from expert_config import cfg_to_dict, load_expert_config
    from expert_init import load_expert_checkpoint, save_expert_checkpoint
    from expert_pretrain import build_agent, build_world_model
    from expert_replay import SourceTaggedProprioReplayBuffer
    from trainer import _log_info_dict, train_agent_step, train_world_model_step
    from utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )


RUN_EXAMPLE = """TERM=xterm WANDB_MODE=offline \\
PYTHONPATH=/home/yhy/surgical_robot_pro1/exts/ur3_lite:/home/yhy/PaMoRL-main:$PYTHONPATH \\
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p \\
  /home/yhy/PaMoRL-main/pwm_isaaclab/verify_dreamer_continue.py \\
  -n dreamer-continue-verify \\
  -seed 0 \\
  -config_path /home/yhy/PaMoRL-main/pwm_isaaclab/config_files/PWM_dreamer_continue_verify.yaml \\
  -env_name Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1 \\
  -device cuda:0 \\
  -checkpoint_path /home/yhy/PaMoRL-main/ckpt/ur3-critic-warmup/20260524_223011_warmup/full_agent_after_critic_warmup.pt \\
  --max_steps 50000 \\
  --buffer_warmup 4096 \\
  --no_run_info_prompt"""


FORCE_OBS_CANDIDATE_KEYS = (
    "force",
    "pipe_force_curr",
    "pipe_force",
    "contact_force",
    "pipe_contact_force",
    "ft_force",
)


def _cfg_to_dict(node):
    if hasattr(node, "items"):
        return {key: _cfg_to_dict(value) for key, value in node.items()}
    return node


def _defrost_set(conf, path, value):
    if value is None:
        return
    target = conf
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def apply_cli_overrides(conf, args):
    conf.defrost()
    _defrost_set(conf, "JointTrainAgent.SampleMaxSteps", args.max_steps)
    _defrost_set(conf, "JointTrainAgent.BufferWarmUp", args.buffer_warmup)
    _defrost_set(conf, "JointTrainAgent.ModelUpdate", args.model_update)
    _defrost_set(conf, "JointTrainAgent.AgentUpdate", args.agent_update)
    _defrost_set(conf, "JointTrainAgent.TrainModelEverySteps", args.train_model_every_steps)
    _defrost_set(conf, "JointTrainAgent.TrainAgentEverySteps", args.train_agent_every_steps)
    _defrost_set(conf, "JointTrainAgent.SaveEverySteps", args.save_every_steps)
    _defrost_set(conf, "JointTrainAgent.VideoLogStep", args.log_every_steps)
    conf.expert.enabled = False
    conf.expert.pretrain_world_model = False
    conf.expert.bc_init = False
    conf.freeze()
    return conf


def build_env(args, conf):
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    import ur3_lite.tasks  # noqa: F401

    make_kwargs = {}
    if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
        make_kwargs = _cfg_to_dict(conf.Env.MakeKwargs)
    num_envs = int(make_kwargs.get("num_envs", conf.JointTrainAgent.NumEnvs))
    use_fabric = bool(make_kwargs.get("use_fabric", True))
    env_seed = int(make_kwargs.get("seed", args.seed))
    env_cfg = parse_env_cfg(args.env_name, device=args.device, num_envs=num_envs, use_fabric=use_fabric)
    env_cfg.seed = env_seed
    env = gymnasium.make(args.env_name, cfg=env_cfg)
    return DreamerVecEnvWrapper(env, device=args.device), simulation_app


def _policy_obs(obs_dict):
    return torch.as_tensor(obs_dict["policy"], dtype=torch.float32)


def _is_first(obs_dict, num_envs, device):
    value = obs_dict.get("is_first")
    if value is None:
        return torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
    return torch.as_tensor(value, dtype=torch.float32, device=device).view(num_envs, 1)


def _extract_force_obs(obs_dict, num_envs, device, force_key=""):
    candidate_keys = (force_key,) if force_key else FORCE_OBS_CANDIDATE_KEYS
    for key in candidate_keys:
        if not key or key not in obs_dict:
            continue
        force = torch.as_tensor(obs_dict[key], dtype=torch.float32, device=device)
        if force.shape[0] != num_envs:
            raise ValueError(f"Force observation `{key}` has first dimension {force.shape[0]}, expected {num_envs}.")
        force = force.reshape(num_envs, -1)
        return force if force.shape[1] == 1 else torch.linalg.norm(force, dim=-1, keepdim=True)
    available = ", ".join(sorted(obs_dict.keys()))
    tried = ", ".join(key for key in candidate_keys if key)
    raise KeyError(f"ForceHead.Enable=True but no force observation was found. Tried [{tried}], available [{available}].")


def infer_checkpoint_dims(path):
    checkpoint = torch.load(path, map_location="cpu")
    metadata = checkpoint.get("expert_metadata") or {}
    obs_dim = metadata.get("obs_dim")
    action_dim = metadata.get("action_dim")
    world_state = checkpoint.get("world_model_state_dict") or {}
    agent_state = checkpoint.get("agent_state_dict") or {}
    encoder_weight = world_state.get("encoder.backbone.0.layer.0.weight")
    if obs_dim is None and torch.is_tensor(encoder_weight) and encoder_weight.ndim == 2:
        obs_dim = int(encoder_weight.shape[1])
    actor_head = agent_state.get("actor.head.weight")
    if action_dim is None and torch.is_tensor(actor_head) and actor_head.ndim == 2:
        action_dim = int(actor_head.shape[0] // 2)
    return {
        "obs_dim": int(obs_dim) if obs_dim is not None else None,
        "action_dim": int(action_dim) if action_dim is not None else None,
        "expert_metadata": metadata,
        "checkpoint_keys": sorted(checkpoint.keys()),
    }


@torch.no_grad()
def parameter_summary(module):
    total_params = 0
    finite = True
    sq_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    checksum = 0.0
    sample_count = 0
    for param in module.parameters():
        data = param.detach().float()
        total_params += data.numel()
        finite = finite and bool(torch.isfinite(data).all().item())
        sq_sum += float(data.square().sum().cpu().item())
        abs_sum += float(data.abs().sum().cpu().item())
        max_abs = max(max_abs, float(data.abs().max().cpu().item()) if data.numel() else 0.0)
        flat = data.flatten()
        take = min(128, flat.numel())
        if take > 0:
            weights = torch.linspace(1.0, 2.0, take, device=flat.device)
            checksum += float((flat[:take] * weights).sum().cpu().item())
            sample_count += take
    return {
        "num_params": int(total_params),
        "finite": bool(finite),
        "l2_norm": math.sqrt(max(sq_sum, 0.0)),
        "mean_abs": abs_sum / max(total_params, 1),
        "max_abs": max_abs,
        "checksum": checksum,
        "checksum_samples": int(sample_count),
    }


def summary_delta(before, after, *, atol=1e-10):
    return {
        "l2_norm_delta": float(after["l2_norm"] - before["l2_norm"]),
        "mean_abs_delta": float(after["mean_abs"] - before["mean_abs"]),
        "max_abs_delta": float(after["max_abs"] - before["max_abs"]),
        "checksum_delta": float(after["checksum"] - before["checksum"]),
        "changed": bool(abs(after["checksum"] - before["checksum"]) > atol or abs(after["l2_norm"] - before["l2_norm"]) > atol),
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False, sort_keys=True)


def write_error_report(checkpoint_dir, exc):
    traceback_text = traceback.format_exc()
    print(colorama.Fore.RED + "Dreamer continue verification failed with traceback:" + colorama.Style.RESET_ALL, file=sys.stderr)
    print(traceback_text, file=sys.stderr, flush=True)
    error_path = os.path.join(checkpoint_dir, "continue_verify_error.json")
    write_json(
        error_path,
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback_text,
        },
    )
    print(colorama.Fore.RED + f"Error report saved: {error_path}" + colorama.Style.RESET_ALL, file=sys.stderr, flush=True)


def report_cleanup_error(label):
    print(
        colorama.Fore.YELLOW
        + f"Cleanup step failed after training error or shutdown: {label}"
        + colorama.Style.RESET_ALL,
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)


def save_plain_state_checkpoints(checkpoint_dir, world_model, agent, env_steps):
    torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_{int(env_steps)}.pth"))
    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_{int(env_steps)}.pth"))


def verify_continue_training(args):
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("WANDB_MODE", "offline")

    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    conf = apply_cli_overrides(load_expert_config(args.config_path), args)
    seed_np_torch(args.seed)
    print(colorama.Fore.CYAN + "Dreamer continue verification run example:" + colorama.Style.RESET_ALL)
    print(RUN_EXAMPLE)

    run_info = collect_training_info(note=args.note, tags=args.tags, prompt=not args.no_run_info_prompt)
    checkpoint_dir = make_unique_run_dir(
        base_name=args.n,
        run_root=args.run_root,
        run_id=args.run_id,
        note=run_info.get("note"),
    )
    write_latest_run_pointer(checkpoint_dir)
    save_run_artifacts(
        run_dir=checkpoint_dir,
        conf=conf,
        config_path=args.config_path,
        args=args,
        run_info=run_info,
        extra={
            "base_run_name": args.n,
            "env_name": args.env_name,
            "seed": args.seed,
            "device": args.device,
            "checkpoint_path": checkpoint_path,
            "run_example": RUN_EXAMPLE,
        },
    )
    print(colorama.Fore.CYAN + f"Run directory: {checkpoint_dir}" + colorama.Style.RESET_ALL)

    wandb.init(
        project=getattr(conf.Wandb, "Project", "IsaacLab-PSSM"),
        group=getattr(conf.Wandb, "Group", args.env_name),
        name=f"{getattr(conf.Wandb, 'Name', args.n)}-{os.path.basename(checkpoint_dir)}",
        dir=checkpoint_dir,
        config=cfg_to_dict(conf),
        mode=getattr(conf.Wandb, "Mode", "offline"),
    )
    logger = Logger()

    simulation_app = None
    try:
        vec_env, simulation_app = build_env(args, conf)
        num_envs = int(vec_env.num_envs)
        obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
        action_dim = int(vec_env.single_action_space.shape[0])
        checkpoint_dims = infer_checkpoint_dims(checkpoint_path)
        if checkpoint_dims["obs_dim"] is not None and checkpoint_dims["obs_dim"] != obs_dim:
            raise ValueError(f"Environment obs_dim={obs_dim} does not match checkpoint obs_dim={checkpoint_dims['obs_dim']}.")
        if checkpoint_dims["action_dim"] is not None and checkpoint_dims["action_dim"] != action_dim:
            raise ValueError(
                f"Environment action_dim={action_dim} does not match checkpoint action_dim={checkpoint_dims['action_dim']}."
            )

        act = getattr(nn, conf.Models.Act)
        world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
        agent = build_agent(conf, action_dim, act, args.device)
        load_expert_checkpoint(checkpoint_path, world_model=world_model, agent=agent, map_location=args.device)
        if hasattr(agent, "sync_slow_critic"):
            agent.sync_slow_critic()

        replay = SourceTaggedProprioReplayBuffer(
            obs_dim,
            action_dim,
            num_envs,
            conf.JointTrainAgent.BufferMaxLength,
            conf.JointTrainAgent.BufferWarmUp,
            args.device,
            include_force=bool(conf.ForceHead.Enable),
            force_dim=1,
            force_key=conf.ForceHead.Key,
        )

        before = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_dims": checkpoint_dims,
            "env": {"obs_dim": obs_dim, "action_dim": action_dim, "num_envs": num_envs},
            "world_model": parameter_summary(world_model),
            "actor": parameter_summary(agent.actor),
            "critic": parameter_summary(agent.critic),
            "slow_critic": parameter_summary(agent.slow_critic),
        }
        write_json(os.path.join(checkpoint_dir, "continue_verify_before.json"), before)

        if args.init_only:
            metrics = {
                "initialized": True,
                "init_only": True,
                "run_dir": checkpoint_dir,
                "checkpoint_path": checkpoint_path,
                "env_obs_dim": obs_dim,
                "env_action_dim": action_dim,
                "checkpoint_obs_dim": checkpoint_dims["obs_dim"],
                "checkpoint_action_dim": checkpoint_dims["action_dim"],
            }
            write_json(os.path.join(checkpoint_dir, "continue_verify_metrics.json"), metrics)
            return metrics

        state = world_model.initial(num_envs)
        current_obs_dict = vec_env.reset()
        current_obs = _policy_obs(current_obs_dict).to(args.device)
        is_first = _is_first(current_obs_dict, num_envs, args.device)

        total_iters = max(int(conf.JointTrainAgent.SampleMaxSteps) // num_envs, 1)
        train_model_every_iters = max(int(conf.JointTrainAgent.TrainModelEverySteps) // num_envs, 1)
        train_agent_every_iters = max(int(conf.JointTrainAgent.TrainAgentEverySteps) // num_envs, 1)
        save_every_steps = max(int(conf.JointTrainAgent.SaveEverySteps), 1)
        next_save_step = save_every_steps
        log_every_steps = max(int(getattr(conf.JointTrainAgent, "VideoLogStep", 2000)), 1)
        next_log_step = log_every_steps
        batch_size = int(conf.JointTrainAgent.BatchSize)
        batch_length = int(conf.JointTrainAgent.BatchLength)
        imagine_batch_size = int(conf.JointTrainAgent.ImagineBatchSize or batch_size)
        imagine_context = int(conf.JointTrainAgent.ImagineContext or batch_length)
        imagine_horizon = int(conf.JointTrainAgent.ImagineHorizon)
        model_update = max(int(conf.JointTrainAgent.ModelUpdate), 1)
        agent_update = max(int(conf.JointTrainAgent.AgentUpdate), 1)
        param_check_interval = int(args.param_check_interval)

        model_update_count = 0
        agent_update_count = 0
        random_steps = 0
        policy_steps = 0
        reward_sum = 0.0
        cost_sum = 0.0
        episodes_completed = 0
        episode_successes = 0
        episode_failures = 0
        episode_timeouts = 0
        recent_returns = deque(maxlen=128)
        recent_episode_success = deque(maxlen=1024)
        episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        saw_nonfinite = False

        logger.log(f"Rollout/IsaacLab/{args.env_name}_reward", 0, 0)
        logger.log("Rollout/buffer_length", 0, 0)

        for iter_idx in range(total_iters):
            env_steps = iter_idx * num_envs
            if replay.ready():
                with torch.no_grad():
                    world_model.eval()
                    agent.eval()
                    feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                    env_action, action = agent.sample_as_env_action(feat, greedy=False)
                    state = world_model.update_inference_state(state, action)
                    policy_steps += num_envs
            else:
                sampled = vec_env.action_space.sample()
                env_action = np.asarray(sampled, dtype=np.float32)
                action = torch.as_tensor(env_action, dtype=torch.float32, device=args.device)
                random_steps += num_envs

            next_obs_dict, reward, done, info = vec_env.step(env_action)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=args.device)
            done = torch.as_tensor(done, dtype=torch.bool, device=args.device)
            _log_info_dict(logger, info, env_steps)

            terminal = torch.as_tensor(
                next_obs_dict.get("is_terminal", torch.zeros_like(done, dtype=torch.int32)),
                dtype=torch.bool,
                device=args.device,
            ).view(-1)
            failure = torch.as_tensor(
                next_obs_dict.get("failure", torch.zeros_like(done, dtype=torch.int32)),
                dtype=torch.bool,
                device=args.device,
            ).view(-1)
            episode_success = info.get("episode_success")
            if episode_success is None:
                episode_success = terminal & ~failure
            else:
                episode_success = torch.as_tensor(episode_success, dtype=torch.bool, device=args.device).view(-1)
            episode_failure = info.get("episode_failure")
            if episode_failure is None:
                episode_failure = terminal & failure
            else:
                episode_failure = torch.as_tensor(episode_failure, dtype=torch.bool, device=args.device).view(-1)
            episode_timeout = info.get("episode_timeout")
            if episode_timeout is None:
                episode_timeout = done & ~terminal
            else:
                episode_timeout = torch.as_tensor(episode_timeout, dtype=torch.bool, device=args.device).view(-1)

            force = None
            if getattr(replay, "include_force", False):
                force = _extract_force_obs(current_obs_dict, num_envs, args.device, getattr(replay, "force_key", ""))
            replay.append(current_obs, action, reward, done, is_first, force=force)

            reward_sum += float(reward.sum().detach().cpu().item())
            episode_returns += reward
            if done.any():
                done_ids = torch.nonzero(done, as_tuple=False).flatten()
                completed_now = int(done_ids.numel())
                success_now = int(episode_success[done_ids].sum().item())
                failure_now = int(episode_failure[done_ids].sum().item())
                timeout_now = int(episode_timeout[done_ids].sum().item())

                episodes_completed += completed_now
                episode_successes += success_now
                episode_failures += failure_now
                episode_timeouts += timeout_now
                recent_returns.extend(episode_returns[done_ids].detach().cpu().tolist())
                recent_episode_success.extend(episode_success[done_ids].float().cpu().tolist())

                for idx in done_ids.tolist():
                    if replay.ready():
                        logger.log(f"Rollout/IsaacLab/{args.env_name}_reward", episode_returns[idx].item(), env_steps)
                        logger.log("Rollout/buffer_length", len(replay), env_steps)
                episode_returns[done_ids] = 0.0

                logger.log("Rollout/episodes_completed", episodes_completed, env_steps)
                logger.log("Rollout/episode_successes", episode_successes, env_steps)
                logger.log("Rollout/episode_failures", episode_failures, env_steps)
                logger.log("Rollout/episode_timeouts", episode_timeouts, env_steps)
                logger.log("Rollout/episode_success_rate", episode_successes / max(episodes_completed, 1), env_steps)
                logger.log("Rollout/episode_failure_rate", episode_failures / max(episodes_completed, 1), env_steps)
                logger.log("Rollout/episode_timeout_rate", episode_timeouts / max(episodes_completed, 1), env_steps)
                logger.log(
                    "Rollout/episode_success_rate_recent",
                    float(np.mean(recent_episode_success)) if recent_episode_success else 0.0,
                    env_steps,
                )

            current_obs_dict = vec_env.reset(seed=done.to(torch.int32))
            current_obs = _policy_obs(current_obs_dict).to(args.device)
            is_first = _is_first(current_obs_dict, num_envs, args.device)

            if replay.ready():
                if iter_idx % train_model_every_iters == 0 and replay.can_sample(batch_length):
                    for _ in range(model_update):
                        samples = replay.sample(batch_size, batch_length)
                        train_world_model_step(samples, world_model, agent, logger, env_steps)
                        model_update_count += 1
                if iter_idx % train_agent_every_iters == 0 and replay.can_sample(imagine_context):
                    for _ in range(agent_update):
                        samples = replay.sample(imagine_batch_size, imagine_context)
                        train_agent_step(samples, world_model, agent, imagine_horizon, logger, env_steps)
                        agent_update_count += 1

                collected_steps = env_steps + num_envs
                logger.log("Train/model_updates", model_update_count, env_steps)
                logger.log("Train/agent_updates", agent_update_count, env_steps)
                logger.log("Train/model_update_ratio", model_update_count / collected_steps, env_steps)
                logger.log("Train/agent_update_ratio", agent_update_count / collected_steps, env_steps)

            if param_check_interval > 0 and iter_idx % param_check_interval == 0:
                for summary_module in (world_model, agent.actor, agent.critic):
                    saw_nonfinite = saw_nonfinite or not parameter_summary(summary_module)["finite"]

            while env_steps + num_envs >= next_save_step:
                save_plain_state_checkpoints(checkpoint_dir, world_model, agent, next_save_step)
                next_save_step += save_every_steps
            while env_steps + num_envs >= next_log_step:
                recent_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
                print(
                    colorama.Fore.CYAN
                    + (
                        f"[continue-train] step={next_log_step} replay={len(replay)} "
                        f"model_updates={model_update_count} agent_updates={agent_update_count} "
                        f"episodes={episodes_completed} recent_return_mean={recent_mean:.3f} "
                        f"random_steps={random_steps} policy_steps={policy_steps}"
                    )
                    + colorama.Style.RESET_ALL,
                    flush=True,
                )
                next_log_step += log_every_steps

        final_step = total_iters * num_envs
        after = {
            "world_model": parameter_summary(world_model),
            "actor": parameter_summary(agent.actor),
            "critic": parameter_summary(agent.critic),
            "slow_critic": parameter_summary(agent.slow_critic),
        }
        write_json(os.path.join(checkpoint_dir, "continue_verify_after.json"), after)

        deltas = {
            "world_model": summary_delta(before["world_model"], after["world_model"]),
            "actor": summary_delta(before["actor"], after["actor"]),
            "critic": summary_delta(before["critic"], after["critic"]),
            "slow_critic": summary_delta(before["slow_critic"], after["slow_critic"]),
        }
        metrics = {
            "initialized": True,
            "init_only": False,
            "run_dir": checkpoint_dir,
            "checkpoint_path": checkpoint_path,
            "final_step": int(final_step),
            "model_update_count": int(model_update_count),
            "agent_update_count": int(agent_update_count),
            "random_steps": int(random_steps),
            "policy_steps": int(policy_steps),
            "replay_length": int(len(replay)),
            "episodes_completed": int(episodes_completed),
            "reward_sum": reward_sum,
            "recent_return_mean": float(np.mean(recent_returns)) if recent_returns else None,
            "cost_sum": cost_sum,
            "saw_nonfinite": bool(saw_nonfinite),
            "world_model_changed": deltas["world_model"]["changed"],
            "actor_changed": deltas["actor"]["changed"],
            "critic_changed": deltas["critic"]["changed"],
            "slow_critic_changed": deltas["slow_critic"]["changed"],
            "deltas": deltas,
            "passed": bool(
                model_update_count > 0
                and agent_update_count > 0
                and deltas["actor"]["changed"]
                and deltas["critic"]["changed"]
                and not saw_nonfinite
            ),
        }
        metrics_path = os.path.join(checkpoint_dir, "continue_verify_metrics.json")
        write_json(metrics_path, metrics)
        final_checkpoint = os.path.join(checkpoint_dir, "full_agent_continue_verify_final.pt")
        save_expert_checkpoint(
            final_checkpoint,
            world_model=world_model,
            agent=agent,
            config=cfg_to_dict(conf),
            expert_metadata=checkpoint_dims.get("expert_metadata", {}),
            extra={
                "continue_verify": metrics,
                "source_checkpoint_path": checkpoint_path,
                "run_example": RUN_EXAMPLE,
            },
        )
        metrics["final_checkpoint"] = final_checkpoint
        write_json(metrics_path, metrics)
        print(colorama.Fore.GREEN + f"Dreamer continue verification metrics: {metrics_path}" + colorama.Style.RESET_ALL)
        print(colorama.Fore.GREEN + f"Dreamer continue verification checkpoint: {final_checkpoint}" + colorama.Style.RESET_ALL)
        return metrics
    except Exception as exc:
        write_error_report(checkpoint_dir, exc)
        raise
    finally:
        try:
            if logger.log_dict:
                wandb.log(logger.log_dict, step=logger.tot_step)
        except Exception:
            report_cleanup_error("wandb.log")
        try:
            wandb.finish()
        except Exception:
            report_cleanup_error("wandb.finish")
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            report_cleanup_error("simulation_app.close")


def build_parser():
    parser = argparse.ArgumentParser(description="Verify continued Dreamer training from a warmup full checkpoint.")
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-checkpoint_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--buffer_warmup", type=int, default=None)
    parser.add_argument("--model_update", type=int, default=None)
    parser.add_argument("--agent_update", type=int, default=None)
    parser.add_argument("--train_model_every_steps", type=int, default=None)
    parser.add_argument("--train_agent_every_steps", type=int, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=None)
    parser.add_argument("--run_root", type=str, default="ckpt")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--tags", type=str, default="dreamer-continue-verify")
    parser.add_argument(
        "--param_check_interval",
        type=int,
        default=1,
        help="Check all model parameters for finite values every N environment iterations. Use 0 to check only at final summary.",
    )
    parser.add_argument("--init_only", action="store_true")
    parser.add_argument("--no_run_info_prompt", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    metrics = verify_continue_training(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    return metrics


if __name__ == "__main__":
    main()
