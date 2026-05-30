from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import colorama
import torch
import torch.nn as nn
import wandb

try:
    from pwm_isaaclab.agents import ActorCriticAgent
    from pwm_isaaclab.expert_config import cfg_to_dict, load_expert_config
    from pwm_isaaclab.expert_init import (
        evaluate_actor_bc_on_expert,
        evaluate_world_model_on_expert,
        load_expert_checkpoint,
        pretrain_actor_bc_from_expert,
        pretrain_world_model_from_expert,
        save_expert_checkpoint,
    )
    from pwm_isaaclab.expert_loader import load_expert_dataset
    from pwm_isaaclab.expert_replay import make_expert_replay
    from pwm_isaaclab.expert_world_model import ExpertWorldModelWithCost
    from pwm_isaaclab.utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )
except ImportError:
    from agents import ActorCriticAgent
    from expert_config import cfg_to_dict, load_expert_config
    from expert_init import (
        evaluate_actor_bc_on_expert,
        evaluate_world_model_on_expert,
        load_expert_checkpoint,
        pretrain_actor_bc_from_expert,
        pretrain_world_model_from_expert,
        save_expert_checkpoint,
    )
    from expert_loader import load_expert_dataset
    from expert_replay import make_expert_replay
    from expert_world_model import ExpertWorldModelWithCost
    from utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )


def _resolve_force_scale(value):
    if isinstance(value, str):
        if value.lower() == "auto":
            print(
                colorama.Fore.YELLOW
                + "ForceLoss.ForceScale='auto' is not estimated in expert_pretrain.py; using 1.0."
                + colorama.Style.RESET_ALL
            )
            return 1.0
        return float(value)
    return float(value)


def build_world_model(conf, obs_dim, action_dim, act, device):
    return ExpertWorldModelWithCost(
        conf.JointTrainAgent.VideoLogStep,
        True,
        obs_dim,
        action_dim,
        conf.Models.Stoch,
        conf.Models.Discrete,
        conf.Models.Hidden,
        conf.Models.WorldModel.Stem,
        conf.Models.WorldModel.MinRes,
        conf.Models.NumBin,
        conf.Models.MaxBin,
        conf.Models.WorldModel.DynScale,
        conf.Models.WorldModel.RepScale,
        conf.Models.WorldModel.ValScale,
        conf.Models.WorldModel.KLFree,
        conf.Models.Gamma,
        conf.Models.Lambda,
        conf.Models.Tau,
        conf.Models.WorldModel.LR,
        conf.Models.WorldModel.Eps,
        conf.BasicSettings.UseAmp,
        act,
        device,
        bool(getattr(conf.ForceHead, "Enable", False)),
        conf.ForceHead.HiddenDim,
        conf.ForceHead.Depth,
        conf.ForceHead.Dropout,
        conf.ForceLoss.Eps,
        _resolve_force_scale(conf.ForceLoss.ForceScale),
        conf.ForceHead.Threshold,
        conf.ForceHead.LossWeight,
        conf.ForceHead.DetachLatent,
        conf.ForceLoss.LambdaCls,
        conf.ForceLoss.LambdaReg,
        conf.ForceLoss.LambdaSign,
        conf.ForceLoss.FocalAlpha,
        conf.ForceLoss.FocalGamma,
        conf.ForceLoss.HuberBeta,
        conf.ForceLoss.RegWeightPower,
        conf.ForceLoss.RegWeightMax,
        conf.ForceHead.SignedForce,
        conf.expert.cost_loss_type,
        conf.expert.cost_loss_weight,
        conf.expert.cost_head_mode,
        conf.expert.cost_cls_weight,
        conf.expert.cost_reg_weight,
        conf.expert.cost_prior_loss_weight,
        conf.expert.cost_focal_alpha,
        conf.expert.cost_focal_gamma,
        conf.expert.cost_pos_weight_max,
        conf.expert.cost_huber_beta,
    ).to(device)


def build_agent(conf, action_dim, act, device):
    return ActorCriticAgent(
        action_dim,
        conf.Models.Stoch * conf.Models.Discrete + conf.Models.Hidden,
        conf.Models.Hidden,
        conf.Models.Agent.EntropyCoef,
        conf.Models.NumBin,
        conf.Models.MaxBin,
        conf.Models.Agent.MinPer,
        conf.Models.Agent.MaxPer,
        conf.Models.Agent.MinStd,
        conf.Models.Agent.MaxStd,
        conf.Models.Agent.EMADecay,
        conf.Models.Gamma,
        conf.Models.Lambda,
        conf.Models.Tau,
        bool(getattr(conf.Models.Agent, "UseSlowCritic", False)),
        conf.Models.Agent.LR,
        conf.Models.Agent.Eps,
        conf.BasicSettings.UseAmp,
        act,
        device,
    ).to(device)


def _split_train_validation(episodes, validation_ratio):
    if not episodes:
        raise ValueError("Expert dataset is empty.")
    if len(episodes) == 1 or validation_ratio <= 0:
        return episodes, episodes
    num_val = max(1, int(round(len(episodes) * float(validation_ratio))))
    num_val = min(num_val, len(episodes) - 1)
    return episodes[:-num_val], episodes[-num_val:]


def _episode_steps(episodes):
    return int(sum(len(episode["reward"]) for episode in episodes))


def _as_path_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item]


def _cost_loader_kwargs(conf):
    return {
        "cost_target_source": conf.expert.cost_target_source,
        "cost_pipe_force_limit": conf.expert.cost_pipe_force_limit,
        "cost_bottom_force_limit": conf.expert.cost_bottom_force_limit,
        "cost_pipe_force_channels": conf.expert.cost_pipe_force_channels,
        "cost_bottom_force_channels": conf.expert.cost_bottom_force_channels,
    }


def _validate_compatible_episodes(episodes, obs_dim, action_dim, label):
    for idx, episode in enumerate(episodes):
        if int(episode["obs"].shape[-1]) != int(obs_dim):
            raise ValueError(
                f"{label} episode {idx} obs_dim={episode['obs'].shape[-1]} does not match expert obs_dim={obs_dim}."
            )
        if int(episode["action"].shape[-1]) != int(action_dim):
            raise ValueError(
                f"{label} episode {idx} action_dim={episode['action'].shape[-1]} "
                f"does not match expert action_dim={action_dim}."
            )


def _load_wm_coverage_datasets(conf, args, expert_metadata):
    paths = _as_path_list(conf.expert.wm_coverage_paths)
    paths.extend(_as_path_list(args.wm_coverage_path))
    if not paths:
        return [], [], []

    max_episodes = args.max_wm_coverage_episodes
    if max_episodes is None or max_episodes <= 0:
        max_episodes = int(getattr(conf.expert, "wm_coverage_max_episodes", 0) or 0)
    max_episodes = max_episodes if max_episodes > 0 else None

    train_episodes = []
    val_episodes = []
    metadata = []
    remaining = max_episodes
    for path in paths:
        load_limit = remaining if remaining is not None else None
        if load_limit is not None and load_limit <= 0:
            break
        dataset = load_expert_dataset(
            path,
            format=conf.expert.wm_coverage_format,
            action_tolerance=conf.expert.action_tolerance,
            max_episodes=load_limit,
            **_cost_loader_kwargs(conf),
        )
        _validate_compatible_episodes(
            dataset,
            expert_metadata["obs_dim"],
            expert_metadata["action_dim"],
            f"wm_coverage:{path}",
        )
        coverage_train, coverage_val = _split_train_validation(
            dataset,
            conf.expert.wm_coverage_validation_ratio,
        )
        train_episodes.extend(coverage_train)
        val_episodes.extend(coverage_val)
        metadata.append(dataset.metadata)
        if remaining is not None:
            remaining -= len(dataset)
    return train_episodes, val_episodes, metadata


def _log_metadata(logger, metadata):
    scalar_keys = (
        "num_episodes",
        "num_transitions",
        "mean_return",
        "mean_cost",
        "min_return",
        "max_return",
        "min_episode_length",
        "max_episode_length",
        "mean_episode_length",
        "action_min_overall",
        "action_max_overall",
        "cost_positive_count",
        "cost_positive_ratio",
        "derived_cost_mean",
        "derived_cost_max",
        "derived_positive_cost_mean",
        "original_cost_positive_count",
        "original_cost_positive_ratio",
        "original_cost_mean",
        "original_cost_max",
        "pipe_force_max",
        "bottom_force_max",
    )
    for key in scalar_keys:
        if key in metadata:
            logger.log(f"expert/{key}", metadata[key], 0)
    for idx, value in enumerate(metadata.get("action_min", [])):
        logger.log(f"expert/action_min_dim_{idx}", float(value), 0)
    for idx, value in enumerate(metadata.get("action_max", [])):
        logger.log(f"expert/action_max_dim_{idx}", float(value), 0)


def _print_metadata(metadata):
    print(
        colorama.Fore.CYAN
        + (
            f"Loaded expert dataset: {metadata['num_episodes']} episodes / "
            f"{metadata['num_transitions']} transitions, "
            f"mean_return={metadata['mean_return']:.4f}, mean_cost={metadata['mean_cost']:.4f}, "
            f"cost_pos_ratio={metadata.get('cost_positive_ratio', 0.0):.6f}, "
            f"action_range=[{metadata['action_min_overall']:.4f}, {metadata['action_max_overall']:.4f}]"
        )
        + colorama.Style.RESET_ALL
    )


def _print_coverage_metadata(metadata_list):
    for metadata in metadata_list:
        print(
            colorama.Fore.CYAN
            + (
                f"Loaded WM coverage dataset: {metadata['num_episodes']} episodes / "
                f"{metadata['num_transitions']} transitions from {metadata['dataset_path']}, "
                f"mean_return={metadata['mean_return']:.4f}, mean_cost={metadata['mean_cost']:.4f}, "
                f"cost_pos_ratio={metadata.get('cost_positive_ratio', 0.0):.6f}, "
                f"action_range=[{metadata['action_min_overall']:.4f}, {metadata['action_max_overall']:.4f}]"
            )
            + colorama.Style.RESET_ALL
        )


def _maybe_set_bc_lr(agent, bc_lr):
    if bc_lr is None:
        return None
    old_lrs = [group["lr"] for group in agent.optimizer.param_groups]
    for group in agent.optimizer.param_groups:
        group["lr"] = float(bc_lr)
    return old_lrs


def _restore_lrs(agent, old_lrs):
    if old_lrs is None:
        return
    for group, lr in zip(agent.optimizer.param_groups, old_lrs):
        group["lr"] = lr


def _load_state_if_exists(path, *, world_model=None, agent=None, device=None):
    if not path:
        return False
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    load_expert_checkpoint(path, world_model=world_model, agent=agent, map_location=device)
    print(colorama.Fore.YELLOW + f"Loaded expert checkpoint: {path}" + colorama.Style.RESET_ALL)
    return True


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument(
        "-config_path",
        type=str,
        default="pwm_isaaclab/config_files/PWM_expert_init.yaml",
    )
    parser.add_argument("-dataset_path", type=str, default=None)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-buffer_device", type=str, default=None)
    parser.add_argument("-max_episodes", type=int, default=0)
    parser.add_argument("-wm_coverage_path", action="append", default=None)
    parser.add_argument("-max_wm_coverage_episodes", type=int, default=0)
    parser.add_argument("--run_root", type=str, default="ckpt")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--no_run_info_prompt", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    conf = load_expert_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    if conf.Task != "JointTrainAgent":
        raise NotImplementedError(f"Task {conf.Task} not implemented for expert pretrain.")

    run_info = collect_training_info(
        note=args.note,
        tags=args.tags,
        prompt=not args.no_run_info_prompt,
    )
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
        extra={"base_run_name": args.n, "seed": args.seed, "device": args.device},
    )
    print(colorama.Fore.CYAN + f"Run directory: {checkpoint_dir}" + colorama.Style.RESET_ALL)

    seed_np_torch(seed=args.seed)
    project = getattr(conf.Wandb, "Project", "IsaacLab-PSSM")
    run_group = getattr(conf.Wandb, "Group", "expert-init")
    base_wandb_name = getattr(conf.Wandb, "Name", args.n)
    wandb_mode = "disabled" if args.no_wandb else getattr(conf.Wandb, "Mode", None)
    init_kwargs = {
        "project": project,
        "group": run_group,
        "name": f"{base_wandb_name}-{os.path.basename(checkpoint_dir)}",
        "dir": checkpoint_dir,
        "config": cfg_to_dict(conf),
    }
    if wandb_mode is not None:
        init_kwargs["mode"] = wandb_mode
    if run_info.get("note"):
        init_kwargs["notes"] = run_info["note"]
    if run_info.get("tags"):
        init_kwargs["tags"] = run_info["tags"]
    wandb.init(**init_kwargs)
    logger = Logger()

    dataset_path = args.dataset_path or conf.expert.path
    if not dataset_path:
        raise ValueError("Expert dataset path is required via -dataset_path or expert.path.")
    max_episodes = args.max_episodes if args.max_episodes > 0 else None
    episodes = load_expert_dataset(
        dataset_path,
        format=conf.expert.format,
        action_tolerance=conf.expert.action_tolerance,
        max_episodes=max_episodes,
        **_cost_loader_kwargs(conf),
    )
    _print_metadata(episodes.metadata)
    if bool(conf.logging.log_expert_init_metrics):
        _log_metadata(logger, episodes.metadata)

    train_episodes, val_episodes = _split_train_validation(episodes, conf.expert.validation_ratio)
    coverage_train_episodes, coverage_val_episodes, coverage_metadata = _load_wm_coverage_datasets(
        conf,
        args,
        episodes.metadata,
    )
    _print_coverage_metadata(coverage_metadata)
    wm_train_episodes = list(train_episodes) + list(coverage_train_episodes)
    wm_val_episodes = list(val_episodes) + list(coverage_val_episodes)

    buffer_device = args.buffer_device or args.device
    include_force = bool(getattr(conf.ForceHead, "Enable", False))
    train_replay = make_expert_replay(
        train_episodes,
        device=buffer_device,
        include_force=include_force,
        force_dim=1,
        force_key=getattr(conf.ForceHead, "Key", ""),
    )
    val_replay = make_expert_replay(
        val_episodes,
        device=buffer_device,
        include_force=include_force,
        force_dim=1,
        force_key=getattr(conf.ForceHead, "Key", ""),
    )
    wm_train_replay = make_expert_replay(
        wm_train_episodes,
        device=buffer_device,
        include_force=include_force,
        force_dim=1,
        force_key=getattr(conf.ForceHead, "Key", ""),
    )
    wm_val_replay = make_expert_replay(
        wm_val_episodes,
        device=buffer_device,
        include_force=include_force,
        force_dim=1,
        force_key=getattr(conf.ForceHead, "Key", ""),
    )
    logger.log("expert/replay_train_steps", train_replay.num_expert_steps(), 0)
    logger.log("expert/replay_validation_steps", val_replay.num_expert_steps(), 0)
    logger.log("expert/wm_coverage_train_steps", _episode_steps(coverage_train_episodes), 0)
    logger.log("expert/wm_coverage_validation_steps", _episode_steps(coverage_val_episodes), 0)
    logger.log("expert/wm_train_steps_total", wm_train_replay.num_expert_steps(), 0)
    logger.log("expert/wm_validation_steps_total", wm_val_replay.num_expert_steps(), 0)

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, episodes.metadata["obs_dim"], episodes.metadata["action_dim"], act, args.device)
    agent = build_agent(conf, episodes.metadata["action_dim"], act, args.device)

    config_payload = cfg_to_dict(conf)
    wm_ckpt = os.path.join(checkpoint_dir, "world_model_expert_pretrained.pt")
    actor_ckpt = os.path.join(checkpoint_dir, "actor_bc_initialized.pt")
    full_ckpt = os.path.join(checkpoint_dir, "full_agent_before_online.pt")
    replay_ckpt = os.path.join(checkpoint_dir, "replay_after_expert_load.pt")

    if bool(conf.expert.save_init_checkpoints):
        train_replay.save_replay(replay_ckpt)
        if coverage_train_episodes:
            wm_train_replay.save_replay(os.path.join(checkpoint_dir, "replay_after_wm_mix_load.pt"))

    loaded_wm = _load_state_if_exists(
        conf.expert.load_pretrained_world_model_path,
        world_model=world_model,
        device=args.device,
    )
    if (
        not loaded_wm
        and bool(conf.expert.skip_pretrain_if_checkpoint_exists)
        and os.path.isfile(wm_ckpt)
    ):
        loaded_wm = _load_state_if_exists(wm_ckpt, world_model=world_model, device=args.device)

    if bool(conf.expert.pretrain_world_model) and not loaded_wm:
        wm_metrics = pretrain_world_model_from_expert(
            wm_train_replay,
            world_model,
            num_steps=conf.expert.pretrain_steps,
            batch_size=conf.JointTrainAgent.BatchSize,
            batch_length=conf.JointTrainAgent.BatchLength,
            logger=logger,
            log_interval=conf.expert.log_interval,
            cost_positive_ratio=conf.expert.cost_balanced_sequence_ratio,
            progress=True,
        )
        print(colorama.Fore.CYAN + f"Final expert WM metrics: {wm_metrics}" + colorama.Style.RESET_ALL)

    if bool(conf.expert.save_init_checkpoints):
        save_expert_checkpoint(
            wm_ckpt,
            world_model=world_model,
            config=config_payload,
            expert_metadata=episodes.metadata,
            extra={
                "action_scaling": "normalized_-1_1",
                "wm_coverage_metadata": coverage_metadata,
                "wm_train_expert_steps": train_replay.num_expert_steps(),
                "wm_train_coverage_steps": _episode_steps(coverage_train_episodes),
            },
        )

    if wm_val_replay.can_sample(conf.JointTrainAgent.BatchLength):
        val_batch = wm_val_replay.sample(
            conf.JointTrainAgent.BatchSize,
            conf.JointTrainAgent.BatchLength,
            return_dict=True,
        )
        wm_val_metrics = evaluate_world_model_on_expert(world_model, val_batch)
        _log_metrics = {f"expert_init/{key}": value for key, value in wm_val_metrics.items()}
        wandb.log(_log_metrics, step=max(int(conf.expert.pretrain_steps), 0))
        print(colorama.Fore.CYAN + f"Expert+coverage WM validation: {wm_val_metrics}" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.YELLOW + "Skipping expert WM validation; validation replay cannot sample." + colorama.Style.RESET_ALL)

    loaded_actor = _load_state_if_exists(conf.expert.load_bc_actor_path, agent=agent, device=args.device)
    if (
        not loaded_actor
        and bool(conf.expert.skip_pretrain_if_checkpoint_exists)
        and os.path.isfile(actor_ckpt)
    ):
        loaded_actor = _load_state_if_exists(actor_ckpt, agent=agent, device=args.device)

    if bool(conf.expert.bc_init) and not loaded_actor:
        old_lrs = _maybe_set_bc_lr(agent, conf.expert.bc_lr)
        bc_metrics = pretrain_actor_bc_from_expert(
            train_replay,
            world_model,
            agent,
            num_steps=conf.expert.bc_steps,
            batch_size=conf.expert.bc_batch_size or conf.JointTrainAgent.BatchSize,
            batch_length=conf.JointTrainAgent.ImagineContext or conf.JointTrainAgent.BatchLength,
            logger=logger,
            log_interval=conf.expert.log_interval,
            progress=True,
        )
        _restore_lrs(agent, old_lrs)
        print(colorama.Fore.CYAN + f"Final expert BC metrics: {bc_metrics}" + colorama.Style.RESET_ALL)

    if val_replay.can_sample(conf.JointTrainAgent.ImagineContext or conf.JointTrainAgent.BatchLength):
        val_batch = val_replay.sample(
            conf.expert.bc_batch_size or conf.JointTrainAgent.BatchSize,
            conf.JointTrainAgent.ImagineContext or conf.JointTrainAgent.BatchLength,
            return_dict=True,
        )
        bc_val_metrics = evaluate_actor_bc_on_expert(world_model, agent, val_batch)
        wandb.log({f"expert_init/{key}": value for key, value in bc_val_metrics.items()}, step=max(int(conf.expert.bc_steps), 0))
        print(colorama.Fore.CYAN + f"Expert BC validation: {bc_val_metrics}" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.YELLOW + "Skipping expert BC validation; validation replay cannot sample." + colorama.Style.RESET_ALL)

    if bool(conf.expert.save_init_checkpoints):
        save_expert_checkpoint(
            actor_ckpt,
            agent=agent,
            config=config_payload,
            expert_metadata=episodes.metadata,
            extra={
                "action_scaling": "normalized_-1_1",
                "bc_dataset": "expert_only",
                "wm_coverage_metadata": coverage_metadata,
            },
        )
        save_expert_checkpoint(
            full_ckpt,
            world_model=world_model,
            agent=agent,
            config=config_payload,
            expert_metadata=episodes.metadata,
            extra={
                "action_scaling": "normalized_-1_1",
                "bc_dataset": "expert_only",
                "wm_coverage_metadata": coverage_metadata,
                "wm_train_expert_steps": train_replay.num_expert_steps(),
                "wm_train_coverage_steps": _episode_steps(coverage_train_episodes),
            },
        )
        print(
            colorama.Fore.GREEN
            + f"Saved expert init checkpoints: {wm_ckpt}, {actor_ckpt}, {full_ckpt}"
            + colorama.Style.RESET_ALL
        )

    if logger.log_dict:
        wandb.log(logger.log_dict, step=logger.tot_step)
    wandb.finish()
