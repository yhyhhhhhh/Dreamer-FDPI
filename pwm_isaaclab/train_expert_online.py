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
    from pwm_isaaclab.env_wrapper import DreamerVecEnvWrapper
    from pwm_isaaclab.expert_config import cfg_to_dict, load_expert_config
    from pwm_isaaclab.expert_init import (
        load_expert_checkpoint,
        pretrain_actor_bc_from_expert,
        pretrain_world_model_from_expert,
        save_expert_checkpoint,
    )
    from pwm_isaaclab.expert_loader import load_expert_dataset
    from pwm_isaaclab.expert_replay import HybridExpertReplay, SourceTaggedProprioReplayBuffer, make_expert_replay
    from pwm_isaaclab.expert_world_model import ExpertWorldModelWithCost
    from pwm_isaaclab.trainer import joint_train_world_model_agent
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
    from env_wrapper import DreamerVecEnvWrapper
    from expert_config import cfg_to_dict, load_expert_config
    from expert_init import (
        load_expert_checkpoint,
        pretrain_actor_bc_from_expert,
        pretrain_world_model_from_expert,
        save_expert_checkpoint,
    )
    from expert_loader import load_expert_dataset
    from expert_replay import HybridExpertReplay, SourceTaggedProprioReplayBuffer, make_expert_replay
    from expert_world_model import ExpertWorldModelWithCost
    from trainer import joint_train_world_model_agent
    from utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )


def _cfg_to_dict(node):
    if hasattr(node, "items"):
        return {key: _cfg_to_dict(value) for key, value in node.items()}
    return node


def _resolve_force_scale(value):
    if isinstance(value, str):
        if value.lower() == "auto":
            print(colorama.Fore.YELLOW + "ForceLoss.ForceScale='auto' is not estimated online; using 1.0." + colorama.Style.RESET_ALL)
            return 1.0
        return float(value)
    return float(value)


def launch_isaaclab(headless=True):
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    return app_launcher.app


def build_env(args, conf):
    import gymnasium
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    import ur3_lite.tasks  # noqa: F401

    make_kwargs = {}
    if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
        make_kwargs = _cfg_to_dict(conf.Env.MakeKwargs)
    num_envs = int(make_kwargs.get("num_envs", conf.JointTrainAgent.NumEnvs))
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


def _load_full_or_split_checkpoints(conf, args, world_model, agent):
    if args.expert_init_checkpoint:
        load_expert_checkpoint(args.expert_init_checkpoint, world_model=world_model, agent=agent, map_location=args.device)
        print(colorama.Fore.YELLOW + f"Loaded full expert init checkpoint: {args.expert_init_checkpoint}" + colorama.Style.RESET_ALL)
        return True

    loaded = False
    if conf.expert.load_pretrained_world_model_path:
        load_expert_checkpoint(conf.expert.load_pretrained_world_model_path, world_model=world_model, map_location=args.device)
        print(colorama.Fore.YELLOW + f"Loaded expert world model: {conf.expert.load_pretrained_world_model_path}" + colorama.Style.RESET_ALL)
        loaded = True
    if conf.expert.load_bc_actor_path:
        load_expert_checkpoint(conf.expert.load_bc_actor_path, agent=agent, map_location=args.device)
        print(colorama.Fore.YELLOW + f"Loaded expert BC actor: {conf.expert.load_bc_actor_path}" + colorama.Style.RESET_ALL)
        loaded = True
    return loaded


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


def _load_wm_coverage_episodes(conf, args, policy_space, action_space):
    paths = _as_path_list(conf.expert.wm_coverage_paths)
    paths.extend(_as_path_list(args.wm_coverage_path))
    if not paths:
        return [], []

    max_episodes = args.max_wm_coverage_episodes
    if max_episodes is None or max_episodes <= 0:
        max_episodes = int(getattr(conf.expert, "wm_coverage_max_episodes", 0) or 0)
    max_episodes = max_episodes if max_episodes > 0 else None

    coverage_episodes = []
    coverage_metadata = []
    remaining = max_episodes
    for path in paths:
        load_limit = remaining if remaining is not None else None
        if load_limit is not None and load_limit <= 0:
            break
        dataset = load_expert_dataset(
            path,
            format=conf.expert.wm_coverage_format,
            env_spec={
                "obs_shape": policy_space.shape,
                "action_shape": action_space.shape,
                "action_space": action_space,
            },
            action_tolerance=conf.expert.action_tolerance,
            max_episodes=load_limit,
            **_cost_loader_kwargs(conf),
        )
        coverage_episodes.extend(dataset)
        coverage_metadata.append(dataset.metadata)
        if remaining is not None:
            remaining -= len(dataset)
    return coverage_episodes, coverage_metadata


def _episode_steps(episodes):
    return int(sum(len(episode["reward"]) for episode in episodes))


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
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-dataset_path", type=str, default=None)
    parser.add_argument("-expert_init_checkpoint", type=str, default=None)
    parser.add_argument("-wm_coverage_path", action="append", default=None)
    parser.add_argument("-max_wm_coverage_episodes", type=int, default=0)
    parser.add_argument("-offline_dataset_dir", type=str, default=None)
    parser.add_argument("--save_offline_episodes", action="store_true")
    parser.add_argument("--run_root", type=str, default="ckpt")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--no_run_info_prompt", action="store_true")
    args = parser.parse_args()

    conf = load_expert_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    if conf.Task != "JointTrainAgent":
        raise NotImplementedError(f"Task {conf.Task} not implemented")

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
        extra={
            "base_run_name": args.n,
            "env_name": args.env_name,
            "seed": args.seed,
            "device": args.device,
        },
    )
    print(colorama.Fore.CYAN + f"Run directory: {checkpoint_dir}" + colorama.Style.RESET_ALL)

    seed_np_torch(seed=args.seed)
    project = getattr(conf.Wandb, "Project", "IsaacLab-PSSM")
    run_group = getattr(conf.Wandb, "Group", args.env_name)
    base_wandb_name = getattr(conf.Wandb, "Name", f"PSSM-{args.env_name}-seed{args.seed}")
    run_name = f"{base_wandb_name}-{os.path.basename(checkpoint_dir)}"
    wandb_mode = getattr(conf.Wandb, "Mode", None)
    init_kwargs = {
        "project": project,
        "group": run_group,
        "name": run_name,
        "dir": checkpoint_dir,
        "config": cfg_to_dict(conf),
    }
    if run_info.get("note"):
        init_kwargs["notes"] = run_info["note"]
    if run_info.get("tags"):
        init_kwargs["tags"] = run_info["tags"]
    if wandb_mode is not None:
        init_kwargs["mode"] = wandb_mode
    wandb.init(**init_kwargs)
    logger = Logger()

    simulation_app = launch_isaaclab(headless=True)
    vec_env = build_env(args, conf)
    num_envs = vec_env.num_envs
    policy_space = vec_env.single_observation_space["policy"]
    action_space = vec_env.single_action_space
    obs_dim = int(policy_space.shape[0])
    action_dim = int(action_space.shape[0])

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)

    online_replay = SourceTaggedProprioReplayBuffer(
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

    expert_replay = None
    wm_pretrain_replay = None
    wm_coverage_metadata = []
    expert_loaded = False
    if bool(conf.expert.enabled):
        dataset_path = args.dataset_path or conf.expert.path
        if dataset_path:
            episodes = load_expert_dataset(
                dataset_path,
                format=conf.expert.format,
                env_spec={
                    "obs_shape": policy_space.shape,
                    "action_shape": action_space.shape,
                    "action_space": action_space,
                },
                action_tolerance=conf.expert.action_tolerance,
                **_cost_loader_kwargs(conf),
            )
            expert_replay = make_expert_replay(
                episodes,
                device=args.device,
                include_force=bool(conf.ForceHead.Enable),
                force_dim=1,
                force_key=conf.ForceHead.Key,
            )
            wm_coverage_episodes, wm_coverage_metadata = _load_wm_coverage_episodes(
                conf,
                args,
                policy_space,
                action_space,
            )
            wm_pretrain_replay = make_expert_replay(
                list(episodes) + list(wm_coverage_episodes),
                device=args.device,
                include_force=bool(conf.ForceHead.Enable),
                force_dim=1,
                force_key=conf.ForceHead.Key,
            )
            expert_loaded = True
            logger.log("expert/num_episodes", episodes.metadata["num_episodes"], 0)
            logger.log("expert/num_steps", episodes.metadata["num_transitions"], 0)
            logger.log("expert/mean_return", episodes.metadata["mean_return"], 0)
            logger.log("expert/mean_cost", episodes.metadata["mean_cost"], 0)
            logger.log("expert/wm_coverage_steps", _episode_steps(wm_coverage_episodes), 0)
            logger.log("expert/wm_train_steps_total", wm_pretrain_replay.num_expert_steps(), 0)
            logger.log("expert/random_prefill_skipped", float(bool(conf.expert.replace_random_prefill)), 0)
            logger.log("expert/replay_size_after_expert_load", expert_replay.num_expert_steps(), 0)
            print(
                colorama.Fore.CYAN
                + (
                    f"Loaded expert replay: {episodes.metadata['num_episodes']} episodes / "
                    f"{episodes.metadata['num_transitions']} steps"
                )
                + colorama.Style.RESET_ALL
            )
        else:
            print(colorama.Fore.YELLOW + "expert.enabled=true but no dataset path was provided; using original random prefill." + colorama.Style.RESET_ALL)

    replay_buffer = HybridExpertReplay(
        online_replay,
        expert_replay,
        replace_random_prefill=bool(conf.expert.enabled and conf.expert.replace_random_prefill and expert_loaded),
        expert_ratio_online=conf.expert.expert_ratio_online,
    )

    loaded_checkpoint = _load_full_or_split_checkpoints(conf, args, world_model, agent)
    if bool(conf.expert.enabled) and expert_loaded and not loaded_checkpoint:
        if bool(conf.expert.pretrain_world_model):
            pretrain_world_model_from_expert(
                wm_pretrain_replay or expert_replay,
                world_model,
                num_steps=conf.expert.pretrain_steps,
                batch_size=conf.JointTrainAgent.BatchSize,
                batch_length=conf.JointTrainAgent.BatchLength,
                logger=logger,
                log_interval=conf.expert.log_interval,
                cost_positive_ratio=conf.expert.cost_balanced_sequence_ratio,
                progress=True,
            )
            save_expert_checkpoint(
                os.path.join(checkpoint_dir, "world_model_expert_pretrained.pt"),
                world_model=world_model,
                config=cfg_to_dict(conf),
                expert_metadata=getattr(episodes, "metadata", {}),
                extra={"wm_coverage_metadata": wm_coverage_metadata},
            )
        if bool(conf.expert.bc_init):
            pretrain_actor_bc_from_expert(
                expert_replay,
                world_model,
                agent,
                num_steps=conf.expert.bc_steps,
                batch_size=conf.expert.bc_batch_size or conf.JointTrainAgent.BatchSize,
                batch_length=conf.JointTrainAgent.ImagineContext or conf.JointTrainAgent.BatchLength,
                logger=logger,
                log_interval=conf.expert.log_interval,
                progress=True,
            )
            save_expert_checkpoint(
                os.path.join(checkpoint_dir, "full_agent_before_online.pt"),
                world_model=world_model,
                agent=agent,
                config=cfg_to_dict(conf),
                expert_metadata=getattr(episodes, "metadata", {}),
                extra={"wm_coverage_metadata": wm_coverage_metadata, "bc_dataset": "expert_only"},
            )

    offline_dataset_dir = args.offline_dataset_dir or conf.JointTrainAgent.OfflineDatasetDir
    save_offline_episodes = (
        bool(conf.JointTrainAgent.SaveOfflineEpisodes)
        or args.save_offline_episodes
        or bool(offline_dataset_dir)
    )
    if save_offline_episodes and not offline_dataset_dir:
        offline_dataset_dir = os.path.join(checkpoint_dir, "offline_episodes")

    joint_train_world_model_agent(
        args.env_name,
        args.n,
        vec_env,
        conf.JointTrainAgent.SampleMaxSteps,
        replay_buffer,
        world_model,
        agent,
        conf.JointTrainAgent.TrainModelEverySteps,
        conf.JointTrainAgent.TrainAgentEverySteps,
        conf.JointTrainAgent.ModelUpdate,
        conf.JointTrainAgent.AgentUpdate,
        conf.JointTrainAgent.BatchSize,
        conf.JointTrainAgent.BatchLength,
        conf.JointTrainAgent.ImagineBatchSize,
        conf.JointTrainAgent.ImagineContext,
        conf.JointTrainAgent.ImagineHorizon,
        conf.JointTrainAgent.SaveEverySteps,
        logger,
        args.device,
        offline_dataset_dir=offline_dataset_dir if save_offline_episodes else None,
        checkpoint_dir=checkpoint_dir,
    )

    if logger.log_dict:
        wandb.log(logger.log_dict, step=logger.tot_step)
    wandb.finish()
    if simulation_app is not None:
        simulation_app.close()
