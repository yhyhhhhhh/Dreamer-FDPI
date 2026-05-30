
import argparse
import os
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import colorama
import gymnasium
import torch
import torch.nn as nn
import wandb

from agents import ActorCriticAgent
from env_wrapper import DreamerVecEnvWrapper
from modules.world_models import ParallelWorldModel
from replay_buffer import ProprioReplayBuffer
from trainer import joint_train_world_model_agent
from utils import (
    Logger,
    collect_training_info,
    load_config,
    make_unique_run_dir,
    save_run_artifacts,
    seed_np_torch,
    write_latest_run_pointer,
)
from omni.isaac.lab.app import AppLauncher
from prettytable import PrettyTable
# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
import ur3_lite.tasks
# /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/train.py   -n cartpole-pssm   -seed 42   -config_path pwm_isaaclab/config_files/PWM.yaml   -env_name Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0   -device cuda:0 --save_offline_episodes


def _cfg_to_dict(node):
    if hasattr(node, "items"):
        return {key: _cfg_to_dict(value) for key, value in node.items()}
    return node


def _resolve_force_scale(value):
    if isinstance(value, str):
        if value.lower() == "auto":
            print(
                colorama.Fore.YELLOW
                + "ForceLoss.ForceScale='auto' is not estimated online; using 1.0. "
                + "Set ForceLoss.ForceScale to a quoted numeric value if needed."
                + colorama.Style.RESET_ALL
            )
            return 1.0
        return float(value)
    return float(value)


def build_env(args, conf):
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
    force_enabled = bool(getattr(conf.ForceHead, "Enable", False))
    return ParallelWorldModel(
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
        force_enabled,
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-offline_dataset_dir", type=str, default=None)
    parser.add_argument("--save_offline_episodes", action="store_true")
    parser.add_argument("--run_root", type=str, default="ckpt")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--no_run_info_prompt", action="store_true")
    args = parser.parse_args()

    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

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
    print(
        colorama.Fore.CYAN
        + f"Run directory: {checkpoint_dir}"
        + colorama.Style.RESET_ALL
    )

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
        "config": _cfg_to_dict(conf),
    }
    if run_info.get("note"):
        init_kwargs["notes"] = run_info["note"]
    if run_info.get("tags"):
        init_kwargs["tags"] = run_info["tags"]
    if wandb_mode is not None:
        init_kwargs["mode"] = wandb_mode
    wandb.init(**init_kwargs)
    wandb.config.update(
        {
            "checkpoint_dir": checkpoint_dir,
            "base_run_name": args.n,
            "run_info": run_info,
        },
        allow_val_change=True,
    )
    logger = Logger()

    if conf.Task != "JointTrainAgent":
        raise NotImplementedError(f"Task {conf.Task} not implemented")

    vec_env = build_env(args, conf)
    num_envs = vec_env.num_envs
    policy_space = vec_env.single_observation_space["policy"]
    action_space = vec_env.single_action_space
    obs_dim = int(policy_space.shape[0])
    action_dim = int(action_space.shape[0])

    if conf.JointTrainAgent.NumEnvs and conf.JointTrainAgent.NumEnvs != num_envs:
        print(
            colorama.Fore.YELLOW
            + f"Config NumEnvs={conf.JointTrainAgent.NumEnvs} but env provides num_envs={num_envs}; using env value."
            + colorama.Style.RESET_ALL
        )
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
            "num_envs": num_envs,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        },
    )

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)
    replay_buffer = ProprioReplayBuffer(
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
    simulation_app.close()
