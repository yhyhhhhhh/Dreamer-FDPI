from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import colorama
import gymnasium
import torch
import torch.nn as nn
import wandb
from yacs.config import CfgNode as CN

from pwm_isaaclab.env_wrapper import DreamerVecEnvWrapper
from pwm_isaaclab.expert_init import load_expert_checkpoint
from pwm_isaaclab.modules.world_models import ParallelWorldModel
from pwm_isaaclab.utils import (
    Logger,
    collect_training_info,
    make_unique_run_dir,
    save_run_artifacts,
    seed_np_torch,
    write_latest_run_pointer,
)
from pwm_isaaclab_dfd.agent_dfd import RiskConditionedActorCriticAgent
from pwm_isaaclab_dfd.dual_policy import DualPolicy
from pwm_isaaclab_dfd.feasibility import LatentFeasibilityModule
from pwm_isaaclab_dfd.replay_buffer_dfd import DFDReplayBuffer
from pwm_isaaclab_dfd.trainer_dfd import joint_train_dfd
from pwm_isaaclab_dfd.utils import cfg_get

from omni.isaac.lab.app import AppLauncher
from prettytable import PrettyTable  # noqa: F401


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.isaac.lab_tasks  # noqa: E402,F401
from omni.isaac.lab_tasks.utils import parse_env_cfg  # noqa: E402
import ur3_lite.tasks  # noqa: E402,F401


def _cfg_to_dict(node):
    if hasattr(node, "items"):
        return {key: _cfg_to_dict(value) for key, value in node.items()}
    return node


def _ensure_node(parent, name):
    if not hasattr(parent, name):
        setattr(parent, name, CN(new_allowed=True))
    node = getattr(parent, name)
    if hasattr(node, "set_new_allowed"):
        node.set_new_allowed(True)
    return node


def _set_default(node, name, value):
    if not hasattr(node, name):
        setattr(node, name, value)


def _as_float(value):
    return float(value)


def _as_int(value):
    return int(value)


def _report_exception(exc, checkpoint_dir=None):
    message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(colorama.Fore.RED + "\nDFD training failed with exception:\n" + message + colorama.Style.RESET_ALL)
    if checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            error_path = os.path.join(checkpoint_dir, "dfd_error.log")
            with open(error_path, "w", encoding="utf-8") as fout:
                fout.write(message)
            print(colorama.Fore.RED + f"Saved DFD error report: {error_path}" + colorama.Style.RESET_ALL)
        except Exception as report_exc:
            print(
                colorama.Fore.YELLOW
                + f"Could not save DFD error report: {report_exc}"
                + colorama.Style.RESET_ALL
            )


def load_dfd_config(config_path):
    conf = CN(new_allowed=True)
    conf.merge_from_file(config_path)
    conf.defrost()

    _ensure_node(conf, "Wandb")
    joint = _ensure_node(conf, "JointTrainAgent")
    _set_default(joint, "SaveOfflineEpisodes", False)
    _set_default(joint, "OfflineDatasetDir", "")

    dfd = _ensure_node(conf, "DFD")
    _set_default(dfd, "use_dual", False)
    _set_default(dfd, "train_feasibility", False)
    _set_default(dfd, "use_risk_conditioned_advantage", False)

    replay = _ensure_node(dfd, "Replay")
    _set_default(replay, "world_model_max_dual_fraction", 0.10)

    cost = _ensure_node(dfd, "Cost")
    _set_default(cost, "bottom_force_threshold", 1.0)
    _set_default(cost, "bottom_force_channels", [2, 5])

    feasibility = _ensure_node(dfd, "Feasibility")
    _set_default(feasibility, "cost_gamma", 0.97)
    _set_default(feasibility, "pf", 0.10)
    _set_default(feasibility, "cg", 0.03)
    _set_default(feasibility, "hidden_dim", 256)
    _set_default(feasibility, "num_layers", 2)
    _set_default(feasibility, "target_tau", 0.005)
    _set_default(feasibility, "lr", 1.0e-4)
    _set_default(feasibility, "eps", 1.0e-8)
    _set_default(feasibility, "update_steps", 1)

    dual = _ensure_node(dfd, "DualPolicy")
    _set_default(dual, "start_step", 100000)
    _set_default(dual, "ratio_start", 0.01)
    _set_default(dual, "ratio_final", 0.02)
    _set_default(dual, "ratio_warmup_steps", 100000)
    _set_default(dual, "target_kl", 0.5)
    _set_default(dual, "max_kl_for_sampling", 2.0)
    _set_default(dual, "lambda_kl_init", 1.0)
    _set_default(dual, "lambda_lr", 1.0e-4)
    _set_default(dual, "lr", 8.0e-5)
    _set_default(dual, "eps", 1.0e-5)
    _set_default(dual, "update_steps", 1)
    _set_default(dual, "init_from_main_actor", False)
    _set_default(dual, "gd_objective", "min")
    _set_default(dual, "dual_g_scale", 1.0)

    risk = _ensure_node(dfd, "MainActorRisk")
    _set_default(risk, "start_step", 150000)
    _set_default(risk, "lambda_cri_start", 0.0)
    _set_default(risk, "lambda_cri_final", 0.02)
    _set_default(risk, "lambda_inf_start", 0.0)
    _set_default(risk, "lambda_inf_final", 0.05)
    _set_default(risk, "lambda_warmup_steps", 50000)
    _set_default(risk, "clip_safe_adv", True)
    _set_default(risk, "safe_adv_min", -5.0)
    _set_default(risk, "safe_adv_max", 5.0)

    conf.freeze()
    return conf


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
        _as_int(conf.JointTrainAgent.VideoLogStep),
        True,
        obs_dim,
        action_dim,
        _as_int(conf.Models.Stoch),
        _as_int(conf.Models.Discrete),
        _as_int(conf.Models.Hidden),
        _as_int(conf.Models.WorldModel.Stem),
        _as_int(conf.Models.WorldModel.MinRes),
        _as_int(conf.Models.NumBin),
        _as_float(conf.Models.MaxBin),
        _as_float(conf.Models.WorldModel.DynScale),
        _as_float(conf.Models.WorldModel.RepScale),
        _as_float(conf.Models.WorldModel.ValScale),
        _as_float(conf.Models.WorldModel.KLFree),
        _as_float(conf.Models.Gamma),
        _as_float(conf.Models.Lambda),
        _as_float(conf.Models.Tau),
        _as_float(conf.Models.WorldModel.LR),
        _as_float(conf.Models.WorldModel.Eps),
        conf.BasicSettings.UseAmp,
        act,
        device,
        force_enabled,
        _as_int(conf.ForceHead.HiddenDim),
        _as_int(conf.ForceHead.Depth),
        _as_float(conf.ForceHead.Dropout),
        _as_float(conf.ForceLoss.Eps),
        _resolve_force_scale(conf.ForceLoss.ForceScale),
        _as_float(conf.ForceHead.Threshold),
        _as_float(conf.ForceHead.LossWeight),
        conf.ForceHead.DetachLatent,
        _as_float(conf.ForceLoss.LambdaCls),
        _as_float(conf.ForceLoss.LambdaReg),
        _as_float(conf.ForceLoss.LambdaSign),
        _as_float(conf.ForceLoss.FocalAlpha),
        _as_float(conf.ForceLoss.FocalGamma),
        _as_float(conf.ForceLoss.HuberBeta),
        _as_float(conf.ForceLoss.RegWeightPower),
        _as_float(conf.ForceLoss.RegWeightMax),
        conf.ForceHead.SignedForce,
    ).to(device)


def build_agent(conf, action_dim, act, device):
    return RiskConditionedActorCriticAgent(
        action_dim,
        _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden),
        _as_int(conf.Models.Hidden),
        _as_float(conf.Models.Agent.EntropyCoef),
        _as_int(conf.Models.NumBin),
        _as_float(conf.Models.MaxBin),
        _as_float(conf.Models.Agent.MinPer),
        _as_float(conf.Models.Agent.MaxPer),
        _as_float(conf.Models.Agent.MinStd),
        _as_float(conf.Models.Agent.MaxStd),
        _as_float(conf.Models.Agent.EMADecay),
        _as_float(conf.Models.Gamma),
        _as_float(conf.Models.Lambda),
        _as_float(conf.Models.Tau),
        bool(getattr(conf.Models.Agent, "UseSlowCritic", False)),
        _as_float(conf.Models.Agent.LR),
        _as_float(conf.Models.Agent.Eps),
        conf.BasicSettings.UseAmp,
        act,
        device,
    ).to(device)


def build_feasibility(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    feas_cfg = conf.DFD.Feasibility
    return LatentFeasibilityModule(
        feat_dim=feat_dim,
        action_dim=action_dim,
        hidden_dim=_as_int(cfg_get(feas_cfg, "hidden_dim", 256)),
        num_layers=_as_int(cfg_get(feas_cfg, "num_layers", 2)),
        cost_gamma=_as_float(cfg_get(feas_cfg, "cost_gamma", 0.97)),
        target_tau=_as_float(cfg_get(feas_cfg, "target_tau", 0.005)),
        lr=_as_float(cfg_get(feas_cfg, "lr", 1.0e-4)),
        eps=_as_float(cfg_get(feas_cfg, "eps", 1.0e-8)),
        use_amp=conf.BasicSettings.UseAmp,
        act=act,
        device=device,
    ).to(device)


def build_dual_policy(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    dual_cfg = conf.DFD.DualPolicy
    return DualPolicy(
        action_dim=action_dim,
        feat_dim=feat_dim,
        hidden=_as_int(conf.Models.Hidden),
        min_std=_as_float(conf.Models.Agent.MinStd),
        max_std=_as_float(conf.Models.Agent.MaxStd),
        lr=_as_float(cfg_get(dual_cfg, "lr", conf.Models.Agent.LR)),
        eps=_as_float(cfg_get(dual_cfg, "eps", conf.Models.Agent.Eps)),
        lambda_kl_init=_as_float(cfg_get(dual_cfg, "lambda_kl_init", 1.0)),
        lambda_lr=_as_float(cfg_get(dual_cfg, "lambda_lr", 1.0e-4)),
        use_amp=conf.BasicSettings.UseAmp,
        act=act,
        device=device,
        gd_objective=cfg_get(dual_cfg, "gd_objective", "min"),
        dual_g_scale=_as_float(cfg_get(dual_cfg, "dual_g_scale", 1.0)),
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
    parser.add_argument("-checkpoint_path", type=str, default=None)
    parser.add_argument("-offline_dataset_dir", type=str, default=None)
    parser.add_argument("--save_offline_episodes", action="store_true")
    parser.add_argument("--run_root", type=str, default="ckpt")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--no_run_info_prompt", action="store_true")
    args = parser.parse_args()

    conf = load_dfd_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    checkpoint_path = None
    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
            "algorithm": "Dual-Feasibility Dreamer",
            "checkpoint_path": checkpoint_path,
        },
    )
    print(colorama.Fore.CYAN + f"Run directory: {checkpoint_dir}" + colorama.Style.RESET_ALL)

    seed_np_torch(seed=args.seed)
    wandb_conf = getattr(conf, "Wandb", None)
    project = cfg_get(wandb_conf, "Project", "IsaacLab-PSSM-DFD")
    run_group = cfg_get(wandb_conf, "Group", args.env_name)
    base_wandb_name = cfg_get(wandb_conf, "Name", f"DFD-{args.env_name}-seed{args.seed}")
    run_name = f"{base_wandb_name}-{os.path.basename(checkpoint_dir)}"
    wandb_mode = cfg_get(wandb_conf, "Mode", None)
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
            "algorithm": "Dual-Feasibility Dreamer",
            "checkpoint_path": checkpoint_path,
        },
    )

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)
    if checkpoint_path:
        load_expert_checkpoint(checkpoint_path, world_model=world_model, agent=agent, map_location=args.device)
        if hasattr(agent, "sync_slow_critic"):
            agent.sync_slow_critic()
        print(
            colorama.Fore.YELLOW
            + f"Loaded DFD initial world_model/agent checkpoint: {checkpoint_path}"
            + colorama.Style.RESET_ALL
        )
    feasibility = build_feasibility(conf, action_dim, act, args.device)
    dual_policy = build_dual_policy(conf, action_dim, act, args.device)
    if bool(cfg_get(conf.DFD.DualPolicy, "init_from_main_actor", False)):
        dual_policy.initialize_from_main_actor(agent)
        print(
            colorama.Fore.YELLOW
            + "Initialized DFD dual actor from the main actor."
            + colorama.Style.RESET_ALL
        )
    replay_buffer = DFDReplayBuffer(
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
    offline_dataset_dir = args.offline_dataset_dir or getattr(conf.JointTrainAgent, "OfflineDatasetDir", "")
    save_offline_episodes = (
        bool(getattr(conf.JointTrainAgent, "SaveOfflineEpisodes", False))
        or args.save_offline_episodes
        or bool(offline_dataset_dir)
    )
    if save_offline_episodes and not offline_dataset_dir:
        offline_dataset_dir = os.path.join(checkpoint_dir, "offline_episodes")

    try:
        joint_train_dfd(
            args.env_name,
            args.n,
            vec_env,
            conf.JointTrainAgent.SampleMaxSteps,
            replay_buffer,
            world_model,
            agent,
            feasibility,
            dual_policy,
            conf.DFD,
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
    except KeyboardInterrupt:
        print(
            colorama.Fore.YELLOW
            + "DFD training interrupted by Ctrl-C; closing IsaacLab resources..."
            + colorama.Style.RESET_ALL
        )
        raise
    except Exception as exc:
        _report_exception(exc, checkpoint_dir)
        raise
    finally:
        try:
            if logger.log_dict and logger.tot_step >= 0:
                wandb.log(logger.log_dict, step=logger.tot_step)
        except Exception as exc:
            print(colorama.Fore.YELLOW + f"Cleanup warning during wandb.log: {exc}" + colorama.Style.RESET_ALL)
        try:
            wandb.finish()
        except Exception as exc:
            print(colorama.Fore.YELLOW + f"Cleanup warning during wandb.finish: {exc}" + colorama.Style.RESET_ALL)
        try:
            vec_env.close()
        except Exception as exc:
            print(colorama.Fore.YELLOW + f"Cleanup warning during vec_env.close: {exc}" + colorama.Style.RESET_ALL)
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception as exc:
            print(
                colorama.Fore.YELLOW
                + f"Cleanup warning during simulation_app.close: {exc}"
                + colorama.Style.RESET_ALL
            )
