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
from pwm_isaaclab.utils import (
    Logger,
    collect_training_info,
    make_unique_run_dir,
    save_run_artifacts,
    seed_np_torch,
    write_latest_run_pointer,
)
from pwm_isaaclab_dfd_v2.agent_dfd_v2 import CostAwareActorCriticAgent
from pwm_isaaclab_dfd_v2.cost_utils import cfg_get
from pwm_isaaclab_dfd_v2.dual_policy_v2 import DualPolicyV2
from pwm_isaaclab_dfd_v2.gd_risk import GdRiskCritic
from pwm_isaaclab_dfd_v2.replay_buffer_dfd_v2 import DFDV2ReplayBuffer
from pwm_isaaclab_dfd_v2.trainer_dfd_v2 import joint_train_dfd_v2
from pwm_isaaclab_dfd_v2.world_model_dfd_v2 import DFDV2WorldModelWithContinuousCost


simulation_app = None


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
    print(colorama.Fore.RED + "\nDFD v2 training failed with exception:\n" + message + colorama.Style.RESET_ALL)
    if checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            error_path = os.path.join(checkpoint_dir, "dfd_v2_error.log")
            with open(error_path, "w", encoding="utf-8") as fout:
                fout.write(message)
            print(colorama.Fore.RED + f"Saved DFD v2 error report: {error_path}" + colorama.Style.RESET_ALL)
        except Exception as report_exc:
            print(colorama.Fore.YELLOW + f"Could not save DFD v2 error report: {report_exc}" + colorama.Style.RESET_ALL)


def load_dfd_v2_config(config_path):
    conf = CN(new_allowed=True)
    conf.merge_from_file(config_path)
    conf.defrost()

    _ensure_node(conf, "Wandb")
    joint = _ensure_node(conf, "JointTrainAgent")
    _set_default(joint, "SaveOfflineEpisodes", False)
    _set_default(joint, "OfflineDatasetDir", "")

    dfd = _ensure_node(conf, "DFD_V2")
    replay = _ensure_node(dfd, "Replay")
    _set_default(replay, "world_model_max_dual_fraction", 0.10)
    _set_default(replay, "cost_positive_ratio", 0.0)

    cost = _ensure_node(dfd, "ContinuousCost")
    _set_default(cost, "Enable", True)
    _set_default(cost, "ForceThreshold", 0.1)
    _set_default(cost, "LowForceScale", 0.05)
    _set_default(cost, "CostForceMax", 15.0)
    _set_default(cost, "ExtremeForceThreshold", 5.0)
    _set_default(cost, "ForceScale", 5.0)
    _set_default(cost, "ClipCost", True)
    _set_default(cost, "CostMin", 0.0)
    _set_default(cost, "CostMax", 1.0)
    _set_default(cost, "BottomForceChannels", [2, 5])

    cost_head = _ensure_node(dfd, "CostHead")
    _set_default(cost_head, "Enable", True)
    _set_default(cost_head, "HiddenDim", 320)
    _set_default(cost_head, "Depth", 3)
    _set_default(cost_head, "LossWeight", 2.0)
    _set_default(cost_head, "HuberBeta", 0.02)
    _set_default(cost_head, "SmallForceThreshold", 0.3)
    _set_default(cost_head, "SmallCostThreshold", 0.05)
    _set_default(cost_head, "SmallCostWeight", 2.0)
    _set_default(cost_head, "ExtremeLossWeight", 0.5)
    _set_default(cost_head, "ExtremeCostWeight", 4.0)
    _set_default(cost_head, "PriorLossWeight", 0.5)

    gd = _ensure_node(dfd, "Gd")
    _set_default(gd, "Enable", True)
    _set_default(gd, "GammaCost", 0.97)
    _set_default(gd, "DoubleCritic", True)
    _set_default(gd, "TargetTau", 0.005)
    _set_default(gd, "RiskMax", 1.0)
    _set_default(gd, "SourceAwareWeight", True)
    _set_default(gd, "DualSourceWeight", 2.0)
    _set_default(gd, "HighCostWeight", 3.0)
    _set_default(gd, "HighCostThreshold", 0.1)
    _set_default(gd, "HiddenDim", 256)
    _set_default(gd, "NumLayers", 2)
    _set_default(gd, "LR", 1.0e-4)
    _set_default(gd, "Eps", 1.0e-8)
    _set_default(gd, "UpdateSteps", 1)

    dual_imag = _ensure_node(dfd, "DualImagination")
    _set_default(dual_imag, "Enable", True)
    _set_default(dual_imag, "StartStep", 100000)
    _set_default(dual_imag, "Horizon", 5)
    _set_default(dual_imag, "Objective", "max_risk")
    _set_default(dual_imag, "KLCoeff", 1.0)
    _set_default(dual_imag, "MaxKLForSampling", 2.0)
    _set_default(dual_imag, "EntropyCoef", 1.0e-4)
    _set_default(dual_imag, "GradClipNorm", 100.0)
    _set_default(dual_imag, "UpdateSteps", 1)

    dual_sampling = _ensure_node(dfd, "DualSampling")
    _set_default(dual_sampling, "Enable", True)
    _set_default(dual_sampling, "StartStep", 120000)
    _set_default(dual_sampling, "RatioStart", 0.01)
    _set_default(dual_sampling, "RatioFinal", 0.03)
    _set_default(dual_sampling, "RatioWarmupSteps", 100000)
    _set_default(dual_sampling, "RequireKLHealthy", True)

    dual_policy = _ensure_node(dfd, "DualPolicy")
    _set_default(dual_policy, "LR", 8.0e-5)
    _set_default(dual_policy, "Eps", 1.0e-5)
    _set_default(dual_policy, "InitFromMainActor", True)

    main_reward = _ensure_node(dfd, "MainCostAwareReward")
    _set_default(main_reward, "Enable", True)
    _set_default(main_reward, "StartStep", 150000)
    _set_default(main_reward, "LambdaCost", 0.03)
    _set_default(main_reward, "UsePredictedCost", True)

    conf.freeze()
    return conf


def _resolve_force_scale(value):
    if isinstance(value, str):
        if value.lower() == "auto":
            print(
                colorama.Fore.YELLOW
                + "ForceLoss.ForceScale='auto' is not estimated online; using 1.0."
                + colorama.Style.RESET_ALL
            )
            return 1.0
        return float(value)
    return float(value)


def build_env(args, conf):
    from omni.isaac.lab_tasks.utils import parse_env_cfg

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
    cost_head = conf.DFD_V2.CostHead
    continuous_cost = conf.DFD_V2.ContinuousCost
    return DFDV2WorldModelWithContinuousCost(
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
        bool(cfg_get(cost_head, "Enable", True)),
        _as_int(cfg_get(cost_head, "HiddenDim", conf.Models.Hidden)),
        _as_int(cfg_get(cost_head, "Depth", 3)),
        _as_float(cfg_get(cost_head, "LossWeight", 2.0)),
        _as_float(cfg_get(cost_head, "HuberBeta", 0.02)),
        _as_float(cfg_get(cost_head, "SmallForceThreshold", 0.3)),
        _as_float(cfg_get(cost_head, "SmallCostThreshold", 0.05)),
        _as_float(cfg_get(cost_head, "SmallCostWeight", 2.0)),
        _as_float(cfg_get(cost_head, "ExtremeLossWeight", 0.5)),
        _as_float(cfg_get(cost_head, "ExtremeCostWeight", 4.0)),
        _as_float(cfg_get(continuous_cost, "ExtremeForceThreshold", 5.0)),
        _as_float(cfg_get(cost_head, "PriorLossWeight", 0.5)),
    ).to(device)


def load_dfd_v2_initial_checkpoint(path, *, world_model=None, agent=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    if world_model is not None:
        state = checkpoint.get("world_model_state_dict", checkpoint)
        cost_head_keys = [key for key in state.keys() if key.startswith("cost_head.")]
        has_v2_cost_head = (
            "cost_head.cost_logit.weight" in state
            and "cost_head.extreme_logit.weight" in state
        )
        if cost_head_keys and not has_v2_cost_head:
            state = {key: value for key, value in state.items() if not key.startswith("cost_head.")}
            print(
                colorama.Fore.YELLOW
                + "Skipped old checkpoint cost_head.* parameters because DFD v2 uses a new continuous normalized cost head."
                + colorama.Style.RESET_ALL
            )
        result = world_model.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith("cost_head.")]
        unexpected = [key for key in result.unexpected_keys if not key.startswith("cost_head.")]
        if missing or unexpected:
            raise RuntimeError(
                "DFD v2 checkpoint load had incompatible non-cost-head keys: "
                f"missing={missing}, unexpected={unexpected}"
            )
    if agent is not None:
        state = checkpoint.get("agent_state_dict", checkpoint)
        agent.load_state_dict(state)
    return checkpoint


def build_agent(conf, action_dim, act, device):
    return CostAwareActorCriticAgent(
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


def build_gd_critic(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    return GdRiskCritic.from_config(
        feat_dim,
        action_dim,
        conf.DFD_V2.Gd,
        use_amp=conf.BasicSettings.UseAmp,
        act=act,
        device=device,
        default_lr=_as_float(conf.Models.Agent.LR),
        default_eps=_as_float(conf.Models.Agent.Eps),
    )


def build_dual_policy(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    dual_cfg = conf.DFD_V2.DualPolicy
    return DualPolicyV2(
        action_dim=action_dim,
        feat_dim=feat_dim,
        hidden=_as_int(conf.Models.Hidden),
        min_std=_as_float(conf.Models.Agent.MinStd),
        max_std=_as_float(conf.Models.Agent.MaxStd),
        lr=_as_float(cfg_get(dual_cfg, "LR", conf.Models.Agent.LR)),
        eps=_as_float(cfg_get(dual_cfg, "Eps", conf.Models.Agent.Eps)),
        use_amp=conf.BasicSettings.UseAmp,
        act=act,
        device=device,
        max_grad_norm=_as_float(cfg_get(conf.DFD_V2.DualImagination, "GradClipNorm", 100.0)),
    ).to(device)


def _launch_isaac(headless=True):
    global simulation_app
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    import omni.isaac.lab_tasks  # noqa: F401
    import ur3_lite.tasks  # noqa: F401


def main():
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

    _launch_isaac(headless=True)
    conf = load_dfd_v2_config(args.config_path)
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
            "algorithm": "Dual-Imagination Continuous-Cost Dreamer v2",
            "checkpoint_path": checkpoint_path,
        },
    )
    print(colorama.Fore.CYAN + f"Run directory: {checkpoint_dir}" + colorama.Style.RESET_ALL)

    seed_np_torch(seed=args.seed)
    wandb_conf = getattr(conf, "Wandb", None)
    project = cfg_get(wandb_conf, "Project", "IsaacLab-PSSM-DFD-V2")
    run_group = cfg_get(wandb_conf, "Group", args.env_name)
    base_wandb_name = cfg_get(wandb_conf, "Name", f"DFD-v2-{args.env_name}-seed{args.seed}")
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
            "algorithm": "Dual-Imagination Continuous-Cost Dreamer v2",
            "checkpoint_path": checkpoint_path,
        },
    )

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)
    if checkpoint_path:
        load_dfd_v2_initial_checkpoint(checkpoint_path, world_model=world_model, agent=agent, map_location=args.device)
        if hasattr(agent, "sync_slow_critic"):
            agent.sync_slow_critic()
        print(colorama.Fore.YELLOW + f"Loaded DFD v2 initial checkpoint: {checkpoint_path}" + colorama.Style.RESET_ALL)

    gd_critic = build_gd_critic(conf, action_dim, act, args.device)
    dual_policy = build_dual_policy(conf, action_dim, act, args.device)
    if bool(cfg_get(conf.DFD_V2.DualPolicy, "InitFromMainActor", True)):
        dual_policy.initialize_from_main_actor(agent)
        print(colorama.Fore.YELLOW + "Initialized DFD v2 dual actor from the main actor." + colorama.Style.RESET_ALL)

    replay_buffer = DFDV2ReplayBuffer(
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
        joint_train_dfd_v2(
            args.env_name,
            args.n,
            vec_env,
            conf.JointTrainAgent.SampleMaxSteps,
            replay_buffer,
            world_model,
            agent,
            gd_critic,
            dual_policy,
            conf.DFD_V2,
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
        print(colorama.Fore.YELLOW + "DFD v2 training interrupted by Ctrl-C." + colorama.Style.RESET_ALL)
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
            print(colorama.Fore.YELLOW + f"Cleanup warning during simulation_app.close: {exc}" + colorama.Style.RESET_ALL)


if __name__ == "__main__":
    main()
