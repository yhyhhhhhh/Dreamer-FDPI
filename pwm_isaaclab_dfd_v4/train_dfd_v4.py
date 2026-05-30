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

try:
    import colorama
except ImportError:  # pragma: no cover
    class _EmptyColors:
        CYAN = RED = YELLOW = RESET_ALL = ""

    class _ColoramaFallback:
        Fore = _EmptyColors()
        Style = _EmptyColors()

    colorama = _ColoramaFallback()


simulation_app = None
torch = None
nn = None
wandb = None
gymnasium = None
CN = None
DreamerVecEnvWrapper = None
Logger = None
collect_training_info = None
make_unique_run_dir = None
save_run_artifacts = None
seed_np_torch = None
write_latest_run_pointer = None
DFDV2WorldModelWithContinuousCost = None
FDPIRegimeActorCriticAgent = None
cfg_get = None
DualPolicyV4 = None
DFDV4ReplayBuffer = None
GpRiskCritic = None
GdRiskCriticV4 = None
joint_train_dfd_v4 = None


def _load_training_deps():
    global torch, nn, wandb, gymnasium, CN
    global DreamerVecEnvWrapper, Logger, collect_training_info, make_unique_run_dir
    global save_run_artifacts, seed_np_torch, write_latest_run_pointer
    global DFDV2WorldModelWithContinuousCost, FDPIRegimeActorCriticAgent, cfg_get
    global DualPolicyV4, DFDV4ReplayBuffer, GpRiskCritic, GdRiskCriticV4, joint_train_dfd_v4

    import gymnasium as _gymnasium
    import torch as _torch
    import torch.nn as _nn
    import wandb as _wandb
    from yacs.config import CfgNode as _CN

    from pwm_isaaclab.env_wrapper import DreamerVecEnvWrapper as _DreamerVecEnvWrapper
    from pwm_isaaclab.utils import (
        Logger as _Logger,
        collect_training_info as _collect_training_info,
        make_unique_run_dir as _make_unique_run_dir,
        save_run_artifacts as _save_run_artifacts,
        seed_np_torch as _seed_np_torch,
        write_latest_run_pointer as _write_latest_run_pointer,
    )
    from pwm_isaaclab_dfd_v2.world_model_dfd_v2 import (
        DFDV2WorldModelWithContinuousCost as _DFDV2WorldModelWithContinuousCost,
    )
    from pwm_isaaclab_dfd_v4.agent_fdpi_regime import FDPIRegimeActorCriticAgent as _FDPIRegimeActorCriticAgent
    from pwm_isaaclab_dfd_v4.cost_utils import cfg_get as _cfg_get
    from pwm_isaaclab_dfd_v4.dual_policy_v4 import DualPolicyV4 as _DualPolicyV4
    from pwm_isaaclab_dfd_v4.replay_buffer_dfd_v4 import DFDV4ReplayBuffer as _DFDV4ReplayBuffer
    from pwm_isaaclab_dfd_v4.risk_critics import GdRiskCriticV4 as _GdRiskCriticV4
    from pwm_isaaclab_dfd_v4.risk_critics import GpRiskCritic as _GpRiskCritic
    from pwm_isaaclab_dfd_v4.trainer_dfd_v4 import joint_train_dfd_v4 as _joint_train_dfd_v4

    torch = _torch
    nn = _nn
    wandb = _wandb
    gymnasium = _gymnasium
    CN = _CN
    DreamerVecEnvWrapper = _DreamerVecEnvWrapper
    Logger = _Logger
    collect_training_info = _collect_training_info
    make_unique_run_dir = _make_unique_run_dir
    save_run_artifacts = _save_run_artifacts
    seed_np_torch = _seed_np_torch
    write_latest_run_pointer = _write_latest_run_pointer
    DFDV2WorldModelWithContinuousCost = _DFDV2WorldModelWithContinuousCost
    FDPIRegimeActorCriticAgent = _FDPIRegimeActorCriticAgent
    cfg_get = _cfg_get
    DualPolicyV4 = _DualPolicyV4
    DFDV4ReplayBuffer = _DFDV4ReplayBuffer
    GpRiskCritic = _GpRiskCritic
    GdRiskCriticV4 = _GdRiskCriticV4
    joint_train_dfd_v4 = _joint_train_dfd_v4


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


def load_dfd_v4_config(config_path):
    global CN
    if CN is None:
        from yacs.config import CfgNode as _CN

        CN = _CN
    conf = CN(new_allowed=True)
    conf.merge_from_file(config_path)
    conf.defrost()

    _ensure_node(conf, "Wandb")
    joint = _ensure_node(conf, "JointTrainAgent")
    _set_default(joint, "SaveOfflineEpisodes", False)
    _set_default(joint, "OfflineDatasetDir", "")

    fdpi = _ensure_node(conf, "FDPIRegimeDreamer")
    replay = _ensure_node(fdpi, "Replay")
    _set_default(replay, "cost_positive_ratio", 0.0)

    cost = _ensure_node(fdpi, "ContinuousCost")
    _set_default(cost, "Enable", True)
    _set_default(cost, "ForceThreshold", 0.1)
    _set_default(cost, "LowForceScale", 0.05)
    _set_default(cost, "CostForceMax", 15.0)
    _set_default(cost, "ForceScale", 5.0)
    _set_default(cost, "ExtremeForceThreshold", 5.0)
    _set_default(cost, "ClipCost", True)
    _set_default(cost, "CostMin", 0.0)
    _set_default(cost, "CostMax", 1.0)
    _set_default(cost, "BottomForceChannels", [2, 5])

    cost_head = _ensure_node(fdpi, "CostHead")
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

    risk = _ensure_node(fdpi, "RiskCritic")
    _set_default(risk, "GammaCost", 0.97)
    _set_default(risk, "RiskMax", 1.0)
    _set_default(risk, "TargetTau", 0.005)
    _set_default(risk, "Pf", 0.10)
    _set_default(risk, "Cg", 0.03)

    for name, dual_weight, high_weight in (("Gp", 1.0, 2.0), ("Gd", 2.0, 3.0)):
        node = _ensure_node(fdpi, name)
        _set_default(node, "Enable", True)
        _set_default(node, "GammaCost", risk.GammaCost)
        _set_default(node, "RiskMax", risk.RiskMax)
        _set_default(node, "TargetTau", risk.TargetTau)
        _set_default(node, "SourceAwareWeight", True)
        _set_default(node, "DualSourceWeight", dual_weight)
        _set_default(node, "HighCostWeight", high_weight)
        _set_default(node, "BoundaryWeight", 2.0)
        _set_default(node, "HighCostThreshold", 0.1)
        _set_default(node, "BoundaryLow", 0.05)
        _set_default(node, "BoundaryHigh", 0.4)
        _set_default(node, "SafetyCriticalRatio", 0.20 if name == "Gp" else 0.40)
        _set_default(node, "HiddenDim", 256)
        _set_default(node, "NumLayers", 2)
        _set_default(node, "LR", 1.0e-4)
        _set_default(node, "Eps", 1.0e-8)
        _set_default(node, "UpdateSteps", 1)

    main = _ensure_node(fdpi, "MainFDPIRegime")
    _set_default(main, "Enable", True)
    _set_default(main, "StartStep", 200000)
    _set_default(main, "LambdaCri", 0.02)
    _set_default(main, "LambdaInf", 0.05)
    _set_default(main, "WarmupSteps", 100000)
    _set_default(main, "EntropyCoef", 1.0e-4)

    dual_policy = _ensure_node(fdpi, "DualPolicy")
    _set_default(dual_policy, "LR", 8.0e-5)
    _set_default(dual_policy, "Eps", 1.0e-5)
    _set_default(dual_policy, "InitFromMainActor", True)

    dual_sampling = _ensure_node(fdpi, "DualSampling")
    _set_default(dual_sampling, "Enable", True)
    _set_default(dual_sampling, "StartStep", 100000)
    _set_default(dual_sampling, "FeasibleRatioWindow", 10000)
    _set_default(dual_sampling, "RatioFea95", 0.50)
    _set_default(dual_sampling, "RatioFea90", 0.35)
    _set_default(dual_sampling, "RatioFea80", 0.20)
    _set_default(dual_sampling, "RatioCriticalHigh", 0.15)
    _set_default(dual_sampling, "RatioUnsafeHigh", 0.05)
    _set_default(dual_sampling, "RatioDefault", 0.10)
    _set_default(dual_sampling, "MaxKLForSampling", 2.0)
    _set_default(dual_sampling, "HighMainCostRate", 0.20)
    _set_default(dual_sampling, "MaxRatioWhenMainCostHigh", 0.10)

    dual_update = _ensure_node(fdpi, "DualUpdate")
    _set_default(dual_update, "Enable", True)
    _set_default(dual_update, "StartStep", 100000)
    _set_default(dual_update, "Type", "imagined_risk_return")
    _set_default(dual_update, "Horizon", 5)
    _set_default(dual_update, "GammaCost", risk.GammaCost)
    _set_default(dual_update, "KLCoeff", 1.0)
    _set_default(dual_update, "EntropyCoef", 1.0e-4)
    _set_default(dual_update, "GradClipNorm", 100.0)
    _set_default(dual_update, "UpdateSteps", 1)

    wm_sampling = _ensure_node(fdpi, "WorldModelSampling")
    _set_default(wm_sampling, "EnableSafetyCriticalSampling", True)
    _set_default(wm_sampling, "UniformRatio", 0.80)
    _set_default(wm_sampling, "SafetyCriticalRatio", 0.20)
    _set_default(wm_sampling, "HighCostThreshold", 0.1)
    _set_default(wm_sampling, "BoundaryLow", 0.05)
    _set_default(wm_sampling, "BoundaryHigh", 0.4)

    conf.freeze()
    return conf


def _resolve_force_scale(value):
    if isinstance(value, str):
        if value.lower() == "auto":
            print(colorama.Fore.YELLOW + "ForceLoss.ForceScale='auto' is not estimated online; using 1.0." + colorama.Style.RESET_ALL)
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
    env_cfg = parse_env_cfg(args.env_name, device=args.device, num_envs=num_envs, use_fabric=use_fabric)
    env_cfg.seed = env_seed
    env = gymnasium.make(args.env_name, cfg=env_cfg)
    return DreamerVecEnvWrapper(env, device=args.device)


def build_world_model(conf, obs_dim, action_dim, act, device):
    force_enabled = bool(getattr(conf.ForceHead, "Enable", False))
    cost_head = conf.FDPIRegimeDreamer.CostHead
    continuous_cost = conf.FDPIRegimeDreamer.ContinuousCost
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


def build_agent(conf, action_dim, act, device):
    return FDPIRegimeActorCriticAgent(
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


def build_gp_critic(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    return GpRiskCritic.from_config(
        feat_dim,
        action_dim,
        conf.FDPIRegimeDreamer.Gp,
        use_amp=conf.BasicSettings.UseAmp,
        act=act,
        device=device,
        default_lr=_as_float(conf.Models.Agent.LR),
        default_eps=_as_float(conf.Models.Agent.Eps),
    )


def build_gd_critic(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    return GdRiskCriticV4.from_config(
        feat_dim,
        action_dim,
        conf.FDPIRegimeDreamer.Gd,
        use_amp=conf.BasicSettings.UseAmp,
        act=act,
        device=device,
        default_lr=_as_float(conf.Models.Agent.LR),
        default_eps=_as_float(conf.Models.Agent.Eps),
    )


def build_dual_policy(conf, action_dim, act, device):
    feat_dim = _as_int(conf.Models.Stoch) * _as_int(conf.Models.Discrete) + _as_int(conf.Models.Hidden)
    dual_cfg = conf.FDPIRegimeDreamer.DualPolicy
    return DualPolicyV4(
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
        max_grad_norm=_as_float(cfg_get(conf.FDPIRegimeDreamer.DualUpdate, "GradClipNorm", 100.0)),
    ).to(device)


def _launch_isaac(headless=True):
    global simulation_app
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    import omni.isaac.lab_tasks  # noqa: F401
    import ur3_lite.tasks  # noqa: F401


def _report_exception(exc, checkpoint_dir=None):
    message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(colorama.Fore.RED + "\nDFD v4 training failed with exception:\n" + message + colorama.Style.RESET_ALL)
    if checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            error_path = os.path.join(checkpoint_dir, "dfd_v4_error.log")
            with open(error_path, "w", encoding="utf-8") as fout:
                fout.write(message)
            print(colorama.Fore.RED + f"Saved DFD v4 error report: {error_path}" + colorama.Style.RESET_ALL)
        except Exception as report_exc:
            print(colorama.Fore.YELLOW + f"Could not save DFD v4 error report: {report_exc}" + colorama.Style.RESET_ALL)


def main():
    warnings.filterwarnings("ignore")
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
    _load_training_deps()
    torch.backends.cudnn.benchmark = False
    conf = load_dfd_v4_config(args.config_path)
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path)) if args.checkpoint_path else None
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
            "algorithm": "FDPI-Regime Dreamer v4",
            "checkpoint_path": checkpoint_path,
        },
    )

    seed_np_torch(seed=args.seed)
    wandb_conf = getattr(conf, "Wandb", None)
    project = cfg_get(wandb_conf, "Project", "IsaacLab-PSSM-DFD-V4")
    run_group = cfg_get(wandb_conf, "Group", args.env_name)
    base_wandb_name = cfg_get(wandb_conf, "Name", f"DFD-v4-{args.env_name}-seed{args.seed}")
    run_name = f"{base_wandb_name}-{os.path.basename(checkpoint_dir)}"
    init_kwargs = {
        "project": project,
        "group": run_group,
        "name": run_name,
        "dir": checkpoint_dir,
        "config": _cfg_to_dict(conf),
    }
    wandb_mode = cfg_get(wandb_conf, "Mode", None)
    if wandb_mode is not None:
        init_kwargs["mode"] = wandb_mode
    if run_info.get("note"):
        init_kwargs["notes"] = run_info["note"]
    if run_info.get("tags"):
        init_kwargs["tags"] = run_info["tags"]
    wandb.init(**init_kwargs)
    logger = Logger()

    vec_env = build_env(args, conf)
    obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
    action_dim = int(vec_env.single_action_space.shape[0])
    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        world_model.load_state_dict(checkpoint.get("world_model_state_dict", checkpoint), strict=False)
        agent.load_state_dict(checkpoint.get("agent_state_dict", checkpoint), strict=False)
        if hasattr(agent, "sync_slow_critic"):
            agent.sync_slow_critic()

    gp_critic = build_gp_critic(conf, action_dim, act, args.device)
    gd_critic = build_gd_critic(conf, action_dim, act, args.device)
    dual_policy = build_dual_policy(conf, action_dim, act, args.device)
    if bool(cfg_get(conf.FDPIRegimeDreamer.DualPolicy, "InitFromMainActor", True)):
        dual_policy.initialize_from_main_actor(agent)

    replay_buffer = DFDV4ReplayBuffer(
        obs_dim,
        action_dim,
        vec_env.num_envs,
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
        joint_train_dfd_v4(
            args.env_name,
            args.n,
            vec_env,
            conf.JointTrainAgent.SampleMaxSteps,
            replay_buffer,
            world_model,
            agent,
            gp_critic,
            gd_critic,
            dual_policy,
            conf.FDPIRegimeDreamer,
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
    except Exception as exc:
        _report_exception(exc, checkpoint_dir)
        raise
    finally:
        try:
            if logger.log_dict and logger.tot_step >= 0:
                wandb.log(logger.log_dict, step=logger.tot_step)
            wandb.finish()
        finally:
            try:
                vec_env.close()
            finally:
                if simulation_app is not None:
                    simulation_app.close()


if __name__ == "__main__":
    main()
