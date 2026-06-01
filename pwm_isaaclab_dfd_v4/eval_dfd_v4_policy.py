from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


simulation_app = None


def _launch_isaac(headless=True):
    global simulation_app
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    import omni.isaac.lab_tasks  # noqa: F401
    import ur3_lite.tasks  # noqa: F401


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, value):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sq_sum += float(np.square(arr).sum())
        self.min = min(self.min, float(arr.min()))
        self.max = max(self.max, float(arr.max()))

    def as_dict(self):
        if self.count <= 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.sum / self.count
        var = max(self.sq_sum / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min,
            "max": self.max,
        }


INFO_KEEP_TOKENS = (
    "reward",
    "success",
    "failure",
    "timeout",
    "force",
    "cost",
    "lift",
    "goal",
    "contact",
)
INFO_SKIP_KEYS = {"terminal_observation"}


def _is_numeric_array(value):
    return np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)


def _to_numpy(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, bool, np.number, np.bool_)):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            return None
    return None


def _flatten_info(prefix, value, out):
    if value is None:
        return
    if isinstance(value, dict):
        for key, sub_value in value.items():
            if key in INFO_SKIP_KEYS:
                continue
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_info(next_prefix, sub_value, out)
        return
    if not any(token in prefix.lower() for token in INFO_KEEP_TOKENS):
        return
    arr = _to_numpy(value)
    if arr is None or arr.ndim == 0 and not _is_numeric_array(arr):
        return
    if not _is_numeric_array(arr):
        return
    out[prefix].add(arr)


def _cfg_get(node, name, default=None):
    if node is None:
        return default
    if hasattr(node, name):
        return getattr(node, name)
    if isinstance(node, dict):
        return node.get(name, default)
    return default


def _set_eval_num_envs(conf, num_envs):
    if num_envs is None:
        return conf
    conf.defrost()
    if not hasattr(conf, "Env"):
        from yacs.config import CfgNode as CN

        conf.Env = CN(new_allowed=True)
    if not hasattr(conf.Env, "MakeKwargs"):
        from yacs.config import CfgNode as CN

        conf.Env.MakeKwargs = CN(new_allowed=True)
    conf.Env.MakeKwargs.num_envs = int(num_envs)
    conf.JointTrainAgent.NumEnvs = int(num_envs)
    conf.freeze()
    return conf


def _episode_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _sample_policy(policy_module, feat, greedy):
    return policy_module.sample(feat, greedy=greedy)


def _apply_gp_shield(
    *,
    feat,
    action,
    policy_module,
    gp_critic,
    greedy,
    enabled,
    threshold,
    candidates,
    min_improvement,
):
    import torch

    base_risk = gp_critic.risk_no_grad(feat, action, clamp=True).reshape(action.shape[0])
    stats = {
        "applied": torch.zeros_like(base_risk),
        "base_risk": base_risk,
        "selected_risk": base_risk,
    }
    if not enabled or int(candidates) <= 0:
        return action, base_risk, stats

    unsafe = base_risk >= float(threshold)
    if not bool(unsafe.any().item()):
        return action, base_risk, stats

    candidate_actions = [action]
    candidate_risks = [base_risk]
    for _ in range(int(candidates)):
        cand_action = _sample_policy(policy_module, feat, greedy=False)
        cand_action = cand_action.to(device=action.device, dtype=action.dtype)
        cand_risk = gp_critic.risk_no_grad(feat, cand_action, clamp=True).reshape(action.shape[0])
        candidate_actions.append(cand_action)
        candidate_risks.append(cand_risk)

    stacked_actions = torch.stack(candidate_actions, dim=0)
    stacked_risks = torch.stack(candidate_risks, dim=0)
    best_risk, best_idx = stacked_risks.min(dim=0)
    env_idx = torch.arange(action.shape[0], device=action.device)
    best_action = stacked_actions[best_idx, env_idx]
    replace = unsafe & (best_risk <= base_risk - float(min_improvement))
    shielded_action = action.clone()
    shielded_action[replace] = best_action[replace]
    selected_risk = torch.where(replace, best_risk, base_risk)
    stats = {
        "applied": replace.float(),
        "base_risk": base_risk,
        "selected_risk": selected_risk,
    }
    return shielded_action, selected_risk, stats


def _save_report(save_dir, result, checkpoint_path):
    os.makedirs(save_dir, exist_ok=True)
    step_text = "unknown"
    name = os.path.basename(checkpoint_path)
    if name.startswith("full_state_v4_") and name.endswith(".pth"):
        step_text = name[len("full_state_v4_") : -len(".pth")]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_dir, f"dfd_v4_eval_{step_text}_{stamp}.json")
    md_path = os.path.join(save_dir, f"dfd_v4_eval_{step_text}_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    summary = result["summary"]
    lines = [
        "# DFD v4 Policy Evaluation",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- policy: `{result['policy']}`",
        f"- greedy: `{result['greedy']}`",
        f"- gp_shield: `{result.get('gp_shield', False)}`",
        f"- num_envs: `{result['num_envs']}`",
        f"- env_steps: `{result['env_steps']}`",
        f"- episodes_completed: `{summary['episodes_completed']}`",
        "",
        "## Task",
        "",
        f"- episode_return_mean: `{summary['episode_return_mean']}`",
        f"- success_rate: `{summary['success_rate']}`",
        f"- failure_rate: `{summary['failure_rate']}`",
        f"- timeout_rate: `{summary['timeout_rate']}`",
        "",
        "## Safety",
        "",
        f"- continuous_cost_mean: `{summary['continuous_cost_mean']}`",
        f"- cost_positive_rate: `{summary['cost_positive_rate']}`",
        f"- high_cost_rate: `{summary['high_cost_rate']}`",
        f"- extreme_cost_rate: `{summary['extreme_cost_rate']}`",
        f"- bottom_force_mean: `{summary['bottom_force_mean']}`",
        f"- bottom_force_max: `{summary['bottom_force_max']}`",
        f"- episode_cost_mean: `{summary['episode_cost_mean']}`",
        "",
        "## GP Regime",
        "",
        f"- gp_mean: `{summary['gp_mean']}`",
        f"- gp_pre_shield_mean: `{summary.get('gp_pre_shield_mean')}`",
        f"- shield_applied_rate: `{summary.get('shield_applied_rate')}`",
        f"- gp_feasible_ratio: `{summary['gp_feasible_ratio']}`",
        f"- gp_critical_ratio: `{summary['gp_critical_ratio']}`",
        f"- gp_infeasible_ratio: `{summary['gp_infeasible_ratio']}`",
        "",
        f"JSON report: `{json_path}`",
    ]
    with open(md_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")
    return json_path, md_path


def evaluate_policy(args):
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    from pwm_isaaclab.trainer import _is_first, _policy_obs, _reset_after_step
    from pwm_isaaclab_dfd_v4.cost_utils import extract_continuous_cost
    from pwm_isaaclab_dfd_v4.train_dfd_v4 import (
        _cfg_to_dict,
        _load_training_deps,
        _load_v4_full_checkpoint,
        build_agent,
        build_dual_policy,
        build_env,
        build_gd_critic,
        build_gp_critic,
        build_world_model,
        load_dfd_v4_config,
    )

    _load_training_deps()
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    conf = load_dfd_v4_config(args.config_path)
    conf = _set_eval_num_envs(conf, args.num_envs)
    if args.seed is not None:
        conf.defrost()
        conf.BasicSettings.Seed = int(args.seed)
        if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
            conf.Env.MakeKwargs.seed = int(args.seed)
        conf.freeze()

    torch.manual_seed(int(conf.BasicSettings.Seed))
    np.random.seed(int(conf.BasicSettings.Seed))

    vec_env = build_env(args, conf)
    try:
        obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
        action_dim = int(vec_env.single_action_space.shape[0])
        act = getattr(nn, conf.Models.Act)

        world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
        agent = build_agent(conf, action_dim, act, args.device)
        gp_critic = build_gp_critic(conf, action_dim, act, args.device)
        gd_critic = build_gd_critic(conf, action_dim, act, args.device)
        dual_policy = build_dual_policy(conf, action_dim, act, args.device)

        _load_v4_full_checkpoint(
            args.v4_full_checkpoint_path,
            world_model=world_model,
            agent=agent,
            gp_critic=gp_critic,
            gd_critic=gd_critic,
            dual_policy=dual_policy,
            replay_buffer=None,
            device=args.device,
            load_optimizer=False,
            load_replay_buffer=False,
            load_rng=False,
        )

        world_model.eval()
        agent.eval()
        gp_critic.eval()
        gd_critic.eval()
        dual_policy.eval()

        fdpi_cfg = conf.FDPIRegimeDreamer
        cost_cfg = fdpi_cfg.ContinuousCost
        risk_cfg = fdpi_cfg.RiskCritic
        wm_sampling_cfg = fdpi_cfg.WorldModelSampling
        pf = float(_cfg_get(risk_cfg, "Pf", 0.40))
        cg = float(_cfg_get(risk_cfg, "Cg", 0.10))
        high_cost_threshold = float(_cfg_get(wm_sampling_cfg, "HighCostThreshold", 0.1))
        bottom_channels = tuple(int(v) for v in _cfg_get(cost_cfg, "BottomForceChannels", [2, 5]))
        shield_threshold = float(args.shield_threshold if args.shield_threshold is not None else pf)

        num_envs = int(vec_env.num_envs)
        state = world_model.initial(num_envs)
        current_obs_dict = vec_env.reset()
        current_obs = _policy_obs(current_obs_dict).to(args.device)
        is_first = _is_first(current_obs_dict, num_envs, args.device)

        step_stats = defaultdict(RunningStats)
        info_stats = defaultdict(RunningStats)
        episode_returns = []
        episode_costs = []
        episode_cost_means = []
        episode_lengths = []
        episode_bottom_force_means = []
        episode_bottom_force_peaks = []
        episode_successes = 0
        episode_failures = 0
        episode_timeouts = 0
        episodes_completed = 0

        ep_return = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        ep_cost = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        ep_bottom_force = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        ep_bottom_force_peak = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        ep_len = torch.zeros(num_envs, dtype=torch.float32, device=args.device)

        total_iters = max(int(math.ceil(float(args.eval_steps) / float(num_envs))), 1)
        if args.eval_episodes is not None:
            total_iters = max(total_iters, 1)

        progress = tqdm(total=total_iters, desc="Evaluating DFD v4 policy")
        iter_idx = 0
        while True:
            if args.eval_episodes is None and iter_idx >= total_iters:
                break
            if args.eval_episodes is not None and episodes_completed >= int(args.eval_episodes):
                break

            with torch.no_grad():
                feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                if args.policy == "dual":
                    policy_module = dual_policy
                else:
                    policy_module = agent
                action = _sample_policy(policy_module, feat, greedy=args.greedy)
                action = action.to(device=args.device, dtype=torch.float32)
                action, gp, shield_stats = _apply_gp_shield(
                    feat=feat,
                    action=action,
                    policy_module=policy_module,
                    gp_critic=gp_critic,
                    greedy=args.greedy,
                    enabled=bool(args.gp_shield),
                    threshold=shield_threshold,
                    candidates=args.shield_candidates,
                    min_improvement=args.shield_min_improvement,
                )
                gd = gd_critic.risk_no_grad(feat, action, clamp=True).reshape(num_envs)
                state = world_model.update_inference_state(state, action)
                env_action = action.detach().cpu().numpy()

            next_obs_dict, reward, done, info = vec_env.step(env_action)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=args.device).view(num_envs)
            done = torch.as_tensor(done, dtype=torch.bool, device=args.device).view(num_envs)

            cost_parts = extract_continuous_cost(
                info,
                next_obs_dict,
                num_envs=num_envs,
                device=args.device,
                force_threshold=float(_cfg_get(cost_cfg, "ForceThreshold", 0.1)),
                low_force_scale=float(_cfg_get(cost_cfg, "LowForceScale", 0.05)),
                cost_force_max=float(_cfg_get(cost_cfg, "CostForceMax", 15.0)),
                force_scale=float(_cfg_get(cost_cfg, "ForceScale", 5.0)),
                extreme_force_threshold=float(_cfg_get(cost_cfg, "ExtremeForceThreshold", 5.0)),
                clip_cost=bool(_cfg_get(cost_cfg, "ClipCost", True)),
                cost_min=float(_cfg_get(cost_cfg, "CostMin", 0.0)),
                cost_max=float(_cfg_get(cost_cfg, "CostMax", 1.0)),
                force_key=str(getattr(conf.ForceHead, "Key", "")),
                bottom_force_channels=bottom_channels,
            )
            continuous_cost = cost_parts["continuous_cost"].view(num_envs)
            binary_cost = cost_parts["binary_cost"].view(num_envs)
            extreme_cost = cost_parts["extreme_cost"].view(num_envs)
            bottom_force = cost_parts["bottom_force"].view(num_envs)
            force_excess = cost_parts["force_excess"].view(num_envs)

            step_stats["reward"].add(reward.detach().cpu().numpy())
            step_stats["continuous_cost"].add(continuous_cost.detach().cpu().numpy())
            step_stats["binary_cost"].add(binary_cost.detach().cpu().numpy())
            step_stats["high_cost"].add((continuous_cost > high_cost_threshold).float().detach().cpu().numpy())
            step_stats["extreme_cost"].add(extreme_cost.detach().cpu().numpy())
            step_stats["bottom_force"].add(bottom_force.detach().cpu().numpy())
            step_stats["force_excess"].add(force_excess.detach().cpu().numpy())
            step_stats["gp_pre_shield"].add(shield_stats["base_risk"].detach().cpu().numpy())
            step_stats["gp"].add(gp.detach().cpu().numpy())
            step_stats["gd"].add(gd.detach().cpu().numpy())
            step_stats["shield_applied"].add(shield_stats["applied"].detach().cpu().numpy())
            step_stats["gp_feasible"].add((gp < (pf - cg)).float().detach().cpu().numpy())
            step_stats["gp_critical"].add(((gp >= (pf - cg)) & (gp < pf)).float().detach().cpu().numpy())
            step_stats["gp_infeasible"].add((gp >= pf).float().detach().cpu().numpy())

            info_for_stats = dict(info) if isinstance(info, dict) else {}
            info_for_stats.setdefault("reward", reward)
            _flatten_info("Info", info_for_stats, info_stats)

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
            episode_success = info.get("episode_success") if isinstance(info, dict) else None
            if episode_success is None:
                episode_success = terminal & ~failure
            else:
                episode_success = torch.as_tensor(episode_success, dtype=torch.bool, device=args.device).view(-1)
            episode_failure = info.get("episode_failure") if isinstance(info, dict) else None
            if episode_failure is None:
                episode_failure = terminal & failure
            else:
                episode_failure = torch.as_tensor(episode_failure, dtype=torch.bool, device=args.device).view(-1)
            episode_timeout = info.get("episode_timeout") if isinstance(info, dict) else None
            if episode_timeout is None:
                episode_timeout = done & ~terminal
            else:
                episode_timeout = torch.as_tensor(episode_timeout, dtype=torch.bool, device=args.device).view(-1)

            ep_return += reward
            ep_cost += continuous_cost
            ep_bottom_force += bottom_force
            ep_bottom_force_peak = torch.maximum(ep_bottom_force_peak, bottom_force)
            ep_len += 1.0

            if done.any():
                done_indices = torch.nonzero(done, as_tuple=False).flatten()
                episodes_completed += int(done_indices.numel())
                episode_successes += int(episode_success[done_indices].sum().item())
                episode_failures += int(episode_failure[done_indices].sum().item())
                episode_timeouts += int(episode_timeout[done_indices].sum().item())
                for idx in done_indices.tolist():
                    length = max(float(ep_len[idx].item()), 1.0)
                    episode_returns.append(float(ep_return[idx].item()))
                    episode_costs.append(float(ep_cost[idx].item()))
                    episode_cost_means.append(float(ep_cost[idx].item()) / length)
                    episode_lengths.append(length)
                    episode_bottom_force_means.append(float(ep_bottom_force[idx].item()) / length)
                    episode_bottom_force_peaks.append(float(ep_bottom_force_peak[idx].item()))
                    ep_return[idx] = 0.0
                    ep_cost[idx] = 0.0
                    ep_bottom_force[idx] = 0.0
                    ep_bottom_force_peak[idx] = 0.0
                    ep_len[idx] = 0.0

            current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, args.device)
            iter_idx += 1
            progress.update(1)
            if args.eval_episodes is not None and episodes_completed >= int(args.eval_episodes):
                break
            if args.max_iters is not None and iter_idx >= int(args.max_iters):
                break
        progress.close()

        step_dict = {key: stat.as_dict() for key, stat in step_stats.items()}
        info_dict = {key: stat.as_dict() for key, stat in info_stats.items()}
        env_steps = int(iter_idx * num_envs)
        summary = {
            "env_steps": env_steps,
            "episodes_completed": int(episodes_completed),
            "success_rate": episode_successes / max(episodes_completed, 1),
            "failure_rate": episode_failures / max(episodes_completed, 1),
            "timeout_rate": episode_timeouts / max(episodes_completed, 1),
            "episode_return_mean": _episode_summary(episode_returns)["mean"],
            "episode_return": _episode_summary(episode_returns),
            "episode_length": _episode_summary(episode_lengths),
            "episode_cost_mean": _episode_summary(episode_cost_means)["mean"],
            "episode_cost": _episode_summary(episode_costs),
            "episode_bottom_force_mean": _episode_summary(episode_bottom_force_means)["mean"],
            "episode_bottom_force_peak": _episode_summary(episode_bottom_force_peaks),
            "continuous_cost_mean": step_dict["continuous_cost"]["mean"],
            "cost_positive_rate": step_dict["binary_cost"]["mean"],
            "high_cost_rate": step_dict["high_cost"]["mean"],
            "extreme_cost_rate": step_dict["extreme_cost"]["mean"],
            "bottom_force_mean": step_dict["bottom_force"]["mean"],
            "bottom_force_max": step_dict["bottom_force"]["max"],
            "force_excess_mean": step_dict["force_excess"]["mean"],
            "gp_mean": step_dict["gp"]["mean"],
            "gp_pre_shield_mean": step_dict["gp_pre_shield"]["mean"],
            "gd_mean": step_dict["gd"]["mean"],
            "shield_applied_rate": step_dict["shield_applied"]["mean"],
            "gp_feasible_ratio": step_dict["gp_feasible"]["mean"],
            "gp_critical_ratio": step_dict["gp_critical"]["mean"],
            "gp_infeasible_ratio": step_dict["gp_infeasible"]["mean"],
        }
        result = {
            "checkpoint_path": os.path.abspath(args.v4_full_checkpoint_path),
            "config_path": os.path.abspath(args.config_path),
            "env_name": args.env_name,
            "device": args.device,
            "policy": args.policy,
            "greedy": bool(args.greedy),
            "gp_shield": bool(args.gp_shield),
            "shield_threshold": shield_threshold,
            "shield_candidates": int(args.shield_candidates),
            "shield_min_improvement": float(args.shield_min_improvement),
            "seed": int(conf.BasicSettings.Seed),
            "num_envs": num_envs,
            "eval_steps_requested": int(args.eval_steps),
            "env_steps": env_steps,
            "config": _cfg_to_dict(conf) if args.save_config else None,
            "summary": summary,
            "step_stats": step_dict,
            "info_stats": info_dict,
        }
        json_path, md_path = _save_report(args.save_dir, result, args.v4_full_checkpoint_path)
        print("\nDFD v4 policy evaluation summary")
        print(f"  checkpoint: {args.v4_full_checkpoint_path}")
        print(f"  policy: {args.policy}, greedy={bool(args.greedy)}, num_envs={num_envs}")
        print(f"  env_steps: {env_steps}, episodes: {episodes_completed}")
        print(f"  success_rate: {summary['success_rate']:.4f}")
        print(f"  episode_return_mean: {summary['episode_return_mean']}")
        print(f"  continuous_cost_mean: {summary['continuous_cost_mean']}")
        print(f"  cost_positive_rate: {summary['cost_positive_rate']}")
        print(f"  high_cost_rate: {summary['high_cost_rate']}")
        print(f"  extreme_cost_rate: {summary['extreme_cost_rate']}")
        print(f"  bottom_force_mean/max: {summary['bottom_force_mean']} / {summary['bottom_force_max']}")
        print(f"  shield_applied_rate: {summary['shield_applied_rate']}")
        print(f"  gp pre/post shield: {summary['gp_pre_shield_mean']} / {summary['gp_mean']}")
        print(f"  gp fea/cri/inf: {summary['gp_feasible_ratio']} / {summary['gp_critical_ratio']} / {summary['gp_infeasible_ratio']}")
        print(f"  saved json: {json_path}")
        print(f"  saved md: {md_path}")
        return result
    finally:
        vec_env.close()


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Evaluate a DFD v4 full-state checkpoint in IsaacLab.")
    parser.add_argument(
        "--v4_full_checkpoint_path",
        type=str,
        required=True,
        help="Path to full_state_v4_*.pth.",
    )
    parser.add_argument(
        "-config_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "pwm_isaaclab_dfd_v4", "config_dfd_v4.yaml"),
    )
    parser.add_argument(
        "-env_name",
        type=str,
        default="Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1",
    )
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=65536, help="Total vectorized env steps to collect.")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Stop after this many completed episodes.")
    parser.add_argument("--max_iters", type=int, default=None, help="Hard cap on vector env iterations.")
    parser.add_argument("--policy", choices=("main", "dual"), default="main")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic sampling instead of greedy mean action.")
    parser.add_argument("--gp_shield", action="store_true", help="Replace high-GP-risk actions with lower-risk sampled candidates.")
    parser.add_argument("--shield_threshold", type=float, default=None, help="GP threshold for applying the shield; defaults to RiskCritic.Pf.")
    parser.add_argument("--shield_candidates", type=int, default=16, help="Number of stochastic candidate actions for GP shield.")
    parser.add_argument("--shield_min_improvement", type=float, default=0.0, help="Minimum GP-risk improvement required to replace an action.")
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "eval_results", "dfd_v4"))
    parser.add_argument("--save_config", action="store_true")
    parser.add_argument("--render", action="store_true", help="Launch non-headless IsaacLab window.")
    args = parser.parse_args()
    args.greedy = not args.stochastic

    if not os.path.isfile(os.path.abspath(os.path.expanduser(args.v4_full_checkpoint_path))):
        raise FileNotFoundError(f"Checkpoint not found: {args.v4_full_checkpoint_path}")
    if not os.path.isfile(os.path.abspath(os.path.expanduser(args.config_path))):
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    _launch_isaac(headless=not args.render)
    try:
        evaluate_policy(args)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
