from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ur3_fdpi.yaml"


def add_isaaclab_extensions() -> None:
    for isaaclab_root in (
        os.environ.get("ISAACLAB_ROOT"),
        "/home/yhy/IsaacLab-1.4.0",
        "/home/yhy/IsaacLab",
    ):
        if not isaaclab_root:
            continue
        root = Path(isaaclab_root).expanduser()
        for extension in ("omni.isaac.lab", "omni.isaac.lab_tasks", "omni.isaac.lab_assets"):
            path = root / "source" / "extensions" / extension
            if path.exists() and str(path) not in sys.path:
                sys.path.append(str(path))


def preparse_config() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args, _ = parser.parse_known_args()
    path = Path(args.config).expanduser()
    if not path.is_absolute():
        project_path = PROJECT_ROOT / path
        cwd_path = Path.cwd() / path
        path = project_path if project_path.exists() else cwd_path
    return path


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    flat = {}
    for group in ("common", "algorithm", "fdpi", "cost", "train", "logging"):
        values = raw.get(group, {})
        if values:
            flat.update(values)
    flat["config"] = str(path)
    return flat


add_isaaclab_extensions()
from omni.isaac.lab.app import AppLauncher  # noqa: E402


def parse_args():
    config_path = preparse_config()
    defaults = load_config(config_path)

    parser = argparse.ArgumentParser(description="Minimal FDPI training script for IsaacLab DirectRLEnv tasks.")
    parser.add_argument("--config", type=str, default=str(config_path))
    parser.add_argument("--task", type=str)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--start_steps", type=int)
    parser.add_argument("--buffer_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--sample_per_iteration", type=int)
    parser.add_argument("--updates_per_iteration", type=int)
    parser.add_argument("--log_every_steps", type=int)
    parser.add_argument("--save_every_steps", type=int)
    parser.add_argument("--max_checkpoints", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--hidden_num", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--cost_gamma", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--target_entropy", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--target_kl", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--dual_thresh", type=float)
    parser.add_argument("--cost_source", choices=("auto", "force_fail", "contact_force"))
    parser.add_argument("--pipe_force_cost_limit", type=float)
    parser.add_argument("--bottom_force_cost_limit", type=float)
    parser.add_argument("--continuous_contact_cost", action="store_true", default=None)
    parser.add_argument("--binary_contact_cost", dest="continuous_contact_cost", action="store_false")
    parser.add_argument("--allow_missing_terminal_observation", action="store_true", default=None)
    parser.add_argument("--require_terminal_observation", dest="allow_missing_terminal_observation", action="store_false")
    parser.add_argument("--disable_fabric", action="store_true", default=None)
    parser.add_argument("--use_fabric", dest="disable_fabric", action="store_false")
    parser.add_argument("--enable_gui", action="store_true", default=False)
    parser.add_argument("--ur3_extension_path", type=str)
    parser.add_argument("--log_root", type=str)
    parser.add_argument("--note", type=str)

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.enable_gui:
        args.headless = False
    if args.num_envs % 2 != 0:
        raise ValueError(f"--num_envs must be even for FDPI dual sampling, got {args.num_envs}.")
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch

    if args_cli.ur3_extension_path:
        ur3_path = Path(args_cli.ur3_extension_path).expanduser()
        if ur3_path.exists() and str(ur3_path) not in sys.path:
            sys.path.insert(0, str(ur3_path))

    import ur3_lite  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg

    from fdpi import FDPIIsaacLabTrainer, TorchReplayBufferIS, TorchSACFPIDual, dump_json, policy_obs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        reset_out = env.reset(seed=args_cli.seed)
        obs_dict = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        device = torch.device(env.unwrapped.device)
        obs_dim = int(policy_obs(obs_dict, device).shape[-1])
        act_dim = int(gym.spaces.flatdim(env.unwrapped.single_action_space))
        hidden_sizes = [args_cli.hidden_dim] * args_cli.hidden_num

        algorithm = TorchSACFPIDual(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            device=device,
            gamma=args_cli.gamma,
            cost_gamma=args_cli.cost_gamma,
            lr=args_cli.lr,
            max_grad_norm=args_cli.max_grad_norm,
            tau=args_cli.tau,
            target_entropy=args_cli.target_entropy,
            pf=args_cli.epsilon,
            target_kl=args_cli.target_kl,
        )
        buffer = TorchReplayBufferIS(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=args_cli.buffer_size,
            device=device,
        )

        run_name = "fdpi_min"
        if args_cli.note:
            run_name += f"_{args_cli.note}"
        run_name += f"_seed{args_cli.seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        log_dir = Path(args_cli.log_root).expanduser() / args_cli.task / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        dump_json(log_dir / "config.json", vars(args_cli) | {"obs_dim": obs_dim, "act_dim": act_dim})

        print(f"[FDPI-MIN] log_dir={log_dir}", flush=True)
        print(f"[FDPI-MIN] obs_dim={obs_dim} act_dim={act_dim} device={device}", flush=True)

        trainer = FDPIIsaacLabTrainer(
            env=env,
            algorithm=algorithm,
            buffer=buffer,
            log_dir=log_dir,
            total_steps=args_cli.total_steps,
            start_steps=args_cli.start_steps,
            batch_size=args_cli.batch_size,
            beta=args_cli.beta,
            dual_thresh=args_cli.dual_thresh,
            sample_per_iteration=args_cli.sample_per_iteration,
            updates_per_iteration=args_cli.updates_per_iteration,
            log_every_steps=args_cli.log_every_steps,
            save_every_steps=args_cli.save_every_steps,
            max_checkpoints=args_cli.max_checkpoints,
            require_terminal_observation=not args_cli.allow_missing_terminal_observation,
            cost_source=args_cli.cost_source,
            pipe_force_cost_limit=args_cli.pipe_force_cost_limit,
            bottom_force_cost_limit=args_cli.bottom_force_cost_limit,
            continuous_contact_cost=args_cli.continuous_contact_cost,
        )
        trainer.train(args_cli.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
