"""Warm up the PaMoRL reward/value critic while keeping actor and world model frozen.

Example:
    WANDB_MODE=offline python pwm_isaaclab/critic_warmup.py \
      -n ur3-critic-warmup \
      -seed 0 \
      -config_path pwm_isaaclab/config_files/PWM_expert_init.yaml \
      -checkpoint_path ckpt/ur3-expert-init-mix-wm/20260524_172358/full_agent_before_online.pt \
      -device cuda:0 \
      --critic_warmup_steps 30000
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import warnings
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import colorama
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

try:
    from pwm_isaaclab import scan
    from pwm_isaaclab.expert_config import cfg_to_dict, load_expert_config
    from pwm_isaaclab.expert_init import load_expert_checkpoint, save_expert_checkpoint
    from pwm_isaaclab.expert_loader import load_expert_dataset
    from pwm_isaaclab.expert_pretrain import (
        _cost_loader_kwargs,
        _validate_compatible_episodes,
        build_agent,
        build_world_model,
    )
    from pwm_isaaclab.expert_replay import make_expert_replay
    from pwm_isaaclab.utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )
except ImportError:
    import scan
    from expert_config import cfg_to_dict, load_expert_config
    from expert_init import load_expert_checkpoint, save_expert_checkpoint
    from expert_loader import load_expert_dataset
    from expert_pretrain import (
        _cost_loader_kwargs,
        _validate_compatible_episodes,
        build_agent,
        build_world_model,
    )
    from expert_replay import make_expert_replay
    from utils import (
        Logger,
        collect_training_info,
        make_unique_run_dir,
        save_run_artifacts,
        seed_np_torch,
        write_latest_run_pointer,
    )


RUN_EXAMPLE = """WANDB_MODE=offline python pwm_isaaclab/critic_warmup.py \\
  -n ur3-critic-warmup \\
  -seed 0 \\
  -config_path pwm_isaaclab/config_files/PWM_expert_init.yaml \\
  -checkpoint_path ckpt/ur3-expert-init-mix-wm/20260524_172358/full_agent_before_online.pt \\
  -device cuda:0 \\
  --critic_warmup_steps 30000"""

CORE_BATCH_KEYS = ("obs", "action", "reward", "done", "is_first")


def _as_path_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item]


def _actual_run_command():
    command = shlex.join([sys.executable, *sys.argv])
    env_prefix = []
    for key in ("WANDB_MODE", "PYTHONPATH"):
        value = os.environ.get(key)
        if value:
            env_prefix.append(f"{key}={shlex.quote(value)}")
    return " ".join([*env_prefix, command])


def _jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _unwrap_optimizer_step(optimizer):
    optimizer_cls = optimizer.__class__
    if hasattr(optimizer_cls.step, "__wrapped__"):
        optimizer_cls.step = optimizer_cls.step.__wrapped__
    if hasattr(optimizer.step, "__wrapped__"):
        optimizer.step = optimizer.step.__wrapped__.__get__(optimizer, optimizer_cls)


def _safe_corrcoef(x, y):
    x = torch.as_tensor(x).detach().float().reshape(-1)
    y = torch.as_tensor(y).detach().float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x.square()) * torch.sum(y.square()))
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-8:
        return None
    return float((torch.sum(x * y) / denom).detach().cpu().item())


def _batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _split_batch_counts(batch_size, ratios):
    ratios = [float(ratio) for ratio in ratios]
    if batch_size < len(ratios):
        raise ValueError(
            f"batch_size={batch_size} is too small for {len(ratios)} replay sources; "
            "use at least one sample per source."
        )
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Replay source ratios must sum to a positive value.")
    raw = np.asarray(ratios, dtype=np.float64) / total * int(batch_size)
    counts = np.floor(raw).astype(np.int64)
    for idx in range(len(counts)):
        if counts[idx] == 0:
            counts[idx] = 1
    while int(counts.sum()) > int(batch_size):
        candidates = np.where(counts > 1)[0]
        if candidates.size == 0:
            break
        idx = int(candidates[np.argmin(raw[candidates] - np.floor(raw[candidates]))])
        counts[idx] -= 1
    remainder = int(batch_size) - int(counts.sum())
    if remainder > 0:
        fractional = raw - np.floor(raw)
        order = np.argsort(-fractional)
        for offset in range(remainder):
            counts[int(order[offset % len(order)])] += 1
    return [int(count) for count in counts]


class CriticReplayMixer:
    """Balanced source sampler for critic warmup start states."""

    def __init__(self, sources, ratios=None):
        if not sources:
            raise ValueError("CriticReplayMixer needs at least one replay source.")
        self.sources = [(str(name), replay) for name, replay in sources]
        self.source_names = [name for name, _ in self.sources]
        self.ratios = [1.0 for _ in self.sources] if ratios is None else list(ratios)
        if len(self.ratios) != len(self.sources):
            raise ValueError("ratios length must match sources length.")

    def can_sample(self, horizon):
        return all(replay.can_sample(horizon) for _, replay in self.sources)

    @torch.no_grad()
    def sample(self, batch_size, horizon):
        counts = _split_batch_counts(batch_size, self.ratios)
        chunks = []
        source_ids = []
        for source_idx, ((source_name, replay), count) in enumerate(zip(self.sources, counts)):
            if not replay.can_sample(horizon):
                raise ValueError(f"Replay source `{source_name}` cannot sample horizon={horizon}.")
            batch = replay.sample(count, horizon, return_dict=True)
            chunks.append(batch)
            source_ids.append(
                torch.full((count,), source_idx, dtype=torch.long, device=batch["obs"].device)
            )

        mixed = {}
        for key in CORE_BATCH_KEYS:
            mixed[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
        if all("cost" in chunk for chunk in chunks):
            mixed["cost"] = torch.cat([chunk["cost"] for chunk in chunks], dim=0)
        mixed["source_ids"] = torch.cat(source_ids, dim=0)
        mixed["source_names"] = list(self.source_names)
        mixed["source_counts"] = dict(zip(self.source_names, counts))
        return mixed


def freeze_for_critic_warmup(world_model, agent):
    world_model.eval()
    agent.eval()
    for param in world_model.parameters():
        param.requires_grad_(False)
    for param in agent.actor.parameters():
        param.requires_grad_(False)
    for param in agent.slow_critic.parameters():
        param.requires_grad_(False)
    for param in agent.critic.parameters():
        param.requires_grad_(True)


@torch.no_grad()
def sync_critic_targets(agent):
    agent.sync_slow_critic()


def _bootstrap_values(agent, feat, raw_value, bootstrap):
    bootstrap = str(bootstrap).lower()
    if bootstrap == "zero":
        return torch.zeros(*feat.shape[:-1], 1, dtype=feat.dtype, device=feat.device)
    if bootstrap == "critic":
        return _decode_twohot(agent, raw_value.detach()).to(dtype=feat.dtype)
    if bootstrap == "slow":
        return _decode_twohot(agent, agent.slow_critic(feat)).to(dtype=feat.dtype)
    raise ValueError(f"Unsupported bootstrap mode: {bootstrap!r}.")


def _decode_twohot(agent, logits):
    # AMP can produce float16 logits while two-hot bins are stored as float32.
    return agent.twohot_loss.decode(logits.float())


def _critic_metrics(agent, raw_value, lambda_return, loss_per, weight, source_ids, source_names):
    with torch.no_grad():
        value = _decode_twohot(agent, raw_value[:, :-1])
        lambda_return = lambda_return.float()
        loss_per = loss_per.float()
        weight = weight.float()
        value_error = value - lambda_return
        finite_weight = torch.clamp(weight, min=0.0)
        weighted_loss = loss_per.sum() / finite_weight.sum().clamp_min(1.0)
        metrics = {
            "critic_loss": float(weighted_loss.detach().float().item()),
            "lambda_return_mean": float(lambda_return.detach().float().mean().item()),
            "lambda_return_std": float(lambda_return.detach().float().std(unbiased=False).item()),
            "critic_value_mean": float(value.detach().float().mean().item()),
            "critic_value_std": float(value.detach().float().std(unbiased=False).item()),
            "value_return_mae": float(value_error.detach().float().abs().mean().item()),
            "value_return_corr": _safe_corrcoef(value, lambda_return),
        }
        source_ids = source_ids.to(loss_per.device)
        for idx, name in enumerate(source_names):
            mask = source_ids == idx
            if not torch.any(mask):
                continue
            src_loss = loss_per[mask].sum() / weight[mask].sum().clamp_min(1.0)
            src_value = value[mask]
            src_return = lambda_return[mask]
            src_error = src_value - src_return
            prefix = f"source/{name}"
            metrics[f"{prefix}/critic_loss"] = float(src_loss.detach().float().item())
            metrics[f"{prefix}/lambda_return_mean"] = float(src_return.detach().float().mean().item())
            metrics[f"{prefix}/critic_value_mean"] = float(src_value.detach().float().mean().item())
            metrics[f"{prefix}/value_return_mae"] = float(src_error.detach().float().abs().mean().item())
            metrics[f"{prefix}/value_return_corr"] = _safe_corrcoef(src_value, src_return)
        return metrics


def critic_warmup_step(
    world_model,
    agent,
    batch,
    optimizer,
    *,
    imagine_horizon,
    bootstrap="zero",
    scaler=None,
):
    source_ids = batch["source_ids"]
    source_names = batch["source_names"]
    world_model.eval()
    agent.actor.eval()
    agent.critic.train()

    feat, _, discount, reward, weight = world_model.imagine_data(
        agent,
        batch["obs"],
        batch["action"],
        batch["reward"],
        batch["done"],
        batch["is_first"],
        int(imagine_horizon),
    )
    agent.critic.train()
    starts_per_sequence = int(feat.shape[0] // max(int(source_ids.numel()), 1))
    expanded_source_ids = source_ids.repeat_interleave(starts_per_sequence)
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=False)
    _unwrap_optimizer_step(optimizer)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=agent.device_type, dtype=agent.tensor_dtype, enabled=agent.use_amp):
        raw_value = agent.critic(feat)
        with torch.no_grad():
            target_value = _bootstrap_values(agent, feat, raw_value, bootstrap)
            lambda_return = torch.transpose(
                scan.parallel_lambda_return(
                    torch.transpose(reward, 0, 1),
                    torch.transpose(target_value[:, :-1], 0, 1),
                    torch.transpose(target_value[:, 1:], 0, 1),
                    torch.transpose(discount, 0, 1),
                    agent.lambd,
                ),
                0,
                1,
            )
        loss_per = agent.twohot_loss(raw_value[:, :-1], lambda_return, reduce=False) * weight
        loss = loss_per.mean()

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=100.0)
    with torch.no_grad():
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)

    metrics = _critic_metrics(
        agent,
        raw_value.detach(),
        lambda_return.detach(),
        loss_per.detach(),
        weight.detach(),
        expanded_source_ids.detach(),
        source_names,
    )
    if hasattr(world_model, "predict_cost"):
        with torch.no_grad():
            with torch.autocast(
                device_type=world_model.device_type,
                dtype=world_model.tensor_dtype,
                enabled=world_model.use_amp,
            ):
                cost_pred, cost_prob, _ = world_model.predict_cost(feat[:, 1:].detach())
            metrics["imagined_cost_mean"] = float(cost_pred.detach().float().mean().item())
            metrics["imagined_cost_prob_mean"] = float(cost_prob.detach().float().mean().item())
    return metrics


def train_critic_warmup(
    world_model,
    agent,
    mixer,
    *,
    num_steps,
    batch_size,
    context_length,
    imagine_horizon,
    critic_lr,
    bootstrap="zero",
    logger=None,
    log_interval=100,
    progress=True,
):
    if int(num_steps) <= 0:
        return {}, []
    if not mixer.can_sample(context_length):
        raise ValueError(f"At least one replay source cannot sample context_length={context_length}.")

    freeze_for_critic_warmup(world_model, agent)
    optimizer = torch.optim.AdamW(
        agent.critic.parameters(),
        lr=float(critic_lr),
        eps=agent.optimizer.param_groups[0].get("eps", 1e-8),
    )
    _unwrap_optimizer_step(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=agent.use_amp)
    history = []
    last_metrics = {}
    iterator = range(1, int(num_steps) + 1)
    for step in tqdm(iterator, disable=not progress):
        batch = mixer.sample(int(batch_size), int(context_length))
        batch = _batch_to_device(batch, agent.device)
        metrics = critic_warmup_step(
            world_model,
            agent,
            batch,
            optimizer,
            imagine_horizon=imagine_horizon,
            bootstrap=bootstrap,
            scaler=scaler,
        )
        last_metrics = metrics
        if step == 1 or step % max(int(log_interval), 1) == 0 or step == int(num_steps):
            record = {"step": step, **metrics}
            history.append(record)
            if logger is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        logger.log(f"critic_warmup/{key}", float(value), step)

    sync_critic_targets(agent)
    return last_metrics, history


def _resolve_source_paths(conf, args):
    coverage_roots = _as_path_list(getattr(conf.expert, "wm_coverage_paths", None))
    coverage_root = coverage_roots[0] if coverage_roots else ""
    expert_path = args.expert_path or conf.expert.path
    random_path = args.random_path or (os.path.join(coverage_root, "random") if coverage_root else "")
    perturb_path = args.perturb_path or (
        os.path.join(coverage_root, "boundary_perturbation") if coverage_root else ""
    )
    return {
        "expert": expert_path,
        "random": random_path,
        "perturb": perturb_path,
    }


def _load_source_dataset(label, path, conf, *, max_episodes=None, reference_metadata=None):
    if not path:
        raise ValueError(f"{label} replay path is required.")
    dataset = load_expert_dataset(
        path,
        format=conf.expert.format if label == "expert" else conf.expert.wm_coverage_format,
        action_tolerance=conf.expert.action_tolerance,
        max_episodes=max_episodes,
        **_cost_loader_kwargs(conf),
    )
    if reference_metadata is not None:
        _validate_compatible_episodes(
            dataset,
            reference_metadata["obs_dim"],
            reference_metadata["action_dim"],
            label,
        )
    return dataset


def _metadata_summary(datasets):
    return {
        name: {
            "dataset_path": dataset.metadata.get("dataset_path"),
            "num_episodes": dataset.metadata.get("num_episodes"),
            "num_transitions": dataset.metadata.get("num_transitions"),
            "mean_return": dataset.metadata.get("mean_return"),
            "mean_cost": dataset.metadata.get("mean_cost"),
            "cost_positive_ratio": dataset.metadata.get("cost_positive_ratio"),
        }
        for name, dataset in datasets.items()
    }


def _log_startup(args, source_paths):
    print(colorama.Fore.CYAN + "Stored run example:" + colorama.Style.RESET_ALL)
    print(RUN_EXAMPLE)
    print(colorama.Fore.CYAN + "Actual run command:" + colorama.Style.RESET_ALL)
    print(_actual_run_command())
    print(colorama.Fore.CYAN + "Critic warmup replay sources:" + colorama.Style.RESET_ALL)
    for name, path in source_paths.items():
        print(f"  {name}: {path}")
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)


def main():
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
    parser.add_argument("-checkpoint_path", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-buffer_device", type=str, default=None)
    parser.add_argument("--critic_warmup_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--imagine_horizon", type=int, default=0)
    parser.add_argument("--critic_lr", type=float, default=0.0)
    parser.add_argument("--bootstrap", choices=("zero", "critic", "slow"), default="zero")
    parser.add_argument("--expert_path", type=str, default=None)
    parser.add_argument("--random_path", type=str, default=None)
    parser.add_argument("--perturb_path", type=str, default=None)
    parser.add_argument("--max_expert_episodes", type=int, default=0)
    parser.add_argument("--max_random_episodes", type=int, default=0)
    parser.add_argument("--max_perturb_episodes", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--run_root", type=str, default="ckpt")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--no_run_info_prompt", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    conf = load_expert_config(args.config_path)
    if conf.Task != "JointTrainAgent":
        raise NotImplementedError(f"Task {conf.Task} not implemented for critic warmup.")

    source_paths = _resolve_source_paths(conf, args)
    _log_startup(args, source_paths)
    seed_np_torch(seed=args.seed)

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
            "seed": args.seed,
            "device": args.device,
            "run_command": _actual_run_command(),
            "run_example": RUN_EXAMPLE,
        },
    )
    print(colorama.Fore.CYAN + f"Run directory: {checkpoint_dir}" + colorama.Style.RESET_ALL)

    project = getattr(conf.Wandb, "Project", "IsaacLab-PSSM")
    run_group = getattr(conf.Wandb, "Group", "critic-warmup")
    wandb_mode = "disabled" if args.no_wandb else getattr(conf.Wandb, "Mode", None)
    init_kwargs = {
        "project": project,
        "group": run_group,
        "name": f"{args.n}-{os.path.basename(checkpoint_dir)}",
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

    datasets = {}
    datasets["expert"] = _load_source_dataset(
        "expert",
        source_paths["expert"],
        conf,
        max_episodes=args.max_expert_episodes if args.max_expert_episodes > 0 else None,
    )
    for label, max_arg in (("random", args.max_random_episodes), ("perturb", args.max_perturb_episodes)):
        datasets[label] = _load_source_dataset(
            label,
            source_paths[label],
            conf,
            max_episodes=max_arg if max_arg > 0 else None,
            reference_metadata=datasets["expert"].metadata,
        )
    for label, dataset in datasets.items():
        print(
            colorama.Fore.CYAN
            + (
                f"Loaded {label} replay: {dataset.metadata['num_episodes']} episodes / "
                f"{dataset.metadata['num_transitions']} steps, "
                f"mean_return={dataset.metadata['mean_return']:.4f}, "
                f"mean_cost={dataset.metadata['mean_cost']:.4f}, "
                f"cost_pos_ratio={dataset.metadata.get('cost_positive_ratio', 0.0):.6f}"
            )
            + colorama.Style.RESET_ALL
        )

    buffer_device = args.buffer_device or args.device
    replays = {
        label: make_expert_replay(dataset, device=buffer_device, include_force=False)
        for label, dataset in datasets.items()
    }
    mixer = CriticReplayMixer(
        [("expert", replays["expert"]), ("random", replays["random"]), ("perturb", replays["perturb"])],
        ratios=[1.0, 1.0, 1.0],
    )

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(
        conf,
        datasets["expert"].metadata["obs_dim"],
        datasets["expert"].metadata["action_dim"],
        act,
        args.device,
    )
    agent = build_agent(conf, datasets["expert"].metadata["action_dim"], act, args.device)
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    load_expert_checkpoint(checkpoint_path, world_model=world_model, agent=agent, map_location=args.device)
    print(colorama.Fore.YELLOW + f"Loaded checkpoint: {checkpoint_path}" + colorama.Style.RESET_ALL)

    batch_size = args.batch_size if args.batch_size > 0 else int(conf.JointTrainAgent.BatchSize)
    context_length = args.context_length if args.context_length > 0 else int(conf.JointTrainAgent.ImagineContext)
    if context_length <= 0:
        context_length = int(conf.JointTrainAgent.BatchLength)
    imagine_horizon = args.imagine_horizon if args.imagine_horizon > 0 else int(conf.JointTrainAgent.ImagineHorizon)
    critic_lr = args.critic_lr if args.critic_lr > 0 else float(conf.Models.Agent.LR)

    final_metrics, history = train_critic_warmup(
        world_model,
        agent,
        mixer,
        num_steps=args.critic_warmup_steps,
        batch_size=batch_size,
        context_length=context_length,
        imagine_horizon=imagine_horizon,
        critic_lr=critic_lr,
        bootstrap=args.bootstrap,
        logger=logger,
        log_interval=args.log_interval,
        progress=True,
    )

    output_checkpoint = os.path.join(checkpoint_dir, "full_agent_after_critic_warmup.pt")
    metrics_path = os.path.join(checkpoint_dir, "critic_warmup_metrics.json")
    extra = {
        "action_scaling": "normalized_-1_1",
        "critic_warmup": {
            "steps": int(args.critic_warmup_steps),
            "batch_size": int(batch_size),
            "context_length": int(context_length),
            "imagine_horizon": int(imagine_horizon),
            "critic_lr": float(critic_lr),
            "bootstrap": args.bootstrap,
            "source_mix": {"expert": 1.0, "random": 1.0, "perturb": 1.0},
            "source_metadata": _metadata_summary(datasets),
        },
        "run_command": _actual_run_command(),
        "run_example": RUN_EXAMPLE,
    }
    save_expert_checkpoint(
        output_checkpoint,
        world_model=world_model,
        agent=agent,
        config=cfg_to_dict(conf),
        expert_metadata=datasets["expert"].metadata,
        extra=extra,
    )

    metrics_payload = {
        "run_command": _actual_run_command(),
        "run_example": RUN_EXAMPLE,
        "args": vars(args),
        "checkpoint_path": checkpoint_path,
        "output_checkpoint": output_checkpoint,
        "source_paths": source_paths,
        "source_metadata": _metadata_summary(datasets),
        "final_metrics": final_metrics,
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as fout:
        json.dump(_jsonable(metrics_payload), fout, indent=2, ensure_ascii=False)

    if logger.log_dict:
        wandb.log(logger.log_dict, step=logger.tot_step)
    wandb.finish()
    print(colorama.Fore.GREEN + f"Saved critic-warmup checkpoint: {output_checkpoint}" + colorama.Style.RESET_ALL)
    print(colorama.Fore.GREEN + f"Saved critic-warmup metrics: {metrics_path}" + colorama.Style.RESET_ALL)


if __name__ == "__main__":
    main()
