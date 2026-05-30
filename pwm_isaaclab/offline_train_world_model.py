import argparse
import json
import os
import warnings

import colorama
import torch
import torch.nn as nn
import wandb

from modules.world_models import ParallelWorldModel
from replay_buffer import load_offline_episodes
from trainer import offline_train_world_model
from utils import (
    Logger,
    collect_training_info,
    load_config,
    make_unique_run_dir,
    save_run_artifacts,
    seed_np_torch,
    write_latest_run_pointer,
)

'''
WANDB_MODE=offline python pwm_isaaclab/offline_train_world_model.py \
  -n ur3lite-wm-offline \
  -seed 42 \
  -config_path pwm_isaaclab/config_files/PWM.yaml \
  -dataset_path /home/yhy/surgical_robot2/latent_safety/log/dreamerv3/world_model_only/0408/232730_test/enhanced_train_eps \
  -device cuda:0 \
  -buffer_device cpu \
  -train_steps 100000 \
  -batch_size 64 \
  -batch_length 64
'''
def build_world_model(conf, obs_dim, action_dim, act, device):
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
    ).to(device)


def _write_dataset_summary(
    checkpoint_dir,
    run_name,
    dataset_meta,
    train_steps,
    batch_size,
    batch_length,
    save_every_steps,
    seed,
    model_device,
    buffer_device,
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary = {
        "run_name": run_name,
        "dataset_path": dataset_meta["dataset_path"],
        "num_episodes": dataset_meta["num_episodes"],
        "total_steps": dataset_meta["total_steps"],
        "obs_dim": dataset_meta["obs_dim"],
        "action_dim": dataset_meta["action_dim"],
        "min_episode_length": dataset_meta["min_episode_length"],
        "max_episode_length": dataset_meta["max_episode_length"],
        "mean_episode_length": dataset_meta["mean_episode_length"],
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "batch_length": int(batch_length),
        "save_every_steps": int(save_every_steps),
        "seed": int(seed),
        "model_device": model_device,
        "buffer_device": buffer_device,
    }
    with open(os.path.join(checkpoint_dir, "offline_dataset_summary.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-dataset_path", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-buffer_device", type=str, default=None)
    parser.add_argument("-train_steps", type=int, required=True)
    parser.add_argument("-batch_size", type=int, default=None)
    parser.add_argument("-batch_length", type=int, default=None)
    parser.add_argument("-save_every_steps", type=int, default=None)
    parser.add_argument("-max_episodes", type=int, default=0)
    parser.add_argument("-world_model_path", type=str, default=None)
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
            "dataset_path": args.dataset_path,
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
    run_group = getattr(conf.Wandb, "Group", "offline-world-model")
    base_wandb_name = getattr(conf.Wandb, "Name", args.n)
    run_name = f"{base_wandb_name}-{os.path.basename(checkpoint_dir)}"
    wandb_mode = getattr(conf.Wandb, "Mode", None)
    init_kwargs = {
        "project": project,
        "group": run_group,
        "name": run_name,
        "dir": checkpoint_dir,
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
    batch_size = args.batch_size or conf.JointTrainAgent.BatchSize
    batch_length = args.batch_length or conf.JointTrainAgent.BatchLength
    save_every_steps = args.save_every_steps or conf.JointTrainAgent.SaveEverySteps
    max_episodes = args.max_episodes if args.max_episodes > 0 else None
    buffer_device = args.buffer_device or args.device

    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive. Pass `-batch_size` or set `JointTrainAgent.BatchSize`.")
    if batch_length <= 0:
        raise ValueError(
            "`batch_length` must be positive. Pass `-batch_length` or set `JointTrainAgent.BatchLength`."
        )
    if args.train_steps <= 0:
        raise ValueError("`train_steps` must be positive.")

    replay_buffer, dataset_meta = load_offline_episodes(
        dataset_path=args.dataset_path,
        device=buffer_device,
        max_episodes=max_episodes,
    )
    save_run_artifacts(
        run_dir=checkpoint_dir,
        conf=conf,
        config_path=args.config_path,
        args=args,
        run_info=run_info,
        extra={
            "base_run_name": args.n,
            "dataset_path": args.dataset_path,
            "seed": args.seed,
            "device": args.device,
            "buffer_device": buffer_device,
            "num_episodes": dataset_meta["num_episodes"],
            "total_steps": dataset_meta["total_steps"],
            "obs_dim": dataset_meta["obs_dim"],
            "action_dim": dataset_meta["action_dim"],
        },
    )
    _write_dataset_summary(
        checkpoint_dir=checkpoint_dir,
        run_name=args.n,
        dataset_meta=dataset_meta,
        train_steps=args.train_steps,
        batch_size=batch_size,
        batch_length=batch_length,
        save_every_steps=save_every_steps,
        seed=args.seed,
        model_device=args.device,
        buffer_device=buffer_device,
    )

    print(
        colorama.Fore.CYAN
        + (
            f"Loaded {dataset_meta['num_episodes']} episodes / {dataset_meta['total_steps']} steps "
            f"from {dataset_meta['dataset_path']}"
        )
        + colorama.Style.RESET_ALL
    )

    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(
        conf=conf,
        obs_dim=dataset_meta["obs_dim"],
        action_dim=dataset_meta["action_dim"],
        act=act,
        device=args.device,
    )

    if args.world_model_path:
        world_model.load_state_dict(torch.load(args.world_model_path, map_location=args.device))
        print(
            colorama.Fore.YELLOW
            + f"Resumed world model from {args.world_model_path}"
            + colorama.Style.RESET_ALL
        )

    offline_train_world_model(
        run_name=args.n,
        replay_buffer=replay_buffer,
        world_model=world_model,
        train_steps=args.train_steps,
        batch_size=batch_size,
        batch_length=batch_length,
        save_every_steps=save_every_steps,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
    )

    if logger.log_dict:
        wandb.log(logger.log_dict, step=logger.tot_step)
    wandb.finish()
