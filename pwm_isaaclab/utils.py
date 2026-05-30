import json
import os
import random
import re
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import wandb
from yacs.config import CfgNode as CN


def seed_np_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _slugify(value, default="run", max_length=80):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-._")
    if not text:
        text = default
    return text[:max_length].strip("-._") or default


def parse_tags(tags):
    if not tags:
        return []
    if isinstance(tags, (list, tuple)):
        raw_tags = tags
    else:
        raw_tags = str(tags).split(",")
    return [tag.strip() for tag in raw_tags if tag and tag.strip()]


def collect_training_info(note=None, tags=None, prompt=True):
    run_info = {
        "note": note or "",
        "tags": parse_tags(tags),
    }
    if not prompt or not sys.stdin.isatty():
        return run_info

    try:
        note_prompt = "Training note"
        if run_info["note"]:
            note_prompt += f" [{run_info['note']}]"
        entered_note = input(f"{note_prompt}: ").strip()
        if entered_note:
            run_info["note"] = entered_note

        tag_default = ",".join(run_info["tags"])
        tag_prompt = "Tags (comma-separated)"
        if tag_default:
            tag_prompt += f" [{tag_default}]"
        entered_tags = input(f"{tag_prompt}: ").strip()
        if entered_tags:
            run_info["tags"] = parse_tags(entered_tags)
    except EOFError:
        pass

    return run_info


def make_unique_run_dir(base_name, run_root="ckpt", run_id=None, note=None):
    group_name = _slugify(base_name, default="run", max_length=100)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    note_slug = _slugify(note, default="", max_length=40) if note else ""
    if run_id:
        leaf_name = _slugify(run_id, default=timestamp, max_length=120)
    elif note_slug:
        leaf_name = f"{timestamp}_{note_slug}"
    else:
        leaf_name = timestamp

    group_dir = os.path.abspath(os.path.expanduser(os.path.join(run_root, group_name)))
    os.makedirs(group_dir, exist_ok=True)

    for idx in range(1000):
        suffix = "" if idx == 0 else f"_{idx:03d}"
        run_dir = os.path.join(group_dir, f"{leaf_name}{suffix}")
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not create a unique run directory under {group_dir}.")


def write_latest_run_pointer(run_dir):
    group_dir = os.path.dirname(os.path.abspath(run_dir))
    pointer_path = os.path.join(group_dir, "latest_run.txt")
    with open(pointer_path, "w", encoding="utf-8") as fout:
        fout.write(os.path.abspath(run_dir) + "\n")


def save_run_artifacts(run_dir, conf, config_path, args, run_info, extra=None):
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fout:
        fout.write(conf.dump())
        fout.write("\n")

    source_config_path = os.path.abspath(os.path.expanduser(config_path))
    if os.path.isfile(source_config_path):
        shutil.copy2(source_config_path, os.path.join(run_dir, "source_config.yaml"))

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint_dir": run_dir,
        "config_path": source_config_path,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "args": vars(args) if args is not None else {},
        "run_info": run_info or {},
    }
    if extra:
        metadata.update(extra)

    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2, ensure_ascii=False)

    note = (run_info or {}).get("note", "")
    tags = (run_info or {}).get("tags", [])
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as fout:
        fout.write("# Training Run\n\n")
        fout.write(f"- Created at: {metadata['created_at']}\n")
        fout.write(f"- Checkpoint dir: {run_dir}\n")
        fout.write(f"- Config path: {source_config_path}\n")
        if tags:
            fout.write(f"- Tags: {', '.join(tags)}\n")
        if note:
            fout.write("\n## Note\n\n")
            fout.write(note.strip() + "\n")


class Logger:
    def __init__(self):
        self.tot_step = -1
        self.log_dict = {}

    def log(self, tag, value, step):
        if step > self.tot_step:
            wandb.log(self.log_dict, step=step)
            self.log_dict = {}
            self.tot_step = step
        self.log_dict.update({tag: value})

    def log_video(self, tag, value, step):
        value = torch.clip(value, min=0, max=1) * 255
        value = value.detach().to(torch.uint8).cpu().numpy()
        self.log(tag, [wandb.Video(value, fps=3, caption=f"step_{step}")], step)


class EMAScalar:
    def __init__(self, decay):
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ObsShape = None
    conf.BasicSettings.UseAmp = False
    conf.BasicSettings.FrameSkip = 0

    conf.Models = CN()
    conf.Models.Hidden = 0
    conf.Models.NumBin = 0
    conf.Models.MaxBin = 0
    conf.Models.Act = ""
    conf.Models.Stoch = 0
    conf.Models.Discrete = 0
    conf.Models.Gamma = 1.0
    conf.Models.Lambda = 0.0
    conf.Models.Tau = 0.0

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.Stem = 0
    conf.Models.WorldModel.MinRes = 0
    conf.Models.WorldModel.DynScale = 0.0
    conf.Models.WorldModel.RepScale = 0.0
    conf.Models.WorldModel.ValScale = 0.0
    conf.Models.WorldModel.KLFree = 0.0
    conf.Models.WorldModel.LR = 0.0
    conf.Models.WorldModel.Eps = 0.0

    conf.Models.Agent = CN()
    conf.Models.Agent.EntropyCoef = 0.0
    conf.Models.Agent.MinStd = 0.0
    conf.Models.Agent.MaxStd = 0.0
    conf.Models.Agent.MinPer = 0.0
    conf.Models.Agent.MaxPer = 0.0
    conf.Models.Agent.LR = 0.0
    conf.Models.Agent.Eps = 0.0
    conf.Models.Agent.EMADecay = 0.0
    conf.Models.Agent.UseSlowCritic = False

    conf.ForceHead = CN()
    conf.ForceHead.Enable = False
    conf.ForceHead.Key = ""
    conf.ForceHead.HiddenDim = 256
    conf.ForceHead.Depth = 4
    conf.ForceHead.Dropout = 0.1
    conf.ForceHead.SignedForce = False
    conf.ForceHead.Threshold = 0.3
    conf.ForceHead.LossWeight = 1.0
    conf.ForceHead.DetachLatent = True

    conf.ForceLoss = CN()
    conf.ForceLoss.Eps = 1e-3
    conf.ForceLoss.ForceScale = 1.0
    conf.ForceLoss.LambdaCls = 1.0
    conf.ForceLoss.LambdaReg = 2.0
    conf.ForceLoss.LambdaSign = 0.5
    conf.ForceLoss.FocalAlpha = 0.75
    conf.ForceLoss.FocalGamma = 2.0
    conf.ForceLoss.HuberBeta = 0.5
    conf.ForceLoss.RegWeightPower = 0.5
    conf.ForceLoss.RegWeightMax = 10.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineContext = 0
    conf.JointTrainAgent.ImagineHorizon = 0
    conf.JointTrainAgent.TrainModelEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.ModelUpdate = 1
    conf.JointTrainAgent.AgentUpdate = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.VideoLogStep = 0
    conf.JointTrainAgent.SaveOfflineEpisodes = False
    conf.JointTrainAgent.OfflineDatasetDir = ""

    conf.Env = CN(new_allowed=True)
    conf.Wandb = CN(new_allowed=True)

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()
    return conf
