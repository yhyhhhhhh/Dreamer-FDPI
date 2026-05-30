# PaMoRL IsaacLab Clean

This is a cleaned PaMoRL IsaacLab project focused on three workflows:

1. Original PaMoRL IsaacLab online training.
2. Offline expert world-model and actor initialization.
3. Offline critic warmup from a full expert checkpoint.

Large generated artifacts are intentionally not included. Keep checkpoints,
offline datasets, wandb runs, logs, and evaluation outputs outside this source
tree, and pass their paths through CLI arguments.

## Environment

```bash
source /home/yhy/anaconda3/etc/profile.d/conda.sh
conda activate isaaclab_14
export PYTHONPATH=/home/yhy/surgical_robot_pro1/exts/ur3_lite:/home/yhy/PaMoRL-IsaacLab-clean:$PYTHONPATH
```

For IsaacLab environments, prefer running scripts through:

```bash
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p <script> <args>
```

## Original Online Training

```bash
TERM=xterm WANDB_MODE=offline \
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p \
  /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/train.py \
  -n pamorl-online \
  -seed 0 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM.yaml \
  -env_name Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0 \
  -device cuda:0
```

## Offline Expert Initialization

Use this path to train/load the expert world model, force/cost head, and actor
from offline expert episodes.

```bash
WANDB_MODE=offline python3 \
  /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/expert_pretrain.py \
  -n ur3-expert-init \
  -seed 0 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -dataset_path /path/to/expert_or_mixed_offline_episodes \
  -device cuda:0 \
  -buffer_device cpu \
  --no_run_info_prompt
```

The script writes checkpoints under `ckpt/<run-name>/` by default. That output
directory is ignored by git and is not part of this clean source project.

## Critic Warmup

Use this path after expert initialization to warm up the value critic while
keeping the actor and world model frozen.

```bash
WANDB_MODE=offline python3 \
  /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/critic_warmup.py \
  -n ur3-critic-warmup \
  -seed 0 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_before_online.pt \
  -device cuda:0 \
  --critic_warmup_steps 30000 \
  --no_run_info_prompt
```

The final warmup checkpoint is saved as `full_agent_after_critic_warmup.pt` in
the run checkpoint directory.

## Evaluation

For expert/world-model evaluation:

```bash
python3 /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_checkpoint.pt \
  -device cuda:0 \
  --eval_dataset \
  -dataset_path /path/to/eval_episodes
```

## What Is Intentionally Excluded

- Old dual-policy experiment scripts and extension packages.
- Dreamer continue verification scripts and configs.
- The standalone force-world-model branch.
- Checkpoints, wandb data, logs, runs, and evaluation result folders.
- Python caches and generated binary artifacts.

If dual-policy safety training is revisited later, rebuild it on top of this
clean project rather than copying the old experimental code back in.

## Checks

```bash
python3 -m py_compile pwm_isaaclab/*.py pwm_isaaclab/modules/*.py
python3 -m unittest tests/test_expert_init.py
python3 pwm_isaaclab/expert_pretrain.py --help
python3 pwm_isaaclab/critic_warmup.py --help
python3 pwm_isaaclab/train_expert_online.py --help
```
