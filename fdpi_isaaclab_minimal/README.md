# Minimal FDPI IsaacLab Trainer

This folder is a small extraction of the IsaacLab training path from the main repository.

It keeps only:

- `fdpi/sac_fpi_dual.py`: Torch SAC-FDPI dual update.
- `fdpi/networks.py`: actor and critic MLPs.
- `fdpi/replay_buffer.py`: replay buffer with FDPI importance weights.
- `fdpi/trainer.py`: minimal IsaacLab rollout, safety cost extraction, updates, logging, checkpointing.
- `train.py`: IsaacLab launcher entrypoint.
- `configs/ur3_fdpi.yaml`: default UR3 IsaacLab task config.

Run it with the IsaacLab launcher, not plain Python:

```bash
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p fdpi_isaaclab_minimal/train.py
```

Useful smoke-test command:

```bash
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p fdpi_isaaclab_minimal/train.py \
  --num_envs 2 \
  --total_steps 128 \
  --start_steps 0 \
  --batch_size 32 \
  --updates_per_iteration 1 \
  --save_every_steps 128
```

To use another IsaacLab task, pass `--task <TASK_ID>`. The task must return observations with
`obs["policy"]`, continuous actions in `[-1, 1]`, and a safety cost through either
`extras["force_fail"]`, `extras["cost"]`, `extras["constraint_violation"]`, pipe/bottom force
fields, or `obs["force"]`.

Checkpoints and TensorBoard logs are written under `logs/fdpi_minimal/<task>/<run_name>/`.

