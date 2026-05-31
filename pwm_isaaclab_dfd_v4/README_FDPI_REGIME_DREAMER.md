# FDPI-Regime Dreamer v4

`pwm_isaaclab_dfd_v4` is an independent ablation branch for FDPI-Regime Dreamer. It does not modify the original `pwm_isaaclab`, `pwm_isaaclab_dfd`, or `pwm_isaaclab_dfd_v2` directories.

The v4 branch keeps Dreamer/PSSM as the world-model and imagination backbone, then adds:

- `Gp`: main-continuation latent risk critic.
- `Gd`: dual-continuation latent risk critic.
- FDPI-style feasible / critical / infeasible main actor regions based on `Gp`.
- Dual real sampling controlled by recent main-source `Gp` feasible ratio.
- Safety-critical replay sampling for world-model and risk-critic updates.
- v3-style bottom-force cost: selected bottom-force channels are thresholded at `0.1`, log-compressed with `LowForceScale=0.05`, normalized by `CostForceMax=15.0`, and clipped to `[0, 1]`.

First-version exclusions are intentional:

- no GR recovery critic;
- no SAC-style reward Q;
- no trajectory-level importance sampling;
- no PPO-Lagrangian branch;
- no imagined dual data for world-model training.

## Run

Use the IsaacLab launcher, as with the other training entrypoints:

```bash
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p pwm_isaaclab_dfd_v4/train_dfd_v4.py \
  -n dfd_v4 \
  -seed 0 \
  -config_path pwm_isaaclab_dfd_v4/config_dfd_v4.yaml \
  -env_name Isaac-UR3-IK-Gripper-v0 \
  -device cuda:0
```

Lite entry:

```bash
WANDB_MODE=online \
RUN_NAME=dfd-v4-lite \
bash pwm_isaaclab_dfd_v4/train_dfd_v4_lite.sh
```

Pass any resume or checkpoint variables through the same wrapper, for example `V4_FULL_CHECKPOINT_PATH`, `RESUME_ENV_STEPS`, `SAMPLE_MAX_STEPS`, or `SAVE_EVERY_STEPS`.

## Smoke Checks

```bash
python -m compileall pwm_isaaclab_dfd_v4
python -m unittest tests.test_dfd_v4_minimal
```

Import smoke:

```bash
python - <<'PY'
import importlib
mods = [
    "pwm_isaaclab_dfd_v4",
    "pwm_isaaclab_dfd_v4.cost_utils",
    "pwm_isaaclab_dfd_v4.risk_critics",
    "pwm_isaaclab_dfd_v4.agent_fdpi_regime",
    "pwm_isaaclab_dfd_v4.trainer_dfd_v4",
]
for m in mods:
    importlib.import_module(m)
print("DFDv4 import smoke test passed")
PY
```
