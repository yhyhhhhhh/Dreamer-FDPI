# FDPI-Lite Dreamer v4

`pwm_isaaclab_dfd_v4_lite` is an independent simplified branch. It does not modify
`pwm_isaaclab_dfd_v4` or earlier DFD branches.

The lite branch keeps the v4 infrastructure:

- Dreamer/PSSM world model with continuous cost head.
- `Gp` and `Gd` latent risk critics.
- Dual policy update and optional dual real-environment sampling.
- Safety-critical replay sampling for world model and risk critics.
- Full-state checkpoints with replay buffer and optimizer state.

The main difference is the main actor objective. Instead of feasible / critical /
infeasible regions, the main actor uses a single GP penalty:

```text
actor_loss = reward_policy_loss
           + lambda_gp_eff * relu(Gp(z, a) - Pf) / (RiskMax - Pf)
           - entropy_coef * entropy

lambda_gp_eff = LambdaGp * clamp((step - StartStep) / RampSteps, 0, 1)
```

This leaves one main safety knob, `MainFDPIRegime.LambdaGp`, and one transition
knob, `MainFDPIRegime.RampSteps`.

## Run

```bash
WANDB_MODE=online bash pwm_isaaclab_dfd_v4_lite/train_dfd_v4_lite.sh
```

Resume from a v4 or v4-lite full checkpoint:

```bash
RUN_NAME=dfd-v4-lite-resume \
V4_FULL_CHECKPOINT_PATH=/path/to/full_state_v4_1999872.pth \
RESUME_ENV_STEPS=1999872 \
SAMPLE_MAX_STEPS=2600000 \
SAVE_EVERY_STEPS=100000 \
WANDB_MODE=online \
bash pwm_isaaclab_dfd_v4_lite/train_dfd_v4_lite.sh
```

## Smoke Checks

```bash
/home/yhy/anaconda3/envs/isaaclab_14/bin/python3.10 -m compileall pwm_isaaclab_dfd_v4_lite
```

Import smoke:

```bash
/home/yhy/anaconda3/envs/isaaclab_14/bin/python3.10 - <<'PY'
import importlib
mods = [
    "pwm_isaaclab_dfd_v4_lite",
    "pwm_isaaclab_dfd_v4_lite.cost_utils",
    "pwm_isaaclab_dfd_v4_lite.risk_critics",
    "pwm_isaaclab_dfd_v4_lite.agent_fdpi_lite",
    "pwm_isaaclab_dfd_v4_lite.trainer_dfd_v4_lite",
]
for m in mods:
    importlib.import_module(m)
print("DFDv4-lite import smoke test passed")
PY
```
