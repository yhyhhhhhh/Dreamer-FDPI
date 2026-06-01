#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# This launcher enables the safe-aggressive config, including
# MainFDPIRegime.TailRiskCoef. By default it does not load any checkpoint.
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/pwm_isaaclab_dfd_v4/config_dfd_v4_safe_aggressive.yaml}"
export RUN_NAME="${RUN_NAME:-dfd-v4-tail-risk-safe-aggressive}"

# Keep both checkpoint knobs explicitly empty by default. This also prevents
# train_dfd_v4.sh from falling back to its legacy warmup checkpoint.
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
export V4_FULL_CHECKPOINT_PATH="${V4_FULL_CHECKPOINT_PATH:-}"
export RESUME_ENV_STEPS="${RESUME_ENV_STEPS:-}"

# SAMPLE_MAX_STEPS is the absolute target total env steps, not extra steps.
# Default: full fresh training target.
export SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-10000000}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-250000}"

if [[ -z "${NOTE:-}" ]]; then
  if [[ -n "${V4_FULL_CHECKPOINT_PATH}" ]]; then
    export NOTE="DFD v4 tail-risk safe-aggressive: continue from V4 full checkpoint"
  elif [[ -n "${CHECKPOINT_PATH}" ]]; then
    export NOTE="DFD v4 tail-risk safe-aggressive: initialize from checkpoint"
  else
    export NOTE="DFD v4 tail-risk safe-aggressive: fresh run without checkpoint"
  fi
fi

if [[ -z "${TAGS:-}" ]]; then
  if [[ -n "${V4_FULL_CHECKPOINT_PATH}" ]]; then
    export TAGS="v4,tail-risk,safe-aggressive,continue-full-state"
  elif [[ -n "${CHECKPOINT_PATH}" ]]; then
    export TAGS="v4,tail-risk,safe-aggressive,init-checkpoint"
  else
    export TAGS="v4,tail-risk,safe-aggressive,no-checkpoint"
  fi
fi
export WANDB_MODE="${WANDB_MODE:-online}"

exec "${SCRIPT_DIR}/train_dfd_v4.sh" "$@"
