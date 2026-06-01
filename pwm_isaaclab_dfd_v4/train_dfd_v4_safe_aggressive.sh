#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/pwm_isaaclab_dfd_v4/config_dfd_v4_safe_aggressive.yaml}"
export RUN_NAME="${RUN_NAME:-dfd-v4-safe-aggressive}"
export NOTE="${NOTE:-DFD v4 safe-aggressive: stronger GP/GD safety and slightly more aggressive dual policy}"
export TAGS="${TAGS:-v4,safe-aggressive,gp-gd,dual}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec "${SCRIPT_DIR}/train_dfd_v4.sh" "$@"
