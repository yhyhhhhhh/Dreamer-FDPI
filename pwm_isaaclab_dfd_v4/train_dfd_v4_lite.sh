#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-dfd-v4-lite}"
export WANDB_MODE="${WANDB_MODE:-online}"
export SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-2000000}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50000}"

exec "${SCRIPT_DIR}/train_dfd_v4.sh" "$@"
