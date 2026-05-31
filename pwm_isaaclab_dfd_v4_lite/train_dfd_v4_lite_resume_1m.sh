#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_V4_DIR="${REPO_ROOT}/ckpt/dfd-v4-fdpi-regime/20260531_055516"
if [[ ! -d "${DEFAULT_V4_DIR}" ]]; then
  DEFAULT_V4_DIR="${REPO_ROOT}/ckpt/dfd-v4-fdpi-regime/20260531_024603"
fi

export RUN_NAME="${RUN_NAME:-dfd-v4-lite-resume-1m}"
export V4_CHECKPOINT_DIR="${V4_CHECKPOINT_DIR:-${DEFAULT_V4_DIR}}"
export V4_CHECKPOINT_STEP="${V4_CHECKPOINT_STEP:-999936}"
export RESUME_ENV_STEPS="${RESUME_ENV_STEPS:-${V4_CHECKPOINT_STEP}}"
export SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-1700000}"
export MAIN_FDPI_START_STEP="${MAIN_FDPI_START_STEP:-1500000}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10000}"

exec "${SCRIPT_DIR}/train_dfd_v4_lite.sh" "$@"
