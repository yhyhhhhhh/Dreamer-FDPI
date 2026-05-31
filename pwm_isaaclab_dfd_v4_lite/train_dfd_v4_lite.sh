#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ISAACLAB_SH="${ISAACLAB_SH:-/home/yhy/IsaacLab-1.4.0/isaaclab.sh}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd -- "$(dirname -- "${ISAACLAB_SH}")" && pwd)}"
ISAACLAB_EXTS="${ISAACLAB_ROOT}/source/extensions"
CONDA_SITE_PACKAGES="${CONDA_SITE_PACKAGES:-/home/yhy/anaconda3/envs/isaaclab_14/lib/python3.10/site-packages}"
UR3_LITE_EXT="${UR3_LITE_EXT:-/home/yhy/surgical_robot_pro1/exts/ur3_lite}"

RUN_NAME="${RUN_NAME:-dfd-v4-lite}"
SEED="${SEED:-0}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/pwm_isaaclab_dfd_v4_lite/config_dfd_v4_lite.yaml}"
ENV_NAME="${ENV_NAME:-Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/ckpt}"
RUN_ID="${RUN_ID:-}"
NOTE="${NOTE:-}"
TAGS="${TAGS:-}"
V4_FULL_CHECKPOINT_PATH="${V4_FULL_CHECKPOINT_PATH:-}"
V4_CHECKPOINT_DIR="${V4_CHECKPOINT_DIR:-}"
V4_CHECKPOINT_STEP="${V4_CHECKPOINT_STEP:-}"
RESUME_ENV_STEPS="${RESUME_ENV_STEPS:-}"
SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-}"
MAIN_FDPI_START_STEP="${MAIN_FDPI_START_STEP:-}"
MAIN_FDPI_RAMP_STEPS="${MAIN_FDPI_RAMP_STEPS:-}"
MAIN_FDPI_LAMBDA_GP="${MAIN_FDPI_LAMBDA_GP:-}"
BUFFER_WARMUP_STEPS="${BUFFER_WARMUP_STEPS:-}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-}"
NO_LOAD_REPLAY_BUFFER="${NO_LOAD_REPLAY_BUFFER:-}"
NO_LOAD_OPTIMIZER="${NO_LOAD_OPTIMIZER:-}"
NO_LOAD_RNG="${NO_LOAD_RNG:-}"

DEFAULT_CHECKPOINT_PATH="/home/yhy/PaMoRL-main/ckpt/ur3-critic-warmup/20260524_223011_warmup/full_agent_after_critic_warmup.pt"
if [[ -n "${V4_CHECKPOINT_DIR}" || -n "${V4_FULL_CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH-}"
else
  CHECKPOINT_PATH="${CHECKPOINT_PATH-${DEFAULT_CHECKPOINT_PATH}}"
fi
CHECKPOINT_PATH="$(printf '%s' "${CHECKPOINT_PATH}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

if [[ "${WANDB_MODE:-}" == "offline" ]]; then
  printf 'ERROR: WANDB_MODE=offline is set. This launcher is configured not to run wandb offline.\n' >&2
  printf 'Unset WANDB_MODE or set WANDB_MODE=online/disabled before running.\n' >&2
  exit 1
fi
export WANDB_MODE="${WANDB_MODE:-online}"

export TERM="${TERM:-xterm}"
export PYTHONPATH="${UR3_LITE_EXT}:${REPO_ROOT}:${ISAACLAB_EXTS}/omni.isaac.lab:${ISAACLAB_EXTS}/omni.isaac.lab_tasks:${ISAACLAB_EXTS}/omni.isaac.lab_assets:${CONDA_SITE_PACKAGES}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

args=(
  -n "${RUN_NAME}"
  -seed "${SEED}"
  -config_path "${CONFIG_PATH}"
  -env_name "${ENV_NAME}"
  -device "${DEVICE}"
  --run_root "${RUN_ROOT}"
  --no_run_info_prompt
)

if [[ -n "${RUN_ID}" ]]; then
  args+=(--run_id "${RUN_ID}")
fi

if [[ -n "${NOTE}" ]]; then
  args+=(--note "${NOTE}")
fi

if [[ -n "${TAGS}" ]]; then
  args+=(--tags "${TAGS}")
fi

if [[ -n "${V4_FULL_CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${V4_FULL_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: V4_FULL_CHECKPOINT_PATH does not exist: %q\n' "${V4_FULL_CHECKPOINT_PATH}" >&2
    exit 1
  fi
  args+=(--v4_full_checkpoint_path "${V4_FULL_CHECKPOINT_PATH}")
fi

if [[ -n "${V4_CHECKPOINT_DIR}" ]]; then
  if [[ ! -d "${V4_CHECKPOINT_DIR}" ]]; then
    printf 'ERROR: V4_CHECKPOINT_DIR does not exist: %q\n' "${V4_CHECKPOINT_DIR}" >&2
    exit 1
  fi
  args+=(--v4_checkpoint_dir "${V4_CHECKPOINT_DIR}")
fi

if [[ -n "${V4_CHECKPOINT_STEP}" ]]; then
  args+=(--v4_checkpoint_step "${V4_CHECKPOINT_STEP}")
fi

if [[ -n "${RESUME_ENV_STEPS}" ]]; then
  args+=(--resume_env_steps "${RESUME_ENV_STEPS}")
fi

if [[ -n "${SAMPLE_MAX_STEPS}" ]]; then
  args+=(--max_steps "${SAMPLE_MAX_STEPS}")
fi

if [[ -n "${MAIN_FDPI_START_STEP}" ]]; then
  args+=(--main_fdpi_start_step "${MAIN_FDPI_START_STEP}")
fi

if [[ -n "${MAIN_FDPI_RAMP_STEPS}" ]]; then
  args+=(--main_fdpi_ramp_steps "${MAIN_FDPI_RAMP_STEPS}")
fi

if [[ -n "${MAIN_FDPI_LAMBDA_GP}" ]]; then
  args+=(--main_fdpi_lambda_gp "${MAIN_FDPI_LAMBDA_GP}")
fi

if [[ -n "${BUFFER_WARMUP_STEPS}" ]]; then
  args+=(--buffer_warmup_steps "${BUFFER_WARMUP_STEPS}")
fi

if [[ -n "${SAVE_EVERY_STEPS}" ]]; then
  args+=(--save_every_steps "${SAVE_EVERY_STEPS}")
fi

if [[ -n "${NO_LOAD_REPLAY_BUFFER}" ]]; then
  args+=(--no_load_replay_buffer)
fi

if [[ -n "${NO_LOAD_OPTIMIZER}" ]]; then
  args+=(--no_load_optimizer)
fi

if [[ -n "${NO_LOAD_RNG}" ]]; then
  args+=(--no_load_rng)
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: CHECKPOINT_PATH does not exist: %q\n' "${CHECKPOINT_PATH}" >&2
    printf 'Set CHECKPOINT_PATH="" to train without a checkpoint.\n' >&2
    exit 1
  fi
  args+=(-checkpoint_path "${CHECKPOINT_PATH}")
fi

exec "${ISAACLAB_SH}" -p "${REPO_ROOT}/pwm_isaaclab_dfd_v4_lite/train_dfd_v4_lite.py" "${args[@]}" "$@"
