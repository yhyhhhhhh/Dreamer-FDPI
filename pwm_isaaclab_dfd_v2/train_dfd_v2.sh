#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ISAACLAB_SH="${ISAACLAB_SH:-/home/yhy/IsaacLab-1.4.0/isaaclab.sh}"
UR3_LITE_EXT="${UR3_LITE_EXT:-/home/yhy/surgical_robot_pro1/exts/ur3_lite}"

RUN_NAME="${RUN_NAME:-dfd-v2-continuous-cost}"
SEED="${SEED:-0}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/pwm_isaaclab_dfd_v2/config_dfd_v2.yaml}"
ENV_NAME="${ENV_NAME:-Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/ckpt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/yhy/PaMoRL-main/ckpt/ur3-critic-warmup/20260524_223011_warmup/full_agent_after_critic_warmup.pt}"
CHECKPOINT_PATH="$(printf '%s' "${CHECKPOINT_PATH}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

export TERM="${TERM:-xterm}"
export PYTHONPATH="${UR3_LITE_EXT}:${REPO_ROOT}:${PYTHONPATH:-}"

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

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: CHECKPOINT_PATH does not exist: %q\n' "${CHECKPOINT_PATH}" >&2
    exit 1
  fi
  args+=(-checkpoint_path "${CHECKPOINT_PATH}")
fi

exec "${ISAACLAB_SH}" -p "${REPO_ROOT}/pwm_isaaclab_dfd_v2/train_dfd_v2.py" "${args[@]}" "$@"
