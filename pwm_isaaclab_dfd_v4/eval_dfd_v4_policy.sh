#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ISAACLAB_SH="${ISAACLAB_SH:-/home/yhy/IsaacLab-1.4.0/isaaclab.sh}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd -- "$(dirname -- "${ISAACLAB_SH}")" && pwd)}"
ISAACLAB_EXTS="${ISAACLAB_ROOT}/source/extensions"
CONDA_SITE_PACKAGES="${CONDA_SITE_PACKAGES:-/home/yhy/anaconda3/envs/isaaclab_14/lib/python3.10/site-packages}"
UR3_LITE_EXT="${UR3_LITE_EXT:-/home/yhy/surgical_robot_pro1/exts/ur3_lite}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/pwm_isaaclab_dfd_v4/config_dfd_v4.yaml}"
ENV_NAME="${ENV_NAME:-Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-16}"
EVAL_STEPS="${EVAL_STEPS:-65536}"
EVAL_EPISODES="${EVAL_EPISODES:-}"
MAX_ITERS="${MAX_ITERS:-}"
POLICY="${POLICY:-main}"
STOCHASTIC="${STOCHASTIC:-}"
GP_SHIELD="${GP_SHIELD:-}"
SHIELD_THRESHOLD="${SHIELD_THRESHOLD:-}"
SHIELD_CANDIDATES="${SHIELD_CANDIDATES:-16}"
SHIELD_MIN_IMPROVEMENT="${SHIELD_MIN_IMPROVEMENT:-0.0}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/eval_results/dfd_v4}"
V4_FULL_CHECKPOINT_PATH="${V4_FULL_CHECKPOINT_PATH:-${REPO_ROOT}/ckpt/dfd-v4-fixed-continue-full/20260601_085833_DFD-v4-fixed-continue-to-original-10M-ta/full_state_v4_2249856.pth}"

if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi
export PYTHONPATH="${UR3_LITE_EXT}:${REPO_ROOT}:${ISAACLAB_EXTS}/omni.isaac.lab:${ISAACLAB_EXTS}/omni.isaac.lab_tasks:${ISAACLAB_EXTS}/omni.isaac.lab_assets:${CONDA_SITE_PACKAGES}:${PYTHONPATH:-}"

if [[ ! -f "${V4_FULL_CHECKPOINT_PATH}" ]]; then
  printf 'ERROR: V4_FULL_CHECKPOINT_PATH does not exist: %q\n' "${V4_FULL_CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'ERROR: CONFIG_PATH does not exist: %q\n' "${CONFIG_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

args=(
  --v4_full_checkpoint_path "${V4_FULL_CHECKPOINT_PATH}"
  -config_path "${CONFIG_PATH}"
  -env_name "${ENV_NAME}"
  -device "${DEVICE}"
  -seed "${SEED}"
  --num_envs "${EVAL_NUM_ENVS}"
  --eval_steps "${EVAL_STEPS}"
  --policy "${POLICY}"
  --save_dir "${SAVE_DIR}"
)

if [[ -n "${EVAL_EPISODES}" ]]; then
  args+=(--eval_episodes "${EVAL_EPISODES}")
fi

if [[ -n "${MAX_ITERS}" ]]; then
  args+=(--max_iters "${MAX_ITERS}")
fi

if [[ -n "${STOCHASTIC}" ]]; then
  args+=(--stochastic)
fi

if [[ -n "${GP_SHIELD}" ]]; then
  args+=(--gp_shield --shield_candidates "${SHIELD_CANDIDATES}" --shield_min_improvement "${SHIELD_MIN_IMPROVEMENT}")
  if [[ -n "${SHIELD_THRESHOLD}" ]]; then
    args+=(--shield_threshold "${SHIELD_THRESHOLD}")
  fi
fi

exec "${ISAACLAB_SH}" -p "${REPO_ROOT}/pwm_isaaclab_dfd_v4/eval_dfd_v4_policy.py" "${args[@]}" "$@"
