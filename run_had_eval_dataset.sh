#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./run_had_eval_dataset.sh [WORLD_SIZE] [SPARSE_VIEW] [LVSM_CKPT_PATH]

Examples:
  USE_LVSM=1 ./run_had_eval_dataset.sh 24 9 /path/to/lvsm/checkpoint_dir
  USE_LVSM=0 ./run_had_eval_dataset.sh 24 9

LVSM_CKPT_PATH may be either a checkpoint directory containing .pt files or a
single .pt checkpoint file. It can also be provided through LVSM_CKPT_PATH or
LVSM_CKPT_DIR environment variables.
USAGE
  exit 0
fi

WORLD_SIZE="${1:-24}"
SPARSE_VIEW="${2:-${SPARSE_VIEW:-9}}"
DEFAULT_LVSM_CKPT_PATH="/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512"
LVSM_CKPT_PATH="${3:-${LVSM_CKPT_PATH:-${LVSM_CKPT_DIR:-${DEFAULT_LVSM_CKPT_PATH}}}}"
PROJECT_ROOT="/home/xi9/code/DreamAware3D_open_source"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"

DATA_ROOT="${DATA_ROOT:-/project/siyuh/common/xiliu/DL3DV-10K-Benchmark}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/siyuh/common/xiliu/HAD_CVPR2026_V2/outputs}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
LVSM_ROOT="${LVSM_ROOT}" \
LVSM_CKPT_PATH="${LVSM_CKPT_PATH}" \
SPARSE_VIEW="${SPARSE_VIEW}" \
VIEW_FUSION="${VIEW_FUSION:-3}" \
USE_LVSM="${USE_LVSM:-0}" \
DATA_FACTOR="${DATA_FACTOR:-4}" \
MAX_STEPS="${MAX_STEPS:-20000}" \
FORCE_RERUN="${FORCE_RERUN:-0}" \
  "${PROJECT_ROOT}/slurm/submit_had_eval_dataset_shards.sh" "${WORLD_SIZE}"
