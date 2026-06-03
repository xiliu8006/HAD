#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./run_had_eval_dataset.sh [WORLD_SIZE] [SPARSE_VIEW] [LVSM_CKPT_PATH]

Examples:
  ./run_had_eval_dataset.sh 24 9 /path/to/hallucination_scoring/checkpoint_dir
  DATASET=mipnerf360 ./run_had_eval_dataset.sh 9 9 /path/to/checkpoint_dir

The checkpoint path may be either a directory containing .pt files or a single
.pt checkpoint file. It can also be provided through LVSM_CKPT_PATH or
LVSM_CKPT_DIR environment variables.
USAGE
  exit 0
fi

WORLD_SIZE="${1:-24}"
SPARSE_VIEW="${2:-${SPARSE_VIEW:-9}}"

# Hard-coded local default checkpoint path used on our machine. Pass the third
# argument, LVSM_CKPT_PATH, or LVSM_CKPT_DIR to use a different checkpoint.
DEFAULT_LVSM_CKPT_PATH="/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512"
LVSM_CKPT_PATH="${3:-${LVSM_CKPT_PATH:-${LVSM_CKPT_DIR:-${DEFAULT_LVSM_CKPT_PATH}}}}"

# Hard-coded local default for the release repository path.
# Change this path if the repository is cloned somewhere else.
PROJECT_ROOT="/home/xi9/code/DreamAware3D_open_source"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
DATASET="${DATASET:-dl3dv}"

if [ "${DATASET}" = "mipnerf360" ]; then
  # Hard-coded Mip-NeRF 360 default path on our machine.
  DEFAULT_DATA_ROOT="${MIPNERF_DATA_ROOT:-/project/siyuh/common/xiliu/MipNeRF360}"
  DEFAULT_SCENES_FILE="${PROJECT_ROOT}/configs/mipnerf360_scenes.txt"
  DEFAULT_VIEW_FUSION=1
  DEFAULT_MAX_STEPS=20000
  DEFAULT_TARGET_SAMPLE_STEP=1
else
  # Hard-coded DL3DV default path on our machine.
  DEFAULT_DATA_ROOT="/project/siyuh/common/xiliu/DL3DV-10K-Benchmark"
  DEFAULT_SCENES_FILE="${PROJECT_ROOT}/configs/dl3dv_eval_scenes.txt"
  DEFAULT_VIEW_FUSION=3
  DEFAULT_MAX_STEPS=20000
  DEFAULT_TARGET_SAMPLE_STEP=2
fi

# Hard-coded output root used on our machine; override OUTPUT_ROOT if needed.
DATASET="${DATASET}" \
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/siyuh/common/xiliu/HAD_CVPR2026_V2/outputs}" \
SCENES_FILE="${SCENES_FILE:-${DEFAULT_SCENES_FILE}}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
LVSM_ROOT="${LVSM_ROOT}" \
LVSM_CKPT_PATH="${LVSM_CKPT_PATH}" \
SPARSE_VIEW="${SPARSE_VIEW}" \
VIEW_FUSION="${VIEW_FUSION:-${DEFAULT_VIEW_FUSION}}" \
USE_LVSM="${USE_LVSM:-1}" \
DATA_FACTOR="${DATA_FACTOR:-4}" \
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}" \
TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-${DEFAULT_TARGET_SAMPLE_STEP}}" \
MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT:-}" \
SPLIT_JSON="${SPLIT_JSON:-}" \
FORCE_RERUN="${FORCE_RERUN:-0}" \
  "${PROJECT_ROOT}/slurm/submit_had_eval_dataset_shards.sh" "${WORLD_SIZE}"
