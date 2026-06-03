#!/usr/bin/env bash
set -eo pipefail

DATASET="${DATASET:-dl3dv}"

# Hard-coded local default for the released training entry point.
# Change this path if the repository is cloned somewhere else.
PROJECT_ROOT="/home/xi9/code/DreamAware3D_open_source"

# Hard-coded local defaults used on our machine. Override these environment
# variables when running with a different dataset, output root, or checkpoint.
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/siyuh/common/xiliu/HAD_CVPR2026_V2/outputs}"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512}"

SCENE="${1:?usage: ./run_train_scene.sh SCENE_ID_OR_DATA_DIR [SPARSE_VIEW] [MAX_STEPS]}"
SPARSE_VIEW="${2:-${SPARSE_VIEW:-9}}"
USE_LVSM="${USE_LVSM:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SPLIT_JSON="${SPLIT_JSON:-}"

if [ "${DATASET}" = "mipnerf360" ]; then
  # Hard-coded Mip-NeRF 360 default path on our machine.
  DATA_ROOT="${DATA_ROOT:-${MIPNERF_DATA_ROOT:-/project/siyuh/common/xiliu/MipNeRF360}}"
  MAX_STEPS="${3:-${MAX_STEPS:-20000}}"
  VIEW_FUSION="${VIEW_FUSION:-1}"
  TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-1}"
else
  # Hard-coded DL3DV default path on our machine.
  DATA_ROOT="${DATA_ROOT:-/project/siyuh/common/xiliu/DL3DV-10K-Benchmark}"
  MAX_STEPS="${3:-${MAX_STEPS:-20000}}"
  VIEW_FUSION="${VIEW_FUSION:-3}"
  TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-2}"
fi
DATA_FACTOR="${DATA_FACTOR:-4}"

if [[ "${SCENE}" == */nerfstudio ]]; then
  DATA_DIR="${SCENE}"
  SCENE_ID="$(basename "$(dirname "${SCENE}")")"
elif [ -d "${SCENE}" ]; then
  DATA_DIR="${SCENE}"
  SCENE_ID="$(basename "${SCENE}")"
elif [ "${DATASET}" = "mipnerf360" ]; then
  SCENE_ID="${SCENE}"
  DATA_DIR="${DATA_ROOT}/${SCENE_ID}"
else
  SCENE_ID="${SCENE}"
  DATA_DIR="${DATA_ROOT}/${SCENE_ID}/nerfstudio"
fi

if [ "${USE_LVSM}" = "1" ]; then
  CONF_FLAG="--use_conf"
else
  CONF_FLAG="--no-use_conf"
fi
LVSM_FLAG="--use_lvsm"

if [ "${DATASET}" = "mipnerf360" ] && [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_mipnerf360_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
elif [ "${DATASET}" = "mipnerf360" ]; then
  METHOD_NAME="dreamaware3d_mipnerf360_no_lvsm_view${SPARSE_VIEW}"
elif [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_lvsm_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
else
  METHOD_NAME="dreamaware3d_no_lvsm_view${SPARSE_VIEW}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_NAME}/${SCENE_ID}"
FINAL_STATS="${OUTPUT_DIR}/stats/val_step$((MAX_STEPS - 1)).json"

if [ "${FORCE_RERUN}" != "1" ] && [ -s "${FINAL_STATS}" ]; then
  echo "Skip completed scene: ${SCENE_ID}"
  echo "Found: ${FINAL_STATS}"
  exit 0
fi

if [ ! -d "${DATA_DIR}" ]; then
  echo "Missing data dir: ${DATA_DIR}" >&2
  exit 2
fi

if [ "${DATASET}" = "mipnerf360" ] && [ -z "${SPLIT_JSON}" ]; then
  MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT:-${DATA_ROOT}}"
  SPLIT_JSON="${MIPNERF_SPLIT_ROOT}/${SCENE_ID}/train_test_split_${SPARSE_VIEW}.json"
fi

SPLIT_ARGS=()
if [ -n "${SPLIT_JSON}" ]; then
  if [ ! -f "${SPLIT_JSON}" ]; then
    echo "Missing split json: ${SPLIT_JSON}" >&2
    echo "Set SPLIT_JSON or MIPNERF_SPLIT_ROOT for MipNeRF360 splits." >&2
    exit 2
  fi
  SPLIT_ARGS=(--split_json "${SPLIT_JSON}")
fi

if [ "${USE_LVSM}" = "1" ] && [ ! -e "${LVSM_CKPT_PATH}" ]; then
  echo "Missing LVSM checkpoint path: ${LVSM_CKPT_PATH}" >&2
  exit 2
fi

# Hard-coded environment setup for our cluster. Adapt these lines if your CUDA
# module name or conda environment is different.
source /etc/profile.d/modules.sh 2>/dev/null || true
module add cuda/11.8.0 2>/dev/null || true
export PATH="$HOME/miniconda3/bin:$PATH"
source activate difix3D

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/gsplat:${PROJECT_ROOT}/examples/gsplat/pycolmap:$(dirname "${LVSM_ROOT}"):$PYTHONPATH"
export LVSM_ROOT="${LVSM_ROOT}"
export LVSM_CKPT_PATH="${LVSM_CKPT_PATH}"

echo "Scene: ${SCENE_ID}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "DATASET=${DATASET} USE_LVSM=${USE_LVSM} SPARSE_VIEW=${SPARSE_VIEW} VIEW_FUSION=${VIEW_FUSION} MAX_STEPS=${MAX_STEPS}"
if [ -n "${SPLIT_JSON}" ]; then
  echo "Split json: ${SPLIT_JSON}"
fi

python "${PROJECT_ROOT}/examples/gsplat/train_dreamaware3d.py" mcmc \
  --data_dir "${DATA_DIR}" \
  --data_factor "${DATA_FACTOR}" \
  --result_dir "${OUTPUT_DIR}" \
  --no-use_eval \
  --no-use_pefect_conf \
  ${CONF_FLAG} \
  --no-partial_setting \
  ${LVSM_FLAG} \
  --no-lvsm_mode \
  --no-normalize-world-space \
  --num_sparse_view "${SPARSE_VIEW}" \
  --target_sample_step "${TARGET_SAMPLE_STEP}" \
  --max_steps "${MAX_STEPS}" \
  --view_fusion "${VIEW_FUSION}" \
  "${SPLIT_ARGS[@]}"
