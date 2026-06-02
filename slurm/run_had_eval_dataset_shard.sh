#!/bin/bash
#SBATCH --job-name=had-dreamaware
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --gpus=a100:1
#SBATCH --output=/home/xi9/code/DreamAware3D_open_source/logs/%x-%j.out
#SBATCH --error=/home/xi9/code/DreamAware3D_open_source/logs/%x-%j.err

set -eo pipefail

RANK="${1:?usage: sbatch run_had_eval_dataset_shard.sh RANK WORLD_SIZE}"
WORLD_SIZE="${2:?usage: sbatch run_had_eval_dataset_shard.sh RANK WORLD_SIZE}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/xi9/code/DreamAware3D_open_source}"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
DEFAULT_LVSM_CKPT_PATH="/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512"
LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-${LVSM_CKPT_DIR:-${DEFAULT_LVSM_CKPT_PATH}}}"
DATA_ROOT="${DATA_ROOT:-/project/siyuh/common/xiliu/DL3DV-10K-Benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/siyuh/common/xiliu/HAD_CVPR2026_V2/outputs}"
SCENES_FILE="${SCENES_FILE:-${PROJECT_ROOT}/configs/had_eval_scenes.txt}"

SPARSE_VIEW="${SPARSE_VIEW:-9}"
VIEW_FUSION="${VIEW_FUSION:-3}"
USE_LVSM="${USE_LVSM:-1}"
DATA_FACTOR="${DATA_FACTOR:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"
if [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_lvsm_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  CONF_FLAG="--use_conf"
  LVSM_FLAG="--use_lvsm"
else
  METHOD_NAME="dreamaware3d_no_lvsm_view${SPARSE_VIEW}"
  CONF_FLAG="--no-use_conf"
  LVSM_FLAG="--no-use_lvsm"
fi
FINAL_STEP="$((MAX_STEPS - 1))"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [ ! -f "${SCENES_FILE}" ]; then
  echo "Missing scene list: ${SCENES_FILE}" >&2
  exit 2
fi

if [ "${USE_LVSM}" = "1" ]; then
  if [ -d "${LVSM_CKPT_PATH}" ]; then
    if [ -z "$(find "${LVSM_CKPT_PATH}" -maxdepth 1 -name '*.pt' -print -quit)" ]; then
      echo "Missing LVSM checkpoint .pt files in ${LVSM_CKPT_PATH}" >&2
      exit 2
    fi
  elif [ ! -f "${LVSM_CKPT_PATH}" ]; then
    echo "Missing LVSM checkpoint path: ${LVSM_CKPT_PATH}" >&2
    exit 2
  fi
fi

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH="$HOME/miniconda3/bin:$PATH"
source activate difix3D

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/${METHOD_NAME}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/gsplat:${PROJECT_ROOT}/examples/gsplat/pycolmap:$(dirname "${LVSM_ROOT}"):$PYTHONPATH"
export LVSM_ROOT="${LVSM_ROOT}"
export LVSM_CKPT_PATH="${LVSM_CKPT_PATH}"

mapfile -t SCENES < <(grep -vE "^\s*(#|$)" "${SCENES_FILE}")

for IDX in "${!SCENES[@]}"; do
  if (( IDX % WORLD_SIZE != RANK )); then
    continue
  fi

  SCENE_ID="${SCENES[IDX]}"
  DATA="${DATA_ROOT}/${SCENE_ID}/nerfstudio"
  OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_NAME}/${SCENE_ID}"
  FINAL_STATS="${OUTPUT_DIR}/stats/val_step${FINAL_STEP}.json"

  if [ "${FORCE_RERUN}" != "1" ] && [ -s "${FINAL_STATS}" ]; then
    echo "Skipping completed scene: ${SCENE_ID}"
    echo "Found: ${FINAL_STATS}"
    continue
  fi

  if [ ! -d "${DATA}" ]; then
    echo "Skipping missing data directory: ${DATA}" >&2
    continue
  fi

  mkdir -p "${OUTPUT_DIR}"
  echo "Processing scene: ${SCENE_ID} rank=${RANK}/${WORLD_SIZE} use_lvsm=${USE_LVSM}"
  echo "Data dir: ${DATA}"
  echo "Output dir: ${OUTPUT_DIR}"
  if [ "${USE_LVSM}" = "1" ]; then
    echo "LVSM checkpoint path: ${LVSM_CKPT_PATH}"
  fi

  if python "${PROJECT_ROOT}/examples/gsplat/train_dreamaware3d.py" mcmc \
      --data_dir "${DATA}" \
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
      --max_steps "${MAX_STEPS}" \
      --view_fusion "${VIEW_FUSION}"; then
    echo "Completed scene: ${SCENE_ID}"
  else
    STATUS="$?"
    echo "Failed scene: ${SCENE_ID} status=${STATUS}" >&2
    echo "Continuing with remaining scenes assigned to rank=${RANK}/${WORLD_SIZE}" >&2
  fi
done
