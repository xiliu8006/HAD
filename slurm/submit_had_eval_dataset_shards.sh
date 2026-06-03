#!/usr/bin/env bash
set -euo pipefail

WORLD_SIZE="${1:-24}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/xi9/code/DreamAware3D_open_source}"
SCRIPT_DIR="${PROJECT_ROOT}/slurm"

DATASET="${DATASET:-dl3dv}"
if [ "${DATASET}" = "mipnerf360" ]; then
  DEFAULT_DATA_ROOT="${MIPNERF_DATA_ROOT:-/project/siyuh/common/xiliu/MipNeRF360}"
  DEFAULT_SCENES_FILE="${PROJECT_ROOT}/configs/mipnerf360_scenes.txt"
  DEFAULT_VIEW_FUSION=1
  DEFAULT_MAX_STEPS=20000
  DEFAULT_TARGET_SAMPLE_STEP=1
else
  DEFAULT_DATA_ROOT="/project/siyuh/common/xiliu/DL3DV-10K-Benchmark"
  DEFAULT_SCENES_FILE="${PROJECT_ROOT}/configs/dl3dv_eval_scenes.txt"
  DEFAULT_VIEW_FUSION=3
  DEFAULT_MAX_STEPS=20000
  DEFAULT_TARGET_SAMPLE_STEP=2
fi

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/siyuh/common/xiliu/HAD_CVPR2026_V2/outputs}"
SCENES_FILE="${SCENES_FILE:-${DEFAULT_SCENES_FILE}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
DEFAULT_LVSM_CKPT_PATH="/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512"
LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-${LVSM_CKPT_DIR:-${DEFAULT_LVSM_CKPT_PATH}}}"
SPARSE_VIEW="${SPARSE_VIEW:-9}"
VIEW_FUSION="${VIEW_FUSION:-${DEFAULT_VIEW_FUSION}}"
USE_LVSM="${USE_LVSM:-1}"
DATA_FACTOR="${DATA_FACTOR:-4}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-${DEFAULT_TARGET_SAMPLE_STEP}}"
MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT:-${DATA_ROOT}}"
SPLIT_JSON="${SPLIT_JSON:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
if [ "${DATASET}" = "mipnerf360" ] && [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_mipnerf360_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
elif [ "${DATASET}" = "mipnerf360" ]; then
  METHOD_NAME="dreamaware3d_mipnerf360_no_lvsm_view${SPARSE_VIEW}"
elif [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_lvsm_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
else
  METHOD_NAME="dreamaware3d_no_lvsm_view${SPARSE_VIEW}"
fi
FINAL_STEP="$((MAX_STEPS - 1))"

mapfile -t SCENES < <(grep -vE "^\s*(#|$)" "${SCENES_FILE}")
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

JOB_IDS=()
for RANK in $(seq 0 "$((WORLD_SIZE - 1))"); do
  PENDING=0
  for IDX in "${!SCENES[@]}"; do
    if (( IDX % WORLD_SIZE != RANK )); then
      continue
    fi

    SCENE_ID="${SCENES[IDX]}"
    FINAL_STATS="${OUTPUT_ROOT}/${METHOD_NAME}/${SCENE_ID}/stats/val_step${FINAL_STEP}.json"
    if [ "${FORCE_RERUN}" = "1" ] || [ ! -s "${FINAL_STATS}" ]; then
      PENDING=$((PENDING + 1))
    fi
  done

  if (( PENDING == 0 )); then
    echo "skip submit rank=${RANK}/${WORLD_SIZE}: all assigned scenes already have val_step${FINAL_STEP}.json"
    continue
  fi

  JOB_ID="$(sbatch --parsable \
    --chdir="${PROJECT_ROOT}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATASET="${DATASET}",DATA_ROOT="${DATA_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",SCENES_FILE="${SCENES_FILE}",LVSM_ROOT="${LVSM_ROOT}",LVSM_CKPT_PATH="${LVSM_CKPT_PATH}",SPARSE_VIEW="${SPARSE_VIEW}",VIEW_FUSION="${VIEW_FUSION}",USE_LVSM="${USE_LVSM}",DATA_FACTOR="${DATA_FACTOR}",MAX_STEPS="${MAX_STEPS}",TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP}",MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT}",SPLIT_JSON="${SPLIT_JSON}",FORCE_RERUN="${FORCE_RERUN}" \
    "${SCRIPT_DIR}/run_had_eval_dataset_shard.sh" "${RANK}" "${WORLD_SIZE}")"
  JOB_IDS+=("${JOB_ID}")
  echo "submitted DreamAware3D ${DATASET} eval rank=${RANK}/${WORLD_SIZE} pending=${PENDING} use_lvsm=${USE_LVSM} job_id=${JOB_ID}"
done

if (( ${#JOB_IDS[@]} == 0 )); then
  echo "no jobs submitted: all scenes already have val_step${FINAL_STEP}.json"
else
  DEPENDENCY="$(IFS=:; echo "${JOB_IDS[*]}")"
  echo "submitted ${#JOB_IDS[@]} single-GPU shard jobs dependency_group=afterany:${DEPENDENCY}"
fi
