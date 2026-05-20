#!/usr/bin/env bash
set -u

# Run from project root regardless of invocation path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdhd}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
  else
    echo "[Warn] conda not found; using current shell Python (${PYTHON_BIN})"
  fi
fi

# Stable v4 reproduction config (best & most stable batch from prior tuning).
MODEL_NAME="${MODEL_NAME:-pdhd}"
DATASET="${DATASET:-hcp}"
EPOCH="${EPOCH:-80}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DIFF_STEPS="${DIFF_STEPS:-50}"
LEARNING_RATE="${LEARNING_RATE:-7e-6}"
MG_DICT="${MG_DICT:-1_4_8}"
NUM_GRAN="${NUM_GRAN:-3}"
SHARE_RATIO_LIST="${SHARE_RATIO_LIST:-1_0.1_0.1}"
WEIGHT_LIST="${WEIGHT_LIST:-0.8_0.1_0.1}"
GRAPH_PERCENTAGE="${GRAPH_PERCENTAGE:-0.95}"
LOSS_WEIGHT_LIST="${LOSS_WEIGHT_LIST:-0.3_0.05}"
FRACTAL_CONDITION_WEIGHT="${FRACTAL_CONDITION_WEIGHT:-1}"
DIFFUSION_CONDITION_WEIGHT="${DIFFUSION_CONDITION_WEIGHT:-1}"
BETA_END="${BETA_END:-0.1}"
DROPOUT_RATE_RNN="${DROPOUT_RATE_RNN:-0.5}"
SEED="${SEED:-2020}"
NUM_REPS="${NUM_REPS:-5}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-96}"
LAGS_SEQ="${LAGS_SEQ:-1,2,4,8,16,32}"
FREQ="${FREQ:-1H}"
USE_TIME_FEATURES="${USE_TIME_FEATURES:-False}"
USE_HGNN="${USE_HGNN:-True}"
NUM_BATCHES_PER_EPOCH="${NUM_BATCHES_PER_EPOCH:-220}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAXIMUM_LEARNING_RATE="${MAXIMUM_LEARNING_RATE:-5e-4}"
CLIP_GRADIENT="${CLIP_GRADIENT:-1.0}"
LOSS_WARMUP_EPOCHS="${LOSS_WARMUP_EPOCHS:-60}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_SEED="${EVAL_SEED:-2020}"
SAVE_CKPT="${SAVE_CKPT:-False}"

# Space-separated GPU IDs, e.g. "0 1 2 3 4"
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4}"
IFS=' ' read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
NUM_GPUS=${#GPU_IDS[@]}

RESULT_PATH="./result/${MODEL_NAME}_${DATASET}"
LOG_PATH="./log/${MODEL_NAME}_${DATASET}"
mkdir -p "${RESULT_PATH}" "${LOG_PATH}"

echo "[Config] model=${MODEL_NAME} dataset=${DATASET} epoch=${EPOCH} batch=${BATCH_SIZE} lr=${LEARNING_RATE} max_lr=${MAXIMUM_LEARNING_RATE} mg=${MG_DICT} share=${SHARE_RATIO_LIST} weight=${WEIGHT_LIST} ctx=${CONTEXT_LENGTH} lags=${LAGS_SEQ} warmup=${LOSS_WARMUP_EPOCHS} eval_samples=${EVAL_NUM_SAMPLES} reps=${NUM_REPS} seed=${SEED}"

idx=0
for rep in $(seq 1 "${NUM_REPS}"); do
  gpu="${GPU_IDS[$((idx % NUM_GPUS))]}"
  run_id=$((1000 + rep))
  tag="stablev4smooth_e${EPOCH}_nw${NUM_WORKERS}_bs${BATCH_SIZE}_hgnn_lr7e6_maxlr5e4_clip1_warm${LOSS_WARMUP_EPOCHS}_rep${rep}"
  log_file="${LOG_PATH}/gran_${NUM_GRAN}_${tag}_mg_${MG_DICT}_gp_${GRAPH_PERCENTAGE}_share_${SHARE_RATIO_LIST}_weight_${WEIGHT_LIST}_lr_${LEARNING_RATE}_seed_${SEED}_ctx_${CONTEXT_LENGTH}_lags_${LAGS_SEQ//,/-}.txt"

  echo "[Run] rep=${rep}/${NUM_REPS} gpu=${gpu} run_id=${run_id} log=$(basename "${log_file}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u src/run_pdhd_hcp.py \
    --result_path "${RESULT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --epoch "${EPOCH}" \
    --cuda_num 0 \
    --dataset "${DATASET}" \
    --freq "${FREQ}" \
    --use_time_features "${USE_TIME_FEATURES}" \
    --diff_steps "${DIFF_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --maximum_learning_rate "${MAXIMUM_LEARNING_RATE}" \
    --clip_gradient "${CLIP_GRADIENT}" \
    --loss_warmup_epochs "${LOSS_WARMUP_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_batches_per_epoch "${NUM_BATCHES_PER_EPOCH}" \
    --num_workers "${NUM_WORKERS}" \
    --context_length "${CONTEXT_LENGTH}" \
    --lags_seq "${LAGS_SEQ}" \
    --mg_dict "${MG_DICT}" \
    --num_gran "${NUM_GRAN}" \
    --share_ratio_list "${SHARE_RATIO_LIST}" \
    --weight_list "${WEIGHT_LIST}" \
    --graph_percentage "${GRAPH_PERCENTAGE}" \
    --run_num "${run_id}" \
    --seed "${SEED}" \
    --eval_seed "${EVAL_SEED}" \
    --eval_num_samples "${EVAL_NUM_SAMPLES}" \
    --use_hgnn "${USE_HGNN}" \
    --save_ckpt "${SAVE_CKPT}" \
    --log_metrics False \
    --loss_weight_list "${LOSS_WEIGHT_LIST}" \
    --fractal_condition_weight "${FRACTAL_CONDITION_WEIGHT}" \
    --diffusion_condition_weight "${DIFFUSION_CONDITION_WEIGHT}" \
    --beta_end "${BETA_END}" \
    --dropout_rate_rnn "${DROPOUT_RATE_RNN}" \
    > "${log_file}" 2>&1 &

  if [ $(((idx + 1) % NUM_GPUS)) -eq 0 ]; then
    wait
    echo "[Batch done] launched $((idx + 1)) runs so far"
  fi
  idx=$((idx + 1))
done

wait
echo "[Done] All reproducibility runs finished."
