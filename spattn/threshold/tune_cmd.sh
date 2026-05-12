#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Usage
usage() {
  echo "Usage: $0 <GPU_IDS> [--no-tune] [--no-redirect] [--sp <value>] [--model <name>] [--subdir <name>] [--extra-common-args \"<args>\"]"
  echo "Examples:"
  echo "  $0 0,1 --no-tune"
  echo "  $0 0,1 --model Wan --subdir positive"
  echo "  $0 0,1 --extra-common-args \"--block-size-q-list 64 --block-size-k-list 16 --group-size-k-list 16\""
}

# Parse first positional argument: GPU IDS
GPU_IDS=${1:-0}
if [[ $# -gt 0 ]]; then shift; fi

# Parse optional arguments
DO_TUNE=1                 # Default: perform tuning
NO_REDIRECT=0             # Default: redirect output to log files
DP_TARGET_SPARSITY=""     # Default: not specified
MODEL="${MODEL:-Wan}"     # Can be overridden via env var, default Wan
SUBDIR="${SUBDIR:-}"      # Can be overridden via env var, default empty
EXTRA_COMMON_ARGS_ALL=()  # Accumulated extra common arguments

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-tune)
      DO_TUNE=0
      shift
      ;;
    --no-redirect)
      NO_REDIRECT=1
      shift
      ;;
    --sp)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --sp requires a value"
        usage
        exit 1
      fi
      DP_TARGET_SPARSITY="$2"
      shift 2
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --model requires a value"
        usage
        exit 1
      fi
      MODEL="$2"
      shift 2
      ;;
    --subdir)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --subdir requires a value"
        usage
        exit 1
      fi
      SUBDIR="$2"
      shift 2
      ;;
    --extra-common-args)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --extra-common-args requires a quoted argument list"
        usage
        exit 1
      fi
      # Split the string into an array and accumulate
      tmp="$2"
      read -r -a tok <<< "${tmp}"
      EXTRA_COMMON_ARGS_ALL+=("${tok[@]}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# Get DP state
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

TOTAL_CORES=$(nproc)
OMP_THREADS=$(( TOTAL_CORES / (NUM_GPUS > 0 ? NUM_GPUS : 1) ))
OMP_THREADS=$(( OMP_THREADS > 0 ? OMP_THREADS : 1 ))
export OMP_NUM_THREADS=${OMP_THREADS}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[INFO] GPUs=${GPU_IDS} (count=${NUM_GPUS}) OMP_NUM_THREADS=${OMP_THREADS} tune=$([[ ${DO_TUNE} -eq 1 ]] && echo on || echo off) redirect=$([[ ${NO_REDIRECT} -eq 0 ]] && echo on || echo off) dp_target_sparsity=$([[ -n "${DP_TARGET_SPARSITY}" ]] && echo ${DP_TARGET_SPARSITY} || echo none) model=${MODEL} subdir=$([[ -n \"${SUBDIR}\" ]] && echo ${SUBDIR} || echo none)"

# Common arguments
PY_SCRIPT="${REPO_ROOT}/spattn/threshold/model_tuner.py"
COMMON_ARGS=(--bsearch_target_recall 0.9 --prompt_to_list 0 1 2 --nd 6)

# Only pass dp_target_sparsity when specified
if [[ -n "${DP_TARGET_SPARSITY}" ]]; then
  COMMON_ARGS+=(--dp_target_sparsity "${DP_TARGET_SPARSITY}")
fi

# Append extra common arguments
if [[ ${#EXTRA_COMMON_ARGS_ALL[@]} -gt 0 ]]; then
  COMMON_ARGS+=("${EXTRA_COMMON_ARGS_ALL[@]}")
fi

TORCHRUN_PREFIX=(uv run --group dit torchrun --standalone --nproc_per_node "${NUM_GPUS}")

# Append arguments based on tune toggle
TUNE_ARGS=()
if [[ "${DO_TUNE}" -eq 0 ]]; then
  TUNE_ARGS+=(--no-tune)
fi

# Log root directory
LOG_ROOT="${REPO_ROOT}/spattn/threshold/logs"
mkdir -p "${LOG_ROOT}"

run_model() {
  local model="$1"; shift
  local qk_dir="$1"; shift
  local extra_args=("$@")

  local model_log_dir="${LOG_ROOT}/${model}"
  mkdir -p "${model_log_dir}"

  # Log filenames differentiated by timestamp
  local ts
  ts=$(date +"%Y%m%d_%H%M%S")

  local stdout_log="${model_log_dir}/${model}_${ts}.out.log"
  local stderr_log="${model_log_dir}/${model}_${ts}.err.log"

  echo "[RUN] model=${model} QK_dir=${qk_dir} logs: ${stdout_log}, ${stderr_log} (redirect=$([[ ${NO_REDIRECT} -eq 0 ]] && echo on || echo off))"

  if [[ "${NO_REDIRECT}" -eq 1 ]]; then
    "${TORCHRUN_PREFIX[@]}" "${PY_SCRIPT}" \
      --QK_dir "${qk_dir}" \
      --model "${model}" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      "${TUNE_ARGS[@]}"
  else
    "${TORCHRUN_PREFIX[@]}" "${PY_SCRIPT}" \
      --QK_dir "${qk_dir}" \
      --model "${model}" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      "${TUNE_ARGS[@]}" \
      1>"${stdout_log}" \
      2>"${stderr_log}"
  fi
}

# Resolve model-specific QK_Cache directory name
MODEL_DIR=""
case "${MODEL,,}" in
  wan) MODEL_DIR="Wan" ;;
  huanyuanvideo|hunyuanvideo|hunyuan) MODEL_DIR="HunyuanVideo" ;;
  *) MODEL_DIR="${MODEL}" ;; # Fallback: use model name as directory
esac

QK_BASE="${REPO_ROOT}/spattn/threshold/QK_Cache/${MODEL_DIR}"
QK_DIR="${QK_BASE}"
EXP_ARGS=()

if [[ -n "${SUBDIR}" ]]; then
  QK_DIR="${QK_BASE}/${SUBDIR}"
  EXP_ARGS+=(--exp_name "${SUBDIR}")
fi

# Execute
run_model "${MODEL}" "${QK_DIR}" "${EXP_ARGS[@]}"
