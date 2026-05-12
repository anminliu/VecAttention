#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: dump_qk_layers.sh [options]
Options:
  --backend hyvideo|wan          Select backend (default: wan)
  --dump-step N                  Target dump_step (default: 5)
  --gpus 0,1,2                  Specify GPU list (default: 0,1,2)
  --prompt-ids 0,1,2            Specify prompt ids (default: 0,1,2)
  --output-root PATH            Output directory (default: debug_results)
  --dump-subdir NAME            Subdirectory under DIT_DUMP_QK_DIR base path (default: none, placed directly under base path)
  -h, --help                    Show help

Notes:
- Uses fixed infer-step=50; sets first-times-fp=dump_step/infer-step to reach target.
- If --dump-subdir=positive and backend=wan, the directory is:
  <REPO_ROOT>/spattn/threshold/QK_Cache/Wan/positive

Examples:
  ./dump_qk_layers.sh --backend wan --dump-step 150 --gpus 0,1,2 --dump-subdir positive
  ./dump_qk_layers.sh --backend hyvideo --dump-step 200 --gpus 3,4,7 --prompt-ids 0,2,4
EOF
}

# Default parameters
BACKEND="${BACKEND:-wan}"           # hyvideo | wan
DUMP_STEP="${DUMP_STEP:-5}"
GPUS="${GPUS:-0,1,2}"
DUMP_SUBDIR="${DUMP_SUBDIR:-}"      # Optional subdirectory, e.g. positive
OUTPUT_ROOT="${OUTPUT_ROOT:-debug_results}"
PROMPT_IDS="${PROMPT_IDS:-0,1,2}"
METHOD="dense"
INFER_STEP=50                       # Fixed at 50
FIRST_TIMES_FP=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --dump-step) DUMP_STEP="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --prompt-ids) PROMPT_IDS="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --dump-subdir) DUMP_SUBDIR="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DITEVAL_DIR="${REPO_ROOT}/eval/DiTEvalKit"

# Normalize backend and set DIT_DUMP_QK_DIR base path
BASE_DIR="${REPO_ROOT}/spattn/threshold/QK_Cache"
case "${BACKEND,,}" in
  hyvideo) DUMP_DIR_BASE="${BASE_DIR}/HunyuanVideo"; BACKEND_ARG="hyvideo";;
  wan)     DUMP_DIR_BASE="${BASE_DIR}/Wan";          BACKEND_ARG="wan";;
  *) echo "Unsupported backend: ${BACKEND} (only hyvideo|wan supported)"; exit 1;;
esac

# Compose final DIT_DUMP_QK_DIR (with optional subdirectory)
if [[ -n "${DUMP_SUBDIR}" ]]; then
  DIT_DUMP_QK_DIR="${DUMP_DIR_BASE}/${DUMP_SUBDIR}"
else
  DIT_DUMP_QK_DIR="${DUMP_DIR_BASE}"
fi
mkdir -p "${DIT_DUMP_QK_DIR}"

# Compute FIRST_TIMES_FP such that dump_step = infer_step * first-times-fp
if ! [[ "${DUMP_STEP}" =~ ^[0-9]+$ ]]; then
  echo "Error: --dump-step must be an integer, got '${DUMP_STEP}'"; exit 1
fi
# Compute floating-point first_times_fp, keeping 6 decimal places
FIRST_TIMES_FP="$(awk "BEGIN {printf \"%.6f\", (${DUMP_STEP}-1)/${INFER_STEP}}")"

# Number of processes = number of specified GPUs
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NPROC="${#GPU_ARR[@]}"
if [[ "${NPROC}" -ne 3 ]]; then
  echo "Note: ${NPROC} GPUs selected; script will launch that many processes (expected 3)."
fi

echo "Backend: ${BACKEND_ARG}"
echo "GPU: ${GPUS} (nproc=${NPROC})"
echo "dump_step=${DUMP_STEP} => infer-step=${INFER_STEP}, first-times-fp=${FIRST_TIMES_FP}"
echo "DIT_DUMP_QK_DIR=${DIT_DUMP_QK_DIR}"
echo

# Run
CUDA_VISIBLE_DEVICES="${GPUS}" uv run --group dit \
  torchrun --standalone --nproc_per_node "${NPROC}" \
  "${DITEVAL_DIR}/run_t2v_eval.py" \
  --output-root "${OUTPUT_ROOT}" \
  --method "${METHOD}" \
  --backend "${BACKEND_ARG}" \
  --prompt-ids "${PROMPT_IDS}" \
  --first-times-fp "${FIRST_TIMES_FP}" \
  --infer-step "${INFER_STEP}" \
  --env "DIT_DUMP_QK_DIR=${DIT_DUMP_QK_DIR}"