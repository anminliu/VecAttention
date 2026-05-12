#!/usr/bin/env bash
# Usage:
# bash run_single_th.sh <gpuids> [methods] [backends] [th] [sample_num] [output_root] [prompt_ids] [resolution] [infer_step]
#
# Arguments:
# gpuids        Required. Comma-separated GPU indices.
# methods       Optional. Comma-separated method list. Default: dense,xattn,SVG,SAP,vecattention
# backends      Optional. Comma-separated backend list. Default: hyvideo,wan
# th            Optional. Single threshold / density / sparsity value to run. Default: 0.9
#               Mapping per method:
#                 xattn            → --xattn-threshold
#                 SVG/svg          → --density
#                 SAP/sap          → --top-p-k
#                 vecattention_wo_DP → --threshold
#                 vecattention     → sparsity target S used to locate the threshold file
# sample_num    Optional. Number of prompts to sample. Ignored if prompt_ids is set.
# output_root   Optional. Override output root directory.
# prompt_ids    Optional. Comma-separated prompt id list (test mode).
# resolution    Optional. Default: 720p
# infer_step    Optional. Default: 50
#
# Examples:
#   bash run_single_th.sh 0 "vecattention" "hyvideo" 0.55
#   bash run_single_th.sh 0,1 "xattn,dense" "hyvideo,wan" 0.9 20

set -euo pipefail

GPU_IDS=${1:-0}
METHODS_OVERRIDE=${2:-""}
BACKENDS_OVERRIDE=${3:-""}
TH=${4:-0.9}
SAMPLE_NUM=${5:-20}
OUTPUT_ROOT_OVERRIDE=${6:-""}
PROMPT_IDS_RAW=${7:-""}
RESOLUTION=${8:-"720p"}
INFER_STEP=${9:-50}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_EVAL="${REPO_ROOT}/eval/DiTEvalKit/run_t2v_eval.py"
PROMPT_PATH="${REPO_ROOT}/eval/DiTEvalKit/hunyuan_all_dimension.txt"



DEFAULT_METHODS=(
  dense
  xattn
  svg
  vecattention
)
DEFAULT_BACKENDS=(hyvideo wan)

if [ -n "${METHODS_OVERRIDE}" ]; then
  IFS=',' read -ra METHODS <<< "${METHODS_OVERRIDE}"
else
  METHODS=("${DEFAULT_METHODS[@]}")
fi
if [ -n "${BACKENDS_OVERRIDE}" ]; then
  IFS=',' read -ra BACKENDS <<< "${BACKENDS_OVERRIDE}"
else
  BACKENDS=("${DEFAULT_BACKENDS[@]}")
fi

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

TOTAL_CORES=$(nproc)
OMP_THREADS=$(( TOTAL_CORES / (NUM_GPUS > 0 ? NUM_GPUS : 1) ))
OMP_THREADS=$(( OMP_THREADS > 0 ? OMP_THREADS : 1 ))
export OMP_NUM_THREADS=${OMP_THREADS}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[INFO] GPUs=${GPU_IDS} (count=${NUM_GPUS}) OMP_NUM_THREADS=${OMP_THREADS}"
echo "[INFO] TH=${TH}"
echo "[INFO] METHODS=${METHODS[*]}"
echo "[INFO] BACKENDS=${BACKENDS[*]}"
echo "[INFO] SAMPLE_NUM=${SAMPLE_NUM} PROMPT_IDS=${PROMPT_IDS_RAW:-<none>} RESOLUTION=${RESOLUTION} INFER_STEP=${INFER_STEP}"

PROMPT_IDS_ARG=()
if [ -n "${PROMPT_IDS_RAW}" ]; then
  PROMPT_IDS_ARG=(--prompt-ids "${PROMPT_IDS_RAW}")
fi

build_common_args() {
  local backend="$1"
  local output_root="$2"
  local first_times_fp="0.06"
  local first_layers_fp="0"
  local arr=(
    --backend "${backend}"
    --prompt-path "${PROMPT_PATH}"
    --resolution "${RESOLUTION}"
    --infer-step "${INFER_STEP}"
    --first-times-fp "${first_times_fp}"
    --first-layers-fp "${first_layers_fp}"
    --skip-existing
    --dist-split-mode round_robin
  )
  if [ ${#PROMPT_IDS_ARG[@]} -gt 0 ]; then
    arr+=("${PROMPT_IDS_ARG[@]}")
  else
    arr+=(--sample-num "${SAMPLE_NUM}")
  fi
  if [ -n "${output_root}" ]; then
    arr+=(--output-root "${output_root}")
  fi
  echo "${arr[@]}"
}

ts() { date +"%Y-%m-%d %H:%M:%S"; }

run_with_torchrun() {
  local backend="$1"; shift
  local method="$1"; shift
  local variant_tag="$1"; shift
  shift
  local output_root="$1"; shift

  local common_args=($(build_common_args "${backend}" "${output_root}"))

  local log_dir="${output_root}/logs/${backend}/${method}"
  mkdir -p "${log_dir}"
  local safe_tag
  safe_tag="$(echo "${variant_tag}" | tr '/' '_' | tr ' ' '_')"
  local log_out="${log_dir}/${safe_tag}.out"
  local log_err="${log_dir}/${safe_tag}.err"

  echo "[RUN ] $(ts) backend=${backend} method=${method} variant=${variant_tag}"
  { echo "===== RUN START $(ts) backend=${backend} method=${method} variant=${variant_tag} ====="; } > "${log_out}"
  { echo "===== RUN START $(ts) backend=${backend} method=${method} variant=${variant_tag} ====="; } > "${log_err}"

  local uv_extra=()
  if [[ "${method}" == "SVG" || "${method}" == "svg" ]]; then
    uv_extra=(--with "triton==3.3.1")
  fi

  set +e
  uv run --group dit "${uv_extra[@]}" torchrun --standalone --nproc_per_node "${NUM_GPUS}" \
    "${RUN_EVAL}" \
    --method "${method}" \
    "${common_args[@]}" \
    "$@" \
    1>> "${log_out}" 2>> "${log_err}"
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "[FAIL] backend=${backend} method=${method} variant=${variant_tag} rc=${rc} (see ${log_err})"
    { echo "===== RUN FAIL $(ts) rc=${rc} ====="; } >> "${log_err}"
  else
    echo "[DONE] $(ts) backend=${backend} method=${method} variant=${variant_tag}"
    { echo "===== RUN DONE $(ts) rc=0 ====="; } >> "${log_out}"
  fi
}

for BACKEND in "${BACKENDS[@]}"; do
  echo "[INFO] Backend=${BACKEND}"
  if [ -n "${OUTPUT_ROOT_OVERRIDE}" ]; then
    OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE}"
  else
    OUTPUT_ROOT="results"
  fi

  for METHOD in "${METHODS[@]}"; do
    case "${METHOD}" in
      dense)
        run_with_torchrun "${BACKEND}" dense "baseline" _ "${OUTPUT_ROOT}" \
         
        ;;
      xattn)
        run_with_torchrun "${BACKEND}" xattn "th_${TH}" _ "${OUTPUT_ROOT}" \
          --xattn-threshold "${TH}" \
          --xattn-stride 8 \
          --xattn-chunk-size 65536 \
         
        ;;
      SVG|svg)
        run_with_torchrun "${BACKEND}" SVG "density_${TH}" _ "${OUTPUT_ROOT}" \
          --density "${TH}" \
          --num-sampled-rows 64 \
          --add-logging-file \
         
        ;;
      vecattention_wo_DP|segsp_wo_DP)
        BLOCK_SIZE_Q="${BLOCK_SIZE_Q:-64}"
        BLOCK_SIZE_K="${BLOCK_SIZE_K:-16}"
        GROUP_K_BLOCK="${GROUP_K_BLOCK:-8192}"
        if [[ "${BACKEND}" == "hyvideo" ]]; then
          VEC_CHUNK_SIZE="${VEC_CHUNK_SIZE:-32768}"
        else
          VEC_CHUNK_SIZE="${VEC_CHUNK_SIZE:-65536}"
        fi
        run_with_torchrun "${BACKEND}" vecattention "th_${TH}_wo_DP" _ "${OUTPUT_ROOT}" \
          --block-size-q "${BLOCK_SIZE_Q}" \
          --block-size-k "${BLOCK_SIZE_K}" \
          --group-k-block "${GROUP_K_BLOCK}" \
          --vec-chunk-size "${VEC_CHUNK_SIZE}" \
          --threshold "${TH}" \
         
        ;;
      vecattention|segsp)
        BLOCK_SIZE_Q="${BLOCK_SIZE_Q:-64}"
        BLOCK_SIZE_K="${BLOCK_SIZE_K:-16}"
        GROUP_K_BLOCK="${GROUP_K_BLOCK:-8192}"
        if [[ "${BACKEND}" == "hyvideo" ]]; then
          VEC_CHUNK_SIZE="${VEC_CHUNK_SIZE:-32768}"
        else
          VEC_CHUNK_SIZE="${VEC_CHUNK_SIZE:-65536}"
        fi

        GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
        WAN_USE_SINGLE_POS=0
        if [[ -n "${GUIDANCE_SCALE}" ]]; then
          if awk "BEGIN{exit !(${GUIDANCE_SCALE} < 1)}"; then
            WAN_USE_SINGLE_POS=1
          fi
        fi

        POS_DIR="${POS_DIR:-}"
        NEG_DIR="${NEG_DIR:-}"
        if [[ -z "${POS_DIR}" || -z "${NEG_DIR}" ]]; then
          case "${BACKEND}" in
            hyvideo)
              POS_DIR="${POS_DIR:-${REPO_ROOT}/spattn/threshold/tune_cache/HuanyuanVideo/Q${BLOCK_SIZE_Q}_K${BLOCK_SIZE_K}_G${GROUP_K_BLOCK}/per_head_min_recall_0.9/results}"
              ;;
            wan)
              if [[ "${WAN_USE_SINGLE_POS}" -eq 1 ]]; then
                POS_DIR="${POS_DIR:-${REPO_ROOT}/spattn/threshold/tune_cache/Wan/Q${BLOCK_SIZE_Q}_K${BLOCK_SIZE_K}_G${GROUP_K_BLOCK}/per_head_min_recall_0.9/results}"
                NEG_DIR="${NEG_DIR:-}"
              else
                POS_DIR="${POS_DIR:-${REPO_ROOT}/spattn/threshold/tune_cache/Wan/Q${BLOCK_SIZE_Q}_K${BLOCK_SIZE_K}_G${GROUP_K_BLOCK}/positive_per_head_min_recall_0.9/results}"
                NEG_DIR="${NEG_DIR:-${REPO_ROOT}/spattn/threshold/tune_cache/Wan/Q${BLOCK_SIZE_Q}_K${BLOCK_SIZE_K}_G${GROUP_K_BLOCK}/negative_per_head_min_recall_0.9/results}"
              fi
              ;;
            *)
              POS_DIR="${POS_DIR:-${REPO_ROOT}/spattn/threshold/tune_cache/HuanyuanVideo/Q${BLOCK_SIZE_Q}_K${BLOCK_SIZE_K}_G${GROUP_K_BLOCK}/per_head_min_recall_0.9/results}"
              ;;
          esac
        fi

        SP="${TH}"
        POS_GLOB="${POS_DIR}/per_head_min_recall_*_S${SP}_*_thresholds.json"
        POS_FILE=""
        if compgen -G "${POS_GLOB}" > /dev/null; then
          POS_FILE=$(compgen -G "${POS_GLOB}" | head -n 1)
        fi

        if [ "${BACKEND}" = "wan" ] && [[ "${WAN_USE_SINGLE_POS}" -ne 1 ]]; then
          NEG_GLOB="${NEG_DIR}/per_head_min_recall_*_S${SP}_*_thresholds.json"
          NEG_FILE=""
          if compgen -G "${NEG_GLOB}" > /dev/null; then
            NEG_FILE=$(compgen -G "${NEG_GLOB}" | head -n 1)
          fi
          if [ -z "${POS_FILE}" ] || [ -z "${NEG_FILE}" ]; then
            echo "[WARN] Threshold file not found: backend=${BACKEND} S=${SP} pos=${POS_FILE:-<none>} neg=${NEG_FILE:-<none>} (skipping)"
            continue
          fi
          run_with_torchrun "${BACKEND}" vecattention "sparsity_${SP}" _ "${OUTPUT_ROOT}" \
            --block-size-q "${BLOCK_SIZE_Q}" \
            --block-size-k "${BLOCK_SIZE_K}" \
            --group-k-block "${GROUP_K_BLOCK}" \
            --vec-chunk-size "${VEC_CHUNK_SIZE}" \
            --threshold-path "${POS_FILE}" \
            --neg-threshold-path "${NEG_FILE}" \
           
        else
          if [ -z "${POS_FILE}" ]; then
            echo "[WARN] Threshold file not found: backend=${BACKEND} S=${SP} dir=${POS_DIR} (skipping)"
            continue
          fi
          run_with_torchrun "${BACKEND}" vecattention "sparsity_${SP}" _ "${OUTPUT_ROOT}" \
            --block-size-q "${BLOCK_SIZE_Q}" \
            --block-size-k "${BLOCK_SIZE_K}" \
            --group-k-block "${GROUP_K_BLOCK}" \
            --vec-chunk-size "${VEC_CHUNK_SIZE}" \
            --threshold-path "${POS_FILE}" \
           
        fi
        ;;
      all)
        echo "[WARN] 'all' is a placeholder, skipping"
        ;;
      *)
        echo "[WARN] Unknown method: ${METHOD}, skipping"
        ;;
    esac
  done
done
