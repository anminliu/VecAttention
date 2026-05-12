#!/usr/bin/env bash
# Quick demo / smoke test for DiTEvalKit video generation evaluation.
# Runs vecattention_wo_DP (no threshold file needed) on 1 prompt with few steps.
#
# Usage: bash demo/run_dit_bench_demo.sh [GPU_IDS] [BACKEND] [INFER_STEP]
#   GPU_IDS     Comma-separated GPU indices (default: 0)
#   BACKEND     hyvideo or wan (default: wan)
#   INFER_STEP  Number of diffusion steps (default: 2, for fast smoke test)
#
# Requires: make ditinit to have been run first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_IDS="${1:-0}"
BACKEND="${2:-wan}"
INFER_STEP="${3:-2}"

DIT_SCRIPT="${REPO_ROOT}/eval/DiTEvalKit/run_single_th.sh"

RESULTS_DIR="${REPO_ROOT}/smoke_test_dit"
echo "[run_ditbench] Clearing previous results: ${RESULTS_DIR}"
rm -rf "${RESULTS_DIR}"

echo "[run_ditbench] GPU_IDS=${GPU_IDS} BACKEND=${BACKEND} INFER_STEP=${INFER_STEP}"
echo "[run_ditbench] Method: vecattention_wo_DP threshold=0.001 prompt_id=0"

# run_single_th.sh signature:
#   $1=gpuids  $2=methods  $3=backends  $4=th  $5=sample_num  $6=output_root  $7=prompt_ids  $8=resolution  $9=infer_step
bash "${DIT_SCRIPT}" \
    "${GPU_IDS}" \
    "vecattention_wo_DP" \
    "${BACKEND}" \
    "0.001" \
    "" \
    "smoke_test_dit" \
    "0" \
    "720p" \
    "${INFER_STEP}"
