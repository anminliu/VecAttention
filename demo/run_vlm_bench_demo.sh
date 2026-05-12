#!/usr/bin/env bash
# Quick demo / smoke test for VLMEvalKit video understanding evaluation.
# Runs a single threshold on a small sample to verify the pipeline works.
#
# Usage: bash demo/run_vlm_bench_demo.sh [GPU_IDS] [THRESHOLD] [NUM_SAMPLES]
#   GPU_IDS      Comma-separated GPU indices (default: 0)
#   THRESHOLD    MinP threshold for vecattention (default: 0.8)
#   NUM_SAMPLES  Number of benchmark samples (default: 4)
#
# Requires: make vlminit to have been run first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_IDS="${1:-0}"
THRESHOLD="${2:-0.8}"
NUM_SAMPLES="${3:-4}"

VLM_SCRIPT="${REPO_ROOT}/eval/VLMEvalKit/run_single_th.sh"
cd "${REPO_ROOT}/eval/VLMEvalKit"

RESULTS_DIR="${REPO_ROOT}/eval/VLMEvalKit/smoke_test"
echo "[run_vlmbench] Clearing previous results: ${RESULTS_DIR}"
rm -rf "${RESULTS_DIR}"

echo "[run_vlmbench] GPU_IDS=${GPU_IDS} THRESHOLD=${THRESHOLD} NUM_SAMPLES=${NUM_SAMPLES}"
echo "[run_vlmbench] Calling: run_single_th.sh vecattention ${GPU_IDS} debug smoke_test qwenvl ${THRESHOLD} ${NUM_SAMPLES}"

bash "${VLM_SCRIPT}" \
    "vecattention" \
    "${GPU_IDS}" \
    debug \
    smoke_test \
    qwenvl \
    "${THRESHOLD}" \
    "${NUM_SAMPLES}"
