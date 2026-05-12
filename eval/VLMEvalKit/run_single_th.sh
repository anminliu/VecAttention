#!/bin/bash
# Usage: bash run_single_th.sh <method> <gpuid> [benchmark] [root_dir] [model] [threshold] [num_samples]
#
# Run a single VLM eval with one threshold value for a given attention method.
#
# Arguments:
#   method      - Attention method: full, origin, vecattention, flex, anchor, xattn16
#   gpuid       - Comma-separated GPU indices (e.g. "0" or "0,1")
#   benchmark   - "all" or "debug" (default: debug)
#   root_dir    - Output directory name (default: debug_results)
#   model       - "qwenvl" or "internvl" (default: qwenvl)
#   threshold   - Method-specific threshold / budget / gamma value
#   num_samples - Number of samples; 0 = all (default depends on benchmark)
set -e
method=$1
gpuid=$2
benchmark=${3:-"debug"}
ROOT_DIR_NAME=${4:-"debug_results"}
model=${5:-"qwenvl"}
th=${6}  # Threshold passed as script argument
num_samples=${7}  # No default; set below based on benchmark

export CUDA_VISIBLE_DEVICES=$gpuid
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0" 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_LOCAL_DATASET=True

# Compute GPU count and OMP thread count
IFS=',' read -ra GPUS <<< "$gpuid"
NPROC=${#GPUS[@]}
TOTAL_CORES=$(nproc)
OMP_THREADS=$((TOTAL_CORES / NPROC))
OMP_THREADS=$((OMP_THREADS > 0 ? OMP_THREADS : 1))
export OMP_NUM_THREADS=$OMP_THREADS
echo "Detected $NPROC GPU(s): $gpuid"
echo "Total CPU cores: $TOTAL_CORES"
echo "Setting OMP_NUM_THREADS=$OMP_THREADS"

# Configure model
case "$model" in
    "internvl")
        MODEL_NAME="InternVL3_5-8B"
        ;;
    "qwenvl")
        MODEL_NAME="Qwen2.5-VL-7B-Instruct-ForVideo"
        ;;
    *)
        echo "Unsupported model: $model"
        exit 1
        ;;
esac

# Configure datasets (coupled to model)
case "$benchmark" in
    "all")
        DEFAULT_NUM_SAMPLES=0
        if [ "$model" == "qwenvl" ]; then
            DATASETS="Video-MME_1fps LongVideoBench_1fps VCRBench_1fps_nopack"
        elif [ "$model" == "internvl" ]; then
            DATASETS="Video-MME_64frame LongVideoBench_64frame VCRBench_64frame_nopack"
        fi
        ;;
    "debug")
        DEFAULT_NUM_SAMPLES=1
        if [ "$model" == "qwenvl" ]; then
            DATASETS="Video-MME_1fps"
        elif [ "$model" == "internvl" ]; then
            DATASETS="Video-MME_64frame"
        fi
        ;;
    *)
        echo "Unsupported benchmark: $benchmark"
        exit 1
        ;;
esac

# Use default num_samples if not specified via argument
if [ -z "$num_samples" ]; then
    num_samples=$DEFAULT_NUM_SAMPLES
fi

# Get script directory and activate the shared venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"


RESULTS_DIR="${SCRIPT_DIR}/${ROOT_DIR_NAME}"

COMMON_ARGS="--model ${MODEL_NAME} \
--data $DATASETS \
--work-dir ${RESULTS_DIR} \
--exp-name ${method} \
--reuse \
--num_samples ${num_samples} 
"

# Method selection using case statement
case "$method" in
    "full")
        torchrun --standalone --nproc-per-node=$NPROC run.py $COMMON_ARGS --fastprefill-metric full
        ;;

    "origin")
        torchrun --standalone --nproc-per-node=$NPROC run.py $COMMON_ARGS --fastprefill-metric origin
        ;;

    "vecattention")
        PRIVATE_ARGS="--fastprefill-block-size-q 64 --fastprefill-block-size-k 16 --fastprefill-group-k-block 16 \
        --fastprefill-chunk-size 65536 \
        "
        torchrun --standalone --nproc-per-node=$NPROC run.py $COMMON_ARGS $PRIVATE_ARGS --fastprefill-metric vecattention --fastprefill-threshold ${th}
        ;;

    "flex")
        PRIVATE_ARGS="--fastprefill-tau 0.1"
        torchrun --standalone --nproc-per-node=$NPROC run.py $COMMON_ARGS $PRIVATE_ARGS --fastprefill-metric flex --fastprefill-gamma ${th}
        ;;

    "anchor")
        PRIVATE_ARGS="--fastprefill-block-size-q 128 --fastprefill-step 16"
        torchrun --standalone --nproc-per-node=$NPROC run.py $COMMON_ARGS $PRIVATE_ARGS --fastprefill-metric anchor --fastprefill-theta ${th}
        ;;

    "xattn16")
        PRIVATE_ARGS="--fastprefill-stride 16"
        torchrun --standalone --nproc-per-node=$NPROC run.py $COMMON_ARGS $PRIVATE_ARGS --fastprefill-metric xattn --fastprefill-threshold ${th}
        ;;

    *)
        echo "Unsupported method: $method"
        exit 1
        ;;
esac