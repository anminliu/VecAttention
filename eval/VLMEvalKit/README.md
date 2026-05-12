# VLMEvalKit — VecAttention Fork

This directory contains a fork of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) (Apache 2.0) adapted for VecAttention evaluation. See `LICENSE` for the original license.

For benchmark results and how to run VLM evaluation, see the [top-level README](../../README.md#evaluation).

---

## What this fork adds

- **`vlmeval/vlm/qwen2_vl/model.py`** — Qwen2.5-VL patched with `spattn` sparse attention (`new_attention_forward`)
- **`vlmeval/vlm/internvl/internvl_chat.py`** — InternVL patched with `spattn` sparse attention
- **`vlmeval/inference_video.py`** — Extended with `FastPrefillConfig` and `SpAttnProfiler` for VecAttention profiling
- **`vlmeval/dataset/video_dataset_config.py`** — Local dataset path configuration

---

## Adding a new benchmark dataset

1. **Create the dataset class** in `vlmeval/dataset/` (subclass `VideoBaseDataset` for video, `ImageBaseDataset` for image):
   - `prepare_dataset()` — point to local data path instead of downloading
   - Support `duration_sample` in the sampling logic if the benchmark has duration splits
   - `evaluate()` / `get_dimension_rating()` — accept a `profile_metrics` kwarg and pass it through for sparsity/latency logging

2. **Register the class** in `vlmeval/dataset/__init__.py` — add the import and include the class in the `supported_datasets` list.

3. **Add dataset config** in `vlmeval/dataset/video_dataset_config.py` — add the local data path mapping.

---

## Adding a new model

1. **Create the model file** in `vlmeval/vlm/` (subclass `BaseModel`):
   - In `generate_inner()`, call `spattn.src.models.load_<model>.new_attention_forward` to patch the model's attention before inference
   - Accept a `prefill_config: FastPrefillConfig` argument and pass it through

2. **Register the model** in `vlmeval/vlm/__init__.py`.

3. **Add a loader** in `spattn/src/models/load_<model>.py` following the pattern of `load_qwen2p5_vl.py` or `load_intern3p5_vl.py`.

---

## Running VecAttention VLM evaluation

```bash
# Full sweep
cd eval/VLMEvalKit
bash vlm_run.sh vecattention-q64k16g16-onlyVision 0 videomme-1fps full_results qwenvl

# Single threshold / quick test
bash vlm_test.sh vecattention-q64k16g16-onlyVision 0 debug smoke_test qwenvl 0.8 4
```
