"""VecAttention vision demo: V-NIAH multimodal prefill benchmark with Qwen2.5-VL.

Loads a Qwen2.5-VL model, patches its attention layers with VecAttention,
and times a single-shot prefill on a vision needle-in-a-haystack prompt:
a single needle image inserted at some depth into a long sequence of haystack
frames sampled from a user-provided video. Prefill-only (no decode).

Usage:
    python demo/vision_demo.py --metric vecattention \
        --haystack_movie_path /path/to/long_video.mp4 --nframe 180 --threshold 0.87
    python demo/vision_demo.py --metric full \
        --haystack_movie_path /path/to/long_video.mp4 --nframe 180
"""

import os
import time
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")

import torch
import torch.nn as nn
import types
import decord
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class _LastTokenLMHead(nn.Module):
    """Project only the last position through lm_head. Used during prefill to
    avoid the [B, S, vocab] allocation that OOMs at long context."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.inner(hidden_states[:, -1:, :])

from eval.check_env import get_env_name
from spattn.src.models.config_args import FastPrefillConfig
from spattn.src.models.load_qwen2p5_vl import new_attention_forward


# Fixed needle: the TilingSelect runtime-breakdown figure, with a multiple-choice
# question whose correct answer is C (TilingSelect + minS).
_HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLE_IMAGE_PATH = os.path.join(_HERE, "..", "assets", "tilingselect-estimate-time-breakdown.png")
NEEDLE_QUESTION = (
    "Find the frame showing the runtime breakdown of different strategies. "
    "Among the four strategies, which one achieves the lowest total runtime? "
    "A. Naive + topP  B. Naive + minS  C. TilingSelect + minS  D. TilingSelect + topP. "
    "Answer with the option's letter from the given choices directly."
)
NEEDLE_ANSWER = "C"


def _check_vlm_env():
    """Verify the current environment is 'vlm' (required by Qwen2.5-VL)."""
    try:
        env = get_env_name()
    except ImportError:
        raise RuntimeError(
            "flashinfer is not installed. Run 'make vlminit' first."
        )
    if env != "vlm":
        raise RuntimeError(
            f"Current environment is '{env}', but this demo requires the 'vlm' "
            "environment. Run 'make vlminit' to set it up (note: vlm and dit "
            "groups conflict and cannot be installed simultaneously)."
        )


def build_config(args) -> FastPrefillConfig:
    if args.metric == "full":
        return FastPrefillConfig(metric="full")
    elif args.metric == "vecattention":
        return FastPrefillConfig(
            metric="vecattention",
            threshold=args.threshold,
            block_size_q=args.block_size_q,
            block_size_k=args.block_size_k,
            group_k_block=args.group_k_block,
            chunk_size=args.chunk_size,
        )
    else:
        raise ValueError(f"Unsupported metric: {args.metric}. Choose from: full, vecattention")


def generate_vision_prompt(processor, args):
    """Build a V-NIAH multimodal prompt: sample `nframe` haystack frames from the
    user-provided video and insert the fixed needle image at `depth * nframe`,
    then attach the needle's question. Returns the processor's batched inputs
    on CUDA."""
    vid = decord.VideoReader(args.haystack_movie_path)
    step = len(vid) / (args.nframe + 1)
    indices = [int(i * step) for i in range(1, args.nframe + 1)]
    haystack = [Image.fromarray(vid[i].asnumpy()) for i in indices]

    insert_pos = int(args.depth * args.nframe)
    needle_img = Image.open(NEEDLE_IMAGE_PATH).convert("RGB")
    frames = haystack[:insert_pos] + [needle_img] + haystack[insert_pos:]

    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": NEEDLE_QUESTION})
    conversation = [{"role": "user", "content": content}]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    image_inputs, _ = process_vision_info(conversation)
    inputs = processor(text=[prompt], images=image_inputs, return_tensors="pt", padding=True)
    return inputs.to("cuda")


def main():
    parser = argparse.ArgumentParser(description="VecAttention V-NIAH vision demo")
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct"),
                        help="Path to Qwen2.5-VL model")
    parser.add_argument("--haystack_movie_path", type=str, required=True,
                        help="Path to a long (>1h recommended) video used as the haystack")
    parser.add_argument("--nframe", type=int, default=200, help="Number of haystack frames")
    parser.add_argument("--depth", type=float, default=0.5, help="Needle insertion depth in [0,1]")
    parser.add_argument("--metric", type=str, default="vecattention",
                        choices=["full", "vecattention"], help="Attention method")
    parser.add_argument("--threshold", type=float, default=0.9, help="MinP threshold")
    parser.add_argument("--block_size_q", type=int, default=64)
    parser.add_argument("--block_size_k", type=int, default=16)
    parser.add_argument("--group_k_block", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=32 * 1024)
    args = parser.parse_args()

    _check_vlm_env()

    config = build_config(args)
    print(f"Metric: {args.metric} | Threshold: {args.threshold} | Frames: {args.nframe} | Depth: {args.depth}")

    # Load model
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model)

    # Patch attention layers
    if config.metric != "full":
        for layer in model.model.layers:
            layer.self_attn.fastprefillconfig = config
            layer.self_attn.forward = types.MethodType(new_attention_forward, layer.self_attn)
        print(f"Patched {len(model.model.layers)} attention layers with {config.metric}")

    # Generate prompt
    print(f"Building V-NIAH prompt ({args.nframe} haystack + 1 needle frames)...")
    inputs = generate_vision_prompt(processor, args)
    actual_len = inputs["input_ids"].shape[1]
    print(f"Question: {NEEDLE_QUESTION}")
    print(f"Reference answer: {NEEDLE_ANSWER}")
    print(f"Actual prompt length: {actual_len} tokens")

    # Single-shot prefill. lm_head is swapped to project only the last position,
    # otherwise the [B, S, vocab_size] logits tensor OOMs at long context
    # (Qwen2.5-VL's forward in this transformers version does not accept
    # `num_logits_to_keep`).
    model.lm_head = _LastTokenLMHead(model.lm_head)
    torch.cuda.synchronize()
    start_prefill = time.time()
    with torch.no_grad():
        output = model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.time() - start_prefill
    print(f"Prefill time: {prefill_time:.2f}s ({actual_len / prefill_time:.0f} tok/s)")

    pred_token_id = output.logits[:, -1, :].argmax(dim=-1).item()
    pred_token = processor.tokenizer.decode([pred_token_id], skip_special_tokens=True)
    print(f"Predicted answer: {pred_token!r}")


if __name__ == "__main__":
    main()
