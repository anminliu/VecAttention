"""VecAttention demo: long-context text inference with Qwen2.5-VL.

Loads a Qwen2.5-VL model, patches its attention layers with VecAttention,
and runs prefill + greedy decode on a long needle-in-a-haystack prompt.
Compare VecAttention vs full attention by running both and checking the output.

Usage:
    python demo/demo.py --metric vecattention --len 65536 --threshold 0.9
    python demo/demo.py --metric full --len 65536
"""

import os
import time
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")

import torch
import types
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from eval.check_env import get_env_name
from spattn.src.models.config_args import FastPrefillConfig
from spattn.src.models.load_qwen2p5_vl import new_attention_forward


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


def generate_long_prompt(processor, target_len: int) -> torch.Tensor:
    """Build a needle-in-a-haystack text prompt of approximately target_len tokens."""
    _here = os.path.dirname(os.path.abspath(__file__))
    needle_path = os.path.join(_here, "vecattention.txt")
    with open(needle_path, "r") as f:
        needle = f.read()

    filler = "A quick brown fox jumps over the lazy dog.\n"
    filler_tokens = len(processor.tokenizer.encode(filler, add_special_tokens=False))
    num_reps = target_len // filler_tokens

    text = (
        "This is a very long document with knowledge of VecAttention. "
        "You need to remember the details for a later question: <book> "
        + filler * (num_reps // 2)
        + needle
        + filler * (num_reps // 2)
        + "</book>\nBased on the content above, briefly describe what VecAttention is.\nAnswer:"
    )

    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()

    # Trim to target length if over
    if input_ids.shape[1] > target_len:
        suffix_len = 200  # keep the question at the end
        input_ids = torch.cat([
            input_ids[:, :target_len - suffix_len],
            input_ids[:, -suffix_len:]
        ], dim=1)
    return input_ids


def main():
    parser = argparse.ArgumentParser(description="VecAttention long-context demo")
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct"),
                        help="Path to Qwen2.5-VL model")
    parser.add_argument("--len", type=int, default=32768, help="Target prompt length in tokens")
    parser.add_argument("--metric", type=str, default="vecattention",
                        choices=["full", "vecattention"], help="Attention method")
    parser.add_argument("--threshold", type=float, default=0.9, help="MinP threshold")
    parser.add_argument("--block_size_q", type=int, default=64)
    parser.add_argument("--block_size_k", type=int, default=16)
    parser.add_argument("--group_k_block", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=32 * 1024)
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    args = parser.parse_args()

    _check_vlm_env()

    config = build_config(args)
    print(f"Metric: {args.metric} | Threshold: {args.threshold} | Seq len: {args.len}")

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
    print(f"Generating prompt (~{args.len} tokens)...")
    input_ids = generate_long_prompt(processor, args.len)
    actual_len = input_ids.shape[1]
    print(f"Actual prompt length: {actual_len} tokens")

    # Prefill
    torch.cuda.synchronize()
    start_prefill = time.time()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            use_cache=True,
        )
    torch.cuda.synchronize()
    prefill_time = time.time() - start_prefill
    past_key_values = output.past_key_values
    print(f"Prefill time: {prefill_time:.2f}s ({actual_len / prefill_time:.0f} tok/s)")

    # Decode
    start_decode = time.time()
    eos_token_id = processor.tokenizer.eos_token_id
    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    torch.cuda.empty_cache()

    with torch.no_grad():
        for _ in tqdm(range(args.max_new_tokens - 1), desc="Decoding", unit="token"):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            token = pred_token_idx.item()
            if token == eos_token_id:
                break
            generated_ids.append(token)

    torch.cuda.synchronize()
    decode_time = time.time() - start_decode

    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nPrefill: {prefill_time:.2f}s | Decode: {decode_time:.2f}s")
    print(f"Generated ({len(generated_ids)} tokens): {output_text}")


if __name__ == "__main__":
    main()
