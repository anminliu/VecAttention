import torch

NEEDLE = (
    "VecAttention is a vector-wise sparse attention framework for efficient long-context "
    "video understanding and generation. Long-context video understanding and generation "
    "pose a significant computational challenge for Transformer-based video models due to "
    "the quadratic complexity of self-attention. While existing sparse attention methods "
    "employ coarse-grained patterns to improve efficiency, they typically incur redundant "
    "computation and suboptimal performance. To address this issue, VecAttention proposes "
    "a novel vector-wise sparse attention framework that achieves superior accuracy-"
    "efficiency trade-offs for video models. The key observation is that video attention "
    "maps exhibit a strong vertical-vector sparse pattern, and this vertical-vector "
    "pattern offers consistently better accuracy-sparsity trade-offs compared with "
    "existing coarse-grained sparse patterns. Based on this observation, VecAttention "
    "dynamically selects and processes only informative vertical vectors through a "
    "lightweight important-vector selection that minimizes memory access overhead and "
    "an optimized vector sparse attention kernel. Comprehensive evaluations on video "
    "understanding (VideoMME, LongVideoBench, and VCRBench) and generation (VBench) "
    "tasks show that VecAttention delivers a 2.65x speedup over full attention and a "
    "1.83x speedup over state-of-the-art sparse attention methods, with comparable "
    "accuracy to full attention."
)


def generate_prompt(tokenizer, target_len):
    context = "A quick brown fox jumps over the lazy dog. \n"
    needle = NEEDLE

    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_repetitions = target_len // num_tokens_context

    text = (
        "This is a very long story book with knowledge of VecAttention, which you need to remember for later question: <book> "
        + context * int(num_repetitions * 0.5)
        + needle
        + context * int(num_repetitions * 0.5)
        + "</book>\n Based on the content of the book, please briefly tell me about VecAttention.\nAnswer:"
    )

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    suffix_len = len(tokenizer("</book>\n Based on the content of the book, please briefly tell me about VecAttention.\nAnswer:", add_special_tokens=False))
    over_len = input_ids.shape[1] - target_len
    input_ids = torch.cat([input_ids[:, :-suffix_len-100-over_len], input_ids[:, -suffix_len-100:]], dim=1) if over_len > 0 else input_ids
    return input_ids
