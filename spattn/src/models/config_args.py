import argparse
from typing import Optional
import os


class FastPrefillConfig(dict):
    """Configuration for prefill attention dispatch.

    Attributes:
        metric: Attention method to use. One of:
            'vecattention', 'xattn', 'full', 'flex', 'anchor', 'origin'.
        output_time: Record per-layer attention latency (for SpAttnProfiler).

        vision_start_position: First vision token index (vecattention only).
            When None, VecAttention is applied to the full sequence.
        vision_end_position: One-past-last vision token index (vecattention only).
            When None, the slice extends to the end of the sequence.
        dump_qk_dir: Directory to dump Q/K tensors for offline analysis.
        input_id: Identifier string appended to dump filenames.
    """

    def __init__(
        self,
        metric: str = "xattn",

        # --- VecAttention ---
        threshold: float = None,        # MinP gap threshold (float or per-head tensor)
        block_size_q: int = 64,         # Q pooling block size; shared with anchor
        block_size_k: int = 16,         # K local block size
        group_k_block: int = 1,
        chunk_size: int = 8 * 1024,     # chunked-prefill window size
        vision_start_position: Optional[int] = None,
        vision_end_position: Optional[int] = None,

        # --- XAttention ---
        stride: int = 16,               # antidiagonal stride; threshold shared above

        # --- FlexPrefill ---
        gamma: float = 0.9,
        tau: float = 0.1,

        # --- AnchorAttention ---
        theta: float = 12,
        step: int = 16,                 # block_size_q shared above

        # --- Profiling / Debug ---
        output_time: bool = True,
        dump_qk_dir: Optional[str] = None,
        input_id: str = "input",
    ):
        super().__init__()
        self.metric = metric
        self.output_time = output_time
        # VecAttention
        self.block_size_q = block_size_q
        self.block_size_k = block_size_k
        self.group_k_block = group_k_block
        self.chunk_size = chunk_size
        self.vision_start_position = vision_start_position
        self.vision_end_position = vision_end_position
        # XAttention
        self.stride = stride
        # FlexPrefill
        self.gamma = gamma
        self.tau = tau
        # AnchorAttention
        self.theta = theta
        self.step = step
        # Debug
        self.dump_qk_dir = dump_qk_dir
        self.input_id = input_id
        # threshold is optional (not all methods need it)
        if threshold is not None:
            self.threshold = threshold


def add_fastprefill_args(parser: argparse.ArgumentParser):
    parser.add_argument('--fastprefill-metric', type=str, default="full")
    # VecAttention
    parser.add_argument('--fastprefill-threshold', type=float, default=0.9)
    parser.add_argument('--fastprefill-block-size-q', type=int, default=64)
    parser.add_argument('--fastprefill-block-size-k', type=int, default=16)
    parser.add_argument('--fastprefill-group-k-block', type=int, default=1)
    parser.add_argument('--fastprefill-chunk-size', type=int, default=64 * 1024)
    # XAttention
    parser.add_argument('--fastprefill-stride', type=int, default=16)
    # FlexPrefill
    parser.add_argument('--fastprefill-gamma', type=float, default=0.9)
    parser.add_argument('--fastprefill-tau', type=float, default=0.1)
    # AnchorAttention
    parser.add_argument('--fastprefill-theta', type=float, default=12)
    parser.add_argument('--fastprefill-step', type=int, default=16)
    # DP threshold tuning: directory to dump Q/K tensors for offline analysis
    parser.add_argument('--fastprefill-dump-qk-dir', type=str, default=None)
    parser.add_argument('--fastprefill-input-id', type=str, default=None,
                        help='Identifier subdirectory for QK dumps (typically prompt id)')
    return parser


def get_fastprefill_config(args):
    kwargs = {
        k.replace("fastprefill_", ""): v
        for k, v in vars(args).items()
        if k.startswith("fastprefill_") and v is not None
    }
    return FastPrefillConfig(**kwargs)


def build_output_file_from_config(base_path, config):
    metric = config.metric
    parts = [f"metric={metric}"]
    if metric == "vecattention":
        parts.append(f"th={config.threshold}")
        parts.append(f"qps={config.block_size_q}")
        parts.append(f"kls={config.block_size_k}")
        parts.append(f"gkb={config.group_k_block}")
    elif metric == "xattn":
        parts.append(f"th={config.threshold}")
        parts.append(f"stride={config.stride}")
    elif metric in ("flex",):
        parts.append(f"th={config.gamma}")
        parts.append(f"tau={config.tau}")
    elif metric == "anchor":
        parts.append(f"bsq={config.block_size_q}")
        parts.append(f"step={config.step}")
        parts.append(f"th={config.theta}")
    base, ext = os.path.splitext(base_path)
    config_str = "_".join(parts)
    return f"{base}_{config_str}{ext}"
