import torch
import math
from typing import Union

from spattn.src.utils import average_vector
from spattn.src.kernels.vecattention_kernels import fuse_qk_softmax_minp_wo_causal
from vllm_flash_attn import sparse_attn_func


def VecAttention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    threshold: Union[float, torch.Tensor] = None,
    q_pooling_size: int = 128,
    k_local_size: int = 128,
    group_k_block: int = 1,
    causal: bool = True,
    chunk_size: int = 16 * 1024,
):
    """Vectorized sparse attention prefill (VecAttention).

    Selects important key-value blocks per query using a fused Triton kernel
    that applies a per-head MinP threshold on averaged Q·K^T scores, then
    runs sparse attention via vllm_flash_attn.sparse_attn_func.

    Args:
        query_states: (batch, num_heads, seq_len, head_dim)
        key_states:   (batch, num_kv_heads, seq_len, head_dim)
        value_states: (batch, num_kv_heads, seq_len, head_dim)
        threshold: MinP threshold. Float for a global value, or
            torch.Tensor of shape (num_layers, num_heads) for per-head values.
        q_pooling_size: Query block size for vector pooling. Must be 64 or 128.
        k_local_size: Key local block size for column selection.
        group_k_block: Number of k-blocks processed together; higher values
            increase parallelism at the cost of selection granularity.
        causal: Whether to apply causal masking.
        chunk_size: Prefill chunk size (tokens). Must be a multiple of q_pooling_size.

    Returns:
        attn_output: (batch, num_heads, seq_len, head_dim)
    """
    assert chunk_size % q_pooling_size == 0, "chunk_size must be a multiple of q_pooling_size"
    assert q_pooling_size in [64, 128], "q_pooling_size must be 64 or 128"
    # NOTE: block_size_K in vllm_flash_attn is fixed to 64
    SPATTN_BLOCK_SIZE_K = 64

    # Convert MinP threshold to minS: gap = -log(threshold)
    if isinstance(threshold, torch.Tensor):
        gap = -torch.log(threshold + 1e-9)
    else:
        gap = -math.log(threshold + 1e-9, math.e)

    batch_size, num_heads, seq_len, head_dim = query_states.shape
    num_q_blocks = math.ceil(seq_len / q_pooling_size)

    attn_output = torch.empty(
        batch_size, num_heads, seq_len, head_dim,
        dtype=query_states.dtype, device=query_states.device
    )

    # Pre-allocate local-block indices (diagonal + initial blocks)
    if causal:
        n = q_pooling_size // SPATTN_BLOCK_SIZE_K
        blk_count = torch.full(
            (batch_size, num_heads, num_q_blocks), 2 * n,
            dtype=torch.int32, device=query_states.device
        )
        blk_count[..., 0] = math.ceil(min(seq_len, q_pooling_size) / SPATTN_BLOCK_SIZE_K)
        if seq_len > q_pooling_size and seq_len % q_pooling_size != 0:
            blk_count[..., -1] = n + math.ceil(
                (seq_len - math.floor(seq_len / q_pooling_size) * q_pooling_size) / SPATTN_BLOCK_SIZE_K
            )
        blk_idx = torch.zeros(
            batch_size, num_heads, num_q_blocks, 2 * n,
            dtype=torch.int32, device=query_states.device
        )
        offsets = torch.arange(n, device=query_states.device, dtype=torch.int32) * SPATTN_BLOCK_SIZE_K
        blk_idx[..., :n] = offsets
        base = (
            torch.arange(0, num_q_blocks, device=query_states.device, dtype=torch.int32) * q_pooling_size
        ).unsqueeze(-1)
        blk_idx[..., n:] = base + offsets
    else:
        blk_count = torch.zeros(
            batch_size, num_heads, num_q_blocks,
            dtype=torch.int32, device=query_states.device
        )
        blk_idx = torch.zeros(
            batch_size, num_heads, num_q_blocks, 1,
            dtype=torch.int32, device=query_states.device
        )

    for i in range(0, seq_len, chunk_size):
        q_chunk = query_states[:, :, i: i + chunk_size, :]

        avg_q_chunk = average_vector(q_chunk, q_pooling_size, use_triton=True)
        col_count, col_idx = fuse_qk_softmax_minp_wo_causal(
            avg_q_chunk, key_states, i // q_pooling_size, gap,
            causal, q_pooling_size, k_local_size,
            wo_initial=causal, group_k_block=group_k_block,
        )

        blk_count_chunk = blk_count[:, :, i // q_pooling_size:(i + chunk_size) // q_pooling_size]
        blk_idx_chunk = blk_idx[:, :, i // q_pooling_size:(i + chunk_size) // q_pooling_size, :]

        k_chunk = key_states[:, :, :i + chunk_size, :] if causal else key_states
        v_chunk = value_states[:, :, :i + chunk_size, :] if causal else value_states

        attn_output[:, :, i:i + chunk_size, :] = sparse_attn_func(
            q_chunk.transpose(1, 2).contiguous(),
            k_chunk.transpose(1, 2).contiguous(),
            v_chunk.transpose(1, 2).contiguous(),
            q_pooling_size,
            blk_count_chunk.contiguous(),
            blk_idx_chunk.contiguous(),
            col_count,
            col_idx,
            return_softmax_lse=False,
            causal=causal,
        ).transpose(1, 2)

    return attn_output
