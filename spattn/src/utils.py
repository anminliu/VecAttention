import math

import torch
import triton
import triton.language as tl


@triton.jit
def bnhd_pool_kernel(
    x_ptr,
    y_ptr,
    # pool type. avg: 0, max: 1, min: 2, max abs: 3, sum: 4
    pool_type: tl.constexpr,
    # shape
    batch_size,
    seq_len,
    num_heads,
    head_dim: tl.constexpr,
    # stride
    stride_xb,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yh,
    stride_yd,
    # META parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
    BLOCK_SIZE_D: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_n * BLOCK_SIZE_N * stride_xn
        + pid_h * BLOCK_SIZE_H * stride_xh
    )

    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    cur_block_size_n = min(seq_len - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)

    x_mask = (
        (off_n < seq_len - pid_n * BLOCK_SIZE_N)[:, None, None]
        & (off_h < num_heads - pid_h * BLOCK_SIZE_H)[None, :, None]
        & (off_d < head_dim)[None, None, :]
    )
    x = tl.load(
        x_ptr
        + off_n[:, None, None] * stride_xn
        + off_h[None, :, None] * stride_xh
        + off_d[None, None, :] * stride_xd,
        mask=x_mask,
        other=0,
    )
    if pool_type == 0:
        y = tl.sum(x, axis=0) / cur_block_size_n
    elif pool_type == 1:
        y = tl.max(x, axis=0)
    elif pool_type == 2:
        y = tl.min(x, axis=0)
    elif pool_type == 3:
        y = tl.max(tl.abs(x), axis=0)
    elif pool_type == 4:
        y = tl.sum(x, axis=0)
    else:
        y = tl.sum(x, axis=0) / cur_block_size_n
    y_ptr = (
        y_ptr + pid_b * stride_yb + pid_n * stride_yn + pid_h * BLOCK_SIZE_H * stride_yh
    )
    y_mask = (off_h < num_heads - pid_h * BLOCK_SIZE_H)[:, None] & (off_d < head_dim)[
        None, :
    ]
    tl.store(
        y_ptr + off_h[:, None] * stride_yh + off_d[None, :] * stride_yd, y, mask=y_mask
    )


def triton_bnhd_pool(x: torch.Tensor, kernel_size: int, pool_type: str = "avg"):
    b, n, h, d = x.shape
    assert d in {16, 32, 64, 128}
    assert kernel_size in {1, 16, 32, 64, 128, 256, 512}
    if kernel_size == 1:
        return x
    m = triton.cdiv(n, kernel_size)
    y = torch.zeros(b, m, h, d, device=x.device, dtype=x.dtype)

    if pool_type == "last":
        if n % kernel_size == 0:
            return x[:, kernel_size - 1 :: kernel_size, ...]
        else:
            return torch.cat(
                (x[:, kernel_size - 1 :: kernel_size, ...], x[:, -1:, ...]), dim=1
            )

    block_size_h = triton.next_power_of_2(h)
    while kernel_size * block_size_h * d > 128 * 128 * 128:
        block_size_h = block_size_h // 2
    assert block_size_h != 0

    block_size_d = triton.next_power_of_2(d)
    pool_str_to_type = {"avg": 0, "max": 1, "min": 2, "maxabs": 3, "sum": 4}
    pool_type = pool_str_to_type[pool_type]

    grid = lambda META: (
        b,
        triton.cdiv(n, META["BLOCK_SIZE_N"]),
        triton.cdiv(h, META["BLOCK_SIZE_H"]),
    )
    bnhd_pool_kernel[grid](
        x,
        y,
        pool_type,
        b,
        n,
        h,
        d,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_D=block_size_d,
    )
    return y


def find_blocks_chunked(
    input_tensor, current_index, threshold, num_to_choose, decoding: bool, mode: str = "both", causal=True
):
    """
        Finds and selects relevant blocks of attention for transformer-based models based on a
        threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
        - current_index (int): The current index in the sequence processing.
        - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
        - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.
        - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
        - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent future information leakage.

        Returns:
        - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    # 0 -- -- -- -- current_index
    # 0 -- -- -- -- -- current_index+1
    # 0 -- -- -- -- -- ----------- current_index + chunk_num - 1
    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, device=input_tensor.device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask
    input_tensor = input_tensor.to(float)

    if threshold is not None:
        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(float)
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            ).expand((batch_size, head_num, chunk_num, 1)).to(input_tensor.device)
        else:
            required_sum = total_sum * threshold
        if causal:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            # Force-select the first block and the last block
            mask[:, :, :, 0] = 1 # This value is likely overwritten by the line below; purpose unclear
            # Anmin Note: The logic here is:
            # 1. At the current index, force an identity matrix so the last block is always selected.
            # 2. For other blocks, sort in descending order and select until the cumulative sum exceeds the predefined threshold; this may or may not include the first block.
            # 3. Finally, force-select the first block.
            mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            other_values = input_tensor.masked_fill(
                mask, 0
            )
            sorted_values, _ = torch.sort(
                other_values, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)

            sorted_values = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask,index,0)
            mask = mask.view(batch_size,head_num*chunk_num,block_num)
            index = index.view(batch_size,head_num*chunk_num,block_num)
            mask[:,torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),index] = True
            mask = mask.view(batch_size,head_num,chunk_num,block_num)
        else:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(
                input_tensor, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        raise NotImplementedError("block num chunk prefill not impleted")

    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except:
        mask[:, :, :, current_index + chunk_num :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor,dtype=bool,device=input_tensor.device)
            lambda_mask[:,:,:,0] = 1
            lambda_mask[:,:,:,current_index:current_index+chunk_num] = torch.eye(chunk_num, device=lambda_mask.device).unsqueeze(0).unsqueeze(0).expand(1,head_num,chunk_num,chunk_num)
            assert(torch.where(lambda_mask,mask,True).all())

    return mask


def causal_mask_in_uneqal_block(seqlen, block_size_q, block_size_k, device):
    """
    Apply a causal mask on an attention map with unequal block sizes.

    Args:
        attention_map (torch.Tensor): Original attention map, shape [seq_len, seq_len].
        block_size_q (int): Row block size.
        block_size_k (int): Column block size.

    Returns:
        diagonal_mask: torch.Tensor: [seqlen, seqlen]
        causal_mask: torch.Tensor: [seqlen, seqlen]
    """
    num_q_blocks = math.ceil(seqlen / block_size_q)
    num_k_blocks = math.ceil(seqlen / block_size_k)

    # Compute start and end indices for row blocks
    q_start = torch.arange(num_q_blocks, device=device) * block_size_q
    # q_end = torch.min((q_start + block_size_q), torch.tensor(seqlen, device=mask.device)) - 1
    q_end = q_start + block_size_q - 1 # inclusive

    # Compute start and end indices for column blocks
    k_start = torch.arange(num_k_blocks, device=device) * block_size_k
    # k_end = torch.min((k_start + block_size_k), torch.tensor(seqlen, device=mask.device)) - 1
    k_end = k_start + block_size_k - 1 #inclusive

    # Broadcast comparison: check if row and column blocks overlap
    # Condition: row block start <= column block end AND column block start <= row block end
    overlap = (q_start.unsqueeze(1) <= k_end.unsqueeze(0)) & (k_start.unsqueeze(0) <= q_end.unsqueeze(1))
    diagonal_mask = overlap

    # Update mask

    assert overlap.any(dim=-1).all(), "Every row must have at least one True"

    # Find the position of the last True in each row
    last_true_idx = overlap.float().cumsum(dim=-1).argmax(dim=-1)  # [num_q_blocks]

    # Build range mask
    col_idx = torch.arange(num_k_blocks, device=overlap.device).unsqueeze(0)  # [1, num_k_blocks]
    last_true_idx = last_true_idx.unsqueeze(1)  # [num_q_blocks, 1]
    mask = col_idx <= last_true_idx  # [num_q_blocks, num_k_blocks]

    return diagonal_mask, mask


def average_vector(q, block_size, use_triton=True):
    """
        q: (batch_size, num_heads, seqlen, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    dtype = q.dtype
    q = q.float() # Use float32 for precise computation

    num_blocks = math.ceil(seq_len / block_size)
    if use_triton:
        q = q.transpose(1,2)
        return triton_bnhd_pool(q, block_size).transpose(1,2).to(dtype)
    else:
        pad_q = num_blocks * block_size - seq_len
        avg_q = (
            torch.nn.functional.pad(q, (0, 0, 0, pad_q), value=0)
            .view(batch_size, num_heads, num_blocks, block_size, head_dim)
            .mean(-2)
        )
        avg_q[:,:,-1, :] = avg_q[:,:,-1,:] * block_size / (block_size - pad_q)
        return avg_q.to(dtype)
