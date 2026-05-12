import torch
import math

from traitlets import Bool
from spattn.src.Flexprefill import triton_bnhd_pool
from flashinfer.logits_processor import LogitsPipe, TopP,MinP

def create_causal_mask(batch_size, head_num, block_size, block_num, divide_block_num):
    """
        Creates a causal attention mask used in transformer-based models.

        Parameters:
        - batch_size (int): The number of sequences in the batch.
        - head_num (int): The number of attention heads.
        - block_size (int): The size of each block in the sequence.
        - block_num (int): The total number of blocks in the sequence.
        - divide_block_num (int): The block index at which causality is applied.

        Returns:
        - torch.Tensor: A mask tensor of shape (batch_size, head_num, block_size, total_size)
        where total_size = block_size * block_num. The mask enforces causal attention by 
        setting certain positions to `-inf` to prevent information leakage from future tokens.
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask

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


# Modified from Flashinfer::tests::test_sampling.py
def score_cover_idx(x: torch.Tensor, score: float, padding_value=0):
    assert x.dtype == torch.float32, "x must be of type float32"
    total_score = x.sum(dim=-1, keepdim=True)
    x, idx = torch.sort(x, dim=-1, descending=False)
    cumsum_x = torch.cumsum(x, dim=-1)
    idx[cumsum_x < (total_score-score)] = padding_value
    return idx


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

def profile_score_concentration(estimated_attention_map, block_causal_mask_head, k=2, p=0.95):
    """
    estimated_attention_map: [seqlen, seqlen]
    """
    def normalized_entropy(estimated_attention_map, block_causal_mask_head):
        # attention_row: [seqlen]
        epsilon = 1e-8
        valid_counts = block_causal_mask_head.sum(dim=-1, keepdim=True).float()
        valid_counts = torch.where(valid_counts == 1, float("inf"), valid_counts)
        entropy = -torch.sum(estimated_attention_map * torch.log(estimated_attention_map + epsilon),dim=-1, keepdim=True)
        entropy = entropy / torch.log(valid_counts + epsilon)
        return entropy.mean().item()
    def gini_coefficient(estimated_attention_map):
        gini = 1 - torch.sum(estimated_attention_map ** 2, dim=-1)
        return gini.mean().item()
    def max_value(estimated_attention_map):
        return estimated_attention_map.max(dim=-1).values.mean().item()
    def topk_ratio(estimated_attention_map, k=k):
        topk_values, _ = torch.topk(estimated_attention_map, k, dim=-1)
        return torch.sum(topk_values,dim=-1).mean().item()
    def variance(estimated_attention_map, block_causal_mask_head):
        valid_counts = block_causal_mask_head.sum(dim=-1, keepdim=True).float()
        mean = estimated_attention_map.sum(dim=-1, keepdim=True) / valid_counts
        var = ((estimated_attention_map - mean) ** 2).masked_fill(~block_causal_mask_head, 0.0).sum(dim=-1) / valid_counts.squeeze(-1)
        return var.mean().item()
    def mean(estimated_attention_map, block_causal_mask_head):
        valid_counts = block_causal_mask_head.sum(dim=-1, keepdim=True).float()
        mean = estimated_attention_map.sum(dim=-1, keepdim=True) / valid_counts 
        return mean.mean().item()
    def sparsity_over_p(estimated_attention_map,global_topp,row_p=p):
        if global_topp:
            avg_attn = estimated_attention_map.view(1, -1)
            block_row_num = estimated_attention_map.shape[0]
            block_topk = score_cover_idx(avg_attn, row_p * block_row_num, padding_value=-1).squeeze(0) #[num_q_blocks * num_k_blocks]
            # head_mask = torch.zeros(num_q_blocks * num_k_blocks, block_size_q, block_size_k, device=q.device, dtype=torch.bool)
        else:
            # # row topp
            block_topk = score_cover_idx(estimated_attention_map, row_p, padding_value=-1) #[num_q_blocks, num_k_blocks]
        valid_blocks =  (block_topk != -1).sum().item()
        total_blocks =  block_causal_mask_head.sum().item()
        return 1.0 - valid_blocks / total_blocks
    return {
        "normalized_entropy": normalized_entropy(estimated_attention_map, block_causal_mask_head),
        "gini_coefficient": gini_coefficient(estimated_attention_map),
        "max_value": max_value(estimated_attention_map),
        f"top{k}_sum": topk_ratio(estimated_attention_map, k=k),
        "variance": variance(estimated_attention_map,block_causal_mask_head),
        "mean": mean(estimated_attention_map,block_causal_mask_head),
        "sparsity_over_globalp": sparsity_over_p(estimated_attention_map, global_topp=True, row_p=p),
        "sparsity_over_rowp": sparsity_over_p(estimated_attention_map, global_topp=False, row_p=p),
    }

def profile_qk_similarity(q_head, k_head, block_size_range):
    """
    Compute intra-block average cosine similarity of q and k for various block sizes.
    q_head, k_head: [seqlen, head_dim]
    block_size_range: iterable, e.g. [16, 32, 64, 128]
    Returns:
        {
            "block_size{block_size}_q": ...,
            "block_size{block_size}_k": ...
        }
    """
    metrics = {}
    seqlen, head_dim = q_head.shape
    device = q_head.device

    for block_size in block_size_range:
        num_blocks = math.ceil(seqlen / block_size)
        pad_len = num_blocks * block_size - seqlen

        # padding
        q_pad = torch.nn.functional.pad(q_head, (0, 0, 0, pad_len), value=0)
        k_pad = torch.nn.functional.pad(k_head, (0, 0, 0, pad_len), value=0)

        # Reshape into blocks
        q_blocks = q_pad.view(num_blocks, block_size, head_dim)  # [num_blocks, block_size, head_dim]
        k_blocks = k_pad.view(num_blocks, block_size, head_dim)

        # Valid mask to exclude padding
        valid_mask = torch.ones(seqlen, dtype=torch.bool, device=device)
        if pad_len > 0:
            valid_mask = torch.cat([valid_mask, torch.zeros(pad_len, dtype=torch.bool, device=device)], dim=0)
        valid_mask = valid_mask.view(num_blocks, block_size)  # [num_blocks, block_size]

        def block_cosine_sim(blocks, valid_mask):
            # [num_blocks, block_size, head_dim]
            blocks_norm = torch.nn.functional.normalize(blocks, dim=-1)
            # [num_blocks, block_size, block_size]
            sim = torch.matmul(blocks_norm, blocks_norm.transpose(-1, -2))
            # Exclude diagonal
            off_diag_mask = ~torch.eye(block_size, dtype=torch.bool, device=device).unsqueeze(0)  # [1, block_size, block_size]
            # Valid mask
            valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & off_diag_mask  # [num_blocks, block_size, block_size]
            sim = sim.masked_fill(~valid, float('nan'))
            # Mean of valid off-diagonal entries per block
            block_mean = torch.nanmean(sim, dim=(1,2))  # [num_blocks]
            # Only count valid blocks
            valid_block = valid_mask.sum(dim=1) > 1
            if valid_block.any():
                return block_mean[valid_block].mean().item()
            else:
                return float('nan')

        metrics[f"block_size{block_size}_q"] = block_cosine_sim(q_blocks, valid_mask)
        metrics[f"block_size{block_size}_k"] = block_cosine_sim(k_blocks, valid_mask)
    return metrics

def minp_idx(probs, min_p, padding_value=0):
    """
    probs: [bsz, seqlen]
    min_p: float
    Returns: idx, keeping only indices where value >= min_p * pmax per row; others set to padding_value
    """
    # Max value per row
    pmax = probs.max(dim=-1, keepdim=True).values  # [bsz, 1]
    threshold = min_p * pmax  # [bsz, 1]
    # Select positions with value >= threshold
    mask = probs >= threshold  # [bsz, seqlen]
    seqlen = probs.shape[-1]
    idx = torch.arange(seqlen, device=probs.device).unsqueeze(0).expand_as(probs)  # [bsz, seqlen]
    idx = torch.where(mask, idx, torch.full_like(idx, padding_value))
    return idx

def average_vector(q, block_size,use_triton=True):
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

def get_important_block_head_mask(avg_attn, threshold, global_topp=False, mode="topp", use_flashinfer=True):
    """
    Generate a block-level head mask from average attention scores.
    Args:
        avg_attn (torch.Tensor): Average attention scores, shape [num_q_blocks, num_k_blocks]. Computed in float32.
        threshold (float): Threshold for block selection.
        global_topp (bool): If True, apply threshold globally; otherwise apply per block row.
        mode (str): "topp" or "minp", determines block selection strategy.
        use_flashinfer (bool): Whether to use the flashinfer implementation.
    Returns:
        torch.Tensor: Boolean mask of shape [num_q_blocks, num_k_blocks] indicating selected blocks.
    """
    assert avg_attn.dtype == torch.float32, "avg_attn must be of type float32"
    num_q_blocks, num_k_blocks = avg_attn.shape
    device = avg_attn.device

    if global_topp:
        attn_input = avg_attn.view(1, -1)
    else:
        attn_input = avg_attn

    if use_flashinfer and global_topp and mode == "topp":
        print(f"Flashinfer doesn't support global TopP (cause p is not in [0,1]) yet, using torch.")
        use_flashinfer = False  # Flashinfer doesn't support global TopP yet

    if mode not in ["topp", "minp"]:
        assert not global_topp, f"Global Threshold is not supported for mode {mode}"
        use_flashinfer = False

    if use_flashinfer:
        pipe = LogitsPipe([TopP()] if mode == "topp" else [MinP()])
        kwargs = {"top_p": threshold} if mode == "topp" else \
                {"min_p": threshold}
        probs = pipe(attn_input, **kwargs)
        if global_topp:
            probs = probs.view(num_q_blocks, num_k_blocks)
        head_mask = probs != 0
    else:
        if mode == "topp":
            block_topk = score_cover_idx(attn_input, threshold * num_q_blocks).squeeze(0) if global_topp \
                         else score_cover_idx(attn_input, threshold)
        elif mode == "minp":
            block_topk = minp_idx(attn_input, threshold)
            if global_topp:
                block_topk = block_topk.squeeze(0)
        elif mode == "topnsigma":
            # attn_input is pre-softmax logits, rather than post-softmax probabilities
            logits_max = attn_input.max(dim=-1, keepdim=True).values
            logits_var = masked_std(attn_input, dim=-1, keepdim=True) # filter nan/inf
            logits_threshold = logits_max - threshold * logits_var
            head_mask = attn_input >= logits_threshold
            return head_mask
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Nearly every row will have index 0 set to True
        if global_topp:
            head_mask = torch.zeros((num_q_blocks * num_k_blocks,), device=device, dtype=torch.bool)
            head_mask[block_topk] = True
            head_mask = head_mask.reshape(num_q_blocks, num_k_blocks)
        else:
            head_mask = torch.zeros((num_q_blocks, num_k_blocks), device=device, dtype=torch.bool)
            head_mask.scatter_(dim=-1, index=block_topk, value=True)

    return head_mask

def get_important_mask_from_logits(
    logits: torch.Tensor,
    threshold: float,
    filter_mode: str,
    block_size_q: int = 1,
    block_size_k: int = 1,
    blocked_softmax: bool = False,
    use_flashinfer: bool = True,
):
    """
    Generate an importance mask from logits.
    When mode="minp", returns indices where value >= min_p * pmax. When a block is all -inf, returns all True.
    """
    inf_mask = logits != float("-inf")
    q_len, k_len = logits.shape
    device = logits.device
    num_q_blocks = math.ceil(q_len / block_size_q)
    num_k_blocks = math.ceil(k_len / block_size_k)
    pad_q = num_q_blocks * block_size_q - q_len
    pad_k = num_k_blocks * block_size_k - k_len
    ## Conditional setup logic
    # Check flashinfer support
    if block_size_q != 1 and filter_mode == "topp" and not blocked_softmax:
        use_flashinfer = False
    if filter_mode not in ["topp", "minp"]:
        assert block_size_q == 1, f"Blocked Threshold is not supported for mode {filter_mode}"
        use_flashinfer = False
    if block_size_q * block_size_k == 1:
        blocked_softmax = False

    logits_padded = torch.nn.functional.pad(
        logits, (0, pad_k, 0, pad_q), value=float("-inf")
    )
    original_logits_padded = logits_padded.clone()
    # Softmax processing
    if not blocked_softmax:
        logits_padded = torch.softmax(logits_padded, dim=-1, dtype=torch.float32)
    # reshape
    def block_reshape(x):
        return x.view(num_q_blocks, block_size_q, num_k_blocks, block_size_k)\
                .permute(0, 2, 1, 3).reshape(num_q_blocks * num_k_blocks, block_size_q * block_size_k)

    if block_size_k * block_size_q != 1:
        logits_padded = block_reshape(logits_padded)
        original_logits_padded = block_reshape(original_logits_padded)
    # Apply softmax again under blocked_softmax mode
    if blocked_softmax:
        probs = torch.softmax(logits_padded, dim=-1, dtype=torch.float32)
    else:
        probs = logits_padded

    # Filtering logic
    def get_head_mask(probs, logits):
        if use_flashinfer:
            pipe = LogitsPipe([TopP()] if filter_mode == "topp" else [MinP()])
            kwargs = {"top_p": threshold} if filter_mode == "topp" else {"min_p": threshold}
            out_probs = pipe(probs, **kwargs)
            return out_probs != 0

        if filter_mode == "topp":
            if block_size_q != 1 and not blocked_softmax:
                mask = score_cover_mask(probs, threshold * block_size_q)
                mask = mask.reshape(num_q_blocks, num_k_blocks, block_size_q * block_size_k)
                probs = probs.reshape(num_q_blocks, num_k_blocks, block_size_q * block_size_k)
                mask[-1] = score_cover_mask(probs[-1], threshold * (block_size_q - pad_q))
                mask = mask.reshape(num_q_blocks * num_k_blocks, block_size_q * block_size_k)
                return mask
            else:
                return score_cover_mask(probs, threshold)
        elif filter_mode == "minp":
            pmax = probs.max(dim=-1, keepdim=True).values
            thres = threshold * pmax
            return probs >= thres
        elif filter_mode == "topnsigma": # use logits
            logits = logits.to(torch.float32)  # Ensure logits are float32
            logits_max = logits.max(dim=-1, keepdim=True).values
            logits_var = masked_std(logits, dim=-1, keepdim=True)
            logits_threshold = logits_max - threshold * logits_var
            return logits >= logits_threshold
        else:
            raise ValueError(f"Unknown mode: {filter_mode}")

    head_mask = get_head_mask(probs, original_logits_padded)
    # Restore mask shape
    if block_size_q * block_size_k == 1:
        return head_mask & inf_mask
    
    head_mask = head_mask.view(
        num_q_blocks, num_k_blocks, block_size_q, block_size_k
    ).permute(0, 2, 1, 3).reshape(
        num_q_blocks * block_size_q, num_k_blocks * block_size_k
    )[:q_len, :k_len]
    return head_mask & inf_mask

def calc_sparsity(mask, causal=True, mode="global",chunk_size=32*1024):
    """
    Calculate sparsity.
    Args:
        mask: torch.Tensor, shape [batch, iters, seqlen]
    Returns:
        float: sparsity ratio
    """
    n_rows, n_cols = mask.shape[-2:]
    assert mask.dtype == torch.bool, "Mask should be boolean type"
    if causal:
        # XXX: torch.tril overflows on large tensors (int overflow in prod(*mask.shape)).
        # Workaround: allocate a fresh tril instead of ones_like.
        # See: https://github.com/pytorch/pytorch/issues/136211
        causal_mask = torch.tril(torch.ones(n_rows, n_cols, dtype=torch.bool, device=mask.device), diagonal=n_cols - n_rows).unsqueeze(0).expand_as(mask)
    else:
        causal_mask = torch.ones_like(mask)
    valid_elements = 0
    total_elements = 0
    row_sparsity = 0
    for i in range(0, n_rows, chunk_size):
        # Calculate sparsity for each chunk
        chunk_mask = mask[..., i:i + chunk_size,:]
        chunk_causal_mask = causal_mask[..., i:i + chunk_size,:]
        if mode == "global":
            valid_elements += chunk_mask.sum(dim=(-1,-2))
            total_elements += chunk_causal_mask.sum(dim=(-1,-2))
        elif mode == "rowwise":
            # Row-wise sparsity calculation
            density = chunk_mask.sum(dim=-1) / chunk_causal_mask.sum(dim=-1)
            row_sparsity += 1.0 - density.mean(dim=-1)
    if mode == "global":
        sparsity = 1.0 - valid_elements / total_elements
        return sparsity.mean().item()
    elif mode == "rowwise":
        # Row-wise sparsity calculation
        return row_sparsity.mean().item()


def calculate_recall_number(mask1, mask2, chunk_size=8192):
    """Calculate recall between two masks: sum(mask1 & mask2) / sum(mask1)"""
    q_len = mask1.shape[-2]
    intersection_nums = 0
    union_nums = 0
    for i in range(0,q_len, chunk_size):
        chunk_mask1 = mask1[..., i:i+chunk_size,:]
        chunk_mask2 = mask2[..., i:i+chunk_size,:]
        intersection_nums += torch.sum(chunk_mask1 & chunk_mask2).float()
        union_nums += torch.sum(chunk_mask1 | chunk_mask2).float()
    return (intersection_nums / union_nums).item()

def calc_recall_score(attention_map, mask, chunk_size=8192, return_mode="mean"):
    """Calculate accumulated score and score per token from attention map and mask"""
    q_len = attention_map.shape[-2]
    accum_score = []
    score_per_token = []
    for i in range(0, q_len, chunk_size):
        chunk_attention = attention_map[..., i:i+chunk_size,:]
        chunk_mask = mask[..., i:i+chunk_size,:]
        chunk_attention = chunk_attention * chunk_mask
        accum_score.append(chunk_attention.sum(dim=-1))
        
        # Handle division by zero for score_per_token using nanmean
        chunk_sum = chunk_attention.sum(dim=-1)
        chunk_mask_sum = chunk_mask.sum(dim=-1)
        # Allow 0/0 to produce nan, then use nanmean to ignore them
        chunk_score_per_token = chunk_sum / chunk_mask_sum  # This will produce nan for 0/0
        score_per_token.append(chunk_score_per_token)
    accum_score = torch.cat(accum_score, dim=-1)
    score_per_token = torch.cat(score_per_token, dim=-1)
    if "mean" == return_mode:
        return accum_score.mean().item(), score_per_token.nanmean().item()
    elif "min_max" == return_mode:
        return (accum_score.mean(dim=-1).min().item(), accum_score.mean(dim=-1).max().item(),accum_score.mean().item()), \
               (score_per_token.nanmean(dim=-1).min().item(), score_per_token.nanmean(dim=-1).max().item(),score_per_token.nanmean().item())
    elif "origin"== return_mode:
        return accum_score, score_per_token
    else:
        raise ValueError(f"Unknown return_mode: {return_mode}")


def masked_std(input: torch.Tensor, dim: int = -1, keepdim: bool = True):
    """
    Compute standard deviation along the given dim, ignoring infinite/invalid values (-inf/nan).
    """
    is_finite_mask = torch.isfinite(input)
    valid_count = is_finite_mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    input_zeroed = input.masked_fill(~is_finite_mask, 0.0)
    mean = input_zeroed.sum(dim=dim, keepdim=keepdim) / valid_count
    mean2 = (input_zeroed ** 2).sum(dim=dim, keepdim=keepdim) / valid_count
    var = mean2 - mean ** 2
    std = var.clamp(min=0.0).sqrt()
    return std

def score_cover_mask(x: torch.Tensor, score: float):
    """
    Return a boolean mask with the same shape as x; selected indices are True, others are False.
    """
    assert x.dtype == torch.float32, "x must be of type float32"
    total_score = x.sum(dim=-1, keepdim=True)
    x_sorted, idx = torch.sort(x, dim=-1, descending=False)
    cumsum_x = torch.cumsum(x_sorted, dim=-1)
    # Select positions where cumsum_x >= (total_score - score)
    selected = cumsum_x >= (total_score - score)
    # Restore to original order
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, idx, selected)
    return mask