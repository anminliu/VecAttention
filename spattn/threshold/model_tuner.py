import os
import subprocess
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []

RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",0))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )

import argparse
from typing import Tuple, Optional
from itertools import product
import math
import json
import ast
import datetime
from spattn.src.kernels.vecattention_kernels import fuse_qk_softmax_minp_wo_causal

import torch
from eval.DiTEvalKit.utils.logger import logger

TUNE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_cache")


def load_tensor(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    q,k = x["query_states"], x["key_states"]
    q = q.detach().to(torch.float32).cpu()
    k = k.detach().to(torch.float32).cpu()
    def check_tensor(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().to(torch.float32).cpu()
        else:
            raise TypeError(f"{path} content is not a Tensor, got {type(x)}")

        # x.shape to [B, H, S, D] / [B, H, S, S]
        if x.ndim == 4:
            pass
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"{path} shape should be [B, H, S, D] or [B, H, S, S], got {x.shape}")
        return x
    
    q = check_tensor(q)
    k = check_tensor(k)
    return q,k

def average_vector(q, block_size):
    """
        q: (batch_size, num_heads, seqlen, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    dtype = q.dtype
    q = q.float() # Use float32 for precise computation

    num_blocks = math.ceil(seq_len / block_size)
    pad_q = num_blocks * block_size - seq_len
    avg_q = (
        torch.nn.functional.pad(q, (0, 0, 0, pad_q), value=0)
        .view(batch_size, num_heads, num_blocks, block_size, head_dim)
        .mean(-2)
    )
    avg_q[:,:,-1, :] = avg_q[:,:,-1,:] * block_size / (block_size - pad_q)
    return avg_q.to(dtype)

def calc_recall(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
        attn: (batch_size, num_heads, chunk_size, seqlen)
        mask: (batch_size, num_heads, chunk_size, seqlen)  binary mask
    returns:
        recall: (batch_size, num_heads, chunk_size)
    """
    batch_size, num_heads, chunk_size, head_dim = q.shape
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(attn, dtype=torch.float32, dim=-1).to(q.dtype)

    assert attn.shape == mask.shape, f"attn {attn.shape} and mask {mask.shape} shape mismatch"
    attn = attn * mask

    recall = attn.sum(-1)
    return recall

def seg_pr_perHead(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size_q: int,
    block_size_k: int = 16,
    group_size_k: int = 16,
    threshold: torch.Tensor = None,
    chunk_size: int = 8 * 1024,
    causal: bool = False,
):
    """
        q: (batch_size, num_heads, seq_len, head_dim)
        k: (batch_size, num_heads, seqlen, head_dim)
        block_size_q: int
        chunk_size: int,

    returns:
        recall: torch.Tensor [num_heads]
        sparsity: torch.Tensor [num_heads]

    """
    SPATTN_BLOCK_SIZE_K = 64 
    batch_size, num_heads, seq_len, head_dim = q.shape
    seqlen = k.shape[2]

    # logger.info(f"Original q shape: {q.shape}")
    if causal:
        attention_mask = torch.ones(1,1,q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=1)
    else:
        attention_mask = torch.zeros(1,1,q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool)

    gap = -torch.log(threshold + 1e-50)
    # logger.info(f"Using gap={gap:.6f} for threshold={threshold}")

    recalls = torch.zeros((batch_size, num_heads, seq_len), dtype=q.dtype, device=q.device)

    sparsity = torch.zeros(num_heads, dtype=q.dtype, device=q.device)

    num_q_blocks = math.ceil(seq_len / block_size_q)

    if causal:
        n = block_size_q // SPATTN_BLOCK_SIZE_K  # ceil_div
        blk_count = torch.full((batch_size, num_heads, num_q_blocks),2 * n, dtype=torch.int32, device=q.device)
        blk_count[...,0] = math.ceil(min(seq_len, block_size_q) / SPATTN_BLOCK_SIZE_K) # for seqlen < block_size_q
        if seq_len > block_size_q and seq_len % block_size_q != 0: # for last q block
            blk_count[...,-1] = n + math.ceil((seq_len - math.floor(seq_len / block_size_q) * block_size_q) / SPATTN_BLOCK_SIZE_K)
        blk_idx = torch.zeros(batch_size, num_heads, num_q_blocks, 2 * n, dtype=torch.int32, device=q.device)
        # First n entries: 0, K, 2K, ...
        offsets = (torch.arange(n, device=q.device, dtype=torch.int32)
                        * SPATTN_BLOCK_SIZE_K)  # [n]
        blk_idx[..., :n] = offsets  # broadcast to [B,H,Q,n]

        base = (torch.arange(0, num_q_blocks, device=q.device, dtype=torch.int32)
                * block_size_q).unsqueeze(-1)  # [Q,1]
        blk_idx[..., n:] = base + offsets 
    else:
        blk_count = torch.zeros(batch_size, num_heads, num_q_blocks, dtype=torch.int32, device=q.device)
        blk_idx = torch.zeros(batch_size, num_heads, num_q_blocks, 1, dtype=torch.int32, device=q.device)

    for chunk_idx in range(0, seq_len, chunk_size):
        q_chunk = q[:, :, chunk_idx : chunk_idx + chunk_size, :]
        chunk_len = q_chunk.shape[2]
        avg_q_chunk = average_vector(q_chunk, block_size_q)

        col_count, col_idx = fuse_qk_softmax_minp_wo_causal(
            avg_q_chunk, k, chunk_idx // block_size_q, gap,
            causal, block_size_q, block_size_k,
            wo_initial=causal, group_k_block=group_size_k,
        )

        blk_count_chunk = blk_count[:,:,chunk_idx//block_size_q:(chunk_idx+chunk_size)//block_size_q]
        blk_idx_chunk = blk_idx[:,:,chunk_idx//block_size_q:(chunk_idx+chunk_size)//block_size_q,:]


        for batch_id in range(batch_size):
            for head_id in range(num_heads):
                attn_weights_chunk = torch.matmul(q_chunk[batch_id, head_id], k[batch_id, head_id].transpose(-2, -1)) / math.sqrt(head_dim)

                causal_mask_chunk = attention_mask[0, 0, chunk_idx:chunk_idx+chunk_size, : k.shape[-2]]
                attn_weights_chunk = attn_weights_chunk + causal_mask_chunk

                attn_weights_chunk = torch.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(q.dtype)

                mask = torch.zeros_like(attn_weights_chunk, dtype=torch.bool)
                for block_id, bc in enumerate(blk_count_chunk[batch_id,head_id]):
                    row_start = block_id * block_size_q
                    if row_start >= chunk_len: continue
                    row_end = min(row_start + block_size_q, chunk_len)

                    offset_in_row = blk_idx_chunk[batch_id,head_id,block_id,:bc]
                    for offset in offset_in_row:
                        col_start = offset
                        col_end = min(offset+SPATTN_BLOCK_SIZE_K, seqlen)
                        mask[row_start:row_end,col_start:col_end] = True

                    column_in_row = col_idx[batch_id,head_id,block_id,:col_count[batch_id,head_id,block_id]]
                    mask[row_start:row_end, column_in_row] = True
                attn_weights_chunk *= mask
                
                causal_bool_mask_chunk = causal_mask_chunk == 0
                sparsity[head_id] += (1.0 - (mask.sum(dim=-1)/causal_bool_mask_chunk.sum(dim=-1)).mean()) * chunk_len
                recalls[batch_id, head_id, chunk_idx:chunk_idx + chunk_size] = attn_weights_chunk.sum(-1)
    
    sparsity /= batch_size
    sparsity /= seq_len

    overall_recall = recalls.sum(dim=(0, 2)) / (batch_size * seq_len)
    return overall_recall, sparsity

# -- Unified quantization utilities -- #
ND = 6
def quantize_threshold(x):
    return round(float(x), ND)

def normalize_key_tuple(k):
    """
    Expects k to be (block_size_q, block_size_k, group_size_k, threshold).
    Converts the first three to int, quantizes the last to round(..., nd).
    """
    bq, bk, gk, th = k
    return (int(bq), int(bk), int(gk), quantize_threshold(th))


# -- Save/Load -- #
def save_res(res, filename="res.json"):
    """
    res: List[Dict[Tuple[int,int,int,float], (recall, sparsity)]]
    Normalizes keys and converts them to strings for JSON serialization.
    """
    output = {}

    for k, v in res.items():
        if isinstance(k, tuple) and len(k) == 4:
            k = normalize_key_tuple(k)
        else:
            raise ValueError(f"Unexpected key format: {k}")

        # Save as string; will use literal_eval when loading back
        key_str = str(k)

        # value -> list
        value_list = list(v)
        output[key_str] = value_list

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # logger.info(f"[✔] res saved to {filename}")


def load_res(filename="res.json"):
    if not os.path.exists(filename):
        # logger.info(f"[!] {filename} does not exist, returning empty dict")
        return {}

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = {}
    
        # data[head_idx] is { key_str: [recall, sparsity], ... }
    for k_str, vs in data.items():
        recall, sparsity = vs  # JSON stores as list
        k = ast.literal_eval(k_str)  # -> tuple
        # Normalize (especially quantize threshold to round(..., nd) float)
        k = normalize_key_tuple(k)

        block_size_q, block_size_k, group_size_k, threshold = k

        res[k] = (recall, sparsity)


    # logger.info(f"[✔] res loaded from {filename}")
    return res


def tune_per_layer_coop_bisect(
    layer_idx: int,
    head_list: list,
    block_size_q: int,
    block_size_k: int,
    group_size_k: int,
    out_dir: str,
    target_recall: float = 0.90,
    QK_dir=None,
    device: Optional[torch.device] = None,
    iters: int = 14,
    th_low_init: float = 0.0,
    th_high_init: float = 0.5,
    causal: bool = False,
):
    """
    Cooperative bisection (one kernel call per iteration):
      - Each head maintains its own threshold interval and best result.
      - Each iteration groups all unconverged, cache-miss heads together
        and makes a single seg_pr call with a threshold vector [H_grp],
        returning [H_grp] recall/sparsity values.
      - Updates each head's low/high/best and convergence state.
      - Extra sampling phase can be batched per group or per head.
    Returns: {head_idx: (best_thr, best_recall, best_sparsity)}
    """
    # ---- Load Q/K ----
    # query_path = os.path.join(QK_dir, f"query_states_layer{layer_idx}.pt")
    # key_path   = os.path.join(QK_dir, f"key_states_layer{layer_idx}.pt")
    qk_path = os.path.join(QK_dir, f"layer_{layer_idx}_qk.pt")
    pid = os.path.basename(QK_dir)

    # q_all = load_tensor(query_path)
    # k_all = load_tensor(key_path)
    q_all, k_all = load_tensor(qk_path)

    # logger.info(f"[{pid}] Loaded Q/K for Layer {layer_idx}: Q {q_all.shape}, K {k_all.shape}")

    assert q_all.shape[0] == k_all.shape[0] and q_all.shape[1] == k_all.shape[1] and q_all.shape[3] == k_all.shape[3], \
        f"query and key shape mismatch, query: {q_all.shape}, key: {k_all.shape}"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_all = q_all.to(device)
    k_all = k_all.to(device)

    layer_dir = os.path.join(out_dir, f"L{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)

    # logger.info(f"[{pid}] Processing Layer {layer_idx} with cooperative bisection ...")

    # ---- Per-head state ----
    th_low  = {h: th_low_init  for h in head_list}
    th_high = {h: th_high_init for h in head_list}
    best    = {h: None         for h in head_list}  # best[h] = (thr, recall, sparsity)
    active  = {h: True         for h in head_list}  # not yet converged

    # Per-head cache (using the same key format)
    caches = {}
    for h in head_list:
        head_cache_path = os.path.join(layer_dir, f"H{h}_res.json")
        caches[h] = load_res(head_cache_path) or {}

    # ---- Helpers ----
    def quantize(th: float) -> float:
        return quantize_threshold(th)

    def qk_for_heads(heads: list):
        """
        Keep shape [B, H_grp, L, D] (do not merge into batch dim).
        """
        q_sub = q_all[:, heads, :, :].contiguous()  # [B, H_grp, Lq, D]
        k_sub = k_all[:, heads, :, :].contiguous()  # [B, H_grp, Lk, D]
        return q_sub, k_sub

    def cache_key(thr_q: float):
        return (int(block_size_q), int(block_size_k), int(group_size_k), thr_q)

    def write_back_caches():
        for h in head_list:
            head_cache_path = os.path.join(layer_dir, f"H{h}_res.json")
            save_res(caches[h], head_cache_path)

    # ---- Main loop (at most one seg_pr_torch call per iteration) ----
    for it in range(iters):
        # logger.info(f"[{pid}] Layer {layer_idx} Bisection Iteration {it} ...")
        # 1) Compute quantized midpoint for all active heads
        mids = {}
        thr_q_map = {}
        for h in head_list:
            if not active[h]:
                continue
            mid = (th_low[h] + th_high[h]) / 2.0
            thr_q = quantize(mid)
            mids[h] = mid
            thr_q_map[h] = thr_q

        # 2) Split into cache hits vs. needs computation
        to_compute = []
        cached_results = {}
        for h, thr_q in thr_q_map.items():
            key = cache_key(thr_q)
            if caches[h].get(key, None) is not None:
                cached_results[h] = caches[h][key]  # (recall, sparsity)
            else:
                to_compute.append(h)

        # 3) Single dispatch: call seg_pr_torch only for to_compute heads (threshold vector)
        results = {}
        if to_compute:
            logger.info(f"[{pid}] Layer {layer_idx} Iter {it}: Computing for heads {to_compute}")
            q_sub, k_sub = qk_for_heads(to_compute)               # [B, H_grp, L, D]
            thr_vec = torch.tensor([thr_q_map[h] for h in to_compute],
                                   dtype=q_sub.dtype, device=q_sub.device)  # [H_grp]

            # seg_pr_torch returns: recall [H_grp], sparsity [H_grp]
            recall_vec, sparsity_vec = seg_pr_perHead(
                q_sub,
                k_sub,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                group_size_k=group_size_k,
                threshold=thr_vec,   # per-head threshold
                causal=causal,
            )

            # Map results back to each head and write to cache
            recall_list   = recall_vec.tolist()
            sparsity_list = sparsity_vec.tolist()
            for i, h in enumerate(to_compute):
                rec, sp = recall_list[i], sparsity_list[i]
                results[h] = (rec, sp)
                caches[h][cache_key(thr_q_map[h])] = (rec, sp)
        # else:
        #     logger.info(f"[{pid}] Layer {layer_idx} Iter {it}: All active heads hit cache, no computation needed.")

        # 4) Merge cache-hit results
        results.update(cached_results)  # {h: (recall, sparsity)}

        # 5) Update each head's best, bisection interval, and convergence state
        for h, (rec, sp) in results.items():
            thr_q = thr_q_map[h]
            thr = mids[h]
            # Update best
            if (best[h] is None) or (abs(rec - target_recall) < abs(best[h][1] - target_recall)):
                best[h] = (thr_q, rec, sp)
            # Advance bisection (use quantized threshold for consistency)
            if rec < target_recall:
                th_high[h] = thr_q
            else:
                th_low[h]  = thr_q

        # 6) Flush to disk + early exit
        # write_back_caches()
        if all(not active[h] for h in head_list):
            break

    # ---- Extra sampling (8 iterations outer loop; <=num_heads per round, one dispatch) ----
    # Only sample heads that already have a best; at most one threshold per head per round
    heads_with_best = [h for h in head_list if best[h] is not None]
    assert len(heads_with_best) == len(head_list), "All heads should have a best result"
    if heads_with_best:
        # Track each head's current threshold (starts from best, multiplied by 0.9 each round)
        cur_thr = {h: best[h][0] for h in heads_with_best}
        # logger.info(f"[{pid}] Layer {layer_idx} Starting Extra Sampling for heads {heads_with_best}, cur_thr = {cur_thr} ...")

        for _ in range(8):
            # logger.info(f"[{pid}] Layer {layer_idx} Extra Sampling Round {_} ...")
            round_heads, round_thrs = [], []
            # Collect one candidate threshold per head for this round (if not cached)
            for h in heads_with_best:
                # Generate next candidate threshold
                candidate = cur_thr[h] * 0.9
                thr_q = quantize(candidate)

                # Add to this round if not yet computed; advance current threshold regardless
                if caches[h].get(cache_key(thr_q), None) is None:
                    round_heads.append(h)
                    round_thrs.append(thr_q)

                # Advance head's current threshold for the next round (*0.9)
                cur_thr[h] = candidate

            if not round_heads:
                # logger.info(f"[{pid}] Layer {layer_idx} Extra Sampling Round {_}: All heads hit cache, no computation needed.")
                continue
            

            # logger.info(f"[{pid}] Layer {layer_idx} Extra Iter {it}: Computing for heads {round_heads}")

            # Single dispatch, batch size <= num_heads
            q_sub = q_all[:, round_heads, :, :].contiguous()  # [B, H_round, Lq, D]
            k_sub = k_all[:, round_heads, :, :].contiguous()  # [B, H_round, Lk, D]
            thr_vec = torch.tensor(round_thrs, dtype=q_sub.dtype, device=q_sub.device)  # [H_round]

            recall_vec, sparsity_vec = seg_pr_perHead(
                q_sub,
                k_sub,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                group_size_k=group_size_k,
                threshold=thr_vec,
            )

            recall_list   = recall_vec.tolist()
            sparsity_list = sparsity_vec.tolist()
            # logger.info(f"[{pid}] Layer {layer_idx} Extra Sampling Round {_}: recall_list = {recall_list}, sparsity_list = {sparsity_list}")
            for i, h in enumerate(round_heads):
                thr_q = round_thrs[i]
                rec, sp = recall_list[i], sparsity_list[i]
                caches[h][cache_key(thr_q)] = (rec, sp)

    write_back_caches()

    return {h: best[h] for h in head_list}

def analyse_dp_results(
    block_size_q,
    block_size_k,
    group_size_k,
    num_layer: int,
    num_head: int,
    dp_cache_path: str,
    TARGET_SPARSITY: Optional[float] = None,
    DP_ND: int = 4,
):
    """
    When TARGET_SPARSITY is None, iterates over [0.00, 1.00] at 0.05 intervals;
    otherwise processes only the specified TARGET_SPARSITY.
    Results saved to: os.path.join(os.path.dirname(dp_cache_path), "results", filename)
    """
    with open(dp_cache_path, "r", encoding="utf-8") as f:
        json_dp = json.load(f)
    dp = []
    for i in range(len(json_dp)):
        dp_dict = {}
        for k_str, v_list in json_dp[str(i)].items():
            k = float(k_str)
            v = tuple(v_list)
            dp_dict[k] = v
        dp.append(dp_dict)

    results_dir = os.path.join(os.path.dirname(dp_cache_path), "results")
    os.makedirs(results_dir, exist_ok=True)

    def process_one_target(target_sparsity: float):
        quantized_target_sparsity = round(target_sparsity, DP_ND)
        interval = 10 ** (-DP_ND)

        tmp1 = tmp2 = quantized_target_sparsity

        while dp[-1].get(tmp1, None) is None and tmp1 >= 0:
            tmp1 -= interval
        tmp1_res = dp[-1].get(tmp1, None)

        while dp[-1].get(tmp2, None) is None and tmp2 <= 1.0:
            tmp2 += interval
        tmp2_res = dp[-1].get(tmp2, None)

        logger.info(f"Dynamic Programming Results for Block size Q={block_size_q}, Block size K={block_size_k}, Group size K={group_size_k}, target_sparsity={quantized_target_sparsity:.{DP_ND}f}")

        if tmp1_res is None and tmp2_res is None:
            logger.warning(f"[SKIP] No valid sparsity found near {quantized_target_sparsity:.{DP_ND}f} in DP results. Skipping.")
            return

        # Choose the closer feasible solution (prefer >= target)
        if tmp2_res is not None:
            final_recall = tmp2_res[0] / (num_layer * num_head)
            final_sparsity = tmp2_res[1] / (num_layer * num_head)
            logger.info(f"  Sparsity >= {quantized_target_sparsity}: Recall = {final_recall:.4f}, Sparsity = {final_sparsity:.10f}")
            chosen = tmp2
        else:
            final_recall = tmp1_res[0] / (num_layer * num_head)
            final_sparsity = tmp1_res[1] / (num_layer * num_head)
            logger.info(f"  Sparsity <= {quantized_target_sparsity}: Recall = {final_recall:.4f}, Sparsity = {final_sparsity:.10f}")
            chosen = tmp1

        # Backtrack thresholds
        output_thresholds = [[-1 for _ in range(num_head)] for _ in range(num_layer)]
        cur_sparsity = chosen
        for i in range(len(dp) - 1, -1, -1):
            _, _, thr, pre_avg_sparsity = dp[i][cur_sparsity]
            layer_idx = i // num_head
            head_idx = i % num_head
            output_thresholds[layer_idx][head_idx] = thr
            cur_sparsity = pre_avg_sparsity

        # Save thresholds to results directory
        filename = f"per_head_min_recall_Q{block_size_q}_K{block_size_k}_G{group_size_k}_S{final_sparsity:.3f}_R{final_recall}_thresholds.json"
        thresholds_cache_path = os.path.join(results_dir, filename)

        # if os.path.exists(thresholds_cache_path):
        #     logger.info(f"[SKIP] {thresholds_cache_path} already exists, skipping write.")
        #     return

        with open(thresholds_cache_path, "w", encoding="utf-8") as f:
            json.dump(output_thresholds, f, indent=2, ensure_ascii=False)
        logger.info(f"[✔] Optimal thresholds saved to {thresholds_cache_path}")

    # Target set: process only the specified value if given, otherwise iterate [0, 1] step 0.05
    if TARGET_SPARSITY is None:
        targets = [round(i * 0.05, 2) for i in range(0, 21)]  # 0.00 ... 1.00
    else:
        targets = [TARGET_SPARSITY]

    for t in targets:
        process_one_target(t)

def dynamic_program_logic(
    block_size_q,
    block_size_k,
    group_size_k,
    num_layer: int,
    num_head: int,
    sampled_res_dir_list: list[str],
    dp_cache_path: str = None,
    DP_ND: int = 4,
):
    """
    dp[i-th head][current sparsity] = max recall
    """

    if os.path.exists(dp_cache_path):
        logger.info(f"[✔] DP cache {dp_cache_path} already exists, skip DP logic.")
        return

    head_list = list(product(range(num_layer), range(num_head)))

    dp = [{} for _ in head_list]

    # logger.info(f"len head_list: {len(list(head_list))}")

    for i, (layer_idx, head_idx) in enumerate(head_list):
        # logger.info(iqq)
        candidates = {}
        for res_dir in sampled_res_dir_list:
            res_path = os.path.join(res_dir, f"L{layer_idx}/H{head_idx}_res.json")
            cache_res = load_res(res_path)
            for k, v in cache_res.items():
                if k[0] != block_size_q or k[1] != block_size_k or k[2] != group_size_k:
                    continue
                thr = k[3]
                recall, sparsity = v
                if candidates.get(thr, None) is None:
                    candidates[thr] = [(recall, sparsity)]
                else:
                    candidates[thr].append((recall, sparsity))
        # logger.info(f"[LAYER{layer_idx}][HEAD{head_idx}] Loaded {len(candidates)} candidate thresholds.")

        assert len(candidates) > 0, f"[LAYER{layer_idx}][HEAD{head_idx}] No candidates found!"

        for thr, res_list in candidates.items():
            recalls = [r for r, s in res_list]
            sparsities = [s for r, s in res_list]
            recall = sum(recalls) / len(recalls)
            sparsity = sum(sparsities) / len(sparsities)
            # logger.info(f"[LAYER{layer_idx}][HEAD{head_idx}] THR={thr:.6f}: recall={recall:.4f}, sparsity={sparsity:.10f}")
            

            # init for head 0
            if i == 0:
                quantized_sparsity = round(sparsity, DP_ND)
                dp[i][quantized_sparsity] = (recall, sparsity, thr, -1)
                continue

            for pre_avg_sparsity, value in dp[i-1].items():
                pre_recall_sum, pre_sparsity_sum, pre_thr, _ = value

                new_recall_sum = pre_recall_sum + recall
                new_sparsity_sum = pre_sparsity_sum + sparsity
                new_avg_sparsity = (new_sparsity_sum) / (i + 1)
                quantized_new_sparsity = round(new_avg_sparsity, DP_ND)

                if dp[i].get(quantized_new_sparsity, None) is None:
                    dp[i][quantized_new_sparsity] = (new_recall_sum, new_sparsity_sum, thr, pre_avg_sparsity)
                else:
                    exist_recall_sum, exist_sparsity_sum, exist_thr, exist_pre_avg_sparsity = dp[i][quantized_new_sparsity]
                    if new_recall_sum > exist_recall_sum:
                        dp[i][quantized_new_sparsity] = (new_recall_sum, new_sparsity_sum, thr, pre_avg_sparsity)


    with open(dp_cache_path, "w", encoding="utf-8") as f:
        json_dp = {}
        for i, dp_dict in enumerate(dp):
            json_dp[i] = {}
            for k, v in dp_dict.items():
                json_dp[i][f"{k:.{DP_ND}f}"] = list(v)
        json.dump(json_dp, f, indent=2, ensure_ascii=False)
    logger.info(f"[✔] DP results saved to {dp_cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--QK_dir", type=str, default=None, help="Directory containing query/key tensors")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--out_dir", type=str, default=TUNE_CACHE_DIR, help="Output directory path")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--bsearch_target_recall", type=float, default=0.9)
    parser.add_argument("--dp_target_sparsity", type=float, default=None)
    parser.add_argument("--model", type=str, choices=["HuanyuanVideo", "Wan"], default="HuanyuanVideo")
    parser.add_argument("--prompt_to_list", type=int, nargs="+", default=[0], help="Prompt sample ID list")
    parser.add_argument("--nd", type=int, default=6, help="Threshold quantization decimal places")

    parser.add_argument("--block-size-q-list", type=int, nargs="+", default=[64], help="Candidate block_size_q list")
    parser.add_argument("--block-size-k-list", type=int, nargs="+", default=[16], help="Candidate block_size_k list")
    parser.add_argument("--group-size-k-list", type=int, nargs="+", default=[8192], help="Candidate group_size_k list")
    args = parser.parse_args()

    ND = args.nd

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    block_size_q_list = args.block_size_q_list
    block_size_k_list = args.block_size_k_list
    group_size_k_list = args.group_size_k_list

    TARGET_RECALL = args.bsearch_target_recall

    if args.model == "HuanyuanVideo":
        NUM_LAYER = 60
        NUM_HEAD = 24
    elif args.model == "Wan":
        NUM_LAYER = 40
        NUM_HEAD = 40
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    LAYER_LIST = list(range(NUM_LAYER))
    HEAD_LIST = list(range(NUM_HEAD))
    PROMPT_ID_LIST = args.prompt_to_list

    out_dir = os.path.join(args.out_dir, f"{args.model}")

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        # torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600))),
            # device_id=torch.device(f'cuda:{LOCAL_RANK}'),
        )

    for block_size_q, block_size_k, group_size_k in product(
        block_size_q_list,
        block_size_k_list,
        group_size_k_list,
    ):
        qkg_dir = os.path.join(
            out_dir,
            f"Q{block_size_q}_K{block_size_k}_G{group_size_k}/per_head_min_recall_{TARGET_RECALL}" if args.exp_name is None else f"Q{block_size_q}_K{block_size_k}_G{group_size_k}/{args.exp_name}_per_head_min_recall_{TARGET_RECALL}",
        )

        if not args.no_tune:
            # Build all tasks (pid, layer)
            all_tasks = [(pid, layer) for pid in PROMPT_ID_LIST for layer in LAYER_LIST]

            # Task partitioning
            if WORLD_SIZE > 1:
                assigned_tasks = [t for idx, t in enumerate(all_tasks) if idx % WORLD_SIZE == RANK]
            else:
                assigned_tasks = all_tasks

            logger.info(f"[DIST] LOCAL_RANK={LOCAL_RANK}/{WORLD_SIZE} assigned_tasks={len(assigned_tasks)} total_tasks={len(all_tasks)}")

            for pid, layer in assigned_tasks:
                pid_dir = os.path.join(qkg_dir, f"{pid}")
                os.makedirs(pid_dir, exist_ok=True)
                qk_states_dir = os.path.join(args.QK_dir, f"{pid}")
                tune_per_layer_coop_bisect(
                    layer,
                    HEAD_LIST,
                    block_size_q,
                    block_size_k,
                    group_size_k,
                    pid_dir,
                    target_recall=TARGET_RECALL,
                    QK_dir=qk_states_dir,
                    causal=args.causal,
                )
                
            if WORLD_SIZE > 1:
                dist.barrier()

        if LOCAL_RANK == 0:
            sampled_res_dir_list_qkg = [
                os.path.join(
                    qkg_dir,
                    f"{pid}",
                ) for pid in PROMPT_ID_LIST
            ]
            file_name = f"per_head_min_sparsity_Q{block_size_q}_K{block_size_k}_G{group_size_k}_dp.json"
            dp_cache_path = os.path.join(
                qkg_dir,
                file_name
            )

            dynamic_program_logic(
                block_size_q,
                block_size_k,
                group_size_k,
                NUM_LAYER,
                NUM_HEAD,
                sampled_res_dir_list_qkg,
                dp_cache_path = dp_cache_path,
            )

            analyse_dp_results(
                block_size_q,
                block_size_k,
                group_size_k,
                NUM_LAYER,
                NUM_HEAD,
                dp_cache_path,
                TARGET_SPARSITY=args.dp_target_sparsity,
            )

            logger.info(f"[✔] All tuning for Q{block_size_q}_K{block_size_k}_G{group_size_k} done.")

