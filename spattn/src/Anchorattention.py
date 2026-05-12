import torch
import triton
import triton.language as tl
import time
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import os
### == from Anchor-Attention/anchor_attn/src/ops/get_mask_test.py
@triton.jit
def get_q_mask_triton(
    Q, K, anchors, true_coords, true_counts, diff,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_az, stride_ah, stride_an, stride_ad,
    stride_cz, stride_ch, stride_cn, stride_cd,
    stride_tcz, stride_tch, stride_tcn, stride_tcd,
    B, H, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SZ: tl.constexpr, step: tl.constexpr,
    qk_scale: tl.constexpr,
    dtype: tl.constexpr,
):
    # Get thread-block indices
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Compute tensor offsets
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    anchors_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    coords_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch
    counts_offset = off_z.to(tl.int64) * stride_tcz + off_h.to(tl.int64) * stride_tch

    # Create block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ * step, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ),
        order=(0, 1),
    )
    anchors_block_ptr = tl.make_block_ptr(
        base=anchors + anchors_offset,
        shape=(N, 1),
        strides=(stride_an, stride_ad),
        offsets=(start_m * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ * step, 1),
        order=(1, 0),
    )
    coords_block_ptr = tl.make_block_ptr(
        base=true_coords + coords_offset,
        shape=(N // BLOCK_SZ // step, N),
        strides=(stride_cn, stride_cd),
        offsets=(start_m, 0),
        block_shape=(1, BLOCK_SZ),
        order=(1, 0),
    )
    counts_block_ptr = tl.make_block_ptr(
        base=true_counts + counts_offset,
        shape=(N // BLOCK_SZ // step, 1),
        strides=(stride_tcn, stride_tcd),
        offsets=(start_m, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )

    # Load and process queries
    q = tl.load(Q_block_ptr)
    q = tl.reshape(q, (step, BLOCK_SZ, D))          # [step, BS, D]
    q = tl.sum(q, axis=1, keep_dims=False) / BLOCK_SZ  # [step, D]
    q = q.to(dtype)

    # Load and process anchors
    anchors = tl.load(anchors_block_ptr)
    anchors = tl.reshape(anchors, (step, BLOCK_SZ, 1))  # [step, BS, 1]
    anchors = tl.sum(anchors, axis=1, keep_dims=False) / BLOCK_SZ  # [step, 1]
    anchors = anchors.to(dtype)

    qk_scale = qk_scale * 1.44269504

    # Initialize indices and counters
    count = tl.zeros((1, 1), dtype=tl.int32)           # Record the number of valid indices
    cur_mask = tl.zeros((step, BLOCK_SZ), dtype=tl.int32)
    f_mask = tl.zeros((1, BLOCK_SZ), dtype=tl.int1)

    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SZ))
    for i in range(BLOCK_SZ, start_m * BLOCK_SZ * step - BLOCK_SZ, BLOCK_SZ):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale                  # [step, BS]
        qk = qk.to(dtype)

        # Compute mask
        cur_mask = ((qk + diff) >= anchors).to(tl.int32)             # [step, BS]
        f_mask = (tl.sum(cur_mask, axis=0, keep_dims=True).to(tl.int32) > 0)  # [1, BS]

        # Generate global indices
        global_indices = tl.arange(0, BLOCK_SZ)[None, :] + i         # [1, BS]

        # Compute valid mask and count
        valid_mask = f_mask > 0                                      # [1, BS]
        valid_count = tl.sum(valid_mask.to(tl.int32))                # scalar

        # Create valid positions: keep global_indices where valid_mask is True, else set to N
        valid_positions = tl.where(valid_mask, global_indices, N)    # [1, BS]

        # Sort valid_positions along the last dimension
        valid_positions = tl.sort(valid_positions)

        count = count + valid_count

        # Store results
        tl.store(coords_block_ptr, valid_positions)
        coords_block_ptr = tl.advance(coords_block_ptr, (0, valid_count))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SZ))

    # Store indices and counts
    tl.store(counts_block_ptr, count)


def get_q_mask(
    Q, K, anchors, block_size, step, diff,
    B, H, seq, h_dim, qk_scale
):
    assert Q.is_contiguous(), "Q should be contiguous"
    assert K.is_contiguous(), "K should be contiguous"
    assert block_size in [16, 32, 64, 128, 256], "BLOCK_SIZE must be a power of 2 and ≥16"
    assert step in [16, 32], "step must be ≥16"
    assert seq % (block_size * step) == 0

    grid = (triton.cdiv(seq, block_size * step), B * H, 1)
    true_coords = torch.zeros(
        (B, H, seq // block_size // step, seq),
        dtype=torch.int32, device=Q.device
    ).contiguous()
    true_counts = torch.zeros(
        (B, H, seq // block_size // step, 1),
        dtype=torch.int32, device=Q.device
    ).contiguous()

    get_q_mask_triton[grid](
        Q, K, anchors, true_coords, true_counts, diff,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        anchors.stride(0), anchors.stride(1), anchors.stride(2), anchors.stride(3),
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, seq, h_dim,
        BLOCK_SZ=block_size, step=step,
        qk_scale=qk_scale,
        dtype=tl.float16 if Q.dtype == torch.float16 else tl.bfloat16
    )
    return true_coords, true_counts

### ===  from Anchor-Attention/anchor_attn/src/ops/anchor_step_1_2_v2.py
@triton.jit
def get_mask_fused_anchor_triton(
    Q, K, V, L_buffer, M_buffer, Acc_buffer,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_lb, stride_lh,
    stride_mb, stride_mh,
    stride_acc_z, stride_acc_h, stride_acc_m, stride_acc_k,
    init_cnt: tl.constexpr,
    local_cnt: tl.constexpr,
    sm_scale: tl.constexpr,
    step: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SZ: tl.constexpr,
    BLOCK_SZ_M: tl.constexpr,
    BLOCK_SZ_N: tl.constexpr,
    dtype: tl.constexpr
):
    # Determine the block index for queries and the combined batch-head index
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_SZ_N)

    # Compute offsets along batch and head dimensions
    off_z = off_hz // H
    off_h = off_hz % H
    qo_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh
    lm_offset = off_z * stride_lb + off_h * stride_lh
    mm_offset = off_z * stride_mb + off_h * stride_mh
    acc_offset = off_z * stride_acc_z + off_h * stride_acc_h

    # Identify global block boundaries for initialization and local scopes
    big_block_startm = (start_m * BLOCK_SZ_M) // (BLOCK_SZ * step)
    init_offset = min(init_cnt * BLOCK_SZ * step, (start_m + 1) * BLOCK_SZ_M) #NOTE: BUG or not???
    local_offset = max((big_block_startm - local_cnt + 1) * BLOCK_SZ * step, 0)

    # Create block pointers for Q, L, M, Acc buffers
    q_ptrs = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_SZ_M, 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0)
    )
    l_ptrs = tl.make_block_ptr(
        base=L_buffer + lm_offset,
        shape=(N,),
        strides=(1,),
        offsets=(start_m * BLOCK_SZ_M,),
        block_shape=(BLOCK_SZ_M,),
        order=(0,)
    )
    m_ptrs = tl.make_block_ptr(
        base=M_buffer + mm_offset,
        shape=(N,),
        strides=(1,),
        offsets=(start_m * BLOCK_SZ_M,),
        block_shape=(BLOCK_SZ_M,),
        order=(0,)
    )
    acc_ptrs = tl.make_block_ptr(
        base=Acc_buffer + acc_offset,
        shape=(N, D),
        strides=(stride_acc_m, stride_acc_k),
        offsets=(start_m * BLOCK_SZ_M, 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0)
    )

    # Block pointers for initial and local key/value windows
    k_init_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ_N),
        order=(0, 1)
    )
    v_init_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SZ_N, D),
        order=(1, 0)
    )
    k_local_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, local_offset),
        block_shape=(D, BLOCK_SZ_N),
        order=(0, 1)
    )
    v_local_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(local_offset, 0),
        block_shape=(BLOCK_SZ_N, D),
        order=(1, 0)
    )

    # Pre-scale queries for log-domain softmax
    q = tl.load(q_ptrs)
    q = (q * sm_scale * 1.44269504).to(dtype)
    offs_m = start_m * BLOCK_SZ_M + tl.arange(0, BLOCK_SZ_M)

    # Initialize running max (m_i), log-sum-exp accumulator (l_i), and output accumulator
    m_i = tl.zeros([BLOCK_SZ_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SZ_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SZ_M, D], dtype=tl.float32)

    # Process initial global window
    for j in range(0, init_offset, BLOCK_SZ_N):
        k_init = tl.load(k_init_ptrs).to(dtype)
        qk = tl.dot(q, k_init)
        causal_mask = (offs_n[None, :] + j) <= offs_m[:, None]
        qk = qk + tl.where(causal_mask, 0, -1e6)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v_init = tl.load(v_init_ptrs)
        acc += tl.dot(p.to(dtype), v_init)
        m_i = m_new
        k_init_ptrs = tl.advance(k_init_ptrs, (0, BLOCK_SZ_N))
        v_init_ptrs = tl.advance(v_init_ptrs, (BLOCK_SZ_N, 0))

    # Process local window around each query block
    for j in range(max(local_offset, init_offset), (big_block_startm + 1) * BLOCK_SZ * step, BLOCK_SZ_N):
        k_local = tl.load(k_local_ptrs)
        qk = tl.dot(q, k_local)
        causal_mask = (offs_n[None, :] + j) <= offs_m[:, None]
        qk = qk + tl.where(causal_mask, 0, -1e6)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v_local = tl.load(v_local_ptrs)
        acc += tl.dot(p.to(dtype), v_local)
        m_i = m_new
        k_local_ptrs = tl.advance(k_local_ptrs, (0, BLOCK_SZ_N))
        v_local_ptrs = tl.advance(v_local_ptrs, (BLOCK_SZ_N, 0))

    # Store updated Acc, L, M buffers
    tl.store(acc_ptrs, acc.to(Acc_buffer.type.element_ty))
    tl.store(l_ptrs, l_i.to(L_buffer.type.element_ty))
    tl.store(m_ptrs, m_i.to(M_buffer.type.element_ty))


def anchor_attn_step_1(
    Q, K, V,
    init_cnt,
    local_cnt,
    block_size,
    step,
    B, H, N, D, sm_scale,
    BLOCK_SZ_M=128, BLOCK_SZ_N=128
):
    # Allocate buffers for L, M, and Accumulator (in fp32)
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    M_buffer = torch.full((B, H, N), -float("inf"), dtype=torch.float32, device=Q.device).contiguous()
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32).contiguous()
    grid = (triton.cdiv(N, BLOCK_SZ_M), B * H, 1)
    get_mask_fused_anchor_triton[grid](
        Q, K, V, L_buffer, M_buffer, Acc_buffer,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),
        init_cnt, local_cnt,
        sm_scale, step,
        B, H, N, D,
        BLOCK_SZ=block_size,
        BLOCK_SZ_M=BLOCK_SZ_M,
        BLOCK_SZ_N=BLOCK_SZ_N,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3,
    )
    return L_buffer, M_buffer, Acc_buffer


def anchor_attn_step_2(
    Q, K, M_buffer,
    block_size,
    step,
    theta, B, H, N, D, sm_scale
):
    return get_q_mask(
        Q, K, M_buffer.unsqueeze(-1),
        block_size, step, theta,
        B, H, N, D, sm_scale
    )


### == from Anchor-Attention/anchor_attn/src/ops/anchor_attn_v4.py
@triton.jit
def _triton_stripe_sparse_attn_fwd_kernel_step3(
    Q, K, V, sm_scale,
    true_coords, true_counts,
    Out, L_buffer, M_buffer, Acc_buffer,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_mb, stride_mh,
    stride_abz, stride_abh, stride_abm, stride_abd,
    stride_tcz, stride_tch, stride_tcm, stride_tck,
    stride_tctz, stride_tcth, stride_tctm, stride_tctk,
    Z, H, N_CTX,
    STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_INDICES: tl.constexpr,
    dtype: tl.constexpr,
):
    # Determine program indices for major dimensions
    start_m = tl.program_id(0)
    big_block_startm = (start_m * BLOCK_M) // STEP
    off_hz = tl.program_id(1)

    # Compute offsets for query, key/value, and buffers
    query_start = start_m * BLOCK_M
    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    ab_offset = (off_hz // H) * stride_abz + (off_hz % H) * stride_abh

    # Offsets for coordinate and count buffers
    coord_offset = (off_hz // H) * stride_tcz + (off_hz % H) * stride_tch + big_block_startm * stride_tcm
    count_offset = (off_hz // H) * stride_tctz + (off_hz % H) * stride_tcth + big_block_startm * stride_tctm

    # Prepare pointers for loading keys and values
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk

    # Load sparsity count for this block
    count_ptr = true_counts + count_offset
    sparse_count = tl.load(count_ptr)

    # Reset query offsets
    offs_m = query_start + tl.arange(0, BLOCK_M)
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    l_ptrs = L_buffer + lm_offset + offs_m
    m_ptrs = M_buffer + lm_offset + offs_m
    a_ptrs = Acc_buffer + ab_offset + offs_m[:, None] * stride_abm + offs_d[None, :] * stride_abd

    # Load initial buffers and scaled queries
    m_i = tl.load(m_ptrs)
    l_i = tl.load(l_ptrs)
    acc = tl.load(a_ptrs)
    q = tl.load(q_ptrs)
    q = (q * sm_scale * 1.44269504).to(dtype)  # Convert to log2 scale

    # Load true coordinate pointers for iteration
    coord_ptr = true_coords + coord_offset

    # Iterate through the selected sparse indices in chunks
    for idx_start in range(0, sparse_count, NUM_INDICES):
        idx_range = idx_start + tl.arange(0, NUM_INDICES)
        valid_mask = idx_range < sparse_count
        block_ids = tl.load(coord_ptr + idx_range, mask=valid_mask, other=0)

        # Compute pointers within the window
        k_window = k_ptrs + block_ids[None, :] * stride_kn
        v_window = v_ptrs + block_ids[:, None] * stride_vn

        # Load masked keys and values
        k = tl.load(k_window, mask=valid_mask[None, :], other=0.0)
        v = tl.load(v_window, mask=valid_mask[:, None], other=0.0)

        # Compute QK^T for this mini-batch
        qk = tl.zeros([BLOCK_M, NUM_INDICES], dtype=dtype)
        qk += tl.dot(q, k)
        qk = tl.where(valid_mask[None, :], qk, float("-inf"))

        # Update running max for numerical stability
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        # Scale accumulator and add new weighted values
        acc *= alpha[:, None]
        acc += tl.dot(p.to(dtype), v)

        # Update L and M buffers
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Finalize attention output
    result = acc / l_i[:, None]
    tl.store(o_ptrs, result.to(dtype))

def _anchor_attn_test_time(Q, K, V,
                        sm_scale: float= None,
                        block_size:int=64, 
                        step:int=16,
                        difference:float=100,
                        return_sparsity_ratio: bool = False,
                        BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128):
    B, H, N, D = Q.shape
    assert N % block_size == 0
    NUM_ROWS_STEP3 = triton.cdiv(N, block_size * step)
    output = torch.zeros_like(Q, device=Q.device)
    
      
    torch.cuda.synchronize()
    start_time = time.time()
    L_buffer, M_buffer, Acc_buffer = anchor_attn_step_1(Q, K, V, 
                                                        1, 1,
                                                        block_size, 
                                                        step,  
                                                        B, H, N, D, sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N)
    true_coords, true_counts = anchor_attn_step_2(Q, K, M_buffer,
                                                    block_size, 
                                                    step, difference, 
                                                    B, H, N, D, sm_scale) 
    torch.cuda.synchronize()
    end_time = time.time()
    search_time = (end_time - start_time) * 1000


    grid_step3 = (triton.cdiv(N, BLOCK_SIZE_M), B * H) 
    
    torch.cuda.synchronize()
    start_time = time.time()
    _triton_stripe_sparse_attn_fwd_kernel_step3[grid_step3](
        Q, K, V, sm_scale,
        true_coords, true_counts,
        output, L_buffer, M_buffer, Acc_buffer,  # Pass Acc_buffer
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),  # Acc_buffer strides
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, N,
        STEP = step * block_size, 
        BLOCK_M = BLOCK_SIZE_M, # config
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    ) 
    torch.cuda.synchronize()
    end_time = time.time()
    compute_time = (end_time - start_time) * 1000
    if return_sparsity_ratio:
        sparsity_ratio = true_counts.sum() / (B * H * N * N / (block_size * step)  // 2)
        return output,sparsity_ratio, search_time, compute_time
    return output,1.0, search_time, compute_time

def _anchor_attn_without_anchor(Q, K, V, 
                        sm_scale: float= None,
                        block_size:int=64, 
                        step:int=16,
                        difference:float=100, 
                        BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128):
    B, H, N, D = Q.shape
    assert N % block_size == 0
    NUM_ROWS_STEP3 = triton.cdiv(N, block_size * step)
    output = torch.zeros_like(Q, device=Q.device)
    
      
    L_buffer, M_buffer, Acc_buffer = anchor_attn_step_1(Q, K, V, 
                                                        1, 1,
                                                        block_size, 
                                                        step,  
                                                        B, H, N, D, 
                                                        sm_scale, 
                                                        BLOCK_SIZE_M, 
                                                        BLOCK_SIZE_N)
    
    true_coords, true_counts = anchor_attn_step_2(Q, K, M_buffer,
                                                    block_size, 
                                                    step, difference, 
                                                    B, H, N, D, 
                                                    sm_scale)  
    
    
    grid_step3 = (triton.cdiv(N, BLOCK_SIZE_M), B * H) 
    _triton_stripe_sparse_attn_fwd_kernel_step3[grid_step3](
        Q, K, V, sm_scale,
        true_coords, true_counts,
        output, L_buffer, M_buffer, Acc_buffer,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),  # Acc_buffer strides
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, N,
        STEP = step * block_size, 
        BLOCK_M = BLOCK_SIZE_M, # config
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    ) 
    sparsity_ratio = true_counts.sum() / (B * H * N * N / (block_size * step)  // 2)
    return output,sparsity_ratio 

def _anchor_attn(Q, K, V,
                    sm_scale: float= None,
                    block_size:int=64, 
                    step:int=16,
                    difference:float=100,
                    return_col_idx_tuple: bool = False,
                    BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128):
    B, H, N, D = Q.shape

    output = torch.zeros_like(Q, device=Q.device)
    L_buffer, M_buffer, Acc_buffer = anchor_attn_step_1(Q, K, V,
                                                        1, 1,
                                                        block_size,
                                                        step,
                                                        B, H, N, D, sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N)

    true_coords, true_counts = anchor_attn_step_2(Q, K, M_buffer,
                                                    block_size, 
                                                    step, difference, 
                                                    B, H, N, D, sm_scale) 
    grid_step3 = (triton.cdiv(N, BLOCK_SIZE_M), B * H) 
    _triton_stripe_sparse_attn_fwd_kernel_step3[grid_step3](
        Q, K, V, sm_scale,
        true_coords, true_counts,
        output, L_buffer, M_buffer, Acc_buffer,  # Pass Acc_buffer
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),  # Acc_buffer strides
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, N,
        STEP = step * block_size, 
        BLOCK_M = BLOCK_SIZE_M, # config
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    ) 
    
    if return_col_idx_tuple:
        return output, (true_coords, true_counts)
    return output,None

def anchor_attn_without_anchor(Q, K, V,
                        sm_scale:float = None,
                        block_size: int = 128, 
                        step: int = 16,
                        difference:float = 12 
                           ):
    difference *= 1.44269504
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert Q.size(-2) % block_size == 0
    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    return _anchor_attn_without_anchor(Q, K, V,sm_scale,block_size,step,difference) 

def anchor_attn(Q, K, V,
                        sm_scale:float = None,
                        block_size: int = 128, 
                        step: int = 16,
                        difference:float = 12,
                        return_col_idx_tuple: bool = True
                           ):
    difference *= 1.44269504
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), \
        "Q, K, V must be contiguous. Please call .contiguous() before passing."

    assert step in [16, 32], \
        f"Invalid step size: {step}. Only step sizes 16 or 32 are supported."

    assert Q.size(-2) % (block_size * step) == 0, \
        f"Sequence length {Q.size(-2)} must be divisible by (block_size * step = {block_size * step})."

    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    sparse_output, col_idx_tuple = _anchor_attn(Q, K, V, 
                                            sm_scale, 
                                            block_size, 
                                            step,
                                            difference,
                                            return_col_idx_tuple,
                                        )
    if return_col_idx_tuple: 
        return sparse_output,col_idx_tuple
    return sparse_output


# the function is same as anchor_attn, but with return computational time, which is used the sync function 
def anchor_attn_test_time(Q, K, V,
                        sm_scale:float = None,
                        block_size: int = 128, 
                        step: int = 16,
                        difference:float = 12,
                        return_computational_ratio: bool = True
                           ):
    difference *= 1.44269504
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert Q.size(-2) % block_size == 0
    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    sparse_output,sparse_ratio,search_time,compute_time = _anchor_attn_test_time(Q, K, V, 
                                            sm_scale, 
                                            block_size, 
                                            step,
                                            difference,
                                            return_computational_ratio,
                                        )
    if return_computational_ratio: 
        return  sparse_output,sparse_ratio,search_time,compute_time
    return sparse_output,search_time,compute_time

def unrepeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, n_heads, slen, head_dim = hidden_states.shape
    return hidden_states.reshape(batch, n_heads//n_rep, n_rep, slen, head_dim)[:,:,0,:,:]

# NOTE: Track upstream branch updates. This code was fetched on 2025/8/7.
def Anchorattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    step,
    block_size_M,
    theta,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    is_causal: bool = True,
):
    bsz, num_heads, q_len, head_dim = query_states.shape

    block_size_N = 128
    pad = q_len % (step * block_size_M)
    attn_output = torch.zeros_like(query_states)
    if pad:
        attn_output[:,:,-pad:,:] = _flash_attention_forward(
            query_states[:,:,-pad:,:].transpose(1,2).contiguous(),
            unrepeat_kv(key_states,num_key_value_groups).transpose(1,2).contiguous(),
            unrepeat_kv(value_states,num_key_value_groups).transpose(1,2).contiguous(),
            None,
            pad,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=None,
            use_top_left_mask=False,
            is_causal=is_causal,
        ).transpose(1,2)
        q = query_states[:,:,:-pad,:].clone().contiguous()
        k = key_states[:,:,:-pad,:].clone().contiguous()
        v = value_states[:,:,:-pad,:].clone().contiguous()
        attn_output[:,:,:-pad,:], _ = anchor_attn(
            q, k, v,
            block_size=block_size_M,
            step=step,
            difference=theta,
            return_col_idx_tuple=True,
        )
    else:
        q = query_states[:,:,:,:].clone().contiguous()
        k = key_states[:,:,:,:].clone().contiguous()
        v = value_states[:,:,:,:].clone().contiguous()
        attn_output[:,:,:,:], _ = anchor_attn(
            q, k, v,
            block_size=block_size_M,
            step=step,
            difference=theta,
            return_col_idx_tuple=True,
        )
    return attn_output
