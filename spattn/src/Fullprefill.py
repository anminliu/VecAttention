import torch
import flashinfer

def Full_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
):
    attn_output = flashinfer.single_prefill_with_kv_cache(
        query_states.transpose(1, 2).squeeze(0),
        key_states.transpose(1, 2).squeeze(0),
        value_states.transpose(1, 2).squeeze(0),
        causal=causal,
    ).unsqueeze(0).transpose(1, 2)
    return attn_output
