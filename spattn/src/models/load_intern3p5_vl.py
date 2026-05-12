    
import flashinfer
import torch
from transformers.models.qwen3.modeling_qwen3 import repeat_kv,apply_rotary_pos_emb,Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from typing import Optional, Tuple
from spattn.src.models.utils import customize_prefill_attention

# Based on modeling_qwen3.py::Qwen3Attention::forward
def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    bsz, q_len, _ = hidden_states.size()
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    output_attentions = kwargs.get("output_attentions", False)
    if key_states.shape == query_states.shape:
        attn_output = customize_prefill_attention(
            self, query_states, key_states, value_states,
            attention_mask=attention_mask, position_ids=kwargs.get("position_ids", None),
            config=self.fastprefillconfig,
        )
    else:
        if key_states.device != query_states.device:
            key_states = key_states.to(query_states.device)
        if value_states.device != query_states.device:
            value_states = value_states.to(query_states.device)
        attn_output = flashinfer.single_decode_with_kv_cache(query_states.squeeze(0).squeeze(1), key_states.squeeze(0).transpose(0, 1).contiguous(), value_states.squeeze(0).transpose(0, 1).contiguous(),kv_layout="NHD").unsqueeze(0).unsqueeze(2)

    if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None