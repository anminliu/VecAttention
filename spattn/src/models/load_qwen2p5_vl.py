import os
import pickle

import flashinfer
import torch
import torch.distributed as dist
from transformers import AutoProcessor, StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    repeat_kv,
    apply_multimodal_rotary_pos_emb,
    Cache,
)
from typing import Optional, Tuple
from spattn.src.models.config_args import FastPrefillConfig
from spattn.src.models.utils import customize_prefill_attention

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")

@torch.no_grad()
def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if q_len > 1:
        attn_output = customize_prefill_attention(
            self, query_states, key_states, value_states,
            attention_mask=attention_mask, position_ids=position_ids[0],
            config=self.fastprefillconfig,
        )
    else:
        if key_states.device != query_states.device:
            key_states = key_states.to(query_states.device)
        if value_states.device != query_states.device:
            value_states = value_states.to(query_states.device)
        attn_output = flashinfer.single_decode_with_kv_cache(query_states.squeeze(0).squeeze(1).contiguous(), key_states.squeeze(0).transpose(0, 1).contiguous(), value_states.squeeze(0).transpose(0, 1).contiguous(),kv_layout="NHD").unsqueeze(0).unsqueeze(2)


    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


@torch.no_grad()
def _forward_to_save(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Qwen2.5-VL attention forward that saves Q/K from a target layer."""
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Save Q/K from the target layer
    if self.layer_idx == self._save_layer_idx:
        self._q_cache[:, :, self._q_valid:self._q_valid + query_states.shape[2], :] = query_states
        self._q_valid += query_states.shape[2]
        if self._q_valid >= self._save_target_len:
            os.makedirs(self._save_data_dir, exist_ok=True)
            with open(os.path.join(self._save_data_dir, f"query_{self._save_target_len}.pkl"), "wb") as f:
                pickle.dump(self._q_cache[:, :, :self._save_target_len, :], f)
        if key_states.shape[2] >= self._save_target_len:
            os.makedirs(self._save_data_dir, exist_ok=True)
            with open(os.path.join(self._save_data_dir, f"key_{self._save_target_len}.pkl"), "wb") as f:
                pickle.dump(key_states[:, :, :self._save_target_len, :], f)

    # Full attention (FlashAttention-2 via model default)
    from flash_attn import flash_attn_func
    q = query_states.transpose(1, 2)  # (bsz, seq, heads, dim)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)
    attn_output = flash_attn_func(q, k, v, causal=True)
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def load_fake_model(
    name_or_path: str = None,
    layer_to_save: int = 12,
    target_len: int = 32768,
    data_dir: str = "data",
):
    """Load Qwen2.5-VL with Q/K-saving attention for benchmark data generation.

    Patches all attention layers to run full attention, while the layer at
    index ``layer_to_save`` additionally accumulates Q and K tensors across
    chunked-prefill steps and writes them to disk once the target length is
    reached.

    Args:
        name_or_path: HuggingFace model ID or local path.
        layer_to_save: Layer index whose Q/K will be saved.
        target_len: Target sequence length in tokens.
        data_dir: Directory for saved Q/K pickle files.

    Returns:
        (model, tokenizer) tuple.
    """
    if name_or_path is None:
        name_or_path = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name_or_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    ).eval()

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    for layer in model.model.layers:
        attn = layer.self_attn
        attn.fastprefillconfig = FastPrefillConfig(metric="full")
        attn._save_layer_idx = layer_to_save
        attn._save_target_len = target_len
        attn._save_data_dir = data_dir
        attn._q_valid = 0
        attn._q_cache = torch.empty(
            (1, num_heads, target_len, head_dim),
            dtype=torch.bfloat16, device="cuda",
        )
        attn.forward = _forward_to_save.__get__(attn)

    processor = AutoProcessor.from_pretrained(name_or_path)
    return model, processor.tokenizer