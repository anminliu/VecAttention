from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer,StaticCache
from transformers.models.llama.modeling_llama import Cache,LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    logging,
)

# from spattn.threshold.llama_threshold import llama_fuse_16,llama_fuse_8,llama_fuse_4
import flashinfer
import time

from spattn.src.models.config_args import FastPrefillConfig
from spattn.src.models.utils import customize_prefill_attention
import os

logger = logging.get_logger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def forward_eval(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
            Performs a forward pass of the attention layer with optimized prefill mechanisms.

            This function integrates various optimizations, including fused attention, 
            efficient caching, and rotary embeddings, to enhance inference speed.

            Parameters:
            - hidden_states (torch.Tensor): The input hidden states of shape (batch_size, seq_len, hidden_dim).
            - attention_mask (torch.Tensor, optional): The attention mask tensor, which defines 
            which positions should be attended to.
            - position_ids (torch.LongTensor, optional): The position indices of tokens in the sequence.
            - past_key_value (Cache, optional): Cached key-value tensors for faster decoding.
            - output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            - use_cache (bool, optional): Whether to use caching for faster inference. Defaults to False.
            - cache_position (torch.LongTensor, optional): The position index used in caching.
            - position_embeddings (Tuple[torch.Tensor, torch.Tensor], optional): Precomputed RoPE 
            embeddings (cos and sin components).
            - **kwargs: Additional arguments for flexibility.

            Returns:
            - Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: 
            The attention output tensor, attention weights (if enabled), and updated past key-value states.
        """

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            reshape_time = time.time() - start_time
            print(f"     Reshape took: {reshape_time:.6f} seconds")
        
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if isinstance(past_key_value, StaticCache):
            key_states = key_states[:,:,:min(cache_position[-1] + 1, key_states.shape[2]),:]
            value_states = value_states[:,:,:min(cache_position[-1]+1,value_states.shape[2]),:]
            
        _, _, k_len, _ = key_states.shape
        _, _, q_len, _ = query_states.shape
        decoding = (q_len != k_len and q_len == 1)
        if not decoding:
            key_states = repeat_kv(key_states, self.num_key_value_groups).to("cuda")
            value_states = repeat_kv(value_states, self.num_key_value_groups).to("cuda")
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            past_kv_time = time.time() - start_time
            print(f"     Past KV update and repeat took: {past_kv_time:.6f} seconds")

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
            print(f"q length: {q_len} k length: {k_len}")

        if os.getenv("DEBUG", "False").lower() == "true":
            from .config_args import build_output_file_from_config
            save_path = f"debug_samples/ruler_multikey3_16K_id0/{self.fastprefillconfig.metric}/qk_layer{self.layer_idx}_llama.pt"
            save_path = build_output_file_from_config(save_path, self.fastprefillconfig)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save((query_states,key_states), save_path)
            if self.layer_idx == 31:
                print(f"Save qk to {save_path}")
                breakpoint()
        
        if not decoding:
            attn_output, attn_weights = customize_prefill_attention(
                self, query_states, key_states, value_states,
                attention_mask=attention_mask, position_ids=position_ids,
                output_attentions=output_attentions, 
                config=self.fastprefillconfig,
            )
        else:
            if key_states.device != query_states.device:
                key_states = key_states.to(query_states.device)
            if value_states.device != query_states.device:
                value_states = value_states.to(query_states.device)

            value_states = value_states.squeeze(0).contiguous()
            query_states = query_states.squeeze(0).squeeze(1)
            key_states = key_states.squeeze(0).contiguous()
            attn_output = flashinfer.single_decode_with_kv_cache(
                query_states, 
                key_states, 
                value_states,
                kv_layout="HND"
            )
            attn_output = attn_output.unsqueeze(0).unsqueeze(2)
        
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            attn_time = time.time() - start_time
            print(f"     Attention computation took: {attn_time:.6f} seconds")

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        del query_states
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            post_attn_time = time.time() - start_time
            print(f"     Post-attention processing took: {post_attn_time:.6f} seconds")

        return attn_output, None#, past_key_value

def load_model(fastprefillconfig=FastPrefillConfig(),name_or_path=""):
    """
        Loads a LLaMA model with FastPrefill optimizations applied.

        This function initializes the model, applies the FastPrefill configuration to attention 
        layers, and loads the tokenizer.

        Parameters:
        - fastprefillconfig (FastPrefillConfig, optional): The configuration for FastPrefill optimizations.
        - name_or_path (str): The model path or identifier for loading the pre-trained model.
        - test_configs (dict, optional): Additional configurations for testing.

        Returns:
        - Tuple[LlamaForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """

    model = LlamaForCausalLM.from_pretrained(
        name_or_path,
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    model.eval()
    for layer in model.model.layers:
        layer.self_attn.fastprefillconfig = fastprefillconfig
        layer.self_attn.forward = forward_eval.__get__(layer.self_attn)
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path
    )
    return model, tokenizer

def forward_to_save(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
            Performs a forward pass of the attention layer with optimized prefill mechanisms.

            This function integrates various optimizations, including fused attention, 
            efficient caching, and rotary embeddings, to enhance inference speed.

            Parameters:
            - hidden_states (torch.Tensor): The input hidden states of shape (batch_size, seq_len, hidden_dim).
            - attention_mask (torch.Tensor, optional): The attention mask tensor, which defines 
            which positions should be attended to.
            - position_ids (torch.LongTensor, optional): The position indices of tokens in the sequence.
            - past_key_value (Cache, optional): Cached key-value tensors for faster decoding.
            - output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            - use_cache (bool, optional): Whether to use caching for faster inference. Defaults to False.
            - cache_position (torch.LongTensor, optional): The position index used in caching.
            - position_embeddings (Tuple[torch.Tensor, torch.Tensor], optional): Precomputed RoPE 
            embeddings (cos and sin components).
            - **kwargs: Additional arguments for flexibility.

            Returns:
            - Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: 
            The attention output tensor, attention weights (if enabled), and updated past key-value states.
        """

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            reshape_time = time.time() - start_time
            print(f"     Reshape took: {reshape_time:.6f} seconds")
        
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if isinstance(past_key_value, StaticCache):
            key_states = key_states[:,:,:min(cache_position[-1] + 1, key_states.shape[2]),:]
            value_states = value_states[:,:,:min(cache_position[-1] + 1,value_states.shape[2]),:]
            
        _, _, k_len, _ = key_states.shape
        _, _, q_len, _ = query_states.shape
        decoding = (q_len != k_len and q_len == 1)
        if not decoding:
            key_states = repeat_kv(key_states, self.num_key_value_groups).to("cuda")
            value_states = repeat_kv(value_states, self.num_key_value_groups).to("cuda")
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            past_kv_time = time.time() - start_time
            print(f"     Past KV update and repeat took: {past_kv_time:.6f} seconds")

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
            print(f"q length: {q_len} k length: {k_len}")

        if not decoding:
            attn_output, attn_weights = customize_prefill_attention(
                self, query_states, key_states, value_states,
                attention_mask=attention_mask, position_ids=position_ids,
                output_attentions=output_attentions, 
                config=self.fastprefillconfig,
            )
        else:
            if key_states.device != query_states.device:
                key_states = key_states.to(query_states.device)
            if value_states.device != query_states.device:
                value_states = value_states.to(query_states.device)

            value_states = value_states.squeeze(0).contiguous()
            query_states = query_states.squeeze(0).squeeze(1)
            key_states = key_states.squeeze(0).contiguous()
            attn_output = flashinfer.single_decode_with_kv_cache(
                query_states, 
                key_states, 
                value_states,
                kv_layout="HND"
            )
            attn_output = attn_output.unsqueeze(0).unsqueeze(2)
        if self.layer_idx == self.layer_to_save:
            import pickle
            import os
            query_path = f"data/query_{self.target_len}.pkl"
            key_path = f"data/key_{self.target_len}.pkl"
            self.q_cache[:,:,self.q_valid:self.q_valid+query_states.shape[-2],:] = query_states
            self.q_valid += query_states.shape[-2]
            os.makedirs(os.path.dirname(query_path), exist_ok=True)
            if self.q_valid == self.target_len:
                with open(query_path, "wb") as f:
                    pickle.dump(self.q_cache, f)
            if key_states.shape[-2] == self.target_len:
                with open(key_path, "wb") as f:
                    pickle.dump(key_states, f)
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            attn_time = time.time() - start_time
            print(f"     Attention computation took: {attn_time:.6f} seconds")

        if self.fastprefillconfig.print_detail:
            start_time = time.time()
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        del query_states
        if self.fastprefillconfig.print_detail:
            torch.cuda.synchronize()
            post_attn_time = time.time() - start_time
            print(f"     Post-attention processing took: {post_attn_time:.6f} seconds")

        return attn_output, None#, past_key_value

def load_fake_model(layer_to_save,target_len,name_or_path=""):
    model = LlamaForCausalLM.from_pretrained(
        name_or_path,
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    for layer in model.model.layers:
        layer.self_attn.fastprefillconfig = FastPrefillConfig(metric="full")
        layer.self_attn.layer_to_save = layer_to_save
        layer.self_attn.target_len = target_len
        layer.self_attn.forward = forward_to_save.__get__(layer.self_attn)
        if layer.self_attn.layer_idx == layer_to_save:
            layer.self_attn.q_valid = 0
            layer.self_attn.q_cache = torch.empty((1, model.config.num_attention_heads, target_len, model.config.head_dim), dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path
    )
    return model, tokenizer