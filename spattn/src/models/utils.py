import os
import threading
import torch
import numpy as np
from typing import Optional, Union
import transformers
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateNonBeamOutput, GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from torch import nn
from termcolor import colored
from spattn.src.models.config_args import FastPrefillConfig
import math
import json
import loguru
logger = loguru.logger

try:
    from spattn.src.Xattention import Xattention_prefill
except:
    print("Xattention Import Fail")
try:
    from spattn.src.Fullprefill import Full_prefill
except:
    print("Full Prefill Import Fail")
try:
    from spattn.src.Flexprefill import Flexprefill_prefill
except:
    print("Flex Prefill Import Fail")
try:
    from spattn.src.VecAttention import VecAttention_prefill
except:
    print("VecAttention Prefill Import Fail")
try:
    from spattn.src.Anchorattention import Anchorattention_prefill
except:
    print("Anchor Attention Prefill Import Fail")

SPARSITY_RECORD = threading.local()
MODEL_METRICS_RECORD = threading.local()

def customize_prefill_attention(
    self, query_states, key_states, value_states,
    attention_mask=None, position_ids=None, causal=True,
    config = FastPrefillConfig(),
):
    """ A wrapper for different efficient attention methods during prefill/inference
    Args:
        self: the attention layer
        query_states: (batch_size, num_heads, seq_len,  head_dim)
        key_states: (batch_size, num_heads, seq_len,  head_dim)
        value_states: (batch_size, num_heads, seq_len,  head_dim)
        attention_mask: (1, 1, seq_len, seq_len) or None
        position_ids: (1, seq_len) for `anchorattention`
        config: configuration for different attention methods
    Returns:
        attn_output: (batch_size, num_heads, seq_len, head_dim)
    Global Variables:
        SPARSITY_RECORD: a threading.local() object to record latency information in attention layer
    """
    assert causal or config.metric in ["full", "xattn", "vecattention"], (
        f"Non-causal attention only supports full, xattn, or vecattention metrics. Got: {config.metric}."
    )
    
    start_evt = None
    end_evt = None
    output_time = config.output_time
    if output_time and query_states.is_cuda and torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        cur_stream = torch.cuda.current_stream(device=query_states.device)
        start_evt.record(stream=cur_stream)

    if config.metric == "flex":
        attn_output = Flexprefill_prefill(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            gamma=config.gamma, tau=config.tau,
        )
        attn_output = attn_output.transpose(1, 2)
    elif config.metric == "vecattention":
        threshold = config.threshold[self.layer_idx] if isinstance(config.threshold, torch.Tensor) else config.threshold
        vision_start = config.vision_start_position
        vision_end = config.vision_end_position

        if vision_start is None and vision_end is None:
            # No vision slicing: VecAttention over the full sequence.
            attn_output = VecAttention_prefill(
                query_states, key_states, value_states,
                threshold=threshold, q_pooling_size=config.block_size_q,
                k_local_size=config.block_size_k, group_k_block=config.group_k_block,
                causal=causal, chunk_size=config.chunk_size,
            )
        else:
            assert causal, "Only causal attention is supported when vision positions are set"
            # Full attention on pre-vision text tokens (if vision_start is set).
            if vision_start is not None:
                attn_output_before = Full_prefill(
                    query_states[:, :, :vision_start, :],
                    key_states[:, :, :vision_start, :],
                    value_states[:, :, :vision_start, :],
                    causal=causal,
                )
            # VecAttention on [0 : vision_end].
            # If vision_end is None, [:None] selects the full sequence.
            attn_output_va = VecAttention_prefill(
                query_states[:, :, :vision_end, :],
                key_states[:, :, :vision_end, :],
                value_states[:, :, :vision_end, :],
                threshold=threshold, q_pooling_size=config.block_size_q,
                k_local_size=config.block_size_k, group_k_block=config.group_k_block,
                causal=causal, chunk_size=config.chunk_size,
            )
            # Overwrite the pre-vision slice with full-attention output.
            if vision_start is not None:
                attn_output_va[:, :, :vision_start, :] = attn_output_before
            # Full attention on post-vision text tokens (if vision_end is set).
            if vision_end is not None:
                attn_output_after = Full_prefill(
                    query_states[:, :, vision_end:, :], key_states, value_states,
                    causal=causal,
                )
                attn_output = torch.cat([attn_output_va, attn_output_after], dim=2)
            else:
                attn_output = attn_output_va

    elif config.metric == "xattn":
        if isinstance(config.threshold, torch.Tensor):
            attn_output = Xattention_prefill(
                query_states, key_states, value_states, config.stride, norm=1,
                threshold=config.threshold[self.layer_idx], use_triton=True, causal=causal,
            )
        else:
            attn_output = Xattention_prefill(
                query_states, key_states, value_states, config.stride, norm=1,
                threshold=config.threshold, use_triton=True, causal=causal,
            )
    elif config.metric == "full":
        attn_output = Full_prefill(
            query_states, key_states, value_states, causal=causal,
        )
    elif config.metric == "anchor":
        attn_output = Anchorattention_prefill(
            query_states, key_states, value_states, self.num_key_value_groups,
            step=config.step, block_size_M=config.block_size_q,
            theta=config.theta, is_causal=causal,
            attention_mask=attention_mask, position_ids=position_ids,
        )
    else:
        raise NotImplementedError(f"Unknown attention metric: {config.metric}")

    if start_evt is not None and end_evt is not None:
        end_evt.record(stream=torch.cuda.current_stream(device=query_states.device))
        if not hasattr(SPARSITY_RECORD, "attention_latency_list"):
            SPARSITY_RECORD.attention_latency_list = []
        SPARSITY_RECORD.attention_latency_list.append((start_evt, end_evt))

    if config.dump_qk_dir:
        dir_name = os.path.join(config.dump_qk_dir, config.input_id)
        os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(os.path.join(dir_name, f"layer_{self.layer_idx}_qk.pt")):
            torch.save({
                "query_states": query_states,
                "key_states": key_states,
            }, os.path.join(dir_name, f"layer_{self.layer_idx}_qk.pt"))
            print(colored(f"Dumped QK for layer {self.layer_idx} to {dir_name}", "magenta"))

    return attn_output

def pop_layer_metrics_from_threadlocal():
    """Pop sparsity metrics recorded in SPARSITY_RECORD.

    Returns:
        metrics: dict, e.g.
            {
              "sparsity": { "by_layer": [...], "mean": ... },
              "attention_latency": { "by_layer": [...], "mean": ... },
              ...
            }
    """

    def _to_py(obj):
        # Event pair -> compute GPU elapsed time (milliseconds)
        if isinstance(obj, tuple) and len(obj) == 2:
            a, b = obj
            if isinstance(a, torch.cuda.Event) and isinstance(b, torch.cuda.Event):
                b.synchronize()
                return float(a.elapsed_time(b))  # ms

        # Convert tensor/ndarray/scalar/sequence to Python primitives (float or list[float] or nested list)
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu()
            if obj.ndim == 0:
                return float(obj.item())
            return [float(x) for x in obj.flatten().tolist()]
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return float(obj.item())
            return [float(x) for x in obj.flatten().tolist()]
        if isinstance(obj, (list, tuple)):
            return [_to_py(x) for x in obj]
        if isinstance(obj, (int, float)):
            return float(obj)
        # Allow dicts to pass through; other types are unsupported
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        raise TypeError(f"Unsupported metric element type: {type(obj)}")

    def _to_float_or_nan(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        # 0-d arrays/tensors are already handled in _to_py; reject non-numeric values here
        raise TypeError(f"Non-numeric value in metrics: {x} (type={type(x)})")

    def _mean_numbers(xs):
        arr = np.asarray([_to_float_or_nan(v) for v in xs], dtype=np.float64)
        with np.errstate(all='ignore'):
            m = np.nanmean(arr)
        return float(m) if arr.size > 0 else None

    def _mean_sequences(seq_list):
        # seq_list: List[List[number]], may differ in length; element-wise nanmean
        if not seq_list:
            return []
        # Validate that elements are flat lists of numbers
        normalized = []
        max_len = 0
        for s in seq_list:
            if isinstance(s, (list, tuple)):
                row = [_to_float_or_nan(v) for v in s]
            else:
                # Wrap scalar row as single-element list
                row = [_to_float_or_nan(s)]
            normalized.append(row)
            max_len = max(max_len, len(row))
        if max_len == 0:
            return []
        mat = np.full((len(normalized), max_len), np.nan, dtype=np.float64)
        for i, row in enumerate(normalized):
            mat[i, :len(row)] = row
        with np.errstate(all='ignore'):
            m = np.nanmean(mat, axis=0)
        return [float(x) for x in m.tolist()]

    def _mean_dicts(dict_list):
        # Aggregate per key; treat None as missing, compute nanmean; raise on non-numeric
        if not dict_list:
            return {}
        # Validate all entries are dicts
        if not all(isinstance(d, dict) for d in dict_list):
            raise TypeError("Mixed types in dict metrics.")
        keys = set().union(*(d.keys() for d in dict_list))
        out = {}
        for k in keys:
            vals = []
            for d in dict_list:
                v = d.get(k, None)
                vals.append(_to_py(v))
            with np.errstate(all='ignore'):
                out[k] = float(np.nanmean(np.asarray(vals, dtype=np.float64)))
        return out

    # Iterate over all *_list attributes on the threadlocal
    if not hasattr(SPARSITY_RECORD, "__dict__"):
        raise AttributeError("SPARSITY_RECORD has no __dict__; ensure it is a threadlocal-like object.")
    metrics = {}
    to_clear = []
    for name, value in SPARSITY_RECORD.__dict__.items():
        if not name.endswith("_list"):
            continue
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Attribute {name} is expected to be a list/tuple, got {type(value)}.")
        base = name[:-5]  # Strip the "_list" suffix
        raw_list = list(value)
        by_layer = [_to_py(v) for v in raw_list]

        # Compute mean (dispatch by element structure)
        mean_val = None
        if len(by_layer) == 0:
            mean_val = None
        else:
            first = by_layer[0]
            if isinstance(first, dict):
                mean_val = _mean_dicts(by_layer)
            elif isinstance(first, (list, tuple)):
                mean_val = _mean_sequences(by_layer)
            elif isinstance(first, (int, float)):
                mean_val = _mean_numbers(by_layer)
            else:
                raise TypeError(f"Unsupported per-layer data type for metric '{base}': {type(first)}")

        metrics[base] = {
            "by_layer": by_layer,
            "mean": mean_val,
        }
        to_clear.append(name)

    # Clear the popped lists
    for name in to_clear:
        setattr(SPARSITY_RECORD, name, [])

    return metrics

def pop_ttft_from_threadlocal():
    """Pop model metrics from MODEL_METRICS_RECORD and return TTFT."""
    if not hasattr(MODEL_METRICS_RECORD, "prefill_start_evt") or not hasattr(MODEL_METRICS_RECORD, "prefill_end_evt"):
        return None
    
    start_evt = MODEL_METRICS_RECORD.prefill_start_evt
    end_evt = MODEL_METRICS_RECORD.prefill_end_evt
    
    if start_evt is None or end_evt is None:
        return None

    try:
        end_evt.synchronize()
        ttft = start_evt.elapsed_time(end_evt)
    except RuntimeError:
        # end_evt was never recorded (e.g., exception during prefill)
        return None
    
    # Clear events
    MODEL_METRICS_RECORD.prefill_start_evt = None
    MODEL_METRICS_RECORD.prefill_end_evt = None
    
    return ttft

def _sample_with_timing(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Modified version of _sample with CUDA event timing for prefill stage.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        if getattr(self, "hf_quantizer", None) is not None:
            is_compileable &= self.hf_quantizer.is_compileable
        is_compileable = is_compileable and not generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            # Start timing for prefill
            if torch.cuda.is_available() and input_ids.is_cuda:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                cur_stream = torch.cuda.current_stream(device=input_ids.device)
                start_evt.record(stream=cur_stream)
                
                # Store events in threadlocal
                MODEL_METRICS_RECORD.prefill_start_evt = start_evt
                MODEL_METRICS_RECORD.prefill_end_evt = end_evt
            
            outputs = self(**model_inputs, return_dict=True)
            
            # End timing for prefill
            if torch.cuda.is_available() and input_ids.is_cuda and hasattr(MODEL_METRICS_RECORD, "prefill_end_evt"):
                end_evt = MODEL_METRICS_RECORD.prefill_end_evt
                if end_evt is not None:
                    end_evt.record(stream=torch.cuda.current_stream(device=input_ids.device))
            
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

def inject_generation_timing():
    """Inject timing functionality into GenerationMixin._sample."""
    # Check transformers version
    expected_commit = "cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d"
    expected_version = "4.52.0.dev0"
    
    current_version = transformers.__version__
    if current_version != expected_version:
        warning_msg = f"Warning: Transformers version mismatch!\nExpected: {expected_version}\nCurrent: {current_version}\nFunction injection may not work correctly."
        # print(colored(warning_msg, 'red'))
        raise RuntimeError(warning_msg)
        # return False
    
    # Inject the modified function
    try:
        original_sample = GenerationMixin._sample
        GenerationMixin._sample = _sample_with_timing
        success_msg = f"Successfully injected timing functionality into GenerationMixin._sample"
        print(colored(success_msg, 'green'))
        return True
    except Exception as e:
        error_msg = f"Failed to inject timing functionality: {str(e)}"
        print(colored(error_msg, 'red'))
        return False

class SpAttnProfiler:
    """ A profiler for attention layers to record sparsity and latency metrics.
    
    Usage:
        1. Call `customize_prefill_attention` in the attention layer's forward method.
        2. After inference, call `pop_layer_metrics_from_threadlocal()` to retrieve metrics.
    """
    
    def __init__(self, *, device=None, pop_layer_metrics=True, pop_model_metrics=True,
                 output_attn_time=True,
                 config: "FastPrefillConfig" = None):
        """Initialize the profiler.

        Args:
            device: CUDA device to use for timing events.
            pop_layer_metrics: If True, pop per-layer metrics (latency) on exit.
            pop_model_metrics: If True, pop TTFT on exit.
            output_attn_time: Whether to record per-layer attention latency during prefill.
            config: Optional FastPrefillConfig whose output_time field will be updated to match.
                The config object is mutated in-place.
        """
        self.device = device
        self.pop_layer_metrics = pop_layer_metrics
        self.pop_model_metrics = pop_model_metrics
        self.output_attn_time = output_attn_time
        self.start_evt = None
        self.end_evt = None
        self.latency = None
        self.ttft = None
        self.layer_metrics = None
        self._stream = None
        if config is not None:
            config.output_time = output_attn_time
    
    def __enter__(self):
        """Enter the profiler context."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for AttentionLatencyCtx but not available.")
        self._stream = torch.cuda.current_stream(device=self.device) if self.device is not None \
            else torch.cuda.current_stream()
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)
        self.start_evt.record(stream=self._stream)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the profiler context."""
        if self.start_evt is not None and self.end_evt is not None:
            self.end_evt.record(stream=self._stream)
            self.end_evt.synchronize()
            self.latency = self.start_evt.elapsed_time(self.end_evt)
        
        if self.pop_layer_metrics:
            self.layer_metrics = pop_layer_metrics_from_threadlocal()
        
        if self.pop_model_metrics:
            self.ttft = pop_ttft_from_threadlocal()