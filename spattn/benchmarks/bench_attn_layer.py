import os
import pickle
import torch
import triton
import math
from typing import Dict, Any, Optional
from tqdm import tqdm
from transformers import StaticCache

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.0 8.9 9.0"

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")

# Import dependencies
try:
    from spattn.benchmarks.generate_prompt import generate_prompt
except ImportError:
    print("Warning: generate_prompt not found, please implement it")
    
try:
    from spattn.src.models.load_qwen2p5_vl import load_fake_model
except ImportError:
    print("Warning: load_fake_model not found, please implement it")

from spattn.src.models.utils import customize_prefill_attention
from spattn.src.models.config_args import FastPrefillConfig

class BenchmarkConfig:
    """Benchmark configuration manager."""
    
    # Sequence length configuration
    SEQ_LENGTHS = [8*1024, 16*1024, 32*1024, 64*1024, 128*1024]
    
    # Method config dict: method name -> (display name, color, line style)
    METHODS = {
        'full': ('Full Attention', 'red', '--'),
        'flex': ('FlexAttention', 'blue', '-'),
        'anchor': ('AnchorAttention', 'orange', '-'),
        'xattn16': ('X-Attention', 'purple', '-'),
        'vecattention': ('VecAttention', 'pink', '-'),
        # New methods can be added here
        # 'new_method': ('New Method Name', 'pink', ':'),
    }
    
    # Currently enabled methods (set in main)
    ENABLED_METHODS = []
    
    # Optimal thresholds per method (set in main)
    OPTIMAL_THRESHOLDS = {}
    
    @classmethod
    def set_benchmark_config(cls, methods: list, seq_lengths: list = None, thresholds: dict = None):
        """Set up the complete benchmark configuration at once.

        Args:
            methods: List of methods to compare.
            seq_lengths: List of sequence lengths (optional).
            thresholds: Dict of method thresholds (optional).
        """
        # Validate that all methods exist
        for method in methods:
            if method not in cls.METHODS:
                raise ValueError(f"Unknown method: {method}. Available methods: {list(cls.METHODS.keys())}")
        
        cls.ENABLED_METHODS = methods.copy()
        
        if seq_lengths is not None:
            cls.SEQ_LENGTHS = seq_lengths.copy()
            
        if thresholds is not None:
            cls.OPTIMAL_THRESHOLDS = thresholds.copy()
        else:
            # Use defaults if no thresholds provided
            cls.OPTIMAL_THRESHOLDS = {}
    
    @classmethod
    def set_threshold(cls, method: str, threshold: float):
        """Set threshold for a single method."""
        cls.OPTIMAL_THRESHOLDS[method] = threshold
    
    @classmethod
    def set_thresholds(cls, thresholds: dict):
        """Set thresholds for multiple methods at once."""
        cls.OPTIMAL_THRESHOLDS.update(thresholds)
    
    @classmethod
    def get_enabled_methods_info(cls):
        """Get info for enabled methods."""
        if not cls.ENABLED_METHODS:
            raise ValueError("No methods enabled. Please call set_benchmark_config() first.")
            
        line_vals = cls.ENABLED_METHODS
        line_names = [cls.METHODS[method][0] for method in cls.ENABLED_METHODS]
        styles = [(cls.METHODS[method][1], cls.METHODS[method][2]) for method in cls.ENABLED_METHODS]
        
        return line_vals, line_names, styles
    
    @classmethod
    def get_threshold(cls, method: str):
        """Get optimal threshold for a method."""
        return cls.OPTIMAL_THRESHOLDS.get(method)
    
    @classmethod
    def add_method_definition(cls, method_key: str, display_name: str, color: str, line_style: str):
        """Add a new method definition (not automatically enabled)."""
        cls.METHODS[method_key] = (display_name, color, line_style)
    
    @classmethod
    def get_current_config_summary(cls):
        """Get current configuration summary."""
        return {
            "enabled_methods": cls.ENABLED_METHODS,
            "seq_lengths": cls.SEQ_LENGTHS,
            "thresholds": cls.OPTIMAL_THRESHOLDS
        }

class AttentionMethodConfig:
    """Manages optimal parameter configs for different attention methods."""
    
    @staticmethod
    def get_config(method: str, threshold: float = None) -> FastPrefillConfig:
        """Get configuration for the specified method.

        Args:
            method: Method name.
            threshold: Threshold parameter; uses default if None.

        Returns:
            FastPrefillConfig instance.
        """
        configs = {
            "full": {
                "metric": "full"
            },
            "flex": {
                "metric": "flex",
                "gamma": 0.95 if threshold is None else threshold,
                "tau": 0.1
            },
            "anchor": {
                "metric": "anchor",
                "block_size_q": 128,
                "step": 16,
                "theta": 12 if threshold is None else threshold
            },
            "xattn16": {
                "metric": "xattn",
                "stride": 16,
                "threshold": 0.9 if threshold is None else threshold
            },
            "vecattention": {
                "metric": "vecattention",
                "block_size_q": 64,
                "block_size_k": 16,
                "group_k_block": 16,
                "chunk_size": 1024 * 32,
                "threshold": 0.8 if threshold is None else threshold
            },
        }
        
        if method not in configs:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(configs.keys())}")
        
        config_dict = configs[method].copy()
        if threshold is not None and "threshold" in config_dict:
            config_dict["threshold"] = threshold
            
        return FastPrefillConfig(output_time=False, **config_dict)

class AttentionDataGenerator:
    """Generates and manages attention layer test data."""
    
    def __init__(self, model_path: str, layer_to_save: int = 12, data_dir: str = "data"):
        self.model_path = model_path
        self.layer_to_save = layer_to_save
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def generate_qkv_data(self, seq_len: int, force_regenerate: bool = False) -> tuple:
        """Generate or load Q, K, V data.

        Args:
            seq_len: Sequence length.
            force_regenerate: Whether to force data regeneration.

        Returns:
            (q, k, v) tensor tuple.
        """
        query_path = os.path.join(self.data_dir, f"query_{seq_len}.pkl")
        key_path = os.path.join(self.data_dir, f"key_{seq_len}.pkl")
        
        if not force_regenerate and os.path.exists(query_path) and os.path.exists(key_path):
            # Load existing data
            with open(query_path, "rb") as f:
                q = pickle.load(f)
            with open(key_path, "rb") as f:
                k = pickle.load(f)
        else:
            # Generate new data
            print(f"Generating Q/K data for sequence length {seq_len}...")
            q, k = self._generate_qk_from_model(seq_len)
            
            # Save data
            with open(query_path, "wb") as f:
                pickle.dump(q, f)
            with open(key_path, "wb") as f:
                pickle.dump(k, f)
        
        # Validate data shape
        assert q.shape[-2] == seq_len, f"Q sequence length mismatch: {q.shape[-2]} vs {seq_len}"
        assert k.shape[-2] == seq_len, f"K sequence length mismatch: {k.shape[-2]} vs {seq_len}"
        
        # Generate V data (random)
        torch.manual_seed(0)
        v = torch.randn(q.shape, dtype=torch.bfloat16).to("cuda").contiguous()
        
        return q, k, v
    
    def _generate_qk_from_model(self, target_len: int) -> tuple:
        """Generate Q, K data from model inference."""
        model, tokenizer = load_fake_model(
            name_or_path=self.model_path, 
            layer_to_save=self.layer_to_save, 
            target_len=target_len
        )
        # TODO: For 1M-length data, implement chunked prefill in forward and disable StaticCache (use_cache=False)
        input_ids = generate_prompt(tokenizer, target_len)
        chunk_size = 4096
        max_len = target_len + 1024
        
        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_len,
            device=model.device,
            dtype=model.dtype
        )
        
        with torch.no_grad():
            for i in tqdm(range(0, input_ids.size(1), chunk_size), desc="Prefilling", unit="chunk"):
                chunk = input_ids[:, i: i + chunk_size]
                output = model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
        
        # Extract Q, K from saved data
        query_path = os.path.join(self.data_dir, f"query_{target_len}.pkl")
        key_path = os.path.join(self.data_dir, f"key_{target_len}.pkl")
        
        with open(query_path, "rb") as f:
            q = pickle.load(f)
        with open(key_path, "rb") as f:
            k = pickle.load(f)
            
        return q, k

class AttentionLayerBenchmark:
    """Attention layer benchmark class."""
    
    def __init__(self, data_generator: AttentionDataGenerator):
        self.data_generator = data_generator
        
    def _create_mock_attention_layer(self, num_heads: int, layer_idx: int = 0):
        """Create a mock attention layer object for calling customize_prefill_attention."""
        class MockAttentionLayer:
            def __init__(self, num_heads, layer_idx):
                self.num_heads = num_heads
                self.layer_idx = layer_idx
                self.num_key_value_groups = 1  # Assuming no GQA
                
        return MockAttentionLayer(num_heads, layer_idx)
    
    def benchmark_single_method(
        self, 
        seq_len: int, 
        method: str, 
        threshold: float = None,
        quantiles: list = [0.5, 0.2, 0.8],
        warmup: int = 10,
        rep: int = 20
    ) -> dict:
        """Benchmark a single method.

        Args:
            seq_len: Sequence length.
            method: Method name.
            threshold: Threshold parameter.
            quantiles: Quantile list.
            warmup: Number of warmup iterations.
            rep: Number of repetitions.

        Returns:
            Dict containing latency information.
        """
        # Generate test data
        q, k, v = self.data_generator.generate_qkv_data(seq_len)
        batch_size, num_heads, _, head_dim = q.shape
        
        # Get method configuration
        config = AttentionMethodConfig.get_config(method, threshold)
        
        # Create mock attention layer
        mock_layer = self._create_mock_attention_layer(num_heads, layer_idx=0)
        position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)
        
        # Define test function
        def test_fn():
            return customize_prefill_attention(
                mock_layer, q, k, v,
                attention_mask=None,
                position_ids=position_ids,
                config=config
            )
        
        # Run benchmark
        ms, min_ms, max_ms = triton.testing.do_bench(
            test_fn, 
            quantiles=quantiles,
            warmup=warmup,
            rep=rep
        )
        
        return {
            "median_ms": ms,
            "min_ms": min_ms, 
            "max_ms": max_ms,
            "method": method,
            "seq_len": seq_len,
            "config": config.__dict__
        }

def get_benchmark_decorator():
    """Dynamically generate benchmark decorator."""
    line_vals, line_names, styles = BenchmarkConfig.get_enabled_methods_info()
    
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_len'],
            x_vals=BenchmarkConfig.SEQ_LENGTHS,
            line_arg='provider',
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel='Latency (ms)',
            plot_name='attention-layers-latency-comparison',
            args={},
        )
    )

def benchmark_attention_layers(seq_len, provider):
    """Performance comparison benchmark for different attention layer methods."""
    # Global data generator instance
    if not hasattr(benchmark_attention_layers, 'data_generator'):
        model_path = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct")
        benchmark_attention_layers.data_generator = AttentionDataGenerator(model_path)
    
    if not hasattr(benchmark_attention_layers, 'benchmark'):
        benchmark_attention_layers.benchmark = AttentionLayerBenchmark(
            benchmark_attention_layers.data_generator
        )
    
    # Get threshold from config manager
    threshold = BenchmarkConfig.get_threshold(provider)
    
    result = benchmark_attention_layers.benchmark.benchmark_single_method(
        seq_len=seq_len,
        method=provider,
        threshold=threshold
    )
    
    return result['median_ms'], result['min_ms'], result['max_ms']

def single_benchmark_attention_layers(
    seq_len: int,
    method: str,
    threshold: float = None,
    model_path: str = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct"),
    layer_to_save: int = 12,
    warmup: int = 10,
    rep: int = 20
) -> dict:
    """Benchmark a single attention method at a specific sequence length.

    Args:
        seq_len: Sequence length.
        method: Method name.
        threshold: Threshold parameter.
        model_path: Model path.
        layer_to_save: Layer index to save.
        warmup: Number of warmup iterations.
        rep: Number of repetitions.

    Returns:
        Detailed performance test results.
    """
    data_generator = AttentionDataGenerator(model_path, layer_to_save)
    benchmark = AttentionLayerBenchmark(data_generator)
    
    result = benchmark.benchmark_single_method(
        seq_len=seq_len,
        method=method, 
        threshold=threshold,
        warmup=warmup,
        rep=rep
    )
    
    return result

def analyze_method_breakdown(
    seq_len: int,
    method: str,
    threshold: float = None,
    model_path: str = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct")
) -> dict:
    """Analyze detailed performance breakdown of a method.

    Args:
        seq_len: Sequence length.
        method: Method name.
        threshold: Threshold parameter.
        model_path: Model path.

    Returns:
        Performance breakdown results.
    """
    print(f"Analyzing {method} performance breakdown at sequence length {seq_len}")
    
    # Test with different repetition counts for more stable results
    rep_configs = [10, 20, 50]
    results = {}
    
    for rep in rep_configs:
        result = single_benchmark_attention_layers(
            seq_len=seq_len,
            method=method,
            threshold=threshold,
            model_path=model_path,
            rep=rep
        )
        results[f"rep_{rep}"] = result
    
    return results

if __name__ == "__main__":
    print("=== Attention Layers Benchmark ===\n")
    
    methods_to_compare = [
        'full',
        # 'flex', 
        # 'anchor',
        # 'xattn16',
        'vecattention'
    ]
    
    custom_seq_lengths = [8*1024, 16*1024,32*1024, 64*1024, 128*1024]
    
    method_thresholds = {
        'flex': 0.95,
        'anchor': 12,
        'xattn16': 0.15,
        'vecattention': 0.8,
    }
    
    # Configure benchmark in one call
    BenchmarkConfig.set_benchmark_config(
        methods=methods_to_compare,
        seq_lengths=custom_seq_lengths,
        thresholds=method_thresholds
    )
    
    config_summary = BenchmarkConfig.get_current_config_summary()
    print("Current benchmark configuration:")
    print(f"  Methods: {config_summary['enabled_methods']}")
    print(f"  Sequence lengths: {config_summary['seq_lengths']}")
    print(f"  Thresholds: {config_summary['thresholds']}")
    print()
    
    decorated_benchmark = get_benchmark_decorator()(benchmark_attention_layers)
    
    print("Running comprehensive benchmark comparison...")
    decorated_benchmark.run(show_plots=True, print_data=True, save_path="figs")

    print("\n=== Benchmark Complete ===")