from typing import Optional


class DiffusionStepContext:
    """Runtime state shared between the inference script and all attention processors.

    All attention processor instances for a single model share one DiffusionStepContext
    (set as a class attribute via replace_wan_attention). The inference script holds the
    same reference and can read/write fields between pipeline steps.

    This keeps DiT-specific runtime state out of FastPrefillConfig, which is a generic
    config class shared with VLM and other use cases.

    Attributes:
        current_rel_step: 0-indexed diffusion step counter. Incremented by layer 0 each
            time the timestep value changes.
        pre_time_step: Timestep value from the previous attention call; used by layer 0
            to detect step boundaries (None before the first step).
        is_cfg: True when classifier-free guidance is active (guidance_scale > 1.0).
            When True, each diffusion step has two forward passes (positive then negative);
            processors switch between pos_threshold and neg_threshold accordingly.
        dump_qk_negative: When True, dump the negative-path pass instead of (or in
            addition to) the positive-path pass.
        dump_qk_started: Set to True by a processor once it determines dumping should
            begin for the current pass.
        dump_qk_done: Set to True by the last layer after completing the dump. The
            inference script checks this flag and calls sys.exit(0) if set.
    """

    def __init__(
        self,
        is_cfg: bool = False,
        dump_qk_negative: bool = False,
    ):
        self.current_rel_step: int = -1
        self.pre_time_step: Optional[float] = None
        self.is_cfg: bool = is_cfg
        self.dump_qk_negative: bool = dump_qk_negative
        self.dump_qk_started: bool = False
        self.dump_qk_done: bool = False
