import os
import argparse
import subprocess
from typing import Dict, Any, List, Optional
import re
# Import torch for distributed support
import torch
import datetime

from eval.check_env import get_env_name
assert get_env_name() == "dit", "Please run this script in DiT local source environment."

# Paths derived from this file's location — CWD-independent
DITEVAL_DIR = os.path.dirname(os.path.abspath(__file__))    # eval/DiTEvalKit
REPO_ROOT = os.path.dirname(os.path.dirname(DITEVAL_DIR))   # repo root

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")

# ==== Backend (engine) configuration ====
BACKENDS = {
    "hyvideo": {
        "entry": os.path.join(DITEVAL_DIR, "hyvideo_t2v_inference.py"),
        "default_model_id": os.path.join(MODEL_DIR, "HunyuanVideo"),
        "default_output_root": "results/hyvideo/t2v",
    },
    "wan": {
        "entry": os.path.join(DITEVAL_DIR, "wan_t2v_inference.py"),
        "default_model_id": os.path.join(MODEL_DIR, "Wan2.1-T2V-14B-Diffusers"),
        "default_output_root": "results/wan/t2v",
    },
}
# ==== Common environment variables ====
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + REPO_ROOT
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.0 8.9 9.0"


# ==== Global defaults (hoisted) ====
DEFAULT_MODEL_ID = os.path.join(MODEL_DIR, "HunyuanVideo")
DEFAULT_SEED = 0
DEFAULT_PROMPT_PATH = os.path.join(DITEVAL_DIR, "hunyuan_all_dimension.txt")
DEFAULT_SAMPLE_NUM = 20
# DEFAULT_OUTPUT_ROOT = "results/hyvideo/t2v"

# Top-level defaults (overridable via CLI)
DEFAULT_RESOLUTION = "720p"
DEFAULT_INFER_STEP = 50
DEFAULT_FIRST_TIMES_FP = 0.06
DEFAULT_FIRST_LAYERS_FP = 0

# Method aliases
_METHOD_ALIAS = {
    "dense": "dense",
    "xattn": "xattn",
    "sap": "SAP",
    "SVG": "SVG",
    "svg": "SVG",
    "segsp": "vecattention",
    "vecattention": "vecattention",
}

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

def norm_method_name(name: str) -> str:
    return _METHOD_ALIAS.get(name.strip(), name.strip())

# Resolution parsing: 720p -> (720, 1280); 480p -> (480, 720)
def parse_resolution(resolution: str) -> tuple[int, int]:
    if not resolution.endswith("p"):
        raise ValueError(f"Unsupported resolution string: {resolution}")
    h = int(resolution[:-1])
    if h == 480:
        w = 720
    elif h >= 720:
        w = 1280
    else:
        w = int(h * 16 / 9)
    return h, w

def infer_target_sparsity(threshold_path: Optional[str]) -> Optional[float]:
    """
    Parse the sparsity value between _S{value}_R from the threshold_path filename.
    Example: ..._S0.69996_R0.9351279693429904_thresholds.json -> 0.69996
    """
    if not threshold_path:
        return None
    fname = os.path.basename(threshold_path)
    m = re.search(r"_S([0-9]*\.?[0-9]+)_R", fname)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def load_prompts(prompt_path: str) -> List[str]:
    prompts = []
    with open(prompt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts

def even_sample_indices(total: int, sample_num: int) -> List[int]:
    interval = max(1, total // max(1, sample_num))
    return list(range(0, total, interval))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# Method-specific defaults (only method-differing fields)
METHOD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dense": {},
    "xattn": {"threshold": 0.90, "stride": 8, "chunk_size": 64 * 1024},
    "SAP": {
        "qc_kmeans": 100,
        "kc_kmeans": 500,
        "top_p_k": 0.9,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 50,
        "kmeans_iter_step": 2,
        "zero_step_kmeans_init": True,
    },
    "SVG": {"density": 0.5, "num_sampled_rows": 64},
    "vecattention": {
        "block_size_q": 64,
        "block_size_k": 16,
        "group_k_block": 8192,
        "chunk_size": 32 * 1024,
        "threshold_path": None,
        "neg_threshold_path": None,
        "threshold": None,
    },
}

# Method-specific default environment variables (overridable via --env)
METHOD_ENV_DEFAULTS: Dict[str, Dict[str, str]] = {
    "dense": {},
    "xattn": {},
    "SAP": {},
    "SVG": {},
    "vecattention": {},
}

# Method-specific default logging flags (overridable via --add-logging-file)
METHOD_LOGGING_DEFAULT: Dict[str, bool] = {
    "dense": False,
    "xattn": False,
    "SAP": True,
    "SVG": True,
    "vecattention": False,
}

def _get_mth_suffix(method: str, mcfg: Dict[str, Any]) -> str:
    if method == "xattn":
        return f"{method}_Stride{mcfg['stride']}"
    elif method == "SAP":
        return f"{method}_QC{mcfg['qc_kmeans']}-KC{mcfg['kc_kmeans']}"
    elif method == "SVG":
        return f"{method}"  # no distinguishing features yet; use method name
    elif method == "vecattention":
        suffix = f"{method}_Q{mcfg['block_size_q']}-K{mcfg['block_size_k']}-G{mcfg['group_k_block']}"
        if not mcfg.get("threshold_path") and not mcfg.get("neg_threshold_path"):
            suffix += "-wo_DP"
        return suffix
    else:
        return method  # dense, etc.

def _build_threshold_suffix(method: str, common: Dict[str, Any], mcfg: Dict[str, Any]) -> str:
    parts = [f"Step_{common['infer_step']}-Res_{common['resolution']}"]
    if common.get("first_times_fp") is not None or common.get("first_layers_fp") is not None:
        parts.append(f"TFP_{common.get('first_times_fp')}-LFP_{common.get('first_layers_fp')}")
    dir_name = "/".join(parts)

    if method == "dense":
        return dir_name
    
    parts = []
    if method == "xattn":
        parts.append(f"Th{mcfg['threshold']}")
    elif method == "SAP":
        parts.append(f"TopP_{mcfg['top_p_k']}")
        parts.append(f"Init_{mcfg['kmeans_iter_init']}-Step_{mcfg['kmeans_iter_step']}-MinR_{mcfg['min_kc_ratio']}")
    elif method == "SVG":
        parts.append(f"Density_{mcfg['density']}")
    elif method == "vecattention":
        if mcfg.get("target_sparsity") is not None:
            parts.append(f"Sp_{mcfg['target_sparsity']}")
        elif mcfg.get("threshold") is not None:
            parts.append(f"Th{mcfg['threshold']}_perHead")
        else:
            parts.append("Th_perHead")
    # dense does not append threshold info
    return dir_name + "/" + "-".join(parts)

def _build_cmd(
    method: str,
    common: Dict[str, Any],
    mcfg: Dict[str, Any],
    prompt: str,
    out_mp4: str,
    out_log: Optional[str],
    add_logging_file: bool,
    prompt_id: Optional[int] = None,
) -> List[str]:
    height, width = parse_resolution(common["resolution"])
    entry_script = common.get("entry", os.path.join(DITEVAL_DIR, "hyvideo_t2v_inference.py"))
    cmd = [
        "python", "-u", entry_script,  # unbuffered output to flush logs before errors
        "--model_id", common.get("model_id", DEFAULT_MODEL_ID),
        "--seed", str(common.get("seed", DEFAULT_SEED)),
        "--height", str(height),
        "--width", str(width),
        "--prompt", prompt,
        "--num_inference_steps", str(common["infer_step"]),
        "--resolution", common["resolution"],
        "--output_file", out_mp4,
    ]
    if common.get("first_times_fp") is not None:
        cmd += ["--first_times_fp", str(common["first_times_fp"])]
    if common.get("first_layers_fp") is not None:
        cmd += ["--first_layers_fp", str(common["first_layers_fp"])]
    if common.get("num_frames") is not None:
        cmd += ["--num_frames", str(common["num_frames"])]

    if method == "dense":
        cmd += ["--fastprefill-metric", "dense"]
    elif method == "xattn":
        cmd += [
            "--fastprefill-metric", "xattn",
            "--fastprefill-threshold", str(mcfg["threshold"]),
            "--fastprefill-stride", str(mcfg["stride"]),
            "--fastprefill-chunk-size", str(mcfg["chunk_size"]),
        ]
    elif method == "SAP":
        cmd += [
            "--fastprefill-metric", "SAP",
            "--num_q_centroids", str(mcfg["qc_kmeans"]),
            "--num_k_centroids", str(mcfg["kc_kmeans"]),
            "--top_p_kmeans", str(mcfg["top_p_k"]),
            "--min_kc_ratio", str(mcfg["min_kc_ratio"]),
            "--kmeans_iter_init", str(mcfg["kmeans_iter_init"]),
            "--kmeans_iter_step", str(mcfg["kmeans_iter_step"]),
        ]
        if mcfg.get("zero_step_kmeans_init", False):
            cmd += ["--zero_step_kmeans_init"]
    elif method == "SVG":
        cmd += [
            "--fastprefill-metric", "SVG",
            "--sparsity", str(mcfg["density"]),
            "--num_sampled_rows", str(mcfg["num_sampled_rows"]),
        ]
    elif method == "vecattention":
        cmd += [
            "--fastprefill-metric", "vecattention",
            "--fastprefill-block-size-q", str(mcfg["block_size_q"]),
            "--fastprefill-block-size-k", str(mcfg["block_size_k"]),
            "--fastprefill-group-k-block", str(mcfg["group_k_block"]),
            "--fastprefill-chunk-size", str(mcfg["chunk_size"]),
        ]
        if mcfg.get("threshold") is not None:
            cmd += ["--fastprefill-threshold", str(mcfg["threshold"])]
        if mcfg.get("threshold_path"):
            cmd += ["--threshold-path", mcfg["threshold_path"]]
        # WAN-only optional negative threshold path
        if mcfg.get("neg_threshold_path"):
            cmd += ["--neg-threshold-path", mcfg["neg_threshold_path"]]
    else:
        raise ValueError(f"Unknown method: {method}")

    if os.environ.get("DIT_DUMP_QK_DIR"):
        cmd += ["--fastprefill-dump-qk-dir", os.environ["DIT_DUMP_QK_DIR"]]
        if prompt_id is not None:
            cmd += ["--fastprefill-input-id", str(prompt_id)]

    if add_logging_file and out_log:
        cmd += ["--logging_file", out_log]

    return cmd

def run_cmd_verbose(cmd: List[str], out_dir: str, debug: bool = False, debug_mode: str = "pdb", debugpy_addr: str = "127.0.0.1:5678"):
    """
    Run a subprocess; on failure, print and save full stdout/stderr.
    When debug=True:
      - pdb: do not redirect output; breakpoint() in target code drops into pdb in the current terminal.
      - debugpy: for python commands, inject debugpy and wait for VS Code remote attach (default 127.0.0.1:5678).
    """
    os.makedirs(out_dir, exist_ok=True)
    stdout_path = os.path.join(out_dir, "stdout.log")
    stderr_path = os.path.join(out_dir, "stderr.log")
    cmd_path = os.path.join(out_dir, "command.log")

    if debug:
        try:
            with open(cmd_path, "w", encoding="utf-8") as f:
                f.write(" ".join(cmd))
        except Exception:
            pass

        is_python = os.path.basename(cmd[0]).startswith("python")
        env = os.environ.copy()

        if debug_mode == "debugpy" and is_python:
            # VS Code: first run “Python: Remote Attach” with port matching debugpy_addr
            new_cmd = [cmd[0], "-m", "debugpy", "--listen", debugpy_addr, "--wait-for-client"] + cmd[1:]
        else:
            # pdb: write breakpoint() in target code; keep terminal interactive (no redirection)
            # Optionally force breakpoint implementation (defaults to pdb.set_trace)
            env.setdefault("PYTHONBREAKPOINT", "pdb.set_trace")
            new_cmd = cmd

        res = subprocess.run(new_cmd, env=env)  # don't capture output; allow interactive use
        if res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, new_cmd)
        return res

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    try:
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd))
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(res.stdout or "")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(res.stderr or "")
    except Exception:
        pass

    if res.returncode != 0:
        print("\n===== Subprocess STDOUT =====\n")
        print(res.stdout or "")
        print("\n===== Subprocess STDERR (Traceback) =====\n")
        print(res.stderr or "")
        print(f"\n[ERROR] Return code: {res.returncode}")
        print(f"[ERROR] Logs: {stdout_path} | {stderr_path}")
        raise subprocess.CalledProcessError(
            res.returncode, cmd, output=res.stdout, stderr=res.stderr
        )
    return res

def run_one(
    method: str,
    prompt_id: int,
    prompt: str,
    common: Dict[str, Any],
    mcfg_overrides: Optional[Dict[str, Any]],
    output_root: str,
    add_logging_file: Optional[bool],
    extra_env: Dict[str, str],
    skip_existing: bool,
    debug_mode: bool,
):
    mth = norm_method_name(method)
    mcfg = dict(METHOD_DEFAULTS.get(mth, {}))
    if mcfg_overrides:
        mcfg.update({k: v for k, v in mcfg_overrides.items() if v is not None})

    merged_env = dict(METHOD_ENV_DEFAULTS.get(mth, {}))
    merged_env.update(extra_env or {})
    for k, v in merged_env.items():
        os.environ[str(k)] = str(v)

    mth_suffix = _get_mth_suffix(mth, mcfg)
    threshold_suffix = _build_threshold_suffix(mth, common, mcfg)

    # Directory layout: output_root/mth_{mth_suffix}/{threshold_suffix}/{prompt_id}
    out_dir = os.path.join(output_root, f"{mth_suffix}", threshold_suffix, str(prompt_id))
    out_mp4 = os.path.join(out_dir, "0.mp4")
    out_log = os.path.join(out_dir, "0.jsonl")
    ensure_dir(out_dir)

    if skip_existing and os.path.exists(out_mp4):
        print(f"Output file {out_mp4} already exists. Skipping...")
        return

    logging_flag = METHOD_LOGGING_DEFAULT.get(mth, False) if add_logging_file is None else bool(add_logging_file)
    cmd = _build_cmd(mth, common, mcfg, prompt, out_mp4, out_log, logging_flag, prompt_id=prompt_id)

    print("\nRunning command:\n", " ".join(cmd), "\n", flush=True)
    run_cmd_verbose(cmd, out_dir, debug_mode)

def run_batch(
    method: str,
    prompt_path: str,
    sample_num: int,
    common: Dict[str, Any],
    mcfg_overrides: Optional[Dict[str, Any]],
    output_root: str,
    add_logging_file: Optional[bool],
    extra_env: Dict[str, str],
    skip_existing: bool,
    prompt_ids: Optional[List[int]] = None,
    dist_split_mode: str = "chunk",
    debug_mode: bool = False,
):
    prompts = load_prompts(prompt_path)
    total = len(prompts)
    if total == 0:
        raise RuntimeError(f"No prompts found in {prompt_path}")

    if prompt_ids:
        indices = [pid for pid in prompt_ids if 0 <= pid < total]
    else:
        indices = even_sample_indices(total, sample_num)

    # Distributed splitting
    original_indices = indices[:]
    if WORLD_SIZE > 1:
        if dist_split_mode == "round_robin":
            indices = [idx for pos, idx in enumerate(original_indices) if pos % WORLD_SIZE == LOCAL_RANK]
        else:  # chunk
            chunk_size = (len(original_indices) + WORLD_SIZE - 1) // WORLD_SIZE
            start = LOCAL_RANK * chunk_size
            end = min(start + chunk_size, len(original_indices))
            indices = original_indices[start:end]
        print(f"[DIST] world_size={WORLD_SIZE} rank={LOCAL_RANK} split_mode={dist_split_mode} "
              f"assigned_indices={indices} total_indices={original_indices}")

    print(f"[INFO] method={method}, total_prompts={total}, indices_assigned={indices}")

    for i in indices:
        print(f"\n=== Rank {LOCAL_RANK} running prompt {i}/{total} ===", flush=True)
        run_one(method, i, prompts[i], common, mcfg_overrides, output_root,
                add_logging_file, extra_env, skip_existing, debug_mode)
        
def parse_env_list(env_list: List[str]) -> Dict[str, str]:
    out = {}
    for item in env_list or []:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified T2V evaluator")
    # General
    p.add_argument("--backend", type=str, default="hyvideo", choices=["hyvideo", "wan"], help="Select T2V backend/engine")
    p.add_argument("--method", type=str, default="dense",
                   choices=["dense", "xattn", "SAP", "SVG", "vecattention", "sap", "svg", "segsp"],
                   help="Evaluation method")
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prompt-path", type=str, default=DEFAULT_PROMPT_PATH)
    p.add_argument("--sample-num", type=int, default=DEFAULT_SAMPLE_NUM)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--prompt-ids", type=str, default=None, help="Comma separated prompt ids to run only these")
    p.add_argument("--dist-split-mode", type=str, default="round_robin",
                   choices=["chunk", "round_robin"], help="torchrun multi-GPU split mode")

    # Top-level (hoisted) common config
    p.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    p.add_argument("--num-frames", type=int, default=None, help="Override num_frames passed to inference")
    p.add_argument("--infer-step", type=int, default=DEFAULT_INFER_STEP)
    p.add_argument("--first-times-fp", type=float, default=DEFAULT_FIRST_TIMES_FP)
    p.add_argument("--first-layers-fp", type=float, default=DEFAULT_FIRST_LAYERS_FP)
    p.add_argument("--debug-run", action="store_true", help="Enable extra debugging env & faulthandler")
    # Execution control
    gskip = p.add_mutually_exclusive_group()
    gskip.add_argument("--skip-existing", action="store_true", default=True, help="Skip if output exists")
    gskip.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--test-prompt-id", type=int, default=None)

    # Top-level logging/env (hoisted)
    p.add_argument("--add-logging-file", dest="add_logging_file", action="store_true", default=None)
    p.add_argument("--no-add-logging-file", dest="add_logging_file", action="store_false")
    p.add_argument("--env", action="append", default=[], help="Extra env KEY=VAL, can repeat")

    # xattn
    p.add_argument("--xattn-threshold", type=float, default=None)
    p.add_argument("--xattn-stride", type=int, default=None)
    p.add_argument("--xattn-chunk-size", type=int, default=None)

    # SAP
    p.add_argument("--qc-kmeans", type=int, default=None)
    p.add_argument("--kc-kmeans", type=int, default=None)
    p.add_argument("--top-p-k", type=float, default=None)
    p.add_argument("--min-kc-ratio", type=float, default=None)
    p.add_argument("--kmeans-iter-init", type=int, default=None)
    p.add_argument("--kmeans-iter-step", type=int, default=None)
    p.add_argument("--zero-step-kmeans-init", action="store_true", default=None)
    p.add_argument("--no-zero-step-kmeans-init", dest="zero_step_kmeans_init", action="store_false")

    # SVG
    p.add_argument("--density", type=float, default=None)
    p.add_argument("--num-sampled-rows", type=int, default=None)

    # vecattention
    p.add_argument("--block-size-q", type=int, default=None)
    p.add_argument("--block-size-k", type=int, default=None)
    p.add_argument("--group-k-block", type=int, default=None)
    p.add_argument("--vec-chunk-size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--threshold-path", type=str, default=None)
    p.add_argument("--neg-threshold-path", type=str, default=None, help="WAN-only optional negative threshold path")
    p.add_argument("--target-sparsity", type=float, default=None, help="For vecattention diff sparsity output suffix Sp_{value}")
    return p

def main():
    args = build_argparser().parse_args()
    method = norm_method_name(args.method)
    backend = args.backend

    if args.debug_run:
        print("[DEBUG] Debug run enabled.")
        os.environ["PYTHONFAULTHANDLER"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600))),
            device_id=torch.device(f'cuda:{LOCAL_RANK}'),
        )
    backend_cfg = BACKENDS[backend]
    model_id = args.model_id or backend_cfg["default_model_id"]
    output_root = os.path.join(args.output_root, f"{backend}", "t2v") or backend_cfg["default_output_root"]

    common = {
        "entry": backend_cfg["entry"],
        "model_id": model_id,
        "seed": args.seed,
        "resolution": args.resolution,
        "infer_step": args.infer_step,
        "first_times_fp": args.first_times_fp,
        "first_layers_fp": args.first_layers_fp,
        "num_frames": args.num_frames,
    }
    extra_env = parse_env_list(args.env)

    mcfg_overrides: Dict[str, Any] = {}
    if method == "xattn":
        mcfg_overrides.update({
            "threshold": args.xattn_threshold,
            "stride": args.xattn_stride,
            "chunk_size": args.xattn_chunk_size,
        })
    elif method == "SAP":
        mcfg_overrides.update({
            "qc_kmeans": args.qc_kmeans,
            "kc_kmeans": args.kc_kmeans,
            "top_p_k": args.top_p_k,
            "min_kc_ratio": args.min_kc_ratio,
            "kmeans_iter_init": args.kmeans_iter_init,
            "kmeans_iter_step": args.kmeans_iter_step,
            "zero_step_kmeans_init": args.zero_step_kmeans_init,
        })
    elif method == "SVG":
        mcfg_overrides.update({
            "density": args.density,
            "num_sampled_rows": args.num_sampled_rows,
        })
    elif method == "vecattention":
        mcfg_overrides.update({
            "block_size_q": args.block_size_q,
            "block_size_k": args.block_size_k,
            "group_k_block": args.group_k_block,
            "chunk_size": args.vec_chunk_size,
            "threshold": args.threshold,
            "threshold_path": args.threshold_path,
            "neg_threshold_path": args.neg_threshold_path,
            "target_sparsity": args.target_sparsity,
        })
        if args.target_sparsity is None:
            inferred_ts = infer_target_sparsity(args.threshold_path)
            if inferred_ts is not None:
                mcfg_overrides["target_sparsity"] = inferred_ts
                print(f"[INFO] Auto-inferred target_sparsity={inferred_ts} from threshold_path filename")
            else:
                print("[INFO] Could not auto-parse target_sparsity from threshold_path filename")
                
    prompt_ids = [int(x) for x in args.prompt_ids.split(",")] if args.prompt_ids else None
    run_batch(
        method=method,
        prompt_path=args.prompt_path,
        sample_num=args.sample_num,
        common=common,
        mcfg_overrides=mcfg_overrides,
        output_root=output_root,
        add_logging_file=args.add_logging_file,
        extra_env=extra_env,
        skip_existing=args.skip_existing,
        prompt_ids=prompt_ids,
        dist_split_mode=args.dist_split_mode,
        debug_mode = args.debug_run,
    )

if __name__ == "__main__":
    main()