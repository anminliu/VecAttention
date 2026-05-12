#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import json
import math
from typing import Dict, Tuple, Optional, List

# Paths derived from this file's location — CWD-independent
_DITEVAL_DIR = os.path.dirname(os.path.abspath(__file__))  # eval/DiTEvalKit

import imageio
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import trange

# ---------- LPIPS (prefer CUDA, fall back to CPU) ----------
try:
    import lpips  # pip install lpips
    _lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
    _lpips_model = lpips.LPIPS(net="vgg").to(_lpips_device)
except Exception as e:
    _lpips_model = None
    _lpips_device = "cpu"
    print(f"[WARN] LPIPS unavailable ({e}); LPIPS will be set to None.")

# ---------- Basic utilities ----------
def load_video(video_path: str) -> torch.Tensor:
    """
    Load video as a float32 Tensor in [0,1], shape (T, C, H, W).
    """
    reader = imageio.get_reader(video_path)
    frames = []
    to_tensor = transforms.ToTensor()
    for frame in reader:
        frames.append(to_tensor(frame))  # (C,H,W), [0,1]
    reader.close()
    if not frames:
        raise ValueError(f"Fail to load frames from {video_path}")
    return torch.stack(frames, dim=0)  # (T,C,H,W)

def _ssim_single(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    img1, img2: (N=1, C, H, W). Sliding-window SSIM estimation.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2 ** 2
    sigma12  = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_quantization_error(video1_tensor: torch.Tensor,
                               video2_tensor: torch.Tensor) -> dict:
    """
    Compute per-frame MSE/PSNR/SSIM/LPIPS and average over frames.
    Input tensors: (T, C, H, W), range [0,1]; should already be on the target device.
    """
    assert video1_tensor.shape == video2_tensor.shape, \
        f"Videos must have the same shape. {video1_tensor.shape} != {video2_tensor.shape}"
    T = video1_tensor.shape[0]

    mse_values: List[float] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []

    eps = 1e-10  # prevent log10(0)
    device = video1_tensor.device
    lpips_dev = _lpips_device if _lpips_model is not None else str(device)

    for i in trange(T, desc="Computing metrics"):
        frame1 = video1_tensor[i:i+1]  # (1,C,H,W)
        frame2 = video2_tensor[i:i+1]

        # --- MSE ---
        mse = F.mse_loss(frame1, frame2, reduction="mean")
        mse_values.append(float(mse.item()))

        # --- PSNR ---
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))
        psnr_values.append(float(psnr.item()))

        # --- SSIM ---
        ssim = _ssim_single(frame1, frame2)
        ssim_values.append(float(ssim.item()))

        # --- LPIPS ---
        if _lpips_model is not None:
            def to_lpips_range(x): return x * 2.0 - 1.0  # [0,1] -> [-1,1]
            lf1 = to_lpips_range(frame1).to(lpips_dev)
            lf2 = to_lpips_range(frame2).to(lpips_dev)
            with torch.no_grad():
                lval = _lpips_model(lf1, lf2)
            lpips_values.append(float(lval.item()))
        else:
            lpips_values.append(float("nan"))

    return {
        "MSE":  sum(mse_values)  / len(mse_values),
        "PSNR": sum(psnr_values) / len(psnr_values),
        "SSIM": sum(ssim_values) / len(ssim_values),
        "LPIPS": (sum(lpips_values) / len(lpips_values))
                 if not any(math.isnan(x) for x in lpips_values) else None,
    }

def parse_sparsity_file(sp_path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse only the global Density and Sparsity; per-layer entries are ignored.
    Returns (density, sparsity).
    """
    if sp_path is None or not os.path.exists(sp_path):
        return (None, None)
    density = None
    sparsity = None
    try:
        with open(sp_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("Density:"):
                    density = float(line.split("Density:")[1].strip())
                elif line.startswith("Sparsity:"):
                    # Only treat "Sparsity: <val>" (single colon) as global sparsity
                    parts = line.split(":")
                    if len(parts) == 2:
                        sparsity = float(parts[1].strip())
    except Exception as e:
        print(f"[WARN] Parse sparsity file failed: {sp_path} ({e})")
    return (density, sparsity)

def parse_attention_latency_file(att_path: Optional[str]) -> Optional[float]:
    """
    Parse an attention_latency file (.txt or _*.txt) and return only the average latency (ms).
    Example file line:
      Average Attention Latency per layer: 612.065214289672 ms
    """
    if att_path is None or not os.path.exists(att_path):
        return None

    avg_ms: Optional[float] = None
    try:
        with open(att_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                # Average
                if line.startswith("Average Attention Latency"):
                    try:
                        val_part = line.split(":")[1]
                        avg_ms = float(val_part.replace("ms", "").strip())
                    except Exception:
                        pass
                    break
    except Exception as e:
        print(f"[WARN] Parse attention latency file failed: {att_path} ({e})")

    return avg_ms

# ---------- Path collection ----------
def collect_ref(ref_dir: str) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    ref_dir/[id]/
      ├── 0.mp4  (id must be numeric)
      ├── attention_latency.txt or attention_latency_*.txt (optional)
    """
    ref_dict: Dict[str, Tuple[str, Optional[str]]] = {}
    for name in sorted(os.listdir(ref_dir)):
        id_dir = os.path.join(ref_dir, name)
        if not os.path.isdir(id_dir):
            continue
        if not name.isdigit():
            print(f"[WARN] non-numeric id in ref_dir: {name} -> skip")
            continue
        mp4 = os.path.join(id_dir, "0.mp4")
        if not os.path.exists(mp4):
            print(f"[WARN] Missing MP4 in ref_dir for id={name}")
            continue

        # attention latency: prefer attention_latency.txt
        att_txt = os.path.join(id_dir, "attention_latency.txt")
        att_list = sorted(glob.glob(os.path.join(id_dir, "attention_latency_*.txt")))
        if os.path.exists(att_txt):
            if att_list:
                print(f"[INFO] [REF] Both attention_latency.txt and attention_latency_*.txt found for id={name}; prefer attention_latency.txt")
            att_path = os.path.abspath(att_txt)
        elif att_list:
            att_path = os.path.abspath(att_list[0])
        else:
            att_path = None

        ref_dict[name] = (os.path.abspath(mp4), att_path)
    return ref_dict

def collect_out(out_dir: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """
    out_dir/
      ├── [id]/0.mp4
      │        sparsity.txt or sparsity_*.txt (optional)
      │        attention_latency.txt or attention_latency_*.txt (optional)
    (id must be numeric)
    """
    out_dict: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {}
    for name in os.listdir(out_dir):
        id_dir = os.path.join(out_dir, name)
        if not os.path.isdir(id_dir):
            continue
        if not name.isdigit():
            print(f"[WARN] non-numeric id in out_dir: {name} -> skip")
            continue
        mp4 = os.path.join(id_dir, "0.mp4")
        if not os.path.exists(mp4):
            print(f"[WARN] Missing MP4 in out_dir for id={name}")
            continue

        # Support both sparsity.txt and sparsity_*.txt; prefer sparsity.txt
        sp_txt = os.path.join(id_dir, "sparsity.txt")
        sp_list = sorted(glob.glob(os.path.join(id_dir, "sparsity_*.txt")))
        if os.path.exists(sp_txt):
            if sp_list:
                print(f"[INFO] Both sparsity.txt and sparsity_*.txt found for id={name}; prefer sparsity.txt")
            sp_path = os.path.abspath(sp_txt)
        elif sp_list:
            sp_path = os.path.abspath(sp_list[0])
        else:
            sp_path = None
            print(f"[WARN] Missing sparsity(.txt or _*.txt) in out_dir for id={name}")

        # attention latency: prefer attention_latency.txt
        att_txt = os.path.join(id_dir, "attention_latency.txt")
        att_list = sorted(glob.glob(os.path.join(id_dir, "attention_latency_*.txt")))
        if os.path.exists(att_txt):
            if att_list:
                print(f"[INFO] Both attention_latency.txt and attention_latency_*.txt found for id={name}; prefer attention_latency.txt")
            att_path = os.path.abspath(att_txt)
        elif att_list:
            att_path = os.path.abspath(att_list[0])
        else:
            att_path = None
            print(f"[WARN] Missing attention_latency(.txt or _*.txt) in out_dir for id={name}")

        out_dict[name] = (os.path.abspath(mp4), sp_path, att_path)
    return out_dict

def read_global_metrics_from_jsonl(jsonl_path: str) -> Optional[dict]:
    """
    Read only the last line of the jsonl file; return its dict if it is a GLOBAL_MEAN entry, else None.
    """
    try:
        if not os.path.exists(jsonl_path):
            return None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return None
        try:
            last_obj = json.loads(lines[-1])
        except Exception:
            return None
        if isinstance(last_obj, dict) and last_obj.get("id") == "GLOBAL_MEAN":
            return last_obj
        return None
    except Exception as e:
        print(f"[WARN] Failed to read jsonl: {jsonl_path} ({e})")
        return None

def process_single(ref_dir: str, out_dir: str, args):
    """
    Evaluation pipeline for a single out_dir (extracted from main).
    Returns the global_metrics dict.
    """
    print(f"[INFO] Processing single out_dir={out_dir}")
    out_dir_abs = os.path.abspath(out_dir)
    out_tag = os.path.basename(os.path.normpath(out_dir_abs))
    default_jsonl = os.path.join(out_dir_abs, f"{out_tag}_metrics.jsonl")
    output_jsonl = os.path.abspath(args.output_jsonl) if args.output_jsonl else default_jsonl
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # If metrics.jsonl already exists, read it and skip recomputation
    existing_global = read_global_metrics_from_jsonl(output_jsonl)
    if existing_global is not None:
        return existing_global
    with open(output_jsonl, "w", encoding="utf-8") as f:
        pass  # truncate file

    # Ref stats file: read-only if it exists; written only when absent
    ref_stats_path = os.path.join(os.path.abspath(ref_dir), "ref_attn_latency_stats.jsonl")
    ref_stats_exists = os.path.exists(ref_stats_path)


    # Collect mappings
    ref_map = collect_ref(ref_dir)
    out_map = collect_out(out_dir)

    ref_ids = set(ref_map.keys())
    out_ids = set(out_map.keys())
    common_ids = sorted(ref_ids & out_ids, key=lambda x: (len(x), x))

    for iid in sorted(ref_ids - out_ids):
        print(f"[WARN] id={iid} only in ref_dir; skip")
    for iid in sorted(out_ids - ref_ids):
        print(f"[WARN] id={iid} only in out_dir; skip")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    failed: List[str] = []
    agg = {
        "MSE": [],
        "PSNR": [],
        "SSIM": [],
        "LPIPS": [],
        "density": [],
        "sparsity": [],
        "attn_latency_avg_ms": [],
    }
    ref_agg = {
        "attn_latency_avg_ms": [],
    }

    selected_ids = [iid for iid in common_ids if (args.max_id is None or int(iid) < args.max_id)]
    if args.max_id is not None:
        print(f"[INFO] Filtering: id < {args.max_id}, {len(common_ids)} -> {len(selected_ids)}")

    for iid in selected_ids:
        prompt_idx = int(iid)
        ref_mp4, ref_att_path = ref_map[iid]
        out_mp4, sp_path, out_att_path = out_map[iid]

        try:
            v1 = load_video(ref_mp4).to(device)
            v2 = load_video(out_mp4).to(device)
            metrics = compute_quantization_error(v1, v2)
            metrics["idx"] = prompt_idx
            metrics["seed"] = args.seed
            metrics["ref_path"] = ref_mp4
            metrics["out_path"] = out_mp4
            metrics["id"] = prompt_idx

            # sparsity/density
            density, sparsity = parse_sparsity_file(sp_path)
            metrics["density"] = density
            metrics["sparsity"] = sparsity

            # out attention latency (average only)
            out_att_avg = parse_attention_latency_file(out_att_path)
            metrics["attn_latency_avg_ms"] = out_att_avg

            # ref attention latency (write to ref_stats, average only)
            ref_att_avg = parse_attention_latency_file(ref_att_path)
            ref_entry = {
                "id": prompt_idx,
                "attn_latency_avg_ms": ref_att_avg,
                "attention_latency_path": ref_att_path,
            }
            with open(ref_stats_path, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(ref_entry, ensure_ascii=False))
                rf.write("\n")

            # Aggregate
            agg["MSE"].append(metrics["MSE"])
            agg["PSNR"].append(metrics["PSNR"])
            agg["SSIM"].append(metrics["SSIM"])
            if metrics["LPIPS"] is not None:
                agg["LPIPS"].append(metrics["LPIPS"])
            if metrics["density"] is not None:
                agg["density"].append(metrics["density"])
            if metrics["sparsity"] is not None:
                agg["sparsity"].append(metrics["sparsity"])
            if out_att_avg is not None:
                agg["attn_latency_avg_ms"].append(out_att_avg)

            if sp_path:
                metrics["sparsity_path"] = sp_path
            if out_att_path:
                metrics["attention_latency_path"] = out_att_path

            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False))
                f.write("\n")

            # ref aggregate
            if ref_att_avg is not None:
                ref_agg["attn_latency_avg_ms"].append(ref_att_avg)

        except Exception as e:
            print(f"[ERR] id={iid} failed: {e}")
            failed.append(iid)

    def safe_mean(x):
        return sum(x) / len(x) if len(x) > 0 else None

    global_metrics = {
        "id": "GLOBAL_MEAN",
        "MSE": safe_mean(agg["MSE"]),
        "PSNR": safe_mean(agg["PSNR"]),
        "SSIM": safe_mean(agg["SSIM"]),
        "LPIPS": safe_mean(agg["LPIPS"]),
        "density": safe_mean(agg["density"]),
        "sparsity": safe_mean(agg["sparsity"]),
        "attn_latency_avg_ms": safe_mean(agg["attn_latency_avg_ms"]),
        "count": len(selected_ids)
    }
    with open(output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(global_metrics, ensure_ascii=False))
        f.write("\n")
    print("\n[GLOBAL MEAN]")
    for k, v in global_metrics.items():
        print(f"{k}: {v}")

    # Ref global stats: write only when the ref_stats file doesn't exist; read otherwise
    if not ref_stats_exists:
        ref_global = {
            "id": "GLOBAL_MEAN",
            "attn_latency_avg_ms": safe_mean(ref_agg["attn_latency_avg_ms"]),
            "count": len(selected_ids)
        }
        with open(ref_stats_path, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(ref_global, ensure_ascii=False))
            rf.write("\n")
    else:
        ref_global = read_global_metrics_from_jsonl(ref_stats_path)

    if ref_global is not None:
        print("\n[REF GLOBAL MEAN]")
        for k, v in ref_global.items():
            print(f"{k}: {v}")

    print(f"\n[SUMMARY] matched = {len(selected_ids)}, failed = {len(failed)}, output = {output_jsonl}")
    print(f"[REF STATS] {ref_stats_path}")
    if failed:
        print("[FAILED IDS] " + ", ".join(failed))
    return global_metrics

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Batch video metrics with global sparsity/density merged")
    ap.add_argument("--ref_dir", required=False, help="Path to ref_dir ([id]-0.mp4); auto-inferred in simple mode")
    ap.add_argument("--out_dir", required=False, help="Path to out_dir ([id]/0.mp4, sparsity_*.txt); omit for batch mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (auto-detect by default)")
    ap.add_argument("--output_jsonl", type=str, default=None,
                    help="Override default output: out_dir/<out_dir_basename>_metrics.jsonl (single-dir mode only)")
    ap.add_argument("--max_id", type=int, default=None, help="Only process samples with id < max_id")
    # ---- Simple batch mode parameters ----
    ap.add_argument("--exp_name", type=str, default="results", help="exp name, default='results'")
    ap.add_argument("--backend", type=str,  help="Simple mode: select backend ({exp_name}/{backend}/t2v/..)")
    ap.add_argument("--exp_setting", type=str,
                    default="Step_50-Res_720p/TFP_0.06-LFP_0.0",
                    help="Simple mode: experiment setting sub-path (may contain multiple levels, e.g. Step_50-Res_720p/TFP_0.06-LFP_0.0)")
    ap.add_argument("--method", type=str, nargs="+",
                    default=["SVG", "xattn_Stride8", "vecattention_Q64-K16-G8192"],
                    help="Simple mode: list of method names; iterates first-level subdirs under each method root as out_dir")
    args = ap.parse_args()

    # --------- Determine whether to enter simple batch mode ----------
    simple_batch_mode = (args.method and args.backend and args.exp_setting and not args.out_dir)

    # --------- If simple mode and ref_dir not provided, auto-infer dense path ----------
    if simple_batch_mode and not args.ref_dir:
        auto_ref = os.path.join(_DITEVAL_DIR, args.exp_name,
                                args.backend, "t2v", "dense", args.exp_setting)
        args.ref_dir = auto_ref
        print(f"[INFO] Simple mode auto ref_dir = {args.ref_dir}")

    if not args.ref_dir:
        raise ValueError("Must provide --ref_dir, or use simple mode with --backend --exp_setting --method")

    # --------- Simple mode: build list of out_dirs to evaluate ----------
    if simple_batch_mode:
        all_batch_results = []
        for method in args.method:
            method_root = os.path.join(_DITEVAL_DIR, args.exp_name,
                                       args.backend, "t2v", method, args.exp_setting)
            if not os.path.isdir(method_root):
                print(f"[WARN] Simple mode method root does not exist: {method_root} -> skip")
                continue
            candidate_out_dirs = []
            for child in sorted(os.listdir(method_root)):
                child_path = os.path.join(method_root, child)
                if os.path.isdir(child_path):
                    candidate_out_dirs.append(child_path)
            if not candidate_out_dirs:
                print(f"[WARN] Simple mode: no first-level subdirectories found under {method_root} to use as out_dir")
                continue
            print(f"[INFO] Method {method}: found {len(candidate_out_dirs)} directories to evaluate:")
            for p in candidate_out_dirs:
                print(f"  - {p}")

            batch_results = []
            for od in candidate_out_dirs:
                try:
                    result = process_single(args.ref_dir, od, args)
                    batch_results.append(("OK", od, result))
                except Exception as e:
                    print(f"[ERR][BATCH] out_dir={od} failed: {e}")
                    batch_results.append(("FAIL", od, None))
            all_batch_results.append((method, batch_results))

        # Summary
        total = 0
        total_ok = 0
        total_fail = 0
        print("\n[SUMMARY][BATCH SIMPLE]")
        for method, batch_results in all_batch_results:
            ok_cnt = sum(1 for s, _, _ in batch_results if s == "OK")
            fail_cnt = sum(1 for s, _, _ in batch_results if s == "FAIL")
            total += len(batch_results)
            total_ok += ok_cnt
            total_fail += fail_cnt
            print(f"  - {method}: total={len(batch_results)}, ok={ok_cnt}, fail={fail_cnt}")
            if fail_cnt:
                print("    [FAILED OUT_DIR LIST]")
                for s, od, _ in batch_results:
                    if s == "FAIL":
                        print(f"      - {od}")
        print(f"  = overall: total={total}, ok={total_ok}, fail={total_fail}")
        return

    # --------- Single-directory normal mode ----------
    if not args.out_dir:
        raise ValueError("Normal mode requires --out_dir (or use simple mode to omit it)")

    process_single(args.ref_dir, args.out_dir, args)

if __name__ == "__main__":
    main()