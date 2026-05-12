import json
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"


def clear_memory_usage():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def print_memory_usage(prefix: str = ""):
    print(
        f"{Color.orange}{prefix} Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB{Color.reset}"
    )


def print_args(args):
    print(f"{Color.magenta}Args:{Color.reset}")
    for key, value in args.__dict__.items():
        print(f"{Color.magenta}{key}: {value}{Color.reset}")


NUM_LAYERS = 60
NUM_HEADS = 24


def _load_threshold_pt(
    thr_path: Optional[str],
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    try:
        # [[thr for head] for layer]
        with open(thr_path, 'r') as f:
            thr_json = json.load(f)
        if not isinstance(thr_json, list):
            raise ValueError(f"Threshold matrix must be a list, got {type(thr_json)}")
        if len(thr_json) != num_layers:
            raise ValueError(f"Threshold matrix length must be {num_layers}, got {len(thr_json)}")
        if len(thr_json[0]) != num_heads:
            raise ValueError(f"Threshold matrix heads per layer must be {num_heads}, got {len(thr_json[0])}")

        thr_tensor = torch.tensor(thr_json, dtype=torch.float32, device=device)
        return thr_tensor.contiguous()
    except Exception as e:
        raise RuntimeError(f"Cannot load threshold file {thr_path}: {e}")


def _save_threshold_pt(
    thr_path: str,
    thr_mat: Union[torch.Tensor, float],
    num_layers: int = NUM_LAYERS,
) -> str:
    valid = False

    if not isinstance(thr_mat, torch.Tensor) and not isinstance(thr_mat, float):
        raise TypeError(f"thr_mat must be Tensor or float, not {type(thr_mat)}")
    if isinstance(thr_mat, float):
        thr_mat = torch.full((num_layers), thr_mat, dtype=torch.float32)
        valid = True
    elif thr_mat.ndim == 1:
        if thr_mat.shape[0] == num_layers:
            valid = True
        elif thr_mat.shape[0] == 1:
            thr_mat = thr_mat.repeat(num_layers)
            valid = True

    if not valid:
        raise ValueError(f"thr_mat shape {thr_mat.shape} cannot be broadcast to ({num_layers})")

    torch.save(thr_mat, thr_path)

    return thr_path
