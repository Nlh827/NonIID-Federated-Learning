import csv
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch


def ensure_dir(path: str):
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def tensor_bytes(x):
    if x is None:
        return 0
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    return 0


def state_dict_bytes(state_dict, selected_keys=None):
    """
    Count bytes of tensors in a state_dict.
    """
    if state_dict is None:
        return 0

    total = 0
    for k, v in state_dict.items():
        if selected_keys is not None and k not in selected_keys:
            continue
        if torch.is_tensor(v):
            total += tensor_bytes(v)
    return total


def mask_bytes(mask):
    """
    Layer-wise mask cost.
    Current mask is dict: param_name -> 0/1.
    Approximate one boolean per layer/parameter tensor.
    """
    if mask is None:
        return 0
    return len(mask)


def bytes_to_mb(x):
    return float(x) / 1024.0 / 1024.0


def cuda_memory_mb():
    if torch.cuda.is_available():
        return bytes_to_mb(torch.cuda.max_memory_allocated())
    return 0.0


def accuracy_auc(acc_list: List[float]):
    """
    Threshold-independent convergence metric.
    If accuracy is in percentage, AUC is also in percentage scale.
    """
    if len(acc_list) == 0:
        return 0.0
    return float(np.mean(acc_list))


def rounds_to_target(acc_list: List[float], target_acc: float):
    """
    Return the first round index starting from 1.
    """
    for i, acc in enumerate(acc_list):
        if acc >= target_acc:
            return i + 1
    return None


def mean_std(values: List[float]):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def append_csv_row(path: str, row: Dict):
    """
    Append one row to csv. Create header automatically.
    """
    ensure_dir(os.path.dirname(path))

    file_exists = os.path.exists(path)
    fieldnames = list(row.keys())

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def cfg_to_dict(cfg):
    """
    Convert yacs CfgNode to normal dict.
    """
    out = {}
    for k, v in cfg.items():
        try:
            if hasattr(v, "items"):
                out[k] = cfg_to_dict(v)
            else:
                out[k] = v
        except Exception:
            out[k] = str(v)
    return out


class RoundTimer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time