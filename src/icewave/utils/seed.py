"""可复现性: 统一随机种子控制 (P1-2)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """固定 random/numpy/torch 种子; deterministic=True 时启用 cuDNN 确定性算法."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except ImportError:  # torch 未安装时仅固定 python/numpy
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 种子, 保证多进程增强可复现."""
    seed = torch_seed_base() + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))


def torch_seed_base() -> int:
    try:
        import torch

        return int(torch.initial_seed()) % (2**32)
    except ImportError:
        return 0
