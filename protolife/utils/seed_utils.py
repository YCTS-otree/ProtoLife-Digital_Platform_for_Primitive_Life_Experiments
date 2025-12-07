"""随机种子管理。"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """统一设置 Python/NumPy/PyTorch 的随机种子，便于复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_seed() -> int:
    """生成一个 32 位整型随机种子。"""

    return random.SystemRandom().randint(0, 2**32 - 1)
