"""CUDA 设备工具函数。"""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """返回可用的 CUDA 设备，否则回退到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
