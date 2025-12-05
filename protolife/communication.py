"""通信机制占位实现。"""
from __future__ import annotations

from typing import Tuple

import torch


def broadcast_messages(messages: torch.Tensor, radius: int) -> torch.Tensor:
    """简单的消息聚合，占位为均值池化。"""

    # messages: (E, A, 2)
    pooled = messages.mean(dim=1, keepdim=True).expand_as(messages)
    return pooled
