"""通信机制占位实现。"""
from __future__ import annotations

import torch


def broadcast_messages(
    messages: torch.Tensor, positions: torch.Tensor, radius: float, base_strength: float = 1.0
) -> torch.Tensor:
    """根据空间距离聚合通信信号，并按对数衰减强度。

    Args:
        messages: `(E, A, D)` 形状的消息向量。
        positions: `(E, A, 2)` 位置坐标张量，顺序为 `(x, y)`。
        radius: 感知半径，超出范围的消息被忽略。
        base_strength: 基础信号强度，作为衰减前的权重。

    Returns:
        `(E, A, D)` 聚合后的消息张量。
    """

    if radius <= 0:
        return torch.zeros_like(messages)

    # 计算所有个体间的欧氏距离矩阵
    xy = positions.float()
    distances = torch.cdist(xy, xy, p=2)  # (E, A, A)

    within_radius = (distances <= radius) & (distances > 0)

    decay = torch.zeros_like(distances)
    decay = torch.where(
        within_radius,
        base_strength / (1.0 + torch.log1p(torch.clamp(distances, min=1e-6))),
        decay,
    )

    weighted = decay.unsqueeze(-1) * messages.unsqueeze(2)  # (E, A, A, D)
    aggregated = weighted.sum(dim=1)  # (E, A, D)
    return aggregated
