"""训练调度器占位。"""
from __future__ import annotations

from typing import Callable


def linear_decay(start: float, end: float, total_steps: int) -> Callable[[int], float]:
    """线性衰减调度器。"""

    def schedule(step: int) -> float:
        ratio = min(max(step / total_steps, 0.0), 1.0)
        return start + (end - start) * ratio

    return schedule
