"""地图编码工具。

每个格子使用 1 字节表示，便于日志压缩和快速回放。
"""
from __future__ import annotations

from typing import Dict

import torch

# 位标记常量，便于未来扩展
BIT_TERRAIN_0 = 1 << 0
BIT_TERRAIN_1 = 1 << 1
BIT_FOOD = 1 << 2
BIT_TOXIN = 1 << 3
BIT_BUILDABLE = 1 << 4
BIT_RESOURCE = 1 << 5


def encode_cell(flags: Dict[str, bool]) -> int:
    """将格子属性字典编码为单字节整数。"""

    value = 0
    value |= BIT_TERRAIN_0 if flags.get("terrain0", False) else 0
    value |= BIT_TERRAIN_1 if flags.get("terrain1", False) else 0
    value |= BIT_FOOD if flags.get("food", False) else 0
    value |= BIT_TOXIN if flags.get("toxin", False) else 0
    value |= BIT_BUILDABLE if flags.get("buildable", False) else 0
    value |= BIT_RESOURCE if flags.get("resource", False) else 0
    return value


def encode_grid(grid: torch.Tensor) -> str:
    """将 `(E, H, W)` 或 `(H, W)` 整数张量编码为十六进制字符串。"""

    flat = grid.view(-1).to(torch.int64)
    return flat.clamp(min=0, max=255).to(torch.uint8).cpu().numpy().tobytes().hex()


def decode_grid(hex_string: str, shape: torch.Size) -> torch.Tensor:
    """从十六进制字符串恢复地图张量。"""

    byte_data = bytes.fromhex(hex_string)
    tensor = torch.tensor(list(byte_data), dtype=torch.uint8).view(shape)
    return tensor
