"""地图编码工具。

每个格子使用 1 字节表示，便于日志压缩和快速回放。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

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


def encode_grid_to_hex(grid: torch.Tensor) -> str:
    """显式命名的封装，便于地图存储/回放。"""

    return encode_grid(grid)


def decode_hex_to_grid(hex_string: str, height: Optional[int] = None, width: Optional[int] = None,
                       shape: Optional[Iterable[int]] = None) -> torch.Tensor:
    """根据尺寸解码十六进制地图字符串。"""

    if shape is None:
        if height is None or width is None:
            raise ValueError("必须提供 height/width 或显式 shape 以解码地图")
        shape = (height, width)
    return decode_grid(hex_string, torch.Size(shape))


def load_hex_map(path: Path) -> str:
    """读取十六进制地图文件，自动去除空白。"""

    content = path.read_text(encoding="utf-8")
    return "".join(line.strip() for line in content.splitlines())


def save_hex_map(path: Path, grid: torch.Tensor) -> None:
    """将网格编码为十六进制并写入文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode_grid_to_hex(grid), encoding="utf-8")
