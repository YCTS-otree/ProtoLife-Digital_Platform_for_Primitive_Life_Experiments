"""简易命令行地图编辑器。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from protolife.encoding import decode_hex_to_grid, encode_grid_to_hex, load_hex_map


CELL_TYPES: Dict[str, int] = {
    "empty": 0,
    "wall": 1,
    "food": 1 << 2,
    "toxin": 1 << 3,
    "resource": 1 << 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife 简易地图编辑器")
    parser.add_argument("--input", type=str, default=None, help="已有地图十六进制文件")
    parser.add_argument("--width", type=int, default=16, help="地图宽度")
    parser.add_argument("--height", type=int, default=16, help="地图高度")
    parser.add_argument("--output", type=str, default=None, help="输出文件名，未填则自动按时间戳生成")
    return parser.parse_args()


def load_grid(args: argparse.Namespace) -> torch.Tensor:
    if args.input:
        path = Path(args.input)
        if path.exists():
            hex_string = load_hex_map(path)
            return decode_hex_to_grid(hex_string, height=args.height, width=args.width)
        print(f"未找到 {path}，创建空白地图")
    return torch.zeros((args.height, args.width), dtype=torch.int64)


def save_grid(grid: torch.Tensor, output_path: Path) -> None:
    hex_string = encode_grid_to_hex(grid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(hex_string, encoding="utf-8")
    print(f"地图已保存至 {output_path}")


def interactive_edit(grid: torch.Tensor) -> torch.Tensor:
    print("输入格式：<x> <y> <type>，例如 `3 4 wall`，输入 `list` 查看可用类型，输入 `q` 保存并退出。")
    while True:
        cmd = input("> ").strip()
        if cmd.lower() in {"q", "quit", "exit"}:
            break
        if cmd.lower() == "list":
            print(f"可用类型: {', '.join(CELL_TYPES.keys())}")
            continue
        parts = cmd.split()
        if len(parts) != 3:
            print("格式错误，请输入 x y type")
            continue
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            print("坐标必须是整数")
            continue
        cell_type = parts[2].lower()
        if cell_type not in CELL_TYPES:
            print("未知类型，输入 list 查看")
            continue
        if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
            print("坐标越界")
            continue
        grid[y, x] = CELL_TYPES[cell_type]
        print(f"设置 ({x}, {y}) 为 {cell_type}")
    return grid


def main() -> None:
    args = parse_args()
    grid = load_grid(args)
    grid = interactive_edit(grid)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("maps") / f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hex"
    save_grid(grid, output_path)


if __name__ == "__main__":
    main()
