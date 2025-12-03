"""简易命令行地图编辑器（带 ASCII 可视化 + 坐标轴）。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import os
import sys
import torch

# 确保可以从仓库根目录导入 protolife 包
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from protolife.encoding import (
    decode_hex_to_grid,
    encode_grid_to_hex,
    load_hex_map,
)


# 位标记：与 encoding / 环境保持一致
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
    """从 hex 文件加载地图，否则创建空白地图。"""
    if args.input:
        path = Path(args.input)
        if path.exists():
            hex_string = load_hex_map(path)
            return decode_hex_to_grid(hex_string, height=args.height, width=args.width)
        print(f"未找到 {path}，创建空白地图")
    return torch.zeros((args.height, args.width), dtype=torch.int64)


def save_grid(grid: torch.Tensor, output_path: Path) -> None:
    """将当前网格编码成 hex 并写入文件。"""
    hex_string = encode_grid_to_hex(grid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(hex_string, encoding="utf-8")
    print(f"地图已保存至 {output_path}")


def cell_to_char(value: int) -> str:
    """根据格子数值映射到一个显示字符。"""
    # 优先级：wall > food > toxin > resource > empty
    if value & CELL_TYPES["wall"]:
        return "■"
    if value & CELL_TYPES["food"]:
        return "F "
    if value & CELL_TYPES["toxin"]:
        return "T "
    if value & CELL_TYPES["resource"]:
        return "R "
    return "  "  # empty


def render_grid_ascii(grid: torch.Tensor) -> None:
    """带坐标轴的 ASCII 地图显示。"""
    h, w = grid.shape

    print("\n当前地图：")
    # X 轴坐标（列下标）
    # 前面留 3 个空格给 Y 轴
    header = "   " + "".join((str(x % 10)+' ') for x in range(w))
    print(header)

    # 每一行：Y 坐标 + 一整行格子字符
    for y in range(h):
        row_chars = []
        for x in range(w):
            v = int(grid[y, x])
            row_chars.append(cell_to_char(v))
        # Y 轴坐标对齐成 2 位
        print(f"{y:2d} " + "".join(row_chars))

    print("\n图例：■=wall, F=food, T=toxin, R=resource, 空格=empty\n")


def fill_border(grid: torch.Tensor, value: int) -> None:
    """将边框全部设置为指定值。"""
    h, w = grid.shape
    # 上下两行
    grid[0, :] = value
    grid[h - 1, :] = value
    # 左右两列
    grid[:, 0] = value
    grid[:, w - 1] = value


def print_help() -> None:
    print("指令说明：")
    print("  x y type   - 设置 (x, y) 为指定类型，例如: 3 4 wall")
    print("  list       - 查看可用类型")
    print("  border     - 使用 wall 填充整张地图的边框")
    print("  border t   - 使用类型 t 填充边框，例如: border food")
    print("  show       - 重新显示当前地图")
    print("  q          - 保存并退出\n")
    print("可用类型:", ", ".join(CELL_TYPES.keys()))
    print()


def interactive_edit(grid: torch.Tensor) -> torch.Tensor:
    """交互式编辑循环。"""
    render_grid_ascii(grid)
    print_help()

    while True:
        cmd = input("> ").strip()
        if not cmd:
            continue

        low = cmd.lower()

        # 退出
        if low in {"q", "quit", "exit"}:
            break

        # 显示可用类型
        if low == "list":
            print("可用类型:", ", ".join(CELL_TYPES.keys()))
            continue

        # 只看地图
        if low == "show":
            render_grid_ascii(grid)
            continue

        # 自动填充边框：border 或 border <type>
        if low.startswith("border"):
            parts = low.split()
            if len(parts) == 1:
                cell_type = "wall"
            elif len(parts) == 2:
                cell_type = parts[1]
            else:
                print("用法: border 或 border <type>")
                continue

            if cell_type not in CELL_TYPES:
                print(f"未知类型 {cell_type}，输入 list 查看可用类型")
                continue

            fill_border(grid, CELL_TYPES[cell_type])
            print(f"已使用 {cell_type} 填充边框。")
            render_grid_ascii(grid)
            continue

        # 普通编辑命令：x y type
        parts = cmd.split()
        if len(parts) != 3:
            print("格式错误，请输入：x y type，例如：3 4 wall")
            continue

        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            print("坐标必须是整数")
            continue

        cell_type = parts[2].lower()
        if cell_type not in CELL_TYPES:
            print("未知类型，输入 list 查看可用类型")
            continue

        if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
            print("坐标越界")
            continue

        grid[y, x] = CELL_TYPES[cell_type]
        print(f"设置 ({x}, {y}) 为 {cell_type}")
        render_grid_ascii(grid)

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
