"""简易命令行地图编辑器。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict
import sys

import numpy as np

import torch
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

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
    parser.add_argument("--gui", action="store_true", help="使用 matplotlib 图形界面进行编辑")
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


def draw_line_on_grid(grid: torch.Tensor, start: tuple[int, int], end: tuple[int, int], value: int) -> None:
    """使用 Bresenham 算法在网格上绘制一条线。"""

    x1, y1 = start
    x2, y2 = end

    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy

    while True:
        if 0 <= x1 < grid.shape[1] and 0 <= y1 < grid.shape[0]:
            grid[y1, x1] = value
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy


def fill_polygon_on_grid(grid: torch.Tensor, vertices: list[tuple[int, int]], value: int) -> None:
    """使用多边形填充（包含边界）设置指定区域。"""

    if len(vertices) < 3:
        raise ValueError("至少需要 3 个顶点进行填充")

    # 使用像素中心坐标判断点是否在多边形内
    h, w = grid.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32) + 0.5,
        torch.arange(w, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).numpy()
    polygon = MplPath(np.array(vertices, dtype=np.float32))
    mask = polygon.contains_points(points, radius=1e-9).reshape(h, w)
    grid[mask] = value
    # 确保顶点在结果中
    for x, y in vertices:
        if 0 <= x < w and 0 <= y < h:
            grid[y, x] = value


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
    print("  line x1 y1 x2 y2 type - 使用直线连接两个点，并填充类型")
    print("  fill x1 y1 x2 y2 [x3 y3 ...] type - 使用多边形填充指定区域（至少 3 个点）")
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

        if low.startswith("line"):
            parts = cmd.split()
            if len(parts) != 6:
                print("用法: line x1 y1 x2 y2 type")
                continue
            try:
                x1, y1, x2, y2 = map(int, parts[1:5])
            except ValueError:
                print("坐标必须是整数")
                continue
            cell_type = parts[5].lower()
            if cell_type not in CELL_TYPES:
                print("未知类型，输入 list 查看")
                continue
            if not all(
                0 <= coord < bound
                for coord, bound in zip((x1, x2), (grid.shape[1], grid.shape[1]))
            ) or not all(
                0 <= coord < bound for coord, bound in zip((y1, y2), (grid.shape[0], grid.shape[0]))
            ):
                print("坐标越界")
                continue
            draw_line_on_grid(grid, (x1, y1), (x2, y2), CELL_TYPES[cell_type])
            print(f"已绘制 {cell_type} 直线：({x1}, {y1}) -> ({x2}, {y2})")
            render_grid_ascii(grid)
            continue

        if low.startswith("fill"):
            parts = cmd.split()
            if len(parts) < 6:
                print("用法: fill x1 y1 x2 y2 [x3 y3 ...] type")
                continue
            cell_type = parts[-1].lower()
            if cell_type not in CELL_TYPES:
                print("未知类型，输入 list 查看")
                continue
            coord_parts = parts[1:-1]
            if len(coord_parts) % 2 != 0 or len(coord_parts) < 6:
                print("至少需要 3 个顶点，坐标数量必须为偶数")
                continue
            try:
                coords = list(map(int, coord_parts))
            except ValueError:
                print("坐标必须是整数")
                continue
            vertices = list(zip(coords[0::2], coords[1::2]))
            if not all(
                0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] for x, y in vertices
            ):
                print("存在越界顶点")
                continue
            fill_polygon_on_grid(grid, vertices, CELL_TYPES[cell_type])
            print(f"已填充 {cell_type} 多边形，顶点: {vertices}")
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
            print("未知类型，输入 list 查看")
            continue
        if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
            print("坐标越界")
            continue
        grid[y, x] = CELL_TYPES[cell_type]
        print(f"设置 ({x}, {y}) 为 {cell_type}")
    return grid


def gui_edit(grid: torch.Tensor) -> torch.Tensor:
    """使用 matplotlib 的点选界面快速编辑网格。"""

    current_type_names = list(CELL_TYPES.keys())
    current_idx = 0
    drawing = False
    mode = "brush"  # brush / line / fill
    line_points: list[tuple[int, int]] = []
    fill_points: list[tuple[int, int]] = []

    def render() -> None:
        display = torch.zeros_like(grid, dtype=torch.float32)
        display = torch.where(grid == CELL_TYPES["wall"], torch.tensor(0.2), display)
        display = torch.where(grid == CELL_TYPES["food"], torch.tensor(0.8), display)
        display = torch.where(grid == CELL_TYPES["toxin"], torch.tensor(0.5), display)
        display = torch.where(grid == CELL_TYPES["resource"], torch.tensor(0.6), display)
        ax.clear()
        title = (
            f"模式: {mode} | 笔刷: {current_type_names[current_idx]}"
            " (n/tab 切换笔刷, l 直线, f 填充, b 画笔, Enter 完成填充, q 退出)"
        )
        ax.set_title(title)
        ax.imshow(display, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.grid(True, color="white", alpha=0.3)

        if mode == "line" and line_points:
            xs, ys = zip(*line_points)
            ax.scatter(xs, ys, c="red", marker="x")
            if len(line_points) == 2:
                ax.plot(xs, ys, color="red", linestyle="--")

        if mode == "fill" and fill_points:
            xs, ys = zip(*fill_points)
            ax.scatter(xs, ys, c="red", marker="o")
            if len(fill_points) > 1:
                closed_xs = list(xs) + [xs[0]]
                closed_ys = list(ys) + [ys[0]]
                ax.plot(closed_xs, closed_ys, color="red", linestyle="--")

        plt.draw()

    def set_mode(new_mode: str) -> None:
        nonlocal mode, drawing
        mode = new_mode
        drawing = False
        line_points.clear()
        fill_points.clear()
        render()

    def onclick(event) -> None:
        if not event.inaxes:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            if mode == "brush":
                grid[y, x] = CELL_TYPES[current_type_names[current_idx]]
                render()
            elif mode == "line":
                line_points.append((x, y))
                if len(line_points) == 2:
                    draw_line_on_grid(
                        grid,
                        line_points[0],
                        line_points[1],
                        CELL_TYPES[current_type_names[current_idx]],
                    )
                    line_points.clear()
                render()
            elif mode == "fill":
                fill_points.append((x, y))
                render()

    def ondrag(event) -> None:
        if not drawing or not event.inaxes or mode != "brush":
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            grid[y, x] = CELL_TYPES[current_type_names[current_idx]]
            render()

    def onkey(event) -> None:
        nonlocal current_idx
        if event.key in {"n", "N", "tab"}:
            current_idx = (current_idx + 1) % len(current_type_names)
            render()
        if event.key in {"l", "L"}:
            set_mode("line")
        if event.key in {"f", "F"}:
            set_mode("fill")
        if event.key in {"b", "B"}:
            set_mode("brush")
        if event.key in {"enter", "return"} and mode == "fill":
            if len(fill_points) < 3:
                print("填充模式至少需要 3 个点，继续点击以添加顶点。")
            else:
                fill_polygon_on_grid(
                    grid, fill_points.copy(), CELL_TYPES[current_type_names[current_idx]]
                )
                fill_points.clear()
                render()
        if event.key in {"q", "escape"}:
            plt.close(fig)

    def onpress(event) -> None:
        nonlocal drawing
        drawing = mode == "brush"
        onclick(event)

    def onrelease(event) -> None:
        nonlocal drawing
        drawing = False

    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect("button_press_event", onpress)
    rid = fig.canvas.mpl_connect("button_release_event", onrelease)
    mid = fig.canvas.mpl_connect("motion_notify_event", ondrag)
    kid = fig.canvas.mpl_connect("key_press_event", onkey)
    render()
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(rid)
    fig.canvas.mpl_disconnect(mid)
    fig.canvas.mpl_disconnect(kid)
    return grid


def main() -> None:
    args = parse_args()
    grid = load_grid(args)
    if args.gui:
        grid = gui_edit(grid)
    else:
        grid = interactive_edit(grid)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("maps") / f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hex"
    save_grid(grid, output_path)


if __name__ == "__main__":
    main()
