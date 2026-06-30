#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""绘制一个或多个 JSONL 日志中的 agent_count。

每个输入文件始终单独绘制；重叠 step 不会互相覆盖。文件增长时图表会自动
刷新，在视图位于最右端时继续跟随最新 step。滚轮缩放，双击或按 R 复位。
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
}


@dataclass
class LogData:
    path: Path
    steps: list[int]
    counts: list[int]
    meta: dict[str, Any]
    created_at_ns: int = 0
    input_order: int = 0

    @property
    def first_step(self) -> int | None:
        return self.steps[0] if self.steps else None

    @property
    def last_step(self) -> int | None:
        return self.steps[-1] if self.steps else None

    @property
    def tag(self) -> str:
        tag = self.meta.get("run_tag")
        return str(tag) if tag else self.path.stem


def normalize_path_input(raw: str) -> Path:
    """清理命令行或交互输入的文件路径。"""

    value = raw.strip()
    if value.lower().startswith("file://"):
        value = value[7:]
        if os.name == "nt" and len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    value = unquote(value)

    if len(value) >= 2 and QUOTE_PAIRS.get(value[0]) == value[-1]:
        value = value[1:-1].strip()
    value = value.strip('"\'“”‘’「」『』')
    return Path(value).expanduser()


def split_input_line_to_paths(raw: str) -> list[Path]:
    raw = raw.strip()
    if not raw:
        return []

    whole = normalize_path_input(raw)
    if whole.exists():
        return [whole]

    translated = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        parts = shlex.split(translated, posix=False)
    except ValueError:
        return [whole]
    return [normalize_path_input(part) for part in parts] if len(parts) > 1 else [whole]


def collect_paths_from_argv(argv: list[str]) -> list[Path]:
    if not argv:
        return ask_log_paths()

    paths: list[Path] = []
    index = 0
    while index < len(argv):
        found: tuple[Path, int] | None = None
        for end in range(len(argv), index, -1):
            candidate = normalize_path_input(" ".join(argv[index:end]))
            if candidate.is_file():
                found = candidate, end
                break
        if found is None:
            paths.append(normalize_path_input(argv[index]))
            index += 1
        else:
            paths.append(found[0])
            index = found[1]
    return paths


def ask_log_paths() -> list[Path]:
    paths: list[Path] = []
    print("请输入 log 文件路径，每行一个；直接回车结束输入并开始绘图。")
    while True:
        raw = input(f"log #{len(paths) + 1}> ")
        if not raw.strip():
            if paths:
                return paths
            raise SystemExit("没有输入任何 log 文件，已退出。")
        for path in split_input_line_to_paths(raw):
            if path.is_file():
                paths.append(path)
                print(f"  已加入：{path}")
            else:
                print(f"  文件不存在或不是文件：{path}")


def file_creation_time_ns(path: Path) -> int:
    stat = path.stat()
    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime is not None:
        return int(birthtime * 1_000_000_000)
    return stat.st_ctime_ns


def read_log(log_path: Path, input_order: int = 0, quiet: bool = False) -> LogData:
    steps: list[int] = []
    counts: list[int] = []
    meta: dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if not quiet:
                    print(f"{log_path.name}: 跳过第 {line_no} 行，JSON 解析失败：{exc}")
                continue

            if isinstance(obj.get("meta"), dict):
                meta.update(obj["meta"])
                continue
            if "step" not in obj:
                continue
            count = obj.get("agent_count")
            if count is None and isinstance(obj.get("agents"), list):
                count = len(obj["agents"])
            if count is None:
                continue
            try:
                step = int(obj["step"])
                agent_count = int(count)
            except (TypeError, ValueError):
                if not quiet:
                    print(f"{log_path.name}: 跳过第 {line_no} 行，step 或 agent_count 无效。")
                continue
            # 坐标从 0 开始；负 step 没有可解释的训练含义，因此不绘制。
            if step >= 0:
                steps.append(step)
                counts.append(agent_count)

    ordered = sorted(zip(steps, counts), key=lambda item: item[0])
    return LogData(
        path=log_path,
        steps=[item[0] for item in ordered],
        counts=[item[1] for item in ordered],
        meta=meta,
        created_at_ns=file_creation_time_ns(log_path),
        input_order=input_order,
    )


def chronological_key(log: LogData) -> tuple[int, int]:
    return log.created_at_ns, log.input_order


def combine_logs(logs: list[LogData]) -> tuple[list[int], list[int], list[LogData]]:
    """汇总所有点且保留重叠 step，不再进行新文件覆盖旧文件。"""

    ordered_logs = sorted(logs, key=lambda log: log.input_order)
    points = sorted(
        (
            (step, log.input_order, point_order, count)
            for log in ordered_logs
            for point_order, (step, count) in enumerate(zip(log.steps, log.counts))
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    return [item[0] for item in points], [item[3] for item in points], ordered_logs


def calc_y_max(logs: list[LogData], counts: list[int] | None = None) -> int:
    """返回显示上界：实际最大个体数 + 2。"""

    values = list(counts or [])
    if not values:
        values = [count for log in logs for count in log.counts]
    return max(max(values, default=0), 0) + 2


def build_title(logs: list[LogData]) -> str:
    names = {str(log.meta.get("run_name")) for log in logs if log.meta.get("run_name")}
    prefix = next(iter(names)) if len(names) == 1 else "simulation"
    latest = max((log.last_step or 0) for log in logs)
    return f"Agent Count vs Step - {prefix} | {len(logs)} logs | latest step {latest}"


def print_summary(logs: list[LogData]) -> None:
    print("\n读取到的 log（每个文件独立绘制）：")
    for index, log in enumerate(logs, start=1):
        print(
            f"  {index}. {log.path.name}: records={len(log.steps)}, "
            f"step={log.steps[0]}->{log.steps[-1]}, "
            f"agent_count={min(log.counts)}->{max(log.counts)}, tag={log.tag}"
        )


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def view_reaches_latest(xlim: tuple[float, float], latest_step: int) -> bool:
    left, right = xlim
    width = max(right - left, 1.0)
    return right >= latest_step - max(width * 0.01, 1.0)


def power_of_two_tick_interval(xlim: tuple[float, float], target_ticks: int = 8) -> int:
    """为当前可见宽度选择 1、2、4、8……形式的主刻度间隔。"""

    width = max(float(xlim[1] - xlim[0]), 1.0)
    rough_interval = max(width / max(target_ticks, 1), 1.0)
    exponent = int(round(math.log2(rough_interval)))
    return 1 << max(exponent, 0)


def _relative_luminance(color: str | tuple[float, ...]) -> float:
    channels = mcolors.to_rgb(color)
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def color_contrast_ratio(first: str | tuple[float, ...], second: str | tuple[float, ...]) -> float:
    light, dark = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def contrasting_colors(background: str | tuple[float, ...], count: int) -> list[str]:
    """返回与背景对比度至少 3:1 的高区分度曲线颜色。"""

    light_background = _relative_luminance(background) >= 0.35
    palette = (
        ["#0072B2", "#D55E00", "#008A67", "#B32E6E", "#6F4E7C", "#202020", "#8C4A35", "#006D77"]
        if light_background
        else ["#56B4E9", "#E69F00", "#00CC96", "#FF6692", "#C19CFF", "#FFFFFF", "#FFA15A", "#19D3F3"]
    )
    usable = [color for color in palette if color_contrast_ratio(color, background) >= 3.0]
    candidate_index = 0
    while len(usable) < count:
        hue = (candidate_index * 0.61803398875 + 0.11) % 1.0
        saturation = 0.82 if light_background else 0.68
        value = 0.60 if light_background else 1.0
        rgb = tuple(mcolors.hsv_to_rgb((hue, saturation, value)))
        anchor = (0.0, 0.0, 0.0) if light_background else (1.0, 1.0, 1.0)
        blend = 0.0
        while color_contrast_ratio(rgb, background) < 3.0 and blend < 0.9:
            blend += 0.08
            rgb = tuple((1.0 - blend) * channel + blend * target for channel, target in zip(rgb, anchor))
        color = mcolors.to_hex(rgb)
        if color not in usable:
            usable.append(color)
        candidate_index += 1
    return usable[:count]


def plot_agent_count(
    logs: list[LogData],
    steps: list[int] | None = None,
    counts: list[int] | None = None,
    y_max: int | None = None,
):
    """绘图并监控日志增长；返回 ``(figure, axes)`` 便于自动化测试。"""

    del steps, counts, y_max  # 旧函数签名兼容；曲线不再合并。
    logs = sorted(logs, key=lambda log: log.input_order)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    colors = contrasting_colors(ax.get_facecolor(), len(logs))
    lines: dict[Path, Any] = {}

    for log, color in zip(logs, colors):
        marker = "o" if len(log.steps) <= 1200 else None
        (line,) = ax.plot(
            log.steps,
            log.counts,
            color=color,
            marker=marker,
            markersize=2.0,
            linewidth=1.35,
            label=f"{log.tag} ({log.path.name})",
        )
        lines[log.path] = line

    latest_step = max((log.last_step or 0) for log in logs)
    display_y_max = calc_y_max(logs)
    ax.set_title(build_title(logs))
    ax.set_xlabel("Step")
    ax.set_ylabel("Agent Count")
    ax.set_xlim(0, max(float(latest_step), 1.0))
    ax.set_ylim(0, display_y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()

    state: dict[str, Any] = {
        "logs": logs,
        "lines": lines,
        "latest_step": latest_step,
        "follow_latest": True,
        "adjusting": False,
        "signatures": {log.path: _file_signature(log.path) for log in logs},
    }

    def update_ticks() -> None:
        ax.xaxis.set_major_locator(MultipleLocator(power_of_two_tick_interval(ax.get_xlim())))

    def reset_view() -> None:
        latest = max(float(state["latest_step"]), 1.0)
        state["adjusting"] = True
        ax.set_xlim(0, latest)
        state["adjusting"] = False
        state["follow_latest"] = True
        update_ticks()
        fig.canvas.draw_idle()

    def on_xlim_changed(_axes) -> None:
        if state["adjusting"]:
            return
        left, right = ax.get_xlim()
        width = max(right - left, 1.0)
        if left < 0:
            state["adjusting"] = True
            ax.set_xlim(0, max(width, 1.0))
            state["adjusting"] = False
        state["follow_latest"] = view_reaches_latest(ax.get_xlim(), state["latest_step"])
        update_ticks()

    def on_scroll(event) -> None:
        if event.inaxes is not ax:
            return
        left, right = ax.get_xlim()
        width = max(right - left, 1.0)
        scale = 0.8 if event.button == "up" else 1.25
        full_width = max(float(state["latest_step"]), 1.0)
        new_width = min(max(width * scale, 1.0), full_width)
        center = event.xdata if event.xdata is not None else right
        ratio = min(max((center - left) / width, 0.0), 1.0)
        new_left = center - new_width * ratio
        new_right = new_left + new_width
        if new_right > full_width:
            new_right = full_width
            new_left = new_right - new_width
        if new_left < 0:
            new_left = 0.0
            new_right = new_width
        state["adjusting"] = True
        ax.set_xlim(new_left, max(new_right, new_left + 1.0))
        state["adjusting"] = False
        state["follow_latest"] = view_reaches_latest(ax.get_xlim(), state["latest_step"])
        update_ticks()
        fig.canvas.draw_idle()

    def on_button_press(event) -> None:
        if event.inaxes is ax and getattr(event, "dblclick", False):
            reset_view()

    def on_key_press(event) -> None:
        if str(getattr(event, "key", "")).lower() == "r":
            reset_view()

    def refresh() -> None:
        signatures = {log.path: _file_signature(log.path) for log in state["logs"]}
        if signatures == state["signatures"]:
            return

        refreshed: list[LogData] = []
        for old_log in state["logs"]:
            if old_log.path.is_file():
                refreshed.append(read_log(old_log.path, old_log.input_order, quiet=True))
        refreshed = [log for log in refreshed if log.steps]
        if not refreshed:
            return

        old_left, old_right = ax.get_xlim()
        old_width = max(old_right - old_left, 1.0)
        new_latest = max((log.last_step or 0) for log in refreshed)
        new_colors = contrasting_colors(ax.get_facecolor(), len(refreshed))

        state["adjusting"] = True
        for log, color in zip(refreshed, new_colors):
            line = state["lines"].get(log.path)
            if line is None:
                (line,) = ax.plot([], [], linewidth=1.35)
                state["lines"][log.path] = line
            line.set_data(log.steps, log.counts)
            line.set_color(color)
            line.set_marker("o" if len(log.steps) <= 1200 else None)
            line.set_label(f"{log.tag} ({log.path.name})")

        ax.set_ylim(0, calc_y_max(refreshed))
        ax.set_title(build_title(refreshed))
        if state["follow_latest"]:
            right = max(float(new_latest), 1.0)
            ax.set_xlim(max(0.0, right - old_width), right)
        else:
            ax.set_xlim(max(0.0, old_left), max(old_right, old_left + 1.0))
        ax.legend()
        state["adjusting"] = False

        state["logs"] = refreshed
        state["latest_step"] = new_latest
        state["signatures"] = signatures
        update_ticks()
        fig.canvas.draw_idle()

    ax.callbacks.connect("xlim_changed", on_xlim_changed)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(refresh)
    timer.start()
    fig._agent_count_refresh_timer = timer
    fig.canvas.mpl_connect("close_event", lambda _event: timer.stop())

    update_ticks()
    fig.tight_layout()
    plt.show()
    return fig, ax


def main() -> None:
    paths = collect_paths_from_argv(sys.argv[1:])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"文件不存在或不是文件：{path}")
        raise FileNotFoundError("存在无效 log 路径，请检查输入。")

    logs = [read_log(path, input_order=index) for index, path in enumerate(paths)]
    logs = [log for log in logs if log.steps]
    if not logs:
        raise RuntimeError("没有读取到有效的 step / agent_count 数据。")

    print_summary(logs)
    plot_agent_count(logs)


if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\n已退出。")
            break
