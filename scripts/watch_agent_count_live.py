#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时监控一个或多个 JSONL log 文件，并刷新显示 agent_count 与 step 的关系。

特点：
- 支持多个续训 log 合并显示，重叠 step 采用创建时间较晚的文件；
- 文件 size/mtime 变化后自动重读并刷新；
- X 轴窗口最多显示 X_WINDOW_STEPS 个 step；
- 默认跟随最新 step；
- 鼠标左键横向拖动可查看历史窗口；
- 滚轮可缩放 X 轴窗口，但不会超过 X_WINDOW_STEPS；
- 按 f 或 r 回到“跟随最新”模式；
- y 轴最大值使用 log meta 中的最大个体容量，兜底使用实际最大 agent_count。

用法：
    python watch_agent_count_live.py current_agents.jsonl
    python watch_agent_count_live.py old_agents.jsonl current_agents.jsonl

如果不传参：
    python watch_agent_count_live.py
然后每行输入一个 log 文件路径，直接回车结束输入并开始监控。
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import matplotlib.pyplot as plt


# ====== 常用可调参数 ======
X_WINDOW_STEPS = 2048          # X 轴最多显示多少 step
REFRESH_INTERVAL_SEC = 0.5     # 刷新间隔，单位：秒
MIN_Y_MAX = 1                  # 防止 y 轴最大值为 0
# ==========================


QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "《": "》",
    "【": "】",
}


@dataclass
class LogData:
    path: Path
    steps: list[int] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
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


@dataclass
class CachedLog:
    path: Path
    input_order: int
    signature: tuple[int, int] | None = None
    data: LogData = field(default_factory=lambda: LogData(path=Path("<empty>")))
    last_error: str | None = None

    def refresh_if_needed(self) -> bool:
        """文件变化时重读。返回 True 表示数据发生过刷新。"""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.last_error = "file not found"
            self.signature = None
            self.data = LogData(path=self.path, input_order=self.input_order)
            return True

        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == self.signature:
            return False

        try:
            self.data = read_log(self.path, input_order=self.input_order, quiet=True)
            self.signature = signature
            self.last_error = None
            return True
        except Exception as exc:  # 实时读文件时，宁可不炸窗
            self.last_error = str(exc)
            return False


def normalize_path_input(raw: str) -> Path:
    """清理用户输入的路径字符串，并转换为 Path。"""
    s = raw.strip()

    if s.lower().startswith("file://"):
        s = s[7:]
        if os.name == "nt" and len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]

    s = unquote(s)

    if len(s) >= 2:
        first, last = s[0], s[-1]
        if first in QUOTE_PAIRS and QUOTE_PAIRS[first] == last:
            s = s[1:-1].strip()

    s = s.strip('"\'“”‘’')
    return Path(s).expanduser()


def split_input_line_to_paths(raw: str) -> list[Path]:
    raw = raw.strip()
    if not raw:
        return []

    whole = normalize_path_input(raw)
    if whole.exists():
        return [whole]

    translated = (
        raw.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    try:
        parts = shlex.split(translated, posix=False)
    except ValueError:
        return [whole]

    if len(parts) <= 1:
        return [whole]

    return [normalize_path_input(part) for part in parts]


def collect_paths_from_argv(argv: list[str]) -> list[Path]:
    if not argv:
        return ask_log_paths()

    paths: list[Path] = []
    i = 0
    while i < len(argv):
        found: tuple[Path, int] | None = None

        for j in range(len(argv), i, -1):
            candidate = normalize_path_input(" ".join(argv[i:j]))
            if candidate.is_file():
                found = (candidate, j)
                break

        if found is not None:
            path, next_i = found
            paths.append(path)
            i = next_i
        else:
            paths.append(normalize_path_input(argv[i]))
            i += 1

    return paths


def ask_log_paths() -> list[Path]:
    paths: list[Path] = []

    print("请输入要监控的 log 文件路径，每行一个；直接回车开始监控。")
    print("提示：可以先输入历史 log，再输入当前正在写入的 log。")

    while True:
        raw = input(f"log #{len(paths) + 1}> ")
        if not raw.strip():
            if paths:
                return paths
            raise SystemExit("没有输入任何 log 文件，已退出。")

        for path in split_input_line_to_paths(raw):
            # 实时监控允许文件暂时不存在，但存在时必须是文件。
            if path.exists() and not path.is_file():
                print(f"  路径存在但不是文件：{path}")
                continue

            paths.append(path)
            if path.exists():
                print(f"  已加入：{path}")
            else:
                print(f"  已加入，但当前文件尚不存在，将继续等待：{path}")


def read_log(log_path: Path, input_order: int = 0, quiet: bool = False) -> LogData:
    steps: list[int] = []
    counts: list[int] = []
    meta: dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            # 正在写入时，最后一行偶尔会半截；静默跳过，避免图表炸毛。
            if not quiet and line_no != len(lines):
                print(f"{log_path.name}: 跳过第 {line_no} 行，JSON 解析失败：{exc}")
            continue

        if "meta" in obj and isinstance(obj["meta"], dict):
            meta.update(obj["meta"])
            continue

        if "step" not in obj:
            continue

        if "agent_count" in obj:
            count = obj["agent_count"]
        elif isinstance(obj.get("agents"), list):
            count = len(obj["agents"])
        else:
            continue

        try:
            steps.append(int(obj["step"]))
            counts.append(int(count))
        except (TypeError, ValueError):
            if not quiet:
                print(f"{log_path.name}: 跳过第 {line_no} 行，step 或 agent_count 不是有效整数。")

    data = sorted(zip(steps, counts), key=lambda x: x[0])
    return LogData(
        path=log_path,
        steps=[x[0] for x in data],
        counts=[x[1] for x in data],
        meta=meta,
        created_at_ns=file_creation_time_ns(log_path),
        input_order=input_order,
    )


def file_creation_time_ns(path: Path) -> int:
    """返回文件创建时间；不支持 birth time 的系统回退到 st_ctime。"""

    stat = path.stat()
    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime is not None:
        return int(birthtime * 1_000_000_000)
    return stat.st_ctime_ns


def chronological_key(log: LogData) -> tuple[int, int]:
    """旧文件先合并、新文件后合并，使重叠 step 采用较晚创建的文件。"""

    return log.created_at_ns, log.input_order


def combine_logs(logs: list[LogData]) -> tuple[list[int], list[int], list[LogData]]:
    ordered_logs = sorted([log for log in logs if log.steps], key=chronological_key)

    merged: dict[int, int] = {}
    for log in ordered_logs:
        for step, count in zip(log.steps, log.counts):
            merged[step] = count

    steps = sorted(merged)
    counts = [merged[step] for step in steps]
    return steps, counts, ordered_logs


def calc_y_max(logs: list[LogData], counts: list[int]) -> int:
    candidates: list[int] = []

    for log in logs:
        meta = log.meta
        num_envs = meta.get("num_envs", 1)
        if not isinstance(num_envs, int) or num_envs < 1:
            num_envs = 1

        for key in ("max_agents_per_env", "agents_per_env"):
            value = meta.get(key)
            if isinstance(value, (int, float)):
                candidates.append(int(value * num_envs))

    if counts:
        candidates.append(max(counts))

    return max(max(candidates) if candidates else MIN_Y_MAX, MIN_Y_MAX)


def latest_xlim(steps: list[int]) -> tuple[float, float]:
    if not steps:
        return 0.0, float(X_WINDOW_STEPS)

    right = float(steps[-1])
    left = max(float(steps[0]), right - float(X_WINDOW_STEPS))

    # 当数据跨度很小，给一点可视宽度，不然刚启动时像心电图仪没插电。
    if right <= left:
        right = left + 1.0

    return left, right


def clamp_xlim(left: float, right: float, steps: list[int]) -> tuple[float, float]:
    """限制 X 轴窗口宽度不超过 X_WINDOW_STEPS，并尽量夹在数据范围内。"""
    if not steps:
        width = min(max(right - left, 1.0), float(X_WINDOW_STEPS))
        return left, left + width

    data_min = float(steps[0])
    data_max = float(steps[-1])
    width = min(max(right - left, 1.0), float(X_WINDOW_STEPS))

    left = float(left)
    right = left + width

    if right > data_max:
        right = data_max
        left = right - width

    if left < data_min:
        left = data_min
        right = left + width

    # 如果数据本身跨度小于窗口，可以显示一点右侧空白，但仍不超过 X_WINDOW_STEPS。
    if right <= left:
        right = left + 1.0

    return left, right


def build_title(logs: list[LogData], steps: list[int]) -> str:
    names = {str(log.meta.get("run_name")) for log in logs if log.meta.get("run_name")}
    prefix = next(iter(names)) if len(names) == 1 else "simulation"

    if steps:
        return f"Live Agent Count - {prefix} | {len(logs)} logs | latest step {steps[-1]}"
    return f"Live Agent Count - {prefix} | waiting for data"


def main() -> None:
    paths = collect_paths_from_argv(sys.argv[1:])
    if not paths:
        raise SystemExit("没有传入任何 log 文件。")

    caches = [CachedLog(path=path, input_order=i) for i, path in enumerate(paths)]

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    marker = None
    (line,) = ax.plot([], [], marker=marker, linewidth=1.2, label="merged agent_count")
    max_line = ax.axhline(y=MIN_Y_MAX, linestyle=":", linewidth=1.0, alpha=0.75, label="max agents")
    status_text = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    ax.set_xlabel("Step")
    ax.set_ylabel("Agent Count")
    ax.set_xlim(0, X_WINDOW_STEPS)
    ax.set_ylim(0, MIN_Y_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    state = {
        "follow_latest": True,
        "dragging": False,
        "press_x": None,
        "orig_xlim": None,
        "steps": [],
    }

    def on_press(event):
        if event.inaxes != ax or event.button != 1 or event.xdata is None:
            return
        state["dragging"] = True
        state["press_x"] = float(event.xdata)
        state["orig_xlim"] = ax.get_xlim()
        state["follow_latest"] = False

    def on_motion(event):
        if not state["dragging"] or event.inaxes != ax or event.xdata is None:
            return
        press_x = state["press_x"]
        orig_xlim = state["orig_xlim"]
        if press_x is None or orig_xlim is None:
            return

        dx = press_x - float(event.xdata)
        left = float(orig_xlim[0]) + dx
        right = float(orig_xlim[1]) + dx
        left, right = clamp_xlim(left, right, state["steps"])
        ax.set_xlim(left, right)
        fig.canvas.draw_idle()

    def on_release(event):
        state["dragging"] = False
        state["press_x"] = None
        state["orig_xlim"] = None

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None:
            return

        state["follow_latest"] = False
        left, right = ax.get_xlim()
        width = right - left

        if event.button == "up":
            new_width = max(1.0, width * 0.8)
        else:
            new_width = min(float(X_WINDOW_STEPS), width / 0.8)

        center = float(event.xdata)
        ratio = (center - left) / max(width, 1e-9)
        new_left = center - new_width * ratio
        new_right = new_left + new_width
        new_left, new_right = clamp_xlim(new_left, new_right, state["steps"])
        ax.set_xlim(new_left, new_right)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in {"f", "r"}:
            state["follow_latest"] = True
            left, right = latest_xlim(state["steps"])
            ax.set_xlim(left, right)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("开始实时监控：")
    for path in paths:
        print(f"  {path}")
    print(f"X_WINDOW_STEPS = {X_WINDOW_STEPS}")
    print("操作：鼠标左键横向拖动查看历史；滚轮缩放；按 f 或 r 回到跟随最新。")

    last_print_time = 0.0

    while plt.fignum_exists(fig.number):
        changed = False
        for cache in caches:
            changed = cache.refresh_if_needed() or changed

        logs = [cache.data for cache in caches if cache.data.steps]
        steps, counts, ordered_logs = combine_logs(logs)
        state["steps"] = steps

        if changed or steps:
            y_max = calc_y_max(ordered_logs, counts)
            line.set_data(steps, counts)
            max_line.set_ydata([y_max, y_max])
            max_line.set_label(f"max agents = {y_max}")

            ax.set_ylim(0, y_max)
            ax.set_title(build_title(ordered_logs, steps))

            if state["follow_latest"]:
                ax.set_xlim(*latest_xlim(steps))
            else:
                left, right = ax.get_xlim()
                ax.set_xlim(*clamp_xlim(left, right, steps))

            if steps:
                status = (
                    f"records={len(steps)} | step={steps[0]}->{steps[-1]} | "
                    f"agent_count={min(counts)}->{max(counts)} | "
                    f"mode={'follow latest' if state['follow_latest'] else 'manual'}"
                )
            else:
                status = "waiting for valid data..."

            errors = [f"{c.path.name}: {c.last_error}" for c in caches if c.last_error]
            if errors:
                status += "\n" + " | ".join(errors[:2])
                if len(errors) > 2:
                    status += f" | +{len(errors) - 2} errors"

            status_text.set_text(status)
            ax.legend(loc="upper right")
            fig.canvas.draw_idle()

            now = time.time()
            if steps and now - last_print_time >= 2.0:
                print(
                    f"step {steps[-1]} | records {len(steps)} | "
                    f"agent_count {counts[-1]} | mode {'follow' if state['follow_latest'] else 'manual'}"
                )
                last_print_time = now

        plt.pause(REFRESH_INTERVAL_SEC)

    plt.ioff()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
