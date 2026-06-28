#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
联合绘制多个 JSONL log 中 agent_count 与 step 的关系。

适合续训/断点恢复场景：
- 可以一次传入多个 log；
- 会按 step 合并，并按文件创建时间决定覆盖优先级；
- 如果不同 log 中出现重复 step，创建时间较晚的文件覆盖较早文件；
- 不保存图片，只用 matplotlib 弹窗显示。

用法：
    python plot_agent_count_v1.py run_1_agents.jsonl run_2_agents.jsonl

如果不传参：
    python plot_agent_count_v1.py
然后每行输入一个 log 文件路径，直接回车结束输入并开始绘图。

路径可以带普通引号、中文引号，也可以包含空格。
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import matplotlib.pyplot as plt


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
    """清理用户输入的路径字符串，并转换为 Path。"""
    s = raw.strip()

    if s.lower().startswith("file://"):
        s = s[7:]
        # Windows 上 file:///C:/... 会多一个开头的 /
        if os.name == "nt" and len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]

    s = unquote(s)

    # 去除首尾成对引号，例如："xxx.jsonl"、'xxx.jsonl'、“xxx.jsonl”
    if len(s) >= 2:
        first, last = s[0], s[-1]
        if first in QUOTE_PAIRS and QUOTE_PAIRS[first] == last:
            s = s[1:-1].strip()

    # 兜底处理一层不成对但常见的首尾引号
    s = s.strip('"\'“”‘’')

    return Path(s).expanduser()


def split_input_line_to_paths(raw: str) -> list[Path]:
    """
    交互输入时解析一行。

    推荐每行一个路径；如果一行里输入了多个带引号的路径，也会尽量拆开。
    对于包含空格的单个路径，优先按整行路径处理。
    """
    raw = raw.strip()
    if not raw:
        return []

    whole = normalize_path_input(raw)
    if whole.exists():
        return [whole]

    # 让中文引号也能被 shlex 当成普通英文引号处理。
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
    """
    从命令行参数收集多个路径。

    正常情况下，shell 会把加过引号的路径作为一个参数传入。
    如果用户忘了给带空格的路径加引号，这里会尝试把相邻参数重新拼成存在的文件路径。
    """
    if not argv:
        return ask_log_paths()

    paths: list[Path] = []
    i = 0
    while i < len(argv):
        found: tuple[Path, int] | None = None

        # 尽量从长到短拼接，解决未加引号的空格路径。
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
    """没有命令行参数时，循环 input()，直到输入为空。"""
    paths: list[Path] = []

    print("请输入 log 文件路径，每行一个；直接回车结束输入并开始绘图。")
    print("示例：20260628_213933_agents.jsonl")

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


def read_log(log_path: Path, input_order: int = 0, quiet: bool = False) -> LogData:
    """读取 JSONL 文件中的 step 和 agent_count。"""
    steps: list[int] = []
    counts: list[int] = []
    meta: dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if not quiet:
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
    steps = [x[0] for x in data]
    counts = [x[1] for x in data]

    return LogData(
        path=log_path,
        steps=steps,
        counts=counts,
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
    """
    合并多个 log。

    返回合并后的 steps/counts，以及按创建时间排序后的 log 列表。
    如果存在重复 step，创建时间较晚的 log 覆盖较早文件的数据。
    """
    ordered_logs = sorted(logs, key=chronological_key)

    merged: dict[int, int] = {}
    for log in ordered_logs:
        for step, count in zip(log.steps, log.counts):
            merged[step] = count

    steps = sorted(merged)
    counts = [merged[step] for step in steps]
    return steps, counts, ordered_logs


def calc_y_max(logs: list[LogData], counts: list[int]) -> int:
    """计算 y 轴最大值：优先使用 meta 里的最大容量，兜底用实际最大值。"""
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

    return max(candidates) if candidates else 1


def build_title(logs: list[LogData], steps: list[int]) -> str:
    names = {str(log.meta.get("run_name")) for log in logs if log.meta.get("run_name")}
    if len(names) == 1:
        prefix = next(iter(names))
    else:
        prefix = "simulation"

    if steps:
        return f"Agent Count vs Step - {prefix} | {len(logs)} logs | step {steps[0]}-{steps[-1]}"
    return f"Agent Count vs Step - {prefix} | {len(logs)} logs"


def print_summary(logs: list[LogData], steps: list[int], counts: list[int], y_max: int) -> None:
    print("\n读取到的 log：")
    for i, log in enumerate(logs, start=1):
        if log.steps:
            print(
                f"  {i}. {log.path.name}: "
                f"records={len(log.steps)}, step={log.steps[0]}->{log.steps[-1]}, "
                f"agent_count={min(log.counts)}->{max(log.counts)}, tag={log.tag}"
            )
        else:
            print(f"  {i}. {log.path.name}: 没有有效数据")

    if steps:
        print("\n合并后：")
        print(f"  records={len(steps)}")
        print(f"  step range={steps[0]} -> {steps[-1]}")
        print(f"  agent_count range={min(counts)} -> {max(counts)}")
        print(f"  y max={y_max}")


def plot_agent_count(logs: list[LogData], steps: list[int], counts: list[int], y_max: int) -> None:
    """弹窗绘制合并后的 agent_count - step 曲线，不保存图片。"""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)

    marker = "o" if len(steps) <= 1200 else None
    ax.plot(
        steps,
        counts,
        marker=marker,
        markersize=2.0,
        linewidth=1.2,
        label="merged agent_count",
    )

    # 续训段边界提示。只画第 2 个及之后 log 的起点，避免图例变成小作文。
    boundary_labeled = False
    for log in logs[1:]:
        if log.first_step is None:
            continue
        ax.axvline(
            x=log.first_step,
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
            label="log boundary" if not boundary_labeled else None,
        )
        boundary_labeled = True
        ax.text(
            log.first_step,
            y_max,
            f" {log.tag}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            alpha=0.75,
        )

    ax.axhline(
        y=y_max,
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
        label=f"max agents = {y_max}",
    )

    ax.set_title(build_title(logs, steps))
    ax.set_xlabel("Step")
    ax.set_ylabel("Agent Count")
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()


def main() -> None:
    paths = collect_paths_from_argv(sys.argv[1:])

    if not paths:
        raise SystemExit("没有传入任何 log 文件。")

    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"文件不存在或不是文件：{path}")
        raise FileNotFoundError("存在无效 log 路径，请检查输入。")

    logs = [read_log(path, input_order=i) for i, path in enumerate(paths)]
    logs = [log for log in logs if log.steps]

    if not logs:
        raise RuntimeError("没有读取到有效的 step / agent_count 数据。")

    steps, counts, ordered_logs = combine_logs(logs)
    y_max = calc_y_max(ordered_logs, counts)

    print_summary(ordered_logs, steps, counts, y_max)
    plot_agent_count(ordered_logs, steps, counts, y_max)


if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\n已退出。")
