"""回放与可视化工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import torch

from .encoding import decode_grid


class ReplayReader:
    """逐条读取日志，供可视化或分析使用。"""

    def __init__(self, map_log: Path, agent_log: Path):
        self.map_log = Path(map_log)
        self.agent_log = Path(agent_log)
        self.metadata = self._read_metadata(self.map_log)
        self.agent_metadata = self._read_metadata(self.agent_log)

    def iter_maps(self) -> Iterator[str]:
        with self.map_log.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                yield line.strip()

    def iter_agents(self) -> Iterator[str]:
        with self.agent_log.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                yield line.strip()

    @staticmethod
    def _read_metadata(path: Path) -> Dict:
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
        except (FileNotFoundError, IndexError):
            return {}
        try:
            parsed = json.loads(first_line)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed.get("meta", parsed)
        return {}


def _extract_expected_shape(metadata: Dict) -> Tuple[int, int]:
    height = metadata.get("height") or metadata.get("world", {}).get("height")
    width = metadata.get("width") or metadata.get("world", {}).get("width")
    if height is None or width is None:
        raise ValueError("日志元数据缺少 height/width，无法自动推断地图尺寸。")
    return int(height), int(width)


def _collect_log_pairs(log_dir: Path) -> List[Tuple[Path, Path]]:
    map_candidates = sorted(log_dir.glob("*_map.log"))
    if not map_candidates and (log_dir / "map.log").exists():
        map_candidates.append(log_dir / "map.log")

    pairs: List[Tuple[Path, Path]] = []
    for map_log in map_candidates:
        if map_log.name.endswith("_map.log"):
            prefix = map_log.name[: -len("_map.log")]
            agent_log = map_log.with_name(f"{prefix}_agents.jsonl")
        else:
            agent_log = map_log.with_name("agents.jsonl")
        if agent_log.exists():
            pairs.append((map_log, agent_log))
    return pairs


def _resolve_log_pairs(target: str) -> List[Tuple[Path, Path]]:
    path = Path(target)
    if path.is_file():
        if path.name.endswith("_map.log"):
            prefix = path.name[: -len("_map.log")]
            agent = path.with_name(f"{prefix}_agents.jsonl")
            if not agent.exists():
                raise FileNotFoundError(f"未找到配对的 agent 日志: {agent}")
            return [(path, agent)]
        if path.name == "map.log":
            agent = path.with_name("agents.jsonl")
            if not agent.exists():
                raise FileNotFoundError(f"未找到配对的 agent 日志: {agent}")
            return [(path, agent)]
    log_dir = path
    if path.is_dir() and (path / "log").is_dir():
        log_dir = path / "log"
    return _collect_log_pairs(log_dir)


def _select_pairs(pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    if len(pairs) <= 1:
        return pairs

    print("发现多个日志文件：")
    for idx, (map_log, _) in enumerate(pairs):
        try:
            meta = ReplayReader(map_log, pairs[idx][1]).metadata
            run_name = meta.get("run_name") or meta.get("run_tag") or map_log.name
            map_size = f"{meta.get('height')}x{meta.get('width')}" if meta else "unknown"
        except Exception:
            run_name = map_log.name
            map_size = "unknown"
        print(f"[{idx}] {map_log.name} (size:{map_size}, run:{run_name})")
    choice = input("选择需要回放的序号（回车播放全部）: ").strip()
    if not choice:
        return sorted(pairs, key=lambda p: p[0].stat().st_mtime)
    try:
        idx = int(choice)
        if 0 <= idx < len(pairs):
            return [pairs[idx]]
    except ValueError:
        pass
    print("输入无效，默认播放全部。")
    return sorted(pairs, key=lambda p: p[0].stat().st_mtime)


def _resolve_marker_size(reader: ReplayReader, fallback: float = 10.0) -> float:
    """从日志元数据中解析个体散点尺寸。"""

    for meta in (reader.metadata, reader.agent_metadata):
        if not meta:
            continue
        size = meta.get("agent_marker_size")
        if size is not None:
            try:
                return float(size)
            except (TypeError, ValueError):
                break
    return float(fallback)


def playback(target: str, interval: float = 0.2) -> None:
    """使用 matplotlib 实时回放日志文件，支持目录或模型路径。"""

    pairs = _resolve_log_pairs(target)
    if not pairs:
        raise FileNotFoundError(f"未在 {target} 下找到日志文件")

    pairs = _select_pairs(pairs)
    plt.ion()
    fig, ax = plt.subplots()
    img = None

    for map_log, agent_log in pairs:
        reader = ReplayReader(map_log, agent_log)
        marker_size = _resolve_marker_size(reader)
        try:
            height, width = _extract_expected_shape(reader.metadata)
        except ValueError:
            try:
                first_line = next(reader.iter_maps())
            except StopIteration:
                continue
            inferred_cells = len(first_line) // 2
            width = int(reader.metadata.get("width", inferred_cells**0.5))
            height = int(inferred_cells / max(width, 1))

        expected_bytes = height * width
        map_iter = reader.iter_maps()
        agent_iter = reader.iter_agents()

        for map_line, agent_line in zip(map_iter, agent_iter):
            trimmed_line = map_line[: expected_bytes * 2]
            grid = decode_grid(trimmed_line, torch.Size((1, height, width)))[0]
            agent_data = json.loads(agent_line)
            agent_state = torch.tensor(agent_data["agents"], dtype=torch.float32)
            display = torch.zeros((height, width))
            display = torch.where((grid & 1) > 0, torch.tensor(0.2), display)
            display = torch.where((grid & (1 << 2)) > 0, torch.tensor(0.8), display)
            display = torch.where((grid & (1 << 3)) > 0, torch.tensor(0.5), display)

            if img is None:
                img = ax.imshow(display, cmap="viridis", vmin=0, vmax=1)
            else:
                img.set_data(display)
            for artist in list(ax.collections):
                artist.remove()
            xs = agent_state[..., 0].view(-1)
            ys = agent_state[..., 1].view(-1)
            ax.scatter(xs, ys, c="red", s=marker_size)
            ax.set_title(f"step {agent_data.get('step', 0)} | {map_log.name}")
            plt.pause(interval)

    plt.ioff()
    plt.show()
