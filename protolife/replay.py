"""回放与可视化占位实现。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple


class ReplayReader:
    """逐条读取日志，供可视化或分析使用。"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.map_log = self.log_dir / "map.log"
        self.agent_log = self.log_dir / "agents.jsonl"

    def iter_maps(self) -> Iterator[str]:
        with self.map_log.open("r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    def iter_agents(self) -> Iterator[str]:
        with self.agent_log.open("r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()
