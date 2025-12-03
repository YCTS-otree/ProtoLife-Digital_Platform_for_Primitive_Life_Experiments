"""回放与可视化工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import torch

from .encoding import decode_grid


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


def playback(log_dir: str, height: int, width: int, interval: float = 0.2) -> None:
    """使用 matplotlib 实时回放 map.log 与 agents.jsonl。"""

    reader = ReplayReader(log_dir)
    map_iter = reader.iter_maps()
    agent_iter = reader.iter_agents()

    plt.ion()
    fig, ax = plt.subplots()
    img = None

    for map_line, agent_line in zip(map_iter, agent_iter):
        grid = decode_grid(map_line, torch.Size((1, height, width)))[0]
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
        ax.collections.clear()
        xs = agent_state[..., 0].view(-1)
        ys = agent_state[..., 1].view(-1)
        ax.scatter(xs, ys, c="red", s=10)
        ax.set_title(f"step {agent_data.get('step', 0)}")
        plt.pause(interval)

    plt.ioff()
    plt.show()
