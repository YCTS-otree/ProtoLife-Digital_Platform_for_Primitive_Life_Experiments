"""实验记录与压缩日志工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .encoding import encode_grid


class ExperimentLogger:
    """负责定期保存地图与个体摘要，便于回放。"""

    def __init__(self, save_dir: str, snapshot_interval: int = 50, env_index: int = 0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.env_index = env_index
        self.map_log = self.save_dir / "map.log"
        self.agent_log = self.save_dir / "agents.jsonl"
        self.step_counter = 0

    def maybe_log(self, map_state: torch.Tensor, agent_state: torch.Tensor) -> None:
        """按照间隔写入快照。"""

        if self.step_counter % self.snapshot_interval == 0:
            self._log_map(map_state)
            self._log_agents(agent_state)
        self.step_counter += 1

    def _log_map(self, map_state: torch.Tensor) -> None:
        encoded = encode_grid(map_state[self.env_index : self.env_index + 1])
        with self.map_log.open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")

    def _log_agents(self, agent_state: torch.Tensor) -> None:
        record = {
            "step": self.step_counter,
            "agents": agent_state[self.env_index].cpu().tolist(),
        }
        with self.agent_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
