"""实验记录与压缩日志工具。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .encoding import encode_grid


class ExperimentLogger:
    """负责定期保存地图与个体摘要，便于回放。"""

    def __init__(
        self,
        save_dir: str,
        snapshot_interval: int = 50,
        env_index: int = 0,
        run_tag: Optional[str] = None,
        metadata: Optional[Dict] = None,
        buffer_on_gpu: bool = False,
        flush_interval: int = 8,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.env_index = env_index
        self.run_tag = run_tag
        self.metadata = metadata or {}
        self.buffer_on_gpu = buffer_on_gpu
        self.flush_interval = max(int(flush_interval), 1)
        self._gpu_buffer: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        tag = self.run_tag or time.strftime("%Y%m%d_%H%M%S")
        self.map_log = self.save_dir / f"{tag}_map.log"
        self.agent_log = self.save_dir / f"{tag}_agents.jsonl"
        self.step_counter = 0
        self._write_header()

    def maybe_log(self, map_state: torch.Tensor, agent_state: torch.Tensor) -> None:
        """按照间隔写入快照。"""

        if self.step_counter % self.snapshot_interval == 0:
            map_snapshot = map_state[self.env_index : self.env_index + 1].detach().clone()
            agent_snapshot = agent_state[self.env_index : self.env_index + 1].detach().clone()
            if self.buffer_on_gpu:
                self._gpu_buffer.append((map_snapshot, agent_snapshot, self.step_counter))
                if len(self._gpu_buffer) >= self.flush_interval:
                    self._flush_buffer()
            else:
                self._write_snapshot(map_snapshot.cpu(), agent_snapshot.cpu(), self.step_counter)
        self.step_counter += 1

    def _write_snapshot(self, map_state: torch.Tensor, agent_state: torch.Tensor, step: int) -> None:
        encoded = encode_grid(map_state)
        with self.map_log.open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")

        record = {
            "step": step,
            "agents": agent_state[0].cpu().tolist(),
        }
        with self.agent_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _flush_buffer(self) -> None:
        if not self._gpu_buffer:
            return

        map_tensors, agent_tensors, steps = zip(*self._gpu_buffer)
        map_stack = torch.cat(map_tensors, dim=0).cpu()
        agent_stack = torch.cat(agent_tensors, dim=0).cpu()
        for idx, step in enumerate(steps):
            self._write_snapshot(map_stack[idx : idx + 1], agent_stack[idx : idx + 1], step)
        self._gpu_buffer.clear()

    def flush(self) -> None:
        """确保缓冲的 GPU 快照被写入磁盘。"""

        self._flush_buffer()

    def _write_header(self) -> None:
        header = {"meta": self.metadata}
        with self.map_log.open("w", encoding="utf-8") as map_f:
            map_f.write(json.dumps(header, ensure_ascii=False) + "\n")
        with self.agent_log.open("w", encoding="utf-8") as agent_f:
            agent_f.write(json.dumps(header, ensure_ascii=False) + "\n")
