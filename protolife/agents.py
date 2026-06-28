"""个体状态与批量管理工具。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class AgentState:
    """单个体的基础状态。"""

    position: torch.Tensor  # (2,)
    energy: torch.Tensor
    health: torch.Tensor
    age: torch.Tensor
    mood: torch.Tensor  # 预留情绪向量
    communication_buffer: torch.Tensor  # (2,)
    genome_id: torch.Tensor
    generation: torch.Tensor


class AgentBatch:
    """批量个体管理，便于 GPU 向量化处理。"""

    def __init__(self, num_envs: int, agents_per_env: int, device: torch.device, memory_dim: int = 0, *, use_lstm: bool = False):
        self.num_envs = num_envs
        self.agents_per_env = agents_per_env
        self.device = device
        self.memory_dim = int(memory_dim)
        self.use_lstm = use_lstm
        self.state = self._allocate_state()

    def _allocate_state(self) -> Dict[str, torch.Tensor]:
        shape = (self.num_envs, self.agents_per_env)
        base_state = {
            "x": torch.zeros(shape, dtype=torch.int64, device=self.device),
            "y": torch.zeros(shape, dtype=torch.int64, device=self.device),
            "energy": torch.zeros(shape, dtype=torch.float32, device=self.device),
            "health": torch.zeros(shape, dtype=torch.float32, device=self.device),
            "age": torch.zeros(shape, dtype=torch.int64, device=self.device),
            "mood": torch.zeros(shape + (2,), dtype=torch.float32, device=self.device),
            "comm": torch.zeros(shape + (2,), dtype=torch.float32, device=self.device),
            "genome_id": torch.zeros(shape, dtype=torch.int64, device=self.device),
            "generation": torch.zeros(shape, dtype=torch.int64, device=self.device),
        }
        if self.memory_dim > 0:
            base_state["memory"] = torch.zeros(shape + (self.memory_dim,), dtype=torch.float32, device=self.device)
            if self.use_lstm:
                base_state["memory_cell"] = torch.zeros(shape + (self.memory_dim,), dtype=torch.float32, device=self.device)
        return base_state

    def reset(
        self,
        height: int,
        width: int,
        base_energy: float = 50,
        base_health: float = 100,
        energy_max: float | None = None,
        health_max: float | None = None,
        initial_agents: int | None = None,
    ) -> None:
        """随机初始化个体位置，仅激活 initial_agents 个槽位。"""

        self.state["x"].random_(0, width)
        self.state["y"].random_(0, height)
        energy_value = base_energy if energy_max is None else min(base_energy, energy_max)
        health_value = base_health if health_max is None else min(base_health, health_max)
        self.state["energy"].fill_(energy_value)
        self.state["health"].fill_(health_value)
        active_count = self.agents_per_env if initial_agents is None else int(initial_agents)
        if active_count < 0 or active_count > self.agents_per_env:
            raise ValueError("initial_agents 必须位于 0 与最大个体数之间")
        if active_count < self.agents_per_env:
            self.state["energy"][:, active_count:].zero_()
            self.state["health"][:, active_count:].zero_()
        self.state["age"].zero_()
        self.state["mood"].zero_()
        self.state["comm"].zero_()
        genome_ids = torch.arange(self.agents_per_env, device=self.device, dtype=torch.int64)
        self.state["genome_id"] = genome_ids.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.state["generation"].zero_()
        if "memory" in self.state:
            self.state["memory"].zero_()
        if "memory_cell" in self.state:
            self.state["memory_cell"].zero_()

    def apply_actions(
        self,
        actions: torch.Tensor,
        height: int,
        width: int,
        map_state: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """根据动作更新坐标；未激活槽位不会移动或增长年龄。"""

        actions = actions.view(self.num_envs, self.agents_per_env)
        if active_mask is None:
            active_mask = torch.ones_like(actions, dtype=torch.bool)
        else:
            active_mask = active_mask.view_as(actions).bool()
        dx = torch.zeros_like(actions)
        dy = torch.zeros_like(actions)
        dx = torch.where(actions == 3, -1, dx)  # 左
        dx = torch.where(actions == 4, 1, dx)  # 右
        dy = torch.where(actions == 1, -1, dy)  # 上
        dy = torch.where(actions == 2, 1, dy)  # 下

        proposed_x = self.state["x"] + dx
        proposed_y = self.state["y"] + dy
        clamped_x = proposed_x.clamp(0, width - 1)
        clamped_y = proposed_y.clamp(0, height - 1)
        out_of_bounds = (proposed_x != clamped_x) | (proposed_y != clamped_y)

        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(actions)
        wall_mask = (map_state[env_ids, clamped_y, clamped_x] & 1).bool()
        collision_mask = (wall_mask | out_of_bounds) & active_mask
        blocked = collision_mask | (~active_mask)
        new_x = torch.where(blocked, self.state["x"], clamped_x)
        new_y = torch.where(blocked, self.state["y"], clamped_y)
        moved = ((new_x != self.state["x"]) | (new_y != self.state["y"])) & active_mask

        self.state["x"] = new_x
        self.state["y"] = new_y
        self.state["age"] += active_mask.to(self.state["age"].dtype)
        return {"moved": moved, "collided": collision_mask}

    def export_state(self) -> torch.Tensor:
        """打包状态为 `(E, A, D)` 张量，便于策略读取。"""

        return torch.stack(
            [
                self.state["x"].float(),
                self.state["y"].float(),
                self.state["energy"],
                self.state["health"],
                self.state["age"].float(),
            ],
            dim=-1,
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """导出完整内部状态，方便 checkpoint。"""

        return {k: v.detach().cpu() for k, v in self.state.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], height: int, width: int) -> None:
        """恢复状态；容量变化时自动截断或用空槽位补齐。"""

        fresh_state = self._allocate_state()
        for key, target in fresh_state.items():
            source = state_dict.get(key)
            if source is None:
                continue
            source = source.to(self.device)
            if source.ndim < 2 or source.size(0) != self.num_envs:
                continue
            if source.shape[2:] != target.shape[2:]:
                continue
            copy_agents = min(source.size(1), self.agents_per_env)
            target[:, :copy_agents].copy_(source[:, :copy_agents])
        self.state = fresh_state
        self.state["x"].clamp_(0, width - 1)
        self.state["y"].clamp_(0, height - 1)
