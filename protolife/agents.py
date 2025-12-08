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

    def __init__(self, num_envs: int, agents_per_env: int, device: torch.device):
        self.num_envs = num_envs
        self.agents_per_env = agents_per_env
        self.device = device
        self.state = self._allocate_state()

    def _allocate_state(self) -> Dict[str, torch.Tensor]:
        shape = (self.num_envs, self.agents_per_env)
        return {
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

    def reset(
        self,
        height: int,
        width: int,
        base_energy: float = 50,
        base_health: float = 100,
        energy_max: float | None = None,
        health_max: float | None = None,
    ) -> None:
        """随机初始化个体位置与基本状态。"""

        self.state["x"].random_(0, width)
        self.state["y"].random_(0, height)
        energy_value = base_energy if energy_max is None else min(base_energy, energy_max)
        health_value = base_health if health_max is None else min(base_health, health_max)
        self.state["energy"].fill_(energy_value)
        self.state["health"].fill_(health_value)
        self.state["age"].zero_()
        self.state["mood"].zero_()
        self.state["comm"].zero_()
        self.state["genome_id"].zero_()
        self.state["generation"].zero_()

    def apply_actions(self, actions: torch.Tensor, height: int, width: int, map_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """根据动作更新坐标，仅处理移动并返回移动/碰撞信息。"""

        actions = actions.view(self.num_envs, self.agents_per_env)
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
        collision_mask = wall_mask | out_of_bounds

        new_x = torch.where(collision_mask, self.state["x"], clamped_x)
        new_y = torch.where(collision_mask, self.state["y"], clamped_y)
        moved = (new_x != self.state["x"]) | (new_y != self.state["y"])

        self.state["x"] = new_x
        self.state["y"] = new_y
        self.state["age"] += 1

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
        """从 checkpoint 恢复内部状态，并确保尺寸匹配当前环境。"""

        self.state = {k: v.to(self.device) for k, v in state_dict.items()}
        if self.state["x"].shape != (self.num_envs, self.agents_per_env):
            raise ValueError("载入的个体状态尺寸与当前设置不一致")
        self.state["x"] = self.state["x"].clamp(0, width - 1)
        self.state["y"] = self.state["y"].clamp(0, height - 1)
