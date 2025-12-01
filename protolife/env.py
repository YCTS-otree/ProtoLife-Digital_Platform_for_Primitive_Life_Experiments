"""环境与物理规则定义。

本文件聚合地图状态、个体交互、能量代谢等硬规则，保证智能决策全部交给策略网络学习。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .agents import AgentBatch
from .encoding import encode_grid
from .rewards import build_action_reward_table


@dataclass
class EnvStepResult:
    """单步环境返回的结果容器。"""

    observations: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    dones: torch.Tensor
    infos: List[Dict]


class ProtoLifeEnv:
    """二维网格环境。

    这里只提供最小原型结构：
    - `reset` 生成随机地图与初始个体。
    - `step` 按动作更新物理规则、计算奖励并返回张量化观测。
    - 为便于 GPU 并行，内部状态倾向使用 batched tensor。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = config.get("world", {}).get("height", 64)
        self.width = config.get("world", {}).get("width", 64)
        self.action_rewards = build_action_reward_table(config.get("action_rewards", {}))
        self.agent_batch = AgentBatch(
            num_envs=config.get("training", {}).get("num_envs", 1),
            agents_per_env=config.get("agents", {}).get("per_env", 1),
            device=self.device,
        )
        self.map_state = torch.zeros(
            (self.agent_batch.num_envs, self.height, self.width), dtype=torch.int64, device=self.device
        )

    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境，返回初始观测。

        这里使用均匀随机初始化地图，个体状态由 `AgentBatch` 自行重置。
        实际实验中可替换为带食物/毒素分布的初始化逻辑。
        """

        self.map_state.zero_()
        self.agent_batch.reset(self.height, self.width)
        return self._build_observations()

    def step(self, actions: torch.Tensor) -> EnvStepResult:
        """执行一批动作并推进环境一帧。

        当前实现仅提供占位流程：
        1. 根据动作更新个体位置（不越界）。
        2. 按动作表发放基础奖励。
        3. 返回观测、奖励、终止标志与信息。
        后续可逐步填充能量代谢、食物/毒素交互、战斗等细节。
        """

        self.agent_batch.apply_actions(actions, self.height, self.width)
        base_rewards = self.action_rewards[actions]
        dones = torch.zeros_like(base_rewards, dtype=torch.bool)
        infos: List[Dict] = [
            {"message": "占位环境，无终止条件"} for _ in range(self.agent_batch.num_envs * self.agent_batch.agents_per_env)
        ]
        observations = self._build_observations()
        return EnvStepResult(observations=observations, rewards=base_rewards, dones=dones, infos=infos)

    def _build_observations(self) -> Dict[str, torch.Tensor]:
        """构建供策略网络使用的张量观测。"""

        return {
            "map": self.map_state.clone(),
            "agents": self.agent_batch.export_state(),
        }

    def render(self) -> None:
        """调试可视化钩子，初版占位。"""

        encoded = encode_grid(self.map_state)
        print(f"Encoded map snapshot: {encoded[:16]} ...")
