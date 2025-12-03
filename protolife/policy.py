"""策略与价值网络占位实现。"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


action_space = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "EAT",
    6: "ATTACK",
    7: "COMMUNICATE",
    8: "BUILD",
    9: "REMOVE",
    10: "REPRODUCE",
}


class MLPPolicy(nn.Module):
    """简化的多层感知机策略，方便后续替换为卷积/注意力结构。"""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, action_dim: int = len(action_space)):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 action logits 与状态价值。"""

        x = self.backbone(obs)
        return self.policy_head(x), self.value_head(x)


def build_policy(config: Dict, obs_dim: int | None = None) -> MLPPolicy:
    """根据配置构建策略网络。"""

    hidden = config.get("model", {}).get("hidden", 128)
    inferred_dim = obs_dim or config.get("model", {}).get("obs_dim", 5)
    return MLPPolicy(obs_dim=inferred_dim, hidden_dim=hidden)
