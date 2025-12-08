"""行为基础奖励配置与构造工具。"""
from __future__ import annotations

from typing import Dict

import torch

BASE_ACTION_REWARD = {
    "STAY": -0.005,
    "MOVE": 0.01,
    "EAT": 0.0,
    "ATTACK": 0.0,
    "COMMUNICATE": 0.0,
    "BUILD": 0.0,
    "REMOVE": 0.0,
    "REPRODUCE": 0.1,
}

BASE_ACTION_ENERGY_COST = {
    "STAY": 0.0,
    "MOVE": 0.0,
    "EAT": 0.0,
    "ATTACK": 0.0,
    "COMMUNICATE": 0.0,
    "BUILD": 0.0,
    "REMOVE": 0.0,
    "REPRODUCE": 0.0,
}

ACTION_NAME_TO_INDEX = {
    "STAY": 0,
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
    "EAT": 5,
    "ATTACK": 6,
    "COMMUNICATE": 7,
    "BUILD": 8,
    "REMOVE": 9,
    "REPRODUCE": 10,
}


def build_action_reward_table(config_rewards: Dict[str, float]) -> torch.Tensor:
    """根据配置构建 shape=(动作数,) 的奖励表。"""

    rewards = BASE_ACTION_REWARD.copy()
    rewards.update(config_rewards)
    table = torch.zeros(len(ACTION_NAME_TO_INDEX), dtype=torch.float32)
    move_alias = {"UP", "DOWN", "LEFT", "RIGHT"}
    for name, idx in ACTION_NAME_TO_INDEX.items():
        if name in move_alias:
            table[idx] = rewards.get("MOVE", BASE_ACTION_REWARD["MOVE"])
            continue
        key = name if name in rewards else name.split("_")[0]
        table[idx] = rewards.get(key, 0.0)
    return table


def build_action_energy_cost_table(config_costs: Dict[str, float]) -> torch.Tensor:
    """根据配置构建 shape=(动作数,) 的能量消耗表。"""

    costs = BASE_ACTION_ENERGY_COST.copy()
    costs.update(config_costs)
    table = torch.zeros(len(ACTION_NAME_TO_INDEX), dtype=torch.float32)
    move_alias = {"UP", "DOWN", "LEFT", "RIGHT"}
    for name, idx in ACTION_NAME_TO_INDEX.items():
        if name in move_alias:
            table[idx] = costs.get("MOVE", BASE_ACTION_ENERGY_COST["MOVE"])
            continue
        key = name if name in costs else name.split("_")[0]
        table[idx] = costs.get(key, 0.0)
    return table
