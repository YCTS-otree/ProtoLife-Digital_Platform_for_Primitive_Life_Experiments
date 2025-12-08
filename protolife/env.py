"""环境与物理规则定义。

本文件聚合地图状态、个体交互、能量代谢等硬规则，保证智能决策全部交给策略网络学习。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .agents import AgentBatch
from .communication import broadcast_messages
from .encoding import (
    BIT_BUILDABLE,
    BIT_FOOD,
    BIT_RESOURCE,
    BIT_TERRAIN_0,
    BIT_TOXIN,
    decode_hex_to_grid,
    encode_grid,
    load_hex_map,
)
from .genetics import can_reproduce
from .rewards import build_action_energy_cost_table, build_action_reward_table


ENV_DEFAULTS = {
    "world": {
        "height": 64,
        "width": 64,
        "map_file": None,
        "food_density": 0.03,
        "toxin_density": 0.01,
        "food_respawn_interval": 0,
        "toxin_respawn_interval": 0,
        "toxin_lifetime": 0,
        "resource_noise": {
            "enabled": False,
            "scale": 12.0,
            "octaves": 1,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "seed": None,
            "food_bias": 0.0,
            "toxin_bias": 0.0,
        },
    },
    "model": {"observation_radius": 2},
    "agents": {
        "per_env": 32,
        "base_energy": 50,
        "base_health": 100,
        "energy_max": 100,
        "health_max": 100,
        "base_metabolism_cost": 1.0,
        "move_cost": 0.2,
        "reproduction_energy_threshold": 80,
        "child_energy_fraction": 0.3,
    },
    "training": {"num_envs": 32, "rollout_steps": 128},
    "logging": {
        "realtime_render": False,
        "agent_marker_size": 10,
        "snapshot_gpu_stage": False,
        "snapshot_flush_interval": 8,
        "show_step": True,
    },
    "rewards": {
        "survival_reward": 0.0001,
        "food_reward": 1.0,
        "food_energy": 10.0,
        "toxin_penalty": -0.5,
        "toxin_health": -5.0,
        "failed_eat_penalty": -0.05,
        "toxin_eat_penalty": -0.2,
        "enable_proximity_reward": True,
        "see_food_reward": 0.005,
        "stand_on_food_reward": 0.02,
        "vision_decay_mode": "linear",
        "vision_decay_coefficient": 1.0,
        "energy_fit_threshold": 50.0,
        "energy_reward_at_fit": 0.05,
        "energy_reward_at_extreme": -0.05,
        "energy_reward_mode": "linear",
        "energy_reward_coefficient": 1.0,
        "health_reward_at_max": 0.05,
        "health_reward_at_zero": -0.2,
        "health_reward_mode": "linear",
        "health_reward_coefficient": 1.0,
        "health_recovery_energy_threshold": 40.0,
        "health_recovery_per_step": 0.5,
        "health_decay_min": -1.0,
        "health_decay_mode": "linear",
        "health_decay_coefficient": 1.0,
    },
    "communication": {"radius": 2, "base_strength": 1.0},
    "combat": {"damage": 10.0, "radius": 2.0, "decay": "none"},
    "reproduction": {"health_cost": 5.0},
}


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

    def __init__(self, config: Dict, default_config: Dict):
        self.config = config
        self.default_config = default_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_seed = self._get("world", "random_seed", None)
        self.random_generator = torch.Generator(device=self.device)
        if self.base_seed is not None:
            self.random_generator.manual_seed(int(self.base_seed))
        self.height = self._get("world", "height", ENV_DEFAULTS["world"]["height"])
        self.width = self._get("world", "width", ENV_DEFAULTS["world"]["width"])
        self.map_file = self._get("world", "map_file", ENV_DEFAULTS["world"].get("map_file"))
        self.observation_radius = self._get("model", "observation_radius", ENV_DEFAULTS["model"]["observation_radius"])
        self.food_density = self._get("world", "food_density", ENV_DEFAULTS["world"]["food_density"])
        self.toxin_density = self._get("world", "toxin_density", ENV_DEFAULTS["world"]["toxin_density"])
        self.resource_noise_cfg = self._get("world", "resource_noise", ENV_DEFAULTS["world"]["resource_noise"])
        self.food_respawn_interval = self._get(
            "world", "food_respawn_interval", ENV_DEFAULTS["world"]["food_respawn_interval"]
        )
        self.toxin_respawn_interval = self._get(
            "world", "toxin_respawn_interval", ENV_DEFAULTS["world"]["toxin_respawn_interval"]
        )
        self.toxin_lifetime = self._get(
            "world",
            "toxin_lifetime",
            self._get("world", "toxin_decay_steps", 0),
        )
        self.energy_max = float(self._get("agents", "energy_max", ENV_DEFAULTS["agents"]["energy_max"]))
        self.health_max = float(self._get("agents", "health_max", ENV_DEFAULTS["agents"]["health_max"]))
        self.energy_reward_fit = float(
            self._get("rewards", "energy_fit_threshold", ENV_DEFAULTS["rewards"]["energy_fit_threshold"])
        )
        self.energy_reward_at_fit = float(
            self._get(
                "rewards",
                "energy_reward_at_fit",
                ENV_DEFAULTS["rewards"]["energy_reward_at_fit"],
            )
        )
        self.energy_reward_extreme = float(
            self._get(
                "rewards",
                "energy_reward_at_extreme",
                ENV_DEFAULTS["rewards"]["energy_reward_at_extreme"],
            )
        )
        self.energy_reward_mode = str(
            self._get("rewards", "energy_reward_mode", ENV_DEFAULTS["rewards"]["energy_reward_mode"])
        ).lower()
        self.energy_reward_coefficient = float(
            self._get(
                "rewards",
                "energy_reward_coefficient",
                ENV_DEFAULTS["rewards"]["energy_reward_coefficient"],
            )
        )
        self.health_reward_at_max = float(
            self._get("rewards", "health_reward_at_max", ENV_DEFAULTS["rewards"]["health_reward_at_max"])
        )
        self.health_reward_at_zero = float(
            self._get("rewards", "health_reward_at_zero", ENV_DEFAULTS["rewards"]["health_reward_at_zero"])
        )
        self.health_reward_mode = str(
            self._get("rewards", "health_reward_mode", ENV_DEFAULTS["rewards"]["health_reward_mode"])
        ).lower()
        self.health_reward_coefficient = float(
            self._get(
                "rewards",
                "health_reward_coefficient",
                ENV_DEFAULTS["rewards"]["health_reward_coefficient"],
            )
        )
        self.health_recovery_energy_threshold = float(
            self._get(
                "rewards",
                "health_recovery_energy_threshold",
                ENV_DEFAULTS["rewards"]["health_recovery_energy_threshold"],
            )
        )
        self.health_recovery_per_step = float(
            self._get(
                "rewards",
                "health_recovery_per_step",
                ENV_DEFAULTS["rewards"]["health_recovery_per_step"],
            )
        )
        self.health_decay_min = float(
            self._get("rewards", "health_decay_min", ENV_DEFAULTS["rewards"]["health_decay_min"])
        )
        self.health_decay_mode = str(
            self._get("rewards", "health_decay_mode", ENV_DEFAULTS["rewards"]["health_decay_mode"])
        ).lower()
        self.health_decay_coefficient = float(
            self._get(
                "rewards",
                "health_decay_coefficient",
                ENV_DEFAULTS["rewards"]["health_decay_coefficient"],
            )
        )
        action_reward_cfg = config.get("action_rewards", self.default_config.get("action_rewards", {}))
        self.action_rewards = build_action_reward_table(action_reward_cfg).to(self.device)
        action_energy_cfg = config.get("action_energy_costs", self.default_config.get("action_energy_costs", {}))
        self.action_energy_costs = build_action_energy_cost_table(action_energy_cfg).to(self.device)
        self.agent_batch = AgentBatch(
            num_envs=self._get("training", "num_envs", ENV_DEFAULTS["training"]["num_envs"]),
            agents_per_env=self._get("agents", "per_env", ENV_DEFAULTS["agents"]["per_env"]),
            device=self.device,
        )
        self._map_template = self._load_map_template().to(self.device)
        self.map_state = torch.zeros(
            (self.agent_batch.num_envs, self.height, self.width), dtype=torch.int64, device=self.device
        )
        self.toxin_age = torch.zeros(
            (self.agent_batch.num_envs, self.height, self.width), dtype=torch.int64, device=self.device
        )
        self.renderer = None
        self.agent_marker_size = self._get(
            "logging",
            "agent_marker_size",
            ENV_DEFAULTS["logging"]["agent_marker_size"],
        )
        self.show_step = bool(self._get("logging", "show_step", ENV_DEFAULTS["logging"]["show_step"]))
        if self._get("logging", "realtime_render", ENV_DEFAULTS["logging"]["realtime_render"]):
            self.renderer = GridRenderer(
                self.height,
                self.width,
                agent_marker_size=self.agent_marker_size,
                show_step=self.show_step,
            )

        self.observation_dim = self._calculate_observation_dim()
        self.step_count = 0

    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境，返回初始观测。

        这里使用均匀随机初始化地图，个体状态由 `AgentBatch` 自行重置。
        实际实验中可替换为带食物/毒素分布的初始化逻辑。
        """

        if self.map_file:
            self.map_state = self._map_template.clone().expand(self.agent_batch.num_envs, -1, -1).contiguous()
        else:
            self.map_state = self._generate_random_map().to(self.device)
        self.toxin_age.zero_()
        self._scatter_resources(respawn=False)
        self._reset_toxin_age_for_existing_toxins()
        self.step_count = 0
        self.agent_batch.reset(
            self.height,
            self.width,
            base_energy=self._get("agents", "base_energy", ENV_DEFAULTS["agents"]["base_energy"]),
            base_health=self._get("agents", "base_health", ENV_DEFAULTS["agents"]["base_health"]),
            energy_max=self.energy_max,
            health_max=self.health_max,
        )
        return self._build_observations()

    def step(self, actions: torch.Tensor) -> EnvStepResult:
        """执行一批动作并推进环境一帧。

        当前实现仅提供占位流程：
        1. 根据动作更新个体位置（不越界）。
        2. 按动作表发放基础奖励。
        3. 返回观测、奖励、终止标志与信息。
        后续可逐步填充能量代谢、食物/毒素交互、战斗等细节。
        """

        actions_2d = actions.view(self.agent_batch.num_envs, self.agent_batch.agents_per_env)
        self.step_count += 1
        move_info = self.agent_batch.apply_actions(actions_2d, self.height, self.width, self.map_state)
        base_rewards = self.action_rewards[actions_2d]

        rewards = base_rewards.view(self.agent_batch.num_envs, self.agent_batch.agents_per_env)
        rewards = rewards.clone()
        energy = self.agent_batch.state["energy"]
        health = self.agent_batch.state["health"]
        x = self.agent_batch.state["x"]
        y = self.agent_batch.state["y"]

        # 通信：按距离对数衰减聚合信号
        self.agent_batch.state["comm"].zero_()
        if self._get("features", "use_communication", True):
            comm_radius = float(
                self._get("communication", "radius", ENV_DEFAULTS["communication"]["radius"])
            )
            comm_strength = float(
                self._get(
                    "communication",
                    "base_strength",
                    ENV_DEFAULTS["communication"]["base_strength"],
                )
            )
            comm_mask = actions_2d == 7
            if comm_mask.any() and comm_radius > 0:
                positions = torch.stack([x, y], dim=-1)
                messages = torch.where(
                    comm_mask.unsqueeze(-1),
                    torch.ones_like(self.agent_batch.state["comm"]),
                    torch.zeros_like(self.agent_batch.state["comm"]),
                )
                aggregated = broadcast_messages(messages, positions, comm_radius, comm_strength)
                self.agent_batch.state["comm"] = aggregated

        # 攻击：按距离衰减伤害
        if self._get("features", "use_combat", True):
            attack_mask = actions_2d == 6
            if attack_mask.any():
                attack_radius = float(
                    self._get("combat", "radius", ENV_DEFAULTS["combat"]["radius"])
                )
                attack_damage = float(
                    self._get("combat", "damage", ENV_DEFAULTS["combat"]["damage"])
                )
                decay_mode = str(
                    self._get("combat", "decay", ENV_DEFAULTS["combat"]["decay"])
                ).lower()
                positions = torch.stack([x, y], dim=-1).float()
                distances = torch.cdist(positions, positions, p=2)
                valid = (distances <= attack_radius) & (distances > 0)

                if decay_mode == "log":
                    damage_scale = attack_damage / (
                        1.0 + torch.log1p(torch.clamp(distances, min=1e-6))
                    )
                elif decay_mode == "linear":
                    damage_scale = attack_damage * torch.clamp(
                        1.0 - distances / max(attack_radius, 1e-6), min=0.0
                    )
                else:
                    damage_scale = torch.where(valid, torch.full_like(distances, attack_damage), torch.zeros_like(distances))

                damage_scale = torch.where(valid, damage_scale, torch.zeros_like(damage_scale))
                attacker_mask = attack_mask.unsqueeze(1).expand_as(damage_scale)
                total_damage = (attacker_mask * damage_scale).sum(dim=1)
                health.sub_(total_damage)

        # 基础代谢与移动消耗
        base_metabolism = self._get("agents", "base_metabolism_cost", ENV_DEFAULTS["agents"]["base_metabolism_cost"])
        move_cost = self._get("agents", "move_cost", ENV_DEFAULTS["agents"]["move_cost"])
        action_cost = self.action_energy_costs[actions_2d]
        energy_cost = base_metabolism + move_cost * move_info["moved"].float() + action_cost
        energy.sub_(energy_cost)

        # 撞墙轻微惩罚
        rewards = torch.where(move_info["collided"], rewards - 0.6, rewards)

        # 繁衍：满足能量阈值时在空闲插槽生成子代，并扣除健康值
        if self._get("features", "use_reproduction", True):
            reproduction_mask = actions_2d == 10
            threshold = self._get(
                "agents",
                "reproduction_energy_threshold",
                ENV_DEFAULTS["agents"]["reproduction_energy_threshold"],
            )
            child_fraction = self._get(
                "agents", "child_energy_fraction", ENV_DEFAULTS["agents"]["child_energy_fraction"]
            )
            health_cost = self._get(
                "reproduction", "health_cost", ENV_DEFAULTS["reproduction"]["health_cost"]
            )
            eligible = reproduction_mask & can_reproduce(energy, threshold)

            for env_idx in range(self.agent_batch.num_envs):
                parents = torch.nonzero(eligible[env_idx], as_tuple=False).view(-1)
                if parents.numel() == 0:
                    continue
                dead_slots = torch.nonzero(
                    (energy[env_idx] <= 0) | (health[env_idx] <= 0), as_tuple=False
                ).view(-1)
                if dead_slots.numel() == 0:
                    continue

                for parent_idx in parents:
                    if dead_slots.numel() == 0:
                        break
                    child_idx = dead_slots[0]
                    dead_slots = dead_slots[1:]

                    transfer_energy = energy[env_idx, parent_idx] * child_fraction
                    energy[env_idx, parent_idx] -= transfer_energy
                    energy[env_idx, child_idx] = transfer_energy
                    health[env_idx, parent_idx] -= health_cost
                    health[env_idx, child_idx] = self._get(
                        "agents", "base_health", ENV_DEFAULTS["agents"]["base_health"]
                    )
                    x[env_idx, child_idx] = x[env_idx, parent_idx]
                    y[env_idx, child_idx] = y[env_idx, parent_idx]
                    self.agent_batch.state["age"][env_idx, child_idx] = 0
                    self.agent_batch.state["generation"][env_idx, child_idx] = (
                        self.agent_batch.state["generation"][env_idx, parent_idx] + 1
                    )
                    self.agent_batch.state["genome_id"][env_idx, child_idx] = self.agent_batch.state[
                        "genome_id"
                    ][env_idx, parent_idx]

        # 交互：食物/毒素
        env_ids = torch.arange(self.agent_batch.num_envs, device=self.device).unsqueeze(1).expand_as(actions_2d)
        current_cells = self.map_state[env_ids, y, x]

        if self._get("rewards", "enable_proximity_reward", ENV_DEFAULTS["rewards"]["enable_proximity_reward"]):
            radius = self.observation_radius
            if radius > 0:
                padded = F.pad(self.map_state.float().unsqueeze(1), (radius, radius, radius, radius))
                kernel = 2 * radius + 1
                patches = F.unfold(padded, kernel_size=kernel)  # (E, K, H*W)
                flat_idx = y * self.width + x
                gather_idx = flat_idx.unsqueeze(1).expand(-1, kernel * kernel, -1)
                gathered = patches.gather(2, gather_idx).long()  # (E, K, A)
                food_patch = (gathered & BIT_FOOD).bool()

                offsets = torch.arange(-radius, radius + 1, device=self.device)
                dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
                distances = torch.sqrt(dx.float() ** 2 + dy.float() ** 2).view(1, -1, 1)
                distances = distances.expand(self.agent_batch.num_envs, -1, self.agent_batch.agents_per_env)

                food_distances = torch.where(
                    food_patch,
                    distances,
                    torch.full_like(distances, float("inf")),
                )
                min_distances, _ = food_distances.min(dim=1)
                food_found = torch.isfinite(min_distances) & (min_distances <= radius)
                valid_distances = torch.where(food_found, min_distances, torch.zeros_like(min_distances))

                decay_mode = self._get(
                    "rewards", "vision_decay_mode", ENV_DEFAULTS["rewards"]["vision_decay_mode"]
                ).lower()
                decay_coeff = self._get(
                    "rewards",
                    "vision_decay_coefficient",
                    ENV_DEFAULTS["rewards"]["vision_decay_coefficient"],
                )
                if decay_mode == "log":
                    decay_scale = 1.0 / (1.0 + decay_coeff * torch.log1p(valid_distances))
                else:
                    decay_scale = torch.clamp(
                        1.0 - decay_coeff * (valid_distances / max(radius, 1)), min=0.0
                    )

                see_reward = self._get(
                    "rewards", "see_food_reward", ENV_DEFAULTS["rewards"]["see_food_reward"]
                )
                stand_on_reward = self._get(
                    "rewards", "stand_on_food_reward", ENV_DEFAULTS["rewards"]["stand_on_food_reward"]
                )
                proximity_bonus = see_reward * decay_scale
                stand_on_mask = (current_cells & BIT_FOOD).bool()
                proximity_bonus = torch.where(
                    stand_on_mask, stand_on_reward * decay_scale, proximity_bonus
                )
                rewards = rewards + torch.where(food_found, proximity_bonus, torch.zeros_like(rewards))

        eat_mask = actions_2d == 5
        food_mask = (current_cells & BIT_FOOD).bool()
        eat_success = eat_mask & food_mask
        failed_eat = eat_mask & (~food_mask)
        rewards = torch.where(eat_success, rewards + self._get("rewards", "food_reward", ENV_DEFAULTS["rewards"]["food_reward"]), rewards)
        rewards = torch.where(failed_eat, rewards + self._get("rewards", "failed_eat_penalty", ENV_DEFAULTS["rewards"]["failed_eat_penalty"]), rewards)
        energy = torch.where(
            eat_success,
            energy + self._get("rewards", "food_energy", ENV_DEFAULTS["rewards"]["food_energy"]),
            energy,
        )
        self.map_state[env_ids, y, x] = torch.where(eat_success, current_cells & (~BIT_FOOD), current_cells)

        toxin_mask = (current_cells & BIT_TOXIN).bool()
        eat_on_toxin = eat_mask & toxin_mask
        rewards = torch.where(toxin_mask, rewards + self._get("rewards", "toxin_penalty", ENV_DEFAULTS["rewards"]["toxin_penalty"]), rewards)
        rewards = torch.where(
            eat_on_toxin,
            rewards + self._get("rewards", "toxin_eat_penalty", ENV_DEFAULTS["rewards"]["toxin_eat_penalty"]),
            rewards,
        )
        health = torch.where(
            toxin_mask,
            health + self._get("rewards", "toxin_health", ENV_DEFAULTS["rewards"]["toxin_health"]),
            health,
        )

        health = self._update_health_from_energy(health, energy)

        energy = torch.clamp(energy, min=0.0, max=self.energy_max)
        health = torch.clamp(health, min=0.0, max=self.health_max)

        rewards = rewards + self._compute_energy_reward(energy)
        rewards = rewards + self._compute_health_reward(health)

        self.agent_batch.state["energy"] = energy
        self.agent_batch.state["health"] = health

        # 生存奖励，鼓励活着
        rewards += self._get("rewards", "survival_reward", ENV_DEFAULTS["rewards"]["survival_reward"])

        # 周期性刷新资源/毒素
        self._maybe_respawn_resources()

        # 毒素衰减
        self._decay_toxins()

        dones = (energy <= 0) | (health <= 0)
        infos: List[Dict] = [
            {"message": "存活状态" if not dones.view(-1)[i] else "能量或健康耗尽"}
            for i in range(self.agent_batch.num_envs * self.agent_batch.agents_per_env)
        ]

        observations = self._build_observations()
        if self.renderer:
            self.renderer.render(self.map_state[0], self.agent_batch.state, step=self.step_count)

        return EnvStepResult(observations=observations, rewards=rewards.view(-1), dones=dones.view(-1), infos=infos)

    def _apply_curve(self, normalized: torch.Tensor, mode: str, coefficient: float) -> torch.Tensor:
        """根据模式调整 [0,1] 归一化值的曲线形状。"""

        normalized = torch.clamp(normalized, 0.0, 1.0)
        coeff = max(float(coefficient), 1e-6)
        if mode == "log":
            return torch.log1p(coeff * normalized) / math.log1p(coeff)
        if mode in {"invlog", "inverse_log"}:
            return torch.expm1(coeff * normalized) / math.expm1(coeff)
        return normalized

    def _interpolate_reward(
        self, normalized: torch.Tensor, start_value: float, end_value: float, mode: str, coefficient: float
    ) -> torch.Tensor:
        curve = self._apply_curve(normalized, mode, coefficient)
        return start_value + curve * (end_value - start_value)

    def _compute_energy_reward(self, energy: torch.Tensor) -> torch.Tensor:
        if self.energy_max <= 0:
            return torch.zeros_like(energy)

        rewards = torch.zeros_like(energy)
        upper_mask = energy >= self.energy_reward_fit
        lower_mask = energy < self.energy_reward_fit

        if upper_mask.any() and self.energy_max > self.energy_reward_fit:
            normalized_upper = torch.clamp(
                (energy[upper_mask] - self.energy_reward_fit) / max(self.energy_max - self.energy_reward_fit, 1e-6),
                0.0,
                1.0,
            )
            rewards_upper = self._interpolate_reward(
                normalized_upper,
                self.energy_reward_at_fit,
                self.energy_reward_extreme,
                self.energy_reward_mode,
                self.energy_reward_coefficient,
            )
            rewards = rewards.masked_scatter(upper_mask, rewards_upper)

        if lower_mask.any() and self.energy_reward_fit > 0:
            normalized_lower = torch.clamp(
                (self.energy_reward_fit - energy[lower_mask]) / max(self.energy_reward_fit, 1e-6),
                0.0,
                1.0,
            )
            rewards_lower = self._interpolate_reward(
                normalized_lower,
                self.energy_reward_at_fit,
                self.energy_reward_extreme,
                self.energy_reward_mode,
                self.energy_reward_coefficient,
            )
            rewards = rewards.masked_scatter(lower_mask, rewards_lower)

        return rewards

    def _compute_health_reward(self, health: torch.Tensor) -> torch.Tensor:
        if self.health_max <= 0:
            return torch.zeros_like(health)

        normalized = torch.clamp((self.health_max - health) / max(self.health_max, 1e-6), 0.0, 1.0)
        return self._interpolate_reward(
            normalized,
            self.health_reward_at_max,
            self.health_reward_at_zero,
            self.health_reward_mode,
            self.health_reward_coefficient,
        )

    def _update_health_from_energy(self, health: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        if self.health_recovery_energy_threshold <= 0:
            return health

        recovery_mask = energy > self.health_recovery_energy_threshold
        if recovery_mask.any() and self.health_recovery_per_step != 0:
            health = torch.where(recovery_mask, health + self.health_recovery_per_step, health)

        decay_mask = energy < self.health_recovery_energy_threshold
        if decay_mask.any() and self.health_decay_min != 0:
            normalized = torch.clamp(
                (self.health_recovery_energy_threshold - energy[decay_mask])
                / max(self.health_recovery_energy_threshold, 1e-6),
                0.0,
                1.0,
            )
            decay_scale = self._apply_curve(normalized, self.health_decay_mode, self.health_decay_coefficient)
            scaled_decay = torch.zeros_like(health)
            scaled_decay[decay_mask] = decay_scale * self.health_decay_min
            health = health + scaled_decay

        return health

    def _build_observations(self) -> Dict[str, torch.Tensor]:
        """构建供策略网络使用的张量观测。"""

        agent_state = self.agent_batch.export_state()
        local_maps = self._extract_local_observation()
        agent_features = self._normalize_agent_features(agent_state)
        flat_local = local_maps.contiguous().view(
            self.agent_batch.num_envs * self.agent_batch.agents_per_env, -1
        )
        flat_agent = agent_features.view(self.agent_batch.num_envs * self.agent_batch.agents_per_env, -1)
        agent_obs = torch.cat([flat_local, flat_agent], dim=-1)

        return {
            "map": self.map_state.clone(),
            "agents": agent_state,
            "agent_obs": agent_obs,
        }

    def render(self) -> None:
        """调试可视化钩子，初版占位。"""

        encoded = encode_grid(self.map_state)
        print(f"Encoded map snapshot: {encoded[:16]} ...")

    def export_state(self) -> Dict[str, torch.Tensor]:
        """导出环境状态用于 checkpoint。"""

        return {
            "map_state": self.map_state.detach().cpu(),
            "agent_state": self.agent_batch.state_dict(),
            "toxin_age": self.toxin_age.detach().cpu(),
        }

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        """从 checkpoint 恢复环境与个体状态。"""

        self.map_state = state["map_state"].to(self.device)
        self._map_template = self.map_state[:1].clone()
        self.agent_batch.load_state_dict(state["agent_state"], self.height, self.width)
        self.toxin_age = state.get("toxin_age", torch.zeros_like(self.map_state)).to(self.device)

    def _load_map_template(self) -> torch.Tensor:
        """加载单张地图模板，若未指定则返回全零地图。"""

        if not self.map_file:
            return torch.zeros((1, self.height, self.width), dtype=torch.int64)

        map_path = Path(self.map_file)
        if not map_path.exists():
            print(f"[警告] map_file={map_path} 不存在，回退到随机地图")
            return torch.zeros((1, self.height, self.width), dtype=torch.int64)

        try:
            hex_string = load_hex_map(map_path)
            grid = decode_hex_to_grid(hex_string, height=self.height, width=self.width).to(torch.int64)
            return grid.unsqueeze(0)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 地图解码失败 {exc}，回退到随机地图")
            return torch.zeros((1, self.height, self.width), dtype=torch.int64)

    def _generate_random_map(self) -> torch.Tensor:
        """按密度随机生成含墙/食物/毒素的地图。"""

        grid = torch.zeros((self.agent_batch.num_envs, self.height, self.width), dtype=torch.int64, device=self.device)
        # 简单边界墙
        grid[:, 0, :] |= BIT_TERRAIN_0
        grid[:, -1, :] |= BIT_TERRAIN_0
        grid[:, :, 0] |= BIT_TERRAIN_0
        grid[:, :, -1] |= BIT_TERRAIN_0
        return grid

    def _perlin_noise(
        self,
        height: int,
        width: int,
        scale: float,
        octaves: int,
        persistence: float,
        lacunarity: float,
        seed: int,
    ) -> torch.Tensor:
        """生成指定大小的 Perlin Noise，返回值范围约在 [0, 1]。"""

        def fade(t: torch.Tensor) -> torch.Tensor:
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        noise = torch.zeros((height, width), device=self.device)
        max_amplitude = 0.0
        amplitude = 1.0
        frequency = 1.0
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))

        xs = torch.arange(width, device=self.device).view(1, -1).expand(height, -1)
        ys = torch.arange(height, device=self.device).view(-1, 1).expand(-1, width)

        octave_count = max(int(octaves), 1)
        for _ in range(octave_count):
            effective_scale = max(scale, 1e-3) / max(frequency, 1e-3)
            grid_w = int(torch.ceil(torch.tensor(width / effective_scale)).item()) + 2
            grid_h = int(torch.ceil(torch.tensor(height / effective_scale)).item()) + 2
            gradients = torch.randn((grid_h, grid_w, 2), device=self.device, generator=generator)
            gradients = F.normalize(gradients, dim=-1)

            nx = xs / effective_scale
            ny = ys / effective_scale
            x0 = torch.floor(nx).long()
            y0 = torch.floor(ny).long()
            x1 = x0 + 1
            y1 = y0 + 1

            dx = nx - x0
            dy = ny - y0

            g00 = gradients[y0, x0]
            g10 = gradients[y0, x1]
            g01 = gradients[y1, x0]
            g11 = gradients[y1, x1]

            dot00 = g00[..., 0] * dx + g00[..., 1] * dy
            dot10 = g10[..., 0] * (dx - 1) + g10[..., 1] * dy
            dot01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1)
            dot11 = g11[..., 0] * (dx - 1) + g11[..., 1] * (dy - 1)

            u = fade(dx)
            v = fade(dy)

            nx0 = dot00 + u * (dot10 - dot00)
            nx1 = dot01 + u * (dot11 - dot01)
            value = nx0 + v * (nx1 - nx0)

            noise += value * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        if max_amplitude > 0:
            noise = noise / max_amplitude

        noise_min, noise_max = noise.min(), noise.max()
        if (noise_max - noise_min) > 1e-8:
            noise = (noise - noise_min) / (noise_max - noise_min)
        else:
            noise = torch.zeros_like(noise)

        return noise

    def _noise_mask_for_density(
        self,
        density: float,
        seed_offset: int,
        bias: float,
        wall_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """使用 Perlin Noise 根据密度生成资源分布掩码。"""

        if density <= 0:
            return None

        cfg = self.resource_noise_cfg or {}
        base_seed = cfg.get("seed", self.base_seed)
        scale = float(cfg.get("scale", 12.0))
        octaves = int(cfg.get("octaves", 1))
        persistence = float(cfg.get("persistence", 0.5))
        lacunarity = float(cfg.get("lacunarity", 2.0))

        masks = []
        for env_idx in range(self.agent_batch.num_envs):
            seed = base_seed if base_seed is not None else int(torch.seed())
            seed = int(seed) + env_idx + seed_offset
            noise = self._perlin_noise(
                self.height,
                self.width,
                scale,
                octaves,
                persistence,
                lacunarity,
                seed,
            )
            if bias:
                noise = torch.clamp(noise + bias, 0.0, 1.0)

            available_mask = None
            if wall_mask is not None:
                available_mask = ~wall_mask[env_idx]

            if available_mask is not None and available_mask.any():
                valid_values = noise[available_mask]
            else:
                valid_values = noise.view(-1)

            if valid_values.numel() == 0:
                masks.append(torch.zeros_like(noise, dtype=torch.bool))
                continue

            threshold = torch.quantile(valid_values, max(0.0, 1 - density))
            env_mask = noise >= threshold

            if available_mask is not None:
                env_mask = env_mask & available_mask

            masks.append(env_mask)

        return torch.stack(masks, dim=0) if masks else None

    def _scatter_resources(self, *, spawn_food: bool = True, spawn_toxin: bool = True, respawn: bool = False) -> None:
        """在地图上随机撒食物和毒素，便于快速获得奖励信号。"""

        if (not spawn_food or self.food_density <= 0) and (not spawn_toxin or self.toxin_density <= 0):
            return

        wall_mask = None
        if respawn:
            wall_mask = (self.map_state & BIT_TERRAIN_0).bool()

        use_noise = (self.resource_noise_cfg or {}).get("enabled", False)

        food_mask = None
        if spawn_food and self.food_density > 0:
            if use_noise:
                food_bias = float((self.resource_noise_cfg or {}).get("food_bias", 0.0))
                food_mask = self._noise_mask_for_density(self.food_density, seed_offset=0, bias=food_bias, wall_mask=wall_mask)
            else:
                food_mask = (
                    torch.rand(
                        (self.agent_batch.num_envs, self.height, self.width),
                        device=self.device,
                        generator=self.random_generator,
                    )
                    < self.food_density
                )
                if wall_mask is not None:
                    food_mask = food_mask & (~wall_mask)

        toxin_mask = None
        if spawn_toxin and self.toxin_density > 0:
            if use_noise:
                toxin_bias = float((self.resource_noise_cfg or {}).get("toxin_bias", 0.0))
                toxin_mask = self._noise_mask_for_density(
                    self.toxin_density, seed_offset=97, bias=toxin_bias, wall_mask=wall_mask
                )
            else:
                toxin_mask = (
                    torch.rand(
                        (self.agent_batch.num_envs, self.height, self.width),
                        device=self.device,
                        generator=self.random_generator,
                    )
                    < self.toxin_density
                )
                if wall_mask is not None:
                    toxin_mask = toxin_mask & (~wall_mask)

        existing_toxin = (self.map_state & BIT_TOXIN).bool()
        if food_mask is not None:
            self.map_state = torch.where(food_mask, self.map_state | BIT_FOOD, self.map_state)
        if toxin_mask is not None:
            self.map_state = torch.where(toxin_mask, self.map_state | BIT_TOXIN, self.map_state)
        if toxin_mask is not None and self.toxin_lifetime:
            new_toxin_cells = toxin_mask & (~existing_toxin)
            self.toxin_age = torch.where(new_toxin_cells, torch.zeros_like(self.toxin_age), self.toxin_age)

    def _reset_toxin_age_for_existing_toxins(self) -> None:
        """确保当前地图上的毒素年龄归零，用于重置或从地图加载。"""

        toxin_mask = (self.map_state & BIT_TOXIN).bool()
        if toxin_mask.any():
            self.toxin_age = torch.where(toxin_mask, torch.zeros_like(self.toxin_age), self.toxin_age)

    def _maybe_respawn_resources(self) -> None:
        """根据配置周期性刷新食物/毒素。"""

        if self.food_respawn_interval and self.food_respawn_interval > 0:
            if self.step_count % self.food_respawn_interval == 0:
                self._scatter_resources(spawn_food=True, spawn_toxin=False, respawn=True)
        if self.toxin_respawn_interval and self.toxin_respawn_interval > 0:
            if self.step_count % self.toxin_respawn_interval == 0:
                self._scatter_resources(spawn_food=False, spawn_toxin=True, respawn=True)

    def _decay_toxins(self) -> None:
        """当毒素存续超过寿命时将其从地图上移除。"""

        if not self.toxin_lifetime or self.toxin_lifetime <= 0:
            return

        toxin_mask = (self.map_state & BIT_TOXIN).bool()
        self.toxin_age = torch.where(toxin_mask, self.toxin_age + 1, torch.zeros_like(self.toxin_age))
        expired = toxin_mask & (self.toxin_age >= self.toxin_lifetime)
        if expired.any():
            self.map_state = torch.where(expired, self.map_state & (~BIT_TOXIN), self.map_state)
            self.toxin_age = torch.where(expired, torch.zeros_like(self.toxin_age), self.toxin_age)

    def _extract_local_observation(self) -> torch.Tensor:
        """裁剪每个体周围 r 范围内的网格并转为 multi-hot 通道。"""

        radius = self.observation_radius
        # 将地图留在 GPU 上，通过 unfold + gather 一次性提取每个个体的局部区域，避免 CPU<->GPU 往返
        padded = F.pad(
            self.map_state.float().unsqueeze(1), (radius, radius, radius, radius)
        )  # (E,1,H+2r,W+2r)
        kernel = 2 * radius + 1
        patches = F.unfold(padded, kernel_size=kernel)  # (E, kernel*kernel, H*W)
        flat_idx = self.agent_batch.state["y"] * self.width + self.agent_batch.state["x"]  # (E, A)
        gather_idx = flat_idx.unsqueeze(1).expand(-1, kernel * kernel, -1)
        gathered = patches.gather(2, gather_idx)  # (E, K, A)
        gathered = gathered.round().long().permute(0, 2, 1)  # (E, A, K)
        channel_tensor = self._cell_to_channels(gathered)
        return channel_tensor

    def _cell_to_channels(self, patch: torch.Tensor) -> torch.Tensor:
        """将单通道 bit map 转为 multi-hot 通道。

        支持输入形状：
        - `(H, W)` 或 `(K,)` 一维 patch
        - `(E, A, K)` 展开后的小块集合
        输出形状与输入一致，末尾附加 channel 维度。
        """

        channels = [
            (patch & BIT_TERRAIN_0) > 0,
            (patch & BIT_BUILDABLE) > 0,
            (patch & BIT_FOOD) > 0,
            (patch & BIT_TOXIN) > 0,
            (patch & BIT_RESOURCE) > 0,
        ]
        stacked = torch.stack([c.float() for c in channels], dim=-1)
        return stacked

    def _normalize_agent_features(self, agent_state: torch.Tensor) -> torch.Tensor:
        """归一化坐标/能量/健康，避免数值尺度差异。"""

        x = agent_state[..., 0] / max(self.width - 1, 1)
        y = agent_state[..., 1] / max(self.height - 1, 1)
        energy = agent_state[..., 2] / max(self.energy_max, 1)
        health = agent_state[..., 3] / max(self.health_max, 1)
        age = agent_state[..., 4] / max(self._get("training", "rollout_steps", ENV_DEFAULTS["training"]["rollout_steps"]), 1)
        return torch.stack([x, y, energy, health, age], dim=-1)

    def _calculate_observation_dim(self) -> int:
        """计算单个 agent 的观测维度，便于构建策略网络。"""

        patch_cells = (2 * self.observation_radius + 1) ** 2
        patch_channels = 5  # terrain/buildable/food/toxin/resource
        agent_feature_dim = 5
        return patch_cells * patch_channels + agent_feature_dim

    def _get(self, section: str, key: str, fallback):
        return self.config.get(section, {}).get(
            key,
            self.default_config.get(section, {}).get(key, ENV_DEFAULTS.get(section, {}).get(key, fallback)),
        )


class GridRenderer:
    """基于 matplotlib 的简易实时可视化，与 map_editor 交互风格一致。"""

    def __init__(self, height: int, width: int, *, agent_marker_size: float = 10, show_step: bool = True) -> None:
        self.height = height
        self.width = width
        self.agent_marker_size = agent_marker_size
        self.show_step = show_step
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img = None

    def _to_display(self, grid: torch.Tensor) -> torch.Tensor:
        """将 bit map 转换为简单的调色板索引。"""

        display = torch.zeros_like(grid, dtype=torch.float32)
        display = torch.where((grid & BIT_TERRAIN_0) > 0, torch.tensor(0.2), display)
        display = torch.where((grid & BIT_FOOD) > 0, torch.tensor(0.8), display)
        display = torch.where((grid & BIT_TOXIN) > 0, torch.tensor(0.5), display)
        display = torch.where((grid & BIT_RESOURCE) > 0, torch.tensor(0.6), display)
        return display

    def render(self, grid: torch.Tensor, agent_state: Dict[str, torch.Tensor], *, step: int | None = None) -> None:
        display = self._to_display(grid.cpu())
        if self.img is None:
            self.img = self.ax.imshow(display, cmap="viridis", vmin=0, vmax=1)
        else:
            self.img.set_data(display)
        for artist in list(self.ax.collections):
            artist.remove()
        xs = agent_state["x"][0].cpu()
        ys = agent_state["y"][0].cpu()
        self.ax.scatter(xs, ys, c="red", s=self.agent_marker_size)
        if self.show_step and step is not None:
            self.ax.set_title(f"Step: {step}")
        plt.pause(0.001)
