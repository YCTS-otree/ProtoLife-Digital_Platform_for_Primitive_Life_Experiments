"""环境与物理规则定义。

本文件聚合地图状态、个体交互、能量代谢等硬规则，保证智能决策全部交给策略网络学习。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .agents import AgentBatch
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
from .rewards import build_action_reward_table


ENV_DEFAULTS = {
    "world": {"height": 64, "width": 64, "map_file": None, "food_density": 0.03, "toxin_density": 0.01},
    "model": {"observation_radius": 2},
    "agents": {
        "per_env": 32,
        "base_energy": 50,
        "base_health": 100,
        "base_metabolism_cost": 1.0,
        "move_cost": 0.2,
    },
    "training": {"num_envs": 32, "rollout_steps": 128},
    "logging": {"realtime_render": False},
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
        self.height = self._get("world", "height", ENV_DEFAULTS["world"]["height"])
        self.width = self._get("world", "width", ENV_DEFAULTS["world"]["width"])
        self.map_file = self._get("world", "map_file", ENV_DEFAULTS["world"].get("map_file"))
        self.observation_radius = self._get("model", "observation_radius", ENV_DEFAULTS["model"]["observation_radius"])
        self.food_density = self._get("world", "food_density", ENV_DEFAULTS["world"]["food_density"])
        self.toxin_density = self._get("world", "toxin_density", ENV_DEFAULTS["world"]["toxin_density"])
        action_reward_cfg = config.get("action_rewards", self.default_config.get("action_rewards", {}))
        self.action_rewards = build_action_reward_table(action_reward_cfg).to(self.device)
        self.agent_batch = AgentBatch(
            num_envs=self._get("training", "num_envs", ENV_DEFAULTS["training"]["num_envs"]),
            agents_per_env=self._get("agents", "per_env", ENV_DEFAULTS["agents"]["per_env"]),
            device=self.device,
        )
        self._map_template = self._load_map_template().to(self.device)
        self.map_state = torch.zeros(
            (self.agent_batch.num_envs, self.height, self.width), dtype=torch.int64, device=self.device
        )
        self.renderer = None
        if self._get("logging", "realtime_render", ENV_DEFAULTS["logging"]["realtime_render"]):
            self.renderer = GridRenderer(self.height, self.width)

        self.observation_dim = self._calculate_observation_dim()

    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境，返回初始观测。

        这里使用均匀随机初始化地图，个体状态由 `AgentBatch` 自行重置。
        实际实验中可替换为带食物/毒素分布的初始化逻辑。
        """

        if self.map_file:
            self.map_state = self._map_template.clone().expand(self.agent_batch.num_envs, -1, -1).contiguous()
        else:
            self.map_state = self._generate_random_map().to(self.device)
        #self._scatter_resources()#取消随机撒
        self.agent_batch.reset(
            self.height,
            self.width,
            base_energy=self._get("agents", "base_energy", ENV_DEFAULTS["agents"]["base_energy"]),
            base_health=self._get("agents", "base_health", ENV_DEFAULTS["agents"]["base_health"]),
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
        move_info = self.agent_batch.apply_actions(actions_2d, self.height, self.width, self.map_state)
        base_rewards = self.action_rewards[actions_2d]

        rewards = base_rewards.view(self.agent_batch.num_envs, self.agent_batch.agents_per_env)
        rewards = rewards.clone()
        energy = self.agent_batch.state["energy"]
        health = self.agent_batch.state["health"]

        # 基础代谢与移动消耗
        base_metabolism = self._get("agents", "base_metabolism_cost", ENV_DEFAULTS["agents"]["base_metabolism_cost"])
        move_cost = self._get("agents", "move_cost", ENV_DEFAULTS["agents"]["move_cost"])
        energy_cost = base_metabolism + move_cost * move_info["moved"].float()
        energy.sub_(energy_cost)

        # 撞墙轻微惩罚
        rewards = torch.where(move_info["collided"], rewards - 0.6, rewards)

        # 交互：食物/毒素
        env_ids = torch.arange(self.agent_batch.num_envs, device=self.device).unsqueeze(1).expand_as(actions_2d)
        x = self.agent_batch.state["x"]
        y = self.agent_batch.state["y"]
        current_cells = self.map_state[env_ids, y, x]

        eat_mask = actions_2d == 5
        food_mask = (current_cells & BIT_FOOD).bool()
        eat_success = eat_mask & food_mask
        rewards = torch.where(eat_success, rewards + 1.0, rewards)
        energy = torch.where(eat_success, energy + 10.0, energy)
        self.map_state[env_ids, y, x] = torch.where(eat_success, current_cells & (~BIT_FOOD), current_cells)

        toxin_mask = (current_cells & BIT_TOXIN).bool()
        rewards = torch.where(toxin_mask, rewards - 0.5, rewards)
        health = torch.where(toxin_mask, health - 5.0, health)

        self.agent_batch.state["energy"] = energy
        self.agent_batch.state["health"] = health

        # 生存奖励，鼓励活着
        rewards += 0.000001

        # 生命值自然恢复
        health += 0.05

        dones = (energy <= 0) | (health <= 0)
        infos: List[Dict] = [
            {"message": "存活状态" if not dones.view(-1)[i] else "能量或健康耗尽"}
            for i in range(self.agent_batch.num_envs * self.agent_batch.agents_per_env)
        ]

        observations = self._build_observations()
        if self.renderer:
            self.renderer.render(self.map_state[0], self.agent_batch.state)

        return EnvStepResult(observations=observations, rewards=rewards.view(-1), dones=dones.view(-1), infos=infos)

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
        }

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        """从 checkpoint 恢复环境与个体状态。"""

        self.map_state = state["map_state"].to(self.device)
        self._map_template = self.map_state[:1].clone()
        self.agent_batch.load_state_dict(state["agent_state"], self.height, self.width)

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

    def _scatter_resources(self) -> None:
        """在地图上随机撒食物和毒素，便于快速获得奖励信号。"""

        if self.food_density <= 0 and self.toxin_density <= 0:
            return

        food_mask = torch.rand((self.agent_batch.num_envs, self.height, self.width), device=self.device) < self.food_density
        toxin_mask = torch.rand((self.agent_batch.num_envs, self.height, self.width), device=self.device) < self.toxin_density
        self.map_state = self.map_state.clone()
        self.map_state = torch.where(food_mask, self.map_state | BIT_FOOD, self.map_state)
        self.map_state = torch.where(toxin_mask, self.map_state | BIT_TOXIN, self.map_state)

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
        energy = agent_state[..., 2] / max(self._get("agents", "base_energy", ENV_DEFAULTS["agents"]["base_energy"]), 1)
        health = agent_state[..., 3] / max(self._get("agents", "base_health", ENV_DEFAULTS["agents"]["base_health"]), 1)
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

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
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

    def render(self, grid: torch.Tensor, agent_state: Dict[str, torch.Tensor]) -> None:
        display = self._to_display(grid.cpu())
        if self.img is None:
            self.img = self.ax.imshow(display, cmap="viridis", vmin=0, vmax=1)
        else:
            self.img.set_data(display)
        for artist in list(self.ax.collections):
            artist.remove()
        xs = agent_state["x"][0].cpu()
        ys = agent_state["y"][0].cpu()
        self.ax.scatter(xs, ys, c="red", s=10)
        plt.pause(0.001)
