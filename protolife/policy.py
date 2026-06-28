"""策略与价值网络实现。"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


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


POLICY_DEFAULTS = {
    "model": {
        "hidden": 128,
        "obs_dim": 5,
        "use_cnn": False,
        "cnn_independent": True,
        "cnn_channels": [32, 64],
        "cnn_feature_dim": 128,
        "cnn_pooling": True,
        "cnn_pool_size": [8, 8],
        "rnn_hidden_dim": 128,
        "rnn_type": "gru",
    }
}


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_pool_size(value) -> Tuple[int, int]:
    if isinstance(value, int):
        size = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        size = (int(value[0]), int(value[1]))
    else:
        raise ValueError("cnn_pool_size 必须是正整数或包含两个正整数的列表")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("cnn_pool_size 必须大于 0")
    return size


class MLPPolicy(nn.Module):
    """简化的多层感知机策略。"""

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
        x = self.backbone(obs)
        return self.policy_head(x), self.value_head(x)


class CNNRecurrentPolicy(nn.Module):
    """单个生命个体拥有的卷积、循环记忆和策略网络。"""

    def __init__(
        self,
        patch_shape: Tuple[int, int, int],
        action_dim: int,
        cnn_channels: Iterable[int],
        cnn_feature_dim: int,
        rnn_hidden_dim: int,
        rnn_type: str = "gru",
        *,
        agent_feature_dim: int = 3,
        cnn_pooling: bool = True,
        cnn_pool_size: Tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.use_cnn = True
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.agent_feature_dim = int(agent_feature_dim)
        self.cnn_pooling = bool(cnn_pooling)
        self.cnn_pool_size = tuple(cnn_pool_size)
        c, h, w = patch_shape
        convs = []
        in_channels = c
        for out_channels in cnn_channels:
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU())
            in_channels = out_channels
        self.cnn = nn.Sequential(*convs)
        if self.cnn_pooling:
            self.pool = nn.AdaptiveAvgPool2d(self.cnn_pool_size)
            projected_h, projected_w = self.cnn_pool_size
        else:
            self.pool = nn.Identity()
            projected_h, projected_w = h, w
        flattened_dim = in_channels * projected_h * projected_w
        self.proj = nn.Linear(flattened_dim, cnn_feature_dim)

        rnn_input_dim = cnn_feature_dim + self.agent_feature_dim
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim, batch_first=True)

        self.policy_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.value_head = nn.Linear(rnn_hidden_dim, 1)

    def _encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.cnn(patch))
        x = x.reshape(x.size(0), -1)
        return torch.relu(self.proj(x))

    def forward(
        self,
        patch: torch.Tensor,
        hidden_state: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
        agent_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
        """输入局部地图与归一化能量/健康/年龄，返回动作、价值和新记忆。"""

        num_envs, agents_per_env = patch.shape[:2]
        batch = num_envs * agents_per_env
        patch_flat = patch.reshape(batch, *patch.shape[2:])
        encoded_patch = self._encode_patch(patch_flat)
        if agent_features is None:
            feature_flat = torch.zeros(
                batch,
                self.agent_feature_dim,
                device=patch.device,
                dtype=patch.dtype,
            )
        else:
            feature_flat = agent_features.reshape(batch, -1).to(dtype=patch.dtype)
            if feature_flat.size(-1) != self.agent_feature_dim:
                raise ValueError(
                    f"个体状态维度应为 {self.agent_feature_dim}，实际为 {feature_flat.size(-1)}"
                )
        encoded = torch.cat([encoded_patch, feature_flat], dim=-1).unsqueeze(1)

        if self.rnn_type == "lstm":
            if hidden_state is None:
                zeros = torch.zeros(batch, self.rnn_hidden_dim, device=patch.device, dtype=patch.dtype)
                h_state = (zeros.unsqueeze(0), zeros.unsqueeze(0))
            else:
                if not isinstance(hidden_state, tuple):
                    raise TypeError("LSTM 需要 (hidden, cell) 状态")
                h, c = hidden_state
                h_state = (h.reshape(1, batch, -1), c.reshape(1, batch, -1))
            rnn_out, (h_out, c_out) = self.rnn(encoded, h_state)
            new_hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] = (
                h_out.reshape(num_envs, agents_per_env, -1),
                c_out.reshape(num_envs, agents_per_env, -1),
            )
        else:
            if hidden_state is None or isinstance(hidden_state, tuple):
                hidden_in = None
            else:
                hidden_in = hidden_state.reshape(1, batch, -1)
            rnn_out, h_out = self.rnn(encoded, hidden_in)
            new_hidden = h_out.reshape(num_envs, agents_per_env, -1)

        core = rnn_out[:, -1, :]
        logits = self.policy_head(core).reshape(num_envs, agents_per_env, -1)
        value = self.value_head(core).reshape(num_envs, agents_per_env, 1)
        return logits, value, new_hidden


class IndependentCNNPolicies(nn.Module):
    """每个环境、每个生命槽位拥有完全独立的一套 CNN+RNN 参数。"""

    def __init__(
        self,
        *,
        num_envs: int,
        agents_per_env: int,
        brain_kwargs: Dict,
    ) -> None:
        super().__init__()
        self.use_cnn = True
        self.cnn_independent = True
        self.num_envs = int(num_envs)
        self.agents_per_env = int(agents_per_env)
        self.brain_count = self.num_envs * self.agents_per_env
        self.rnn_type = str(brain_kwargs.get("rnn_type", "gru")).lower()
        self.rnn_hidden_dim = int(brain_kwargs["rnn_hidden_dim"])
        self.brains = nn.ModuleList(
            CNNRecurrentPolicy(**brain_kwargs) for _ in range(self.brain_count)
        )

    def _brain_index(self, env_idx: int, agent_idx: int) -> int:
        return int(env_idx) * self.agents_per_env + int(agent_idx)

    @staticmethod
    def _batched_linear(inputs: torch.Tensor, layers: list[nn.Linear]) -> torch.Tensor:
        weights = torch.stack([layer.weight for layer in layers], dim=0)
        biases = torch.stack([layer.bias for layer in layers], dim=0)
        return torch.bmm(weights, inputs.unsqueeze(-1)).squeeze(-1) + biases

    def _encode_all_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """使用分组卷积一次并行执行所有互不共享参数的 CNN。"""

        x = patches
        brain_count = self.brain_count
        conv_pair_count = len(self.brains[0].cnn) // 2
        for pair_idx in range(conv_pair_count):
            conv_idx = pair_idx * 2
            layers = [brain.cnn[conv_idx] for brain in self.brains]
            out_channels = layers[0].out_channels
            weights = torch.cat([layer.weight for layer in layers], dim=0)
            biases = torch.cat([layer.bias for layer in layers], dim=0)
            _, in_channels, height, width = x.shape
            grouped_input = x.reshape(1, brain_count * in_channels, height, width)
            grouped_output = F.conv2d(
                grouped_input,
                weights,
                biases,
                padding=layers[0].padding,
                groups=brain_count,
            )
            x = torch.relu(
                grouped_output.reshape(brain_count, out_channels, height, width)
            )

        x = self.brains[0].pool(x).reshape(brain_count, -1)
        projected = self._batched_linear(x, [brain.proj for brain in self.brains])
        return torch.relu(projected)

    def _advance_all_memories(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None,
        cell: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden is None:
            hidden = torch.zeros(
                self.brain_count,
                self.rnn_hidden_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        rnn_layers = [brain.rnn for brain in self.brains]
        weight_ih = torch.stack([layer.weight_ih_l0 for layer in rnn_layers], dim=0)
        weight_hh = torch.stack([layer.weight_hh_l0 for layer in rnn_layers], dim=0)
        bias_ih = torch.stack([layer.bias_ih_l0 for layer in rnn_layers], dim=0)
        bias_hh = torch.stack([layer.bias_hh_l0 for layer in rnn_layers], dim=0)
        input_gates = torch.bmm(weight_ih, inputs.unsqueeze(-1)).squeeze(-1) + bias_ih
        hidden_gates = torch.bmm(weight_hh, hidden.unsqueeze(-1)).squeeze(-1) + bias_hh

        if self.rnn_type == "lstm":
            if cell is None:
                cell = torch.zeros_like(hidden)
            gates = input_gates + hidden_gates
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            output_gate = torch.sigmoid(output_gate)
            new_cell = forget_gate * cell + input_gate * candidate
            new_hidden = output_gate * torch.tanh(new_cell)
            return new_hidden, new_cell

        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset_gate * hidden_new)
        new_hidden = (1.0 - update_gate) * candidate + update_gate * hidden
        return new_hidden, None

    def forward(
        self,
        patch: torch.Tensor,
        hidden_state: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
        agent_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
        num_envs, agents_per_env = patch.shape[:2]
        if num_envs != self.num_envs or agents_per_env != self.agents_per_env:
            raise ValueError(
                "输入个体数量与独立大脑数量不一致："
                f"期望 ({self.num_envs}, {self.agents_per_env})，"
                f"实际 ({num_envs}, {agents_per_env})"
            )

        patch_flat = patch.reshape(self.brain_count, *patch.shape[2:])
        encoded_patch = self._encode_all_patches(patch_flat)
        if agent_features is None:
            feature_flat = torch.zeros(
                self.brain_count, 3, device=patch.device, dtype=patch.dtype
            )
        else:
            feature_flat = agent_features.reshape(self.brain_count, -1).to(
                device=patch.device, dtype=patch.dtype
            )
        encoded = torch.cat([encoded_patch, feature_flat], dim=-1)

        if isinstance(hidden_state, tuple):
            hidden_flat = hidden_state[0].reshape(self.brain_count, -1)
            cell_flat = hidden_state[1].reshape(self.brain_count, -1)
        elif hidden_state is not None:
            hidden_flat = hidden_state.reshape(self.brain_count, -1)
            cell_flat = None
        else:
            hidden_flat = None
            cell_flat = None

        new_hidden, new_cell = self._advance_all_memories(
            encoded, hidden_flat, cell_flat
        )
        logits = self._batched_linear(
            new_hidden, [brain.policy_head for brain in self.brains]
        ).reshape(num_envs, agents_per_env, -1)
        values = self._batched_linear(
            new_hidden, [brain.value_head for brain in self.brains]
        ).reshape(num_envs, agents_per_env, 1)
        hidden_out = new_hidden.reshape(num_envs, agents_per_env, -1)
        if new_cell is not None:
            return logits, values, (
                hidden_out,
                new_cell.reshape(num_envs, agents_per_env, -1),
            )
        return logits, values, hidden_out

    @torch.no_grad()
    def inherit_policy_head(
        self,
        env_idx: int,
        parent_idx: int,
        child_idx: int,
        mutation_std: float,
    ) -> list[nn.Parameter]:
        """子代复制父代策略头并叠加小幅高斯变异，返回被替换的参数。"""

        parent = self.brains[self._brain_index(env_idx, parent_idx)].policy_head
        child = self.brains[self._brain_index(env_idx, child_idx)].policy_head
        child.load_state_dict(parent.state_dict())
        std = max(float(mutation_std), 0.0)
        if std > 0:
            for parameter in child.parameters():
                parameter.add_(torch.randn_like(parameter) * std)
        return list(child.parameters())


def build_policy(
    config: Dict,
    default_config: Dict,
    obs_dim: int | None = None,
    patch_shape: Optional[Tuple[int, int, int]] = None,
) -> nn.Module:
    """根据配置构建共享 MLP 或完全独立的 CNN 个体大脑。"""

    model_cfg = default_config.get("model", {}) | config.get("model", {})
    use_cnn = _as_bool(model_cfg.get("use_cnn", POLICY_DEFAULTS["model"]["use_cnn"]))
    if use_cnn:
        if patch_shape is None:
            raise ValueError("use_cnn=True 时必须提供 patch_shape")
        cnn_channels = model_cfg.get("cnn_channels", POLICY_DEFAULTS["model"]["cnn_channels"])
        cnn_feature_dim = int(model_cfg.get("cnn_feature_dim", POLICY_DEFAULTS["model"]["cnn_feature_dim"]))
        rnn_hidden_dim = int(model_cfg.get("rnn_hidden_dim", POLICY_DEFAULTS["model"]["rnn_hidden_dim"]))
        rnn_type = str(model_cfg.get("rnn_type", POLICY_DEFAULTS["model"]["rnn_type"])).lower()
        cnn_pooling = _as_bool(model_cfg.get("cnn_pooling", POLICY_DEFAULTS["model"]["cnn_pooling"]))
        cnn_pool_size = _normalize_pool_size(
            model_cfg.get("cnn_pool_size", POLICY_DEFAULTS["model"]["cnn_pool_size"])
        )
        brain_kwargs = {
            "patch_shape": patch_shape,
            "action_dim": len(action_space),
            "cnn_channels": cnn_channels,
            "cnn_feature_dim": cnn_feature_dim,
            "rnn_hidden_dim": rnn_hidden_dim,
            "rnn_type": rnn_type,
            "agent_feature_dim": 3,
            "cnn_pooling": cnn_pooling,
            "cnn_pool_size": cnn_pool_size,
        }
        independent = _as_bool(
            model_cfg.get("cnn_independent", POLICY_DEFAULTS["model"]["cnn_independent"])
        )
        if independent:
            training_cfg = default_config.get("training", {}) | config.get("training", {})
            configured_agents = config.get("agents", {})
            default_agents = default_config.get("agents", {})
            if "max_per_env" in configured_agents:
                agents_per_env = configured_agents["max_per_env"]
            elif "per_env" in configured_agents:
                agents_per_env = configured_agents["per_env"]
            else:
                agents_per_env = default_agents.get(
                    "max_per_env", default_agents.get("per_env", 32)
                )
            return IndependentCNNPolicies(
                num_envs=int(training_cfg.get("num_envs", 1)),
                agents_per_env=int(agents_per_env),
                brain_kwargs=brain_kwargs,
            )
        return CNNRecurrentPolicy(**brain_kwargs)

    hidden = model_cfg.get(
        "hidden", default_config.get("model", {}).get("hidden", POLICY_DEFAULTS["model"]["hidden"])
    )
    inferred_dim = obs_dim or model_cfg.get("obs_dim", POLICY_DEFAULTS["model"]["obs_dim"])
    return MLPPolicy(obs_dim=inferred_dim, hidden_dim=hidden)
