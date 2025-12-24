"""策略与价值网络占位实现。"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

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


POLICY_DEFAULTS = {
    "model": {
        "hidden": 128,
        "obs_dim": 5,
        "use_cnn": False,
        "cnn_channels": [32, 64],
        "cnn_feature_dim": 128,
        "rnn_hidden_dim": 128,
        "rnn_type": "gru",
    }
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


class CNNRecurrentPolicy(nn.Module):
    """卷积 + 循环结构的策略/价值网络，支持多环境多智能体批量输入。"""

    def __init__(
        self,
        patch_shape: Tuple[int, int, int],
        action_dim: int,
        cnn_channels: Iterable[int],
        cnn_feature_dim: int,
        rnn_hidden_dim: int,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        self.use_cnn = True
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_dim = rnn_hidden_dim
        c, h, w = patch_shape
        convs = []
        in_channels = c
        for out_channels in cnn_channels:
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU())
            in_channels = out_channels
        self.cnn = nn.Sequential(*convs)
        flattened_dim = in_channels * h * w
        self.proj = nn.Linear(flattened_dim, cnn_feature_dim)

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=cnn_feature_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=cnn_feature_dim, hidden_size=rnn_hidden_dim, batch_first=True)

        self.policy_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.value_head = nn.Linear(rnn_hidden_dim, 1)

    def _encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (B, C, H, W)
        x = self.cnn(patch)
        x = x.reshape(x.size(0), -1)
        return torch.relu(self.proj(x))

    def forward(
        self,
        patch: torch.Tensor,
        hidden_state: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
        """接受局部 patch 与循环状态，返回 logits、value 与新隐藏状态。

        输入 patch 形状：(E, A, C, H, W)，hidden_state: GRU 为 (E,A,H)，LSTM 为 ( (E,A,H), (E,A,H) )
        """

        num_envs, agents_per_env = patch.shape[:2]
        batch = num_envs * agents_per_env
        patch_flat = patch.view(batch, *patch.shape[2:])
        encoded = self._encode_patch(patch_flat).unsqueeze(1)  # (B, 1, F)

        if self.rnn_type == "lstm":
            h_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
            if hidden_state is None:
                zeros = torch.zeros(batch, self.rnn_hidden_dim, device=patch.device, dtype=patch.dtype)
                h_state = (zeros.unsqueeze(0), zeros.unsqueeze(0))
            else:
                assert isinstance(hidden_state, tuple)
                h, c = hidden_state
                h_state = (h.view(1, batch, -1), c.view(1, batch, -1))
            rnn_out, (h_out, c_out) = self.rnn(encoded, h_state)
            new_hidden: Tuple[torch.Tensor, torch.Tensor] = (
                h_out.view(num_envs, agents_per_env, -1),
                c_out.view(num_envs, agents_per_env, -1),
            )
        else:
            if hidden_state is None or isinstance(hidden_state, tuple):
                hidden_in = None
            else:
                hidden_in = hidden_state.view(1, batch, -1)
            rnn_out, h_out = self.rnn(encoded, hidden_in)
            new_hidden = h_out.view(num_envs, agents_per_env, -1)

        core = rnn_out[:, -1, :]
        logits = self.policy_head(core).view(num_envs, agents_per_env, -1)
        value = self.value_head(core).view(num_envs, agents_per_env, 1)
        return logits, value, new_hidden


def build_policy(
    config: Dict,
    default_config: Dict,
    obs_dim: int | None = None,
    patch_shape: Optional[Tuple[int, int, int]] = None,
) -> nn.Module:
    """根据配置构建策略网络。"""

    model_cfg = default_config.get("model", {}) | config.get("model", {})
    use_cnn = model_cfg.get("use_cnn", POLICY_DEFAULTS["model"]["use_cnn"])
    if use_cnn:
        if patch_shape is None:
            raise ValueError("use_cnn=True 时必须提供 patch_shape")
        cnn_channels = model_cfg.get("cnn_channels", POLICY_DEFAULTS["model"]["cnn_channels"])
        cnn_feature_dim = model_cfg.get("cnn_feature_dim", POLICY_DEFAULTS["model"]["cnn_feature_dim"])
        rnn_hidden_dim = model_cfg.get("rnn_hidden_dim", POLICY_DEFAULTS["model"]["rnn_hidden_dim"])
        rnn_type = str(model_cfg.get("rnn_type", POLICY_DEFAULTS["model"]["rnn_type"])).lower()
        return CNNRecurrentPolicy(
            patch_shape=patch_shape,
            action_dim=len(action_space),
            cnn_channels=cnn_channels,
            cnn_feature_dim=cnn_feature_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_type=rnn_type,
        )

    hidden = model_cfg.get(
        "hidden", default_config.get("model", {}).get("hidden", POLICY_DEFAULTS["model"]["hidden"])
    )
    inferred_dim = obs_dim or model_cfg.get("obs_dim", POLICY_DEFAULTS["model"]["obs_dim"])
    return MLPPolicy(obs_dim=inferred_dim, hidden_dim=hidden)
