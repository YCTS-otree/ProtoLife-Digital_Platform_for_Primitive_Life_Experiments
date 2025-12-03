"""训练与环境 checkpoint 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    path: Path,
    env_state: Dict[str, torch.Tensor],
    policy_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """保存包含环境、模型与训练元信息的完整快照。"""

    payload = {
        "env_state": env_state,
        "policy_state_dict": policy_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: Optional[torch.device] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, Any]], Dict[str, Any]]:
    """读取 checkpoint 并返回各组件状态。"""

    checkpoint = torch.load(path, map_location=map_location)
    return (
        checkpoint["env_state"],
        checkpoint["policy_state_dict"],
        checkpoint.get("optimizer_state_dict"),
        checkpoint.get("meta", {}),
    )
