"""YAML 配置管理工具。"""
from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    """加载 YAML 配置文件。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
