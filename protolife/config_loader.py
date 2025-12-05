"""YAML 配置管理工具。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _read_yaml(path: Path) -> Dict:
    """读取单个 YAML，若不存在则返回空字典。"""

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_default_config() -> Dict:
    """加载默认配置，使用缓存避免重复 IO。"""

    return _read_yaml(DEFAULT_CONFIG_PATH)


def load_config(path: str) -> Tuple[Dict, Dict]:
    """加载指定配置与默认配置，返回 (用户配置, 默认配置)。"""

    user_config = _read_yaml(Path(path)) if path else {}
    return user_config, load_default_config()
