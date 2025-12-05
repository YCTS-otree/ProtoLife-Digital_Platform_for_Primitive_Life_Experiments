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
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        # 常见于 Windows 配置中直接写入 "C:\path\file" 导致反斜杠未转义。
        escaped = text.replace("\\", "\\\\")
        try:
            return yaml.safe_load(escaped) or {}
        except yaml.YAMLError:
            raise ValueError(
                f"无法解析 YAML: {path}. 如使用 Windows 路径请改用正斜杠或双反斜杠。"
            ) from exc


@lru_cache(maxsize=1)
def load_default_config() -> Dict:
    """加载默认配置，使用缓存避免重复 IO。"""

    return _read_yaml(DEFAULT_CONFIG_PATH)


def load_config(path: str) -> Tuple[Dict, Dict]:
    """加载指定配置与默认配置，返回 (用户配置, 默认配置)。"""

    user_config = _read_yaml(Path(path)) if path else {}
    return user_config, load_default_config()
