"""YAML 配置管理工具。"""
from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path
from typing import Dict, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
RENDER_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "render.yaml"


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

    default_config = _read_yaml(DEFAULT_CONFIG_PATH)
    render_config = _read_yaml(RENDER_CONFIG_PATH)
    return _deep_merge(default_config, render_config)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并配置，override 覆盖 base。"""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Tuple[Dict, Dict]:
    """加载指定配置与默认配置，返回 (用户配置, 默认配置)。"""

    user_config = _read_yaml(Path(path)) if path else {}
    return user_config, load_default_config()


def _format_template_value(value) -> str:
    """将单个配置值格式化为紧凑且合法的 YAML。"""

    wrapped = yaml.safe_dump(
        [value],
        allow_unicode=True,
        default_flow_style=True,
        sort_keys=False,
    ).strip()
    return wrapped[1:-1].strip()


def _render_config_template(config: Dict) -> str:
    """把配置值写入默认模板，同时保留模板的布局与注释。"""

    template_parts = [DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").rstrip()]
    if RENDER_CONFIG_PATH.exists():
        template_parts.append(
            "# 渲染设置\n" + RENDER_CONFIG_PATH.read_text(encoding="utf-8").strip()
        )
    lines = "\n\n".join(template_parts).splitlines()
    key_stack: list[tuple[int, str]] = []
    rendered = []

    for line in lines:
        match = re.match(r"^(\s*)([^#\s][^:]*?):(\s*)(.*)$", line)
        if not match:
            rendered.append(line)
            continue

        indent_text, key, _, remainder = match.groups()
        indent = len(indent_text)
        while key_stack and key_stack[-1][0] >= indent:
            key_stack.pop()
        path = [stack_key for _, stack_key in key_stack] + [key]

        value = config
        found = True
        for path_key in path:
            if not isinstance(value, dict) or path_key not in value:
                found = False
                break
            value = value[path_key]

        if not found:
            rendered.append(line)
            continue
        if isinstance(value, dict):
            key_stack.append((indent, key))
            rendered.append(line)
            continue

        comment_match = re.search(r"(\s+#.*)$", remainder)
        comment = comment_match.group(1) if comment_match else ""
        rendered.append(
            f"{indent_text}{key}: {_format_template_value(value)}{comment}"
        )

    return "\n".join(rendered) + "\n"


def save_config_with_comments(path: Path, config: Dict) -> None:
    """保存完整配置，并保留 default.yaml/render.yaml 的顺序和注释。"""

    path.write_text(_render_config_template(config), encoding="utf-8")
