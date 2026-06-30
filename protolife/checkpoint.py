"""训练与环境 checkpoint 工具。"""

from __future__ import annotations

import random
import re
import math
import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


_SPLIT_CHECKPOINT_RE = re.compile(
    r"^(?P<prefix>.*?)(?:env|model|optim)_step_(?P<step>\d+)\.(?:pt|pth)$"
)
_LEGACY_CHECKPOINT_RE = re.compile(
    r"^(?P<prefix>.*?)full_step_(?P<step>\d+)\.pt$"
)
_RNG_STATE_META_KEY = "_rng_state"
DEFAULT_CHECKPOINT_MIN_REMAINING_BYTES = 2 * 1024**3
DEFAULT_CHECKPOINT_SIZE_SAFETY_FACTOR = 1.10


class InsufficientCheckpointSpaceError(RuntimeError):
    """checkpoint 预计写入后无法保留安全空间。"""

    def __init__(self, *, free_bytes: int, required_bytes: int, path: Path) -> None:
        self.free_bytes = int(free_bytes)
        self.required_bytes = int(required_bytes)
        self.path = Path(path)
        super().__init__(
            f"checkpoint 空间不足: 可用 {free_bytes / 1024**3:.2f} GiB，"
            f"安全写入需要 {required_bytes / 1024**3:.2f} GiB，目录={path}"
        )


def estimate_payload_bytes(*payloads: Any) -> int:
    """按唯一 tensor storage 估算序列化体积，避免真正写盘后才发现空间不足。"""

    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int, int]] = set()

    def visit(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            size = int(storage.nbytes())
            key = (str(value.device), int(storage.data_ptr()), size)
            if key in seen_storages:
                return 0
            seen_storages.add(key)
            return size + 128
        if isinstance(value, np.ndarray):
            return int(value.nbytes) + 128
        if isinstance(value, (str, bytes, bytearray)):
            return len(value) + 64
        if value is None or isinstance(value, (bool, int, float)):
            return 32
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, dict):
            return 128 + sum(visit(key) + visit(item) for key, item in value.items())
        if isinstance(value, (list, tuple, set)):
            return 64 + sum(visit(item) for item in value)
        return 128

    return sum(visit(payload) for payload in payloads)


def ensure_checkpoint_space(
    checkpoint_dir: Path,
    *payloads: Any,
    min_remaining_bytes: int = DEFAULT_CHECKPOINT_MIN_REMAINING_BYTES,
    size_safety_factor: float = DEFAULT_CHECKPOINT_SIZE_SAFETY_FACTOR,
    free_bytes: int | None = None,
) -> tuple[int, int]:
    """在写入任何字节前确认磁盘足以容纳整组文件及保留空间。"""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if size_safety_factor < 1.0:
        raise ValueError("checkpoint size_safety_factor 不能小于 1")
    estimated = estimate_payload_bytes(*payloads)
    required = math.ceil(estimated * size_safety_factor) + int(min_remaining_bytes)
    available = (
        int(free_bytes)
        if free_bytes is not None
        else int(shutil.disk_usage(checkpoint_dir).free)
    )
    if available < required:
        raise InsufficientCheckpointSpaceError(
            free_bytes=available,
            required_bytes=required,
            path=checkpoint_dir,
        )
    return estimated, available


def capture_rng_state() -> Dict[str, Any]:
    """捕获续训可能用到的 Python、NumPy、PyTorch CPU/CUDA 随机状态。"""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> bool:
    """尽可能恢复所有随机源；完整恢复返回 ``True``。"""

    if not state:
        return False

    complete = True
    restorers = (
        ("python", lambda value: random.setstate(value)),
        ("numpy", lambda value: np.random.set_state(value)),
        ("torch_cpu", lambda value: torch.set_rng_state(value.cpu())),
    )
    for name, restore in restorers:
        value = state.get(name)
        if value is None:
            complete = False
            continue
        try:
            restore(value)
        except Exception as exc:  # noqa: BLE001
            complete = False
            warnings.warn(f"恢复 {name} RNG 状态失败: {exc}", RuntimeWarning, stacklevel=2)

    saved_cuda_states = state.get("torch_cuda") or []
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(saved_cuda_states) != current_cuda_count:
        complete = False
        warnings.warn(
            "checkpoint CUDA RNG 设备数与当前环境不同："
            f"保存={len(saved_cuda_states)}，当前={current_cuda_count}；将恢复重叠设备。",
            RuntimeWarning,
            stacklevel=2,
        )
    for device_index, cuda_state in enumerate(saved_cuda_states[:current_cuda_count]):
        try:
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)
        except Exception as exc:  # noqa: BLE001
            complete = False
            warnings.warn(
                f"恢复 CUDA:{device_index} RNG 状态失败: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return complete


def checkpoint_file_is_structurally_valid(path: Path) -> bool:
    """快速检查 PyTorch ZIP 的目录尾；不读取数 GB 的张量内容。"""

    path = Path(path)
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
        # 早期 torch.save 可使用非 ZIP 格式，交给 torch.load 保持兼容。
        if magic != b"PK\x03\x04":
            return True
        with zipfile.ZipFile(path) as archive:
            archive.infolist()
        return True
    except (OSError, zipfile.BadZipFile):
        return False


def _state_values_equal(expected: Any, actual: Any) -> bool:
    """递归比较 checkpoint 内容，供删除旧文件前做严格校验。"""

    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(expected, actual)
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and np.array_equal(expected, actual)
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(_state_values_equal(expected[key], actual[key]) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, type(expected))
            and len(expected) == len(actual)
            and all(_state_values_equal(left, right) for left, right in zip(expected, actual))
        )
    return expected == actual


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """先完整写入同目录临时文件，再替换目标，避免留下半个 checkpoint。"""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_torch_payload(
    path: Path,
    payload: Any,
    *,
    min_remaining_bytes: int = DEFAULT_CHECKPOINT_MIN_REMAINING_BYTES,
    size_safety_factor: float = DEFAULT_CHECKPOINT_SIZE_SAFETY_FACTOR,
) -> None:
    """带空间预检、原子替换和 ZIP 结构验证地保存单个 PyTorch 文件。"""

    path = Path(path)
    ensure_checkpoint_space(
        path.parent,
        payload,
        min_remaining_bytes=min_remaining_bytes,
        size_safety_factor=size_safety_factor,
    )
    _atomic_torch_save(payload, path)
    if not checkpoint_file_is_structurally_valid(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint 写后结构校验失败，已删除: {path}")


def save_checkpoint(
    path: Path,
    env_state: Dict[str, torch.Tensor],
    policy_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """保存旧式单文件完整快照。

    该接口仅为兼容已有调用和旧 checkpoint 保留。新训练应使用
    :func:`save_split_checkpoint`，避免重复保存模型和优化器。
    """

    payload = {
        "env_state": env_state,
        "policy_state_dict": policy_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "rng_state": capture_rng_state(),
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_payload(path, payload)


def split_checkpoint_paths(
    checkpoint_dir: Path, step: int, filename_prefix: str = ""
) -> dict[str, Path]:
    """返回新式三文件 checkpoint 的标准路径。"""

    return {
        "env": checkpoint_dir / f"{filename_prefix}env_step_{step}.pth",
        "model": checkpoint_dir / f"{filename_prefix}model_step_{step}.pth",
        "optim": checkpoint_dir / f"{filename_prefix}optim_step_{step}.pth",
    }


def save_split_checkpoint(
    checkpoint_dir: Path,
    step: int,
    env_state: Dict[str, torch.Tensor],
    policy_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    filename_prefix: str = "",
    min_remaining_bytes: int = DEFAULT_CHECKPOINT_MIN_REMAINING_BYTES,
    size_safety_factor: float = DEFAULT_CHECKPOINT_SIZE_SAFETY_FACTOR,
) -> dict[str, Path]:
    """保存不含重复数据的环境、模型和优化器三文件 checkpoint。"""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = split_checkpoint_paths(checkpoint_dir, step, filename_prefix)
    checkpoint_meta = dict(meta or {})
    checkpoint_meta["step"] = int(step)
    rng_state = capture_rng_state()
    has_complete_rng = "random_generator_state" in env_state
    checkpoint_meta["checkpoint_format"] = (
        "split_v2_rng" if has_complete_rng else "split_v1"
    )
    env_payload = {
        "env_state": env_state,
        "rng_state": rng_state,
        "meta": checkpoint_meta,
    }
    payloads = {
        "model": policy_state_dict,
        "optim": optimizer_state_dict,
        "env": env_payload,
    }
    ensure_checkpoint_space(
        checkpoint_dir,
        *payloads.values(),
        min_remaining_bytes=min_remaining_bytes,
        size_safety_factor=size_safety_factor,
    )

    temporary_paths = {
        key: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for key, path in paths.items()
    }
    try:
        # 三个临时文件全部写完并验证后才落位，避免出现看似完整的半套 checkpoint。
        for key in ("model", "optim", "env"):
            torch.save(payloads[key], temporary_paths[key])
            if not checkpoint_file_is_structurally_valid(temporary_paths[key]):
                raise RuntimeError(f"checkpoint 临时文件结构校验失败: {temporary_paths[key]}")
        for key in ("model", "optim", "env"):
            temporary_paths[key].replace(paths[key])
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
    return paths


def migrate_legacy_checkpoint(
    path: Path,
    *,
    delete_full: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """将旧 ``full_step`` 转为三文件格式，并可在校验后删除原文件。"""

    path = Path(path)
    match = _LEGACY_CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"不是旧式 full checkpoint: {path}")
    if not path.exists():
        raise FileNotFoundError(path)

    filename_step = int(match.group("step"))
    filename_prefix = match.group("prefix")
    checkpoint = torch.load(path, map_location="cpu")
    meta = dict(checkpoint.get("meta", {}))
    metadata_step = int(meta.get("step", filename_step))
    if metadata_step != filename_step:
        raise ValueError(
            f"checkpoint 文件名 step={filename_step} 与元数据 step={metadata_step} 不一致"
        )

    paths = split_checkpoint_paths(path.parent, filename_step, filename_prefix)
    meta["step"] = filename_step
    has_complete_rng = checkpoint.get("rng_state") is not None and (
        "random_generator_state" in checkpoint["env_state"]
    )
    meta["checkpoint_format"] = "split_v2_rng" if has_complete_rng else "split_v1"
    payloads = {
        "env": {
            "env_state": checkpoint["env_state"],
            "rng_state": checkpoint.get("rng_state"),
            "meta": meta,
        },
        "model": checkpoint["policy_state_dict"],
        "optim": checkpoint.get("optimizer_state_dict"),
    }
    for key, destination in paths.items():
        if overwrite or not destination.exists():
            _atomic_torch_save(payloads[key], destination)

    failed: list[str] = []
    env_payload = torch.load(paths["env"], map_location="cpu")
    if not _state_values_equal(checkpoint["env_state"], env_payload.get("env_state")):
        failed.append("环境")
    if not _state_values_equal(checkpoint.get("rng_state"), env_payload.get("rng_state")):
        failed.append("RNG")
    migrated_step = int(env_payload.get("meta", {}).get("step", -1))
    del env_payload

    saved_policy = torch.load(paths["model"], map_location="cpu")
    if not _state_values_equal(checkpoint["policy_state_dict"], saved_policy):
        failed.append("模型")
    del saved_policy

    saved_optimizer = torch.load(paths["optim"], map_location="cpu")
    if not _state_values_equal(checkpoint.get("optimizer_state_dict"), saved_optimizer):
        failed.append("优化器")
    del saved_optimizer

    if failed:
        raise ValueError(
            f"迁移后的 {'/'.join(failed)} 与 full 文件不一致；未删除原文件。"
            "如需以 full 内容覆盖旧三件套，请传入 overwrite=True。"
        )
    if migrated_step != filename_step:
        raise ValueError(f"迁移后的 checkpoint step 校验失败；未删除原文件: {path}")

    if delete_full:
        path.unlink()
    return paths


def checkpoint_step(path: Path) -> int | None:
    """从旧式或新式 checkpoint 文件名解析 step。"""

    match = _SPLIT_CHECKPOINT_RE.match(path.name) or _LEGACY_CHECKPOINT_RE.match(path.name)
    return int(match.group("step")) if match else None


def checkpoint_filename_prefix(path: Path) -> str:
    """返回 checkpoint 文件中位于 env/model/optim/full 之前的前缀。"""

    match = _SPLIT_CHECKPOINT_RE.match(path.name) or _LEGACY_CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"无法解析 checkpoint 文件名: {path}")
    return match.group("prefix")


def resolve_split_checkpoint(path: Path) -> dict[str, Path]:
    """将新式三件套中的任一文件解析为整组路径。"""

    step = checkpoint_step(path)
    if step is None or _LEGACY_CHECKPOINT_RE.match(path.name):
        raise ValueError(f"不是新式 checkpoint 文件: {path}")
    return split_checkpoint_paths(path.parent, step, checkpoint_filename_prefix(path))


def load_checkpoint(
    path: Path,
    map_location: Optional[torch.device] = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Optional[Dict[str, Any]],
    Dict[str, Any],
]:
    """读取旧式 full 文件或新式三文件 checkpoint。"""

    path = Path(path)
    if _LEGACY_CHECKPOINT_RE.match(path.name) or not _SPLIT_CHECKPOINT_RE.match(path.name):
        checkpoint = torch.load(path, map_location=map_location)
        if not isinstance(checkpoint, dict) or not {
            "env_state",
            "policy_state_dict",
        }.issubset(checkpoint):
            raise ValueError(f"checkpoint 格式不受支持: {path}")
        meta = dict(checkpoint.get("meta", {}))
        meta[_RNG_STATE_META_KEY] = checkpoint.get("rng_state")
        return (
            checkpoint["env_state"],
            checkpoint["policy_state_dict"],
            checkpoint.get("optimizer_state_dict"),
            meta,
        )

    paths = resolve_split_checkpoint(path)
    missing = [item for item in paths.values() if not item.exists()]
    if missing:
        names = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(f"checkpoint 三件套不完整，缺少: {names}")

    env_payload = torch.load(paths["env"], map_location=map_location)
    policy_state = torch.load(paths["model"], map_location=map_location)
    optimizer_state = torch.load(paths["optim"], map_location=map_location)
    meta = dict(env_payload.get("meta", {}))
    meta[_RNG_STATE_META_KEY] = env_payload.get("rng_state")
    return (
        env_payload["env_state"],
        policy_state,
        optimizer_state,
        meta,
    )


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """查找最近修改且可恢复的 checkpoint；同 step 优先新式文件。"""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    candidates: list[tuple[int, float, int, Path]] = []
    for env_path in checkpoint_dir.glob("*env_step_*.pth"):
        step = checkpoint_step(env_path)
        if step is None:
            continue
        prefix = checkpoint_filename_prefix(env_path)
        if prefix.startswith("无效_"):
            continue
        paths = split_checkpoint_paths(checkpoint_dir, step, prefix)
        existing = [item for item in paths.values() if item.exists()]
        invalid = [
            item for item in existing if not checkpoint_file_is_structurally_valid(item)
        ]
        if invalid:
            warnings.warn(
                "已跳过损坏的 checkpoint 文件: "
                + ", ".join(str(item) for item in invalid),
                RuntimeWarning,
                stacklevel=2,
            )
        if len(existing) == len(paths) and not invalid:
            newest_mtime = max(item.stat().st_mtime for item in paths.values())
            candidates.append((step, newest_mtime, 1, paths["env"]))

    for full_path in checkpoint_dir.glob("*full_step_*.pt"):
        step = checkpoint_step(full_path)
        if step is None:
            continue
        if checkpoint_filename_prefix(full_path).startswith("无效_"):
            continue
        if not checkpoint_file_is_structurally_valid(full_path):
            warnings.warn(
                f"已跳过损坏的 checkpoint 文件: {full_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        candidates.append((step, full_path.stat().st_mtime, 0, full_path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1], item[0], item[2]))[3]
