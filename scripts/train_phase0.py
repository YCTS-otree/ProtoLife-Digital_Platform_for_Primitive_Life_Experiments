"""Phase0 训练脚本占位。

示例用法：
    python -m scripts.train_phase0 --config config/phase0_survival.yaml
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import json
import os
import re
import secrets
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import yaml

from protolife.config_loader import load_config, save_config_with_comments
from protolife.checkpoint import (
    InsufficientCheckpointSpaceError,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_split_checkpoint,
    save_torch_payload,
)
from protolife.env import ProtoLifeEnv
from protolife.policy import action_space, build_policy
from protolife.utils.seed_utils import generate_seed, set_seed


MIN_VALID_SIMULATION_STEPS = 1024
SIMULATION_COUNT_PATH = Path("Simulation_counting.json")
LEGACY_SIMULATION_COUNT_PATH = Path("Simulation_counting.txt")
SIMULATION_COUNT_SCHEMA_VERSION = 2
INVALID_ARTIFACT_MARKER = "无效_"
_CONSOLE_CAPTURE = None
_SIMULATION_CONTEXT: dict = {}
_ACTIVE_PROFILER = None


class _TeeStream:
    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, text):
        self.primary.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _start_console_capture(path: Path) -> None:
    global _CONSOLE_CAPTURE
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)
    _CONSOLE_CAPTURE = (original_stdout, original_stderr, log_file)


def _stop_console_capture() -> None:
    global _CONSOLE_CAPTURE
    if _CONSOLE_CAPTURE is None:
        return
    original_stdout, original_stderr, log_file = _CONSOLE_CAPTURE
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        _CONSOLE_CAPTURE = None


def _stop_active_profiler() -> Path | None:
    global _ACTIVE_PROFILER
    if _ACTIVE_PROFILER is None:
        return None
    profiler = _ACTIVE_PROFILER
    _ACTIVE_PROFILER = None
    output_dir = Path(profiler.output_dir)
    profiler.stop()
    return output_dir


def _record_valid_simulation_unlocked(
    *,
    started_at: datetime,
    start_step: int,
    total_steps: int,
    run_steps: int,
    stop_reason: str,
    model_name: str = "未知模型",
    resumed: bool = False,
    resume_from: str | None = None,
    path: Path = SIMULATION_COUNT_PATH,
) -> int | None:
    if run_steps <= MIN_VALID_SIMULATION_STEPS:
        return None

    path = Path(path)
    json_existed = path.exists()
    data = _load_simulation_count_data(path)
    # The top-level count is deliberately authoritative: users can correct it
    # without having to manufacture/delete historical records.  The legacy txt
    # is consulted exactly once, when the JSON file is first created.
    existing_count = int(data.get("effective_simulation_count", 0))
    legacy_path = _find_legacy_simulation_count_path(path)
    legacy_count = 0
    legacy_records: list[dict] = []
    if not json_existed:
        legacy_count = _read_legacy_simulation_count(legacy_path)
        legacy_records = _read_legacy_simulation_records(legacy_path)
        existing_count = max(existing_count, legacy_count)
    effective_count = existing_count + 1

    simulations = data.setdefault("simulations", [])
    simulations.extend(legacy_records)
    simulations.append(
        {
            "sequence": effective_count,
            "started_at": started_at.isoformat(timespec="seconds"),
            "date": started_at.strftime("%Y-%m-%d"),
            "time": started_at.strftime("%H:%M:%S"),
            "model_name": model_name,
            "training_type": "resume" if resumed else "new",
            "resumed": bool(resumed),
            "resume_from": resume_from,
            "start_step": int(start_step),
            "end_step": int(total_steps),
            "run_steps": int(run_steps),
            "stop_reason": stop_reason,
        }
    )
    data["schema_version"] = SIMULATION_COUNT_SCHEMA_VERSION
    data["effective_simulation_count"] = effective_count
    data["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if legacy_count:
        migration = data.setdefault("legacy_migration", {})
        migration["source"] = str(legacy_path.resolve())
        migration["detected_count"] = legacy_count
        migration["migrated_record_count"] = len(legacy_records)
        migration["historical_count_without_detail"] = max(
            0, legacy_count - len(legacy_records)
        )
        migration.setdefault(
            "first_migrated_at", datetime.now().astimezone().isoformat(timespec="seconds")
        )
        migration["note"] = "原 txt 保留不变；JSON 为新的权威记录"
    _atomic_write_json(path, data)
    return effective_count


def _find_legacy_simulation_count_path(json_path: Path) -> Path:
    """兼容手工改名过的 Simulation_counting*.txt，选择累计值最大的版本。"""

    preferred = json_path.with_name(LEGACY_SIMULATION_COUNT_PATH.name)
    candidates = list(json_path.parent.glob("Simulation_counting*.txt"))
    if preferred not in candidates:
        candidates.append(preferred)
    existing = [item for item in candidates if item.is_file()]
    if not existing:
        return preferred
    return max(
        existing,
        key=lambda item: (_read_legacy_simulation_count(item), item.stat().st_mtime_ns),
    )


@contextmanager
def _simulation_count_lock(path: Path, timeout_seconds: float = 30.0):
    """用同目录独占锁保证并行训练不会领取相同的有效编号。"""

    lock_path = path.with_name(f".{path.name}.lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 300
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待模拟计数锁超时: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _record_valid_simulation(
    *,
    started_at: datetime,
    start_step: int,
    total_steps: int,
    run_steps: int,
    stop_reason: str,
    model_name: str = "未知模型",
    resumed: bool = False,
    resume_from: str | None = None,
    path: Path = SIMULATION_COUNT_PATH,
) -> int | None:
    if run_steps <= MIN_VALID_SIMULATION_STEPS:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _simulation_count_lock(path):
        return _record_valid_simulation_unlocked(
            started_at=started_at,
            start_step=start_step,
            total_steps=total_steps,
            run_steps=run_steps,
            stop_reason=stop_reason,
            model_name=model_name,
            resumed=resumed,
            resume_from=resume_from,
            path=path,
        )


def _new_simulation_count_data() -> dict:
    return {
        "schema_version": SIMULATION_COUNT_SCHEMA_VERSION,
        "effective_simulation_count": 0,
        "simulations": [],
    }


def _load_simulation_count_data(path: Path) -> dict:
    if not path.exists():
        return _new_simulation_count_data()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取模拟计数 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"模拟计数 JSON 顶层必须是对象: {path}")
    try:
        count = int(data.get("effective_simulation_count", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"effective_simulation_count 必须是非负整数: {path}"
        ) from exc
    if count < 0:
        raise RuntimeError(f"effective_simulation_count 不能为负数: {path}")
    simulations = data.get("simulations", [])
    if not isinstance(simulations, list):
        raise RuntimeError(f"simulations 必须是数组: {path}")
    data["effective_simulation_count"] = count
    data["simulations"] = simulations
    return data


def _read_legacy_simulation_count(path: Path) -> int:
    text = _read_legacy_simulation_text(path)
    if text is None:
        return 0

    # A manually corrected cumulative field must win over the old line-count
    # scheme. Sequence numbers and line count remain useful fallbacks.
    cumulative = [
        int(value)
        for value in re.findall(r"累计有效模拟\s*[=:：]\s*(\d+)", text)
    ]
    sequences = [
        int(value) for value in re.findall(r"有效模拟\s*#\s*(\d+)", text)
    ]
    matching_lines = sum(1 for line in text.splitlines() if "有效模拟" in line)
    return max([0, matching_lines, *cumulative, *sequences])


def _read_legacy_simulation_text(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _read_legacy_simulation_records(path: Path) -> list[dict]:
    text = _read_legacy_simulation_text(path)
    if text is None:
        return []
    records: list[dict] = []
    for line in text.splitlines():
        match = re.match(r"\s*有效模拟\s*#\s*(\d+)", line)
        if match is None:
            continue
        fields: dict[str, str] = {}
        for part in line.split("|")[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key.strip()] = value.strip()

        date = fields.get("日期", "")
        clock = fields.get("时间", "")
        started_at = f"{date}T{clock}" if date and clock else None

        def parse_int(key: str) -> int | None:
            try:
                return int(fields[key])
            except (KeyError, TypeError, ValueError):
                return None

        resume_from = fields.get("原PT路径")
        if resume_from in (None, "", "无"):
            resume_from = None
        training_type_text = fields.get("类型", "")
        resumed = training_type_text == "续训"
        records.append(
            {
                "sequence": int(match.group(1)),
                "started_at": started_at,
                "date": date or None,
                "time": clock or None,
                "model_name": fields.get("模型", "未知模型"),
                "training_type": "resume" if resumed else "new",
                "resumed": resumed,
                "resume_from": resume_from,
                "start_step": parse_int("开始step"),
                "end_step": parse_int("结束step"),
                "run_steps": parse_int("本次训练步数"),
                "stop_reason": fields.get("停止原因", "未知"),
                "migrated_from_legacy": True,
            }
        )
    return records


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _register_run_artifacts(*paths: Path) -> None:
    artifacts = _SIMULATION_CONTEXT.setdefault("artifacts", [])
    known = {str(item) for item in artifacts}
    for path in paths:
        value = str(Path(path))
        if value not in known:
            artifacts.append(value)
            known.add(value)


def _attach_artifacts_to_simulation(
    sequence: int, artifacts: list[Path], path: Path | None = None
) -> None:
    path = Path(path or SIMULATION_COUNT_PATH)
    with _simulation_count_lock(path):
        data = _load_simulation_count_data(path)
        matching = [
            record
            for record in data.get("simulations", [])
            if int(record.get("sequence", -1)) == int(sequence)
        ]
        if not matching:
            raise RuntimeError(f"计数 JSON 中找不到有效模拟 #{sequence}，无法登记产物")
        matching[-1]["artifacts"] = [str(item.resolve()) for item in artifacts]
        data["schema_version"] = SIMULATION_COUNT_SCHEMA_VERSION
        _atomic_write_json(path, data)


def _finalize_run_artifacts() -> list[Path]:
    """有效训练改为 ``#N_``；无效训练保留 ``无效_`` 标记。"""

    if not _SIMULATION_CONTEXT or _SIMULATION_CONTEXT.get("artifacts_finalized"):
        return []
    original_paths = [Path(item) for item in _SIMULATION_CONTEXT.get("artifacts", [])]
    sequence = _SIMULATION_CONTEXT.get("sequence")
    if sequence is None:
        _SIMULATION_CONTEXT["artifacts_finalized"] = True
        return [item for item in original_paths if item.exists()]

    finalized: list[Path] = []
    invalid_prefix = str(
        _SIMULATION_CONTEXT.get("invalid_artifact_prefix", INVALID_ARTIFACT_MARKER)
    )
    for source in original_paths:
        if not source.name.startswith(INVALID_ARTIFACT_MARKER):
            destination = source
        else:
            suffix = (
                source.name[len(invalid_prefix) :]
                if source.name.startswith(invalid_prefix)
                else source.name[len(INVALID_ARTIFACT_MARKER) :]
            )
            destination = source.with_name(
                f"#{int(sequence)}_{suffix}"
            )
        if source.exists() and source != destination:
            if destination.exists():
                raise FileExistsError(f"有效模拟产物已存在，拒绝覆盖: {destination}")
            source.replace(destination)
        if destination.exists():
            finalized.append(destination)

    _attach_artifacts_to_simulation(
        int(sequence),
        finalized,
        Path(_SIMULATION_CONTEXT.get("simulation_count_path", SIMULATION_COUNT_PATH)),
    )
    _SIMULATION_CONTEXT["artifacts"] = [str(item) for item in finalized]
    _SIMULATION_CONTEXT["artifacts_finalized"] = True
    return finalized


def _finalize_simulation_context(stop_reason: str | None = None) -> int | None:
    if not _SIMULATION_CONTEXT or _SIMULATION_CONTEXT.get("finalized"):
        return None
    if not _SIMULATION_CONTEXT.get("count_simulation", True):
        _SIMULATION_CONTEXT["finalized"] = True
        return None
    if stop_reason is not None:
        _SIMULATION_CONTEXT["stop_reason"] = stop_reason
    count = _record_valid_simulation(
        started_at=_SIMULATION_CONTEXT["started_at"],
        start_step=int(_SIMULATION_CONTEXT.get("start_step", 0)),
        total_steps=int(_SIMULATION_CONTEXT.get("total_steps", 0)),
        run_steps=int(_SIMULATION_CONTEXT.get("run_steps", 0)),
        stop_reason=str(_SIMULATION_CONTEXT.get("stop_reason", "未知")),
        model_name=str(_SIMULATION_CONTEXT.get("model_name", "未知模型")),
        resumed=bool(_SIMULATION_CONTEXT.get("resumed", False)),
        resume_from=_SIMULATION_CONTEXT.get("resume_from"),
    )
    _SIMULATION_CONTEXT["sequence"] = count
    _SIMULATION_CONTEXT["finalized"] = True
    return count


TRAIN_DEFAULTS = {
    "world": {"random_seed": 0},
    "training": {
        "checkpoint_dir": "checkpoints",
        "initial_model": None,
        "auto_resume": True,
        "rollout_steps": 128,
        "n_steps": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "ppo_epochs": 4,
        "ppo_minibatch_steps": 4,
        "value_coef": 0.5,
        "normalize_advantages": True,
        "save_interval": 100,
        "checkpoint_min_free_gb": 2.0,
        "checkpoint_size_safety_factor": 1.10,
        "entropy_coef": 0.01,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "action_noise": {"gaussian_std": 0.0, "epsilon": 0.0},
        "print_actions": False,
        "print_interval": 128,
    },
    "logging": {"save_dir": "runs/default", "snapshot_interval": 50},
}


def _load_state_dict_lenient(model: torch.nn.Module, state_dict: dict) -> bool:
    """加载 state_dict，自动跳过尺寸不匹配的权重并返回是否有跳过。"""

    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    dropped_keys = []
    for key, tensor in state_dict.items():
        if key not in current:
            continue
        if tensor.shape != current[key].shape:
            dropped_keys.append(key)
            continue
        compatible[key] = tensor

    model.load_state_dict(compatible, strict=False)
    missing_keys = [key for key in current if key not in compatible]
    if dropped_keys:
        print(
            "[警告] 以下权重尺寸不匹配，已跳过加载（可能是 observation_radius 调整导致）：",
            dropped_keys,
        )
    if missing_keys:
        print(f"[警告] checkpoint 缺少 {len(missing_keys)} 个当前模型参数，使用新初始化值")
    return bool(dropped_keys or missing_keys)


def _extract_model_state(payload: dict) -> dict:
    """兼容单脑模型、完整策略模型以及完整 checkpoint。"""

    for key in ("brain_state_dict", "policy_state_dict", "model_state_dict", "state_dict"):
        state = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(state, dict):
            return state
    return payload


def _load_initial_model(policy: torch.nn.Module, path: Path, device: torch.device) -> None:
    """将一个单个体模型复制到所有独立大脑，或载入完整策略模型。"""

    payload = torch.load(path, map_location=device)
    state = _extract_model_state(payload)
    if not isinstance(state, dict):
        raise ValueError(f"初始模型格式不受支持: {path}")

    is_independent = getattr(policy, "cnn_independent", False)
    is_full_independent_state = any(str(key).startswith("brains.") for key in state)
    if is_independent and not is_full_independent_state:
        skipped = False
        for brain in policy.brains:
            skipped = _load_state_dict_lenient(brain, state) or skipped
        if skipped:
            print("[警告] 初始单脑模型有部分参数尺寸不匹配，已跳过")
        print(f"已将初始单脑模型复制到 {policy.brain_count} 个个体: {path}")
        return

    _load_state_dict_lenient(policy, state)
    print(f"已加载完整初始模型: {path}")


def _select_last_survivor(env: ProtoLifeEnv) -> dict | None:
    """在 GPU 上保留最佳幸存者索引；仅在真正灭绝时同步到 CPU。"""

    energy = env.agent_batch.state["energy"]
    health = env.agent_batch.state["health"]
    alive = energy > 0
    if env.use_health:
        alive = alive & (health > 0)
    score = energy / max(env.energy_max, 1e-6)
    if env.use_health:
        score = score + health / max(env.health_max, 1e-6)
    score = torch.where(alive, score, torch.full_like(score, float("-inf")))
    flat_index = score.reshape(-1).argmax()
    return {
        "flat_index": flat_index.detach().clone(),
        "agents_per_env": env.agent_batch.agents_per_env,
        "energy": energy.reshape(-1)[flat_index].detach().clone(),
        "health": health.reshape(-1)[flat_index].detach().clone(),
    }


def _save_last_survivor_model(
    policy: torch.nn.Module,
    checkpoint_dir: Path,
    survivor: dict,
    step: int,
    filename_prefix: str = "",
    min_remaining_bytes: int = 2 * 1024**3,
    size_safety_factor: float = 1.10,
) -> Path:
    """保存灭绝前最后存活个体的完整独立大脑。"""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"{filename_prefix}last_survivor_step_{step}.pth"
    if "flat_index" in survivor:
        flat_index = int(survivor["flat_index"].item())
        env_idx, agent_idx = divmod(flat_index, int(survivor["agents_per_env"]))
    else:
        env_idx = int(survivor["env_idx"])
        agent_idx = int(survivor["agent_idx"])
    energy_value = survivor["energy"]
    health_value = survivor["health"]
    payload = {
        "step": int(step),
        "env_idx": env_idx,
        "agent_idx": agent_idx,
        "energy_before_final_step": float(
            energy_value.item() if isinstance(energy_value, torch.Tensor) else energy_value
        ),
        "health_before_final_step": float(
            health_value.item() if isinstance(health_value, torch.Tensor) else health_value
        ),
    }
    if getattr(policy, "cnn_independent", False):
        brain_index = env_idx * policy.agents_per_env + agent_idx
        payload["brain_state_dict"] = policy.brains[brain_index].state_dict()
    else:
        payload["policy_state_dict"] = policy.state_dict()
    save_torch_payload(
        destination,
        payload,
        min_remaining_bytes=min_remaining_bytes,
        size_safety_factor=size_safety_factor,
    )
    return destination


def get_cfg(config: dict, default_config: dict, section: str, key: str, fallback):
    return config.get(section, {}).get(
        key,
        default_config.get(section, {}).get(key, TRAIN_DEFAULTS.get(section, {}).get(key, fallback)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife Phase0 训练")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径，不提供时会使用 model/<name>/<name>.yaml",
    )
    parser.add_argument("--save-interval", type=int, default=None, help="多少步保存一次模型/存档")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint 输出目录")
    parser.add_argument("--model-dir", type=str, default=None, help="模型目录，支持指向已有模型继续训练")
    parser.add_argument("--model-name", type=str, default=None, help="模型名称，若缺省则交互式输入")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从旧 full 文件或新式 env/model/optim 三件套中的任一文件继续推演",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="忽略已有 checkpoint，从初始模型或随机参数开始"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="无 checkpoint 时使用的初始模型；单脑模型会复制给所有个体",
    )
    parser.add_argument("--print-interval", type=int, default=None, help="训练日志打印间隔")
    parser.add_argument(
        "--profiler",
        nargs="?",
        const="config/PyTorch_Profiler.yaml",
        default=None,
        metavar="YAML",
        help=(
            "启用按需 PyTorch Profiler；省略 YAML 时使用 "
            "config/PyTorch_Profiler.yaml"
        ),
    )
    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w.-]", "_", name)
    return name


def _merge_dict(default: dict, override: dict) -> dict:
    result = copy.deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_random_seed(config: dict, default_config: dict) -> int:
    world_cfg = config.setdefault("world", {})
    seed = world_cfg.get("random_seed")
    if seed is None:
        seed = default_config.get("world", {}).get("random_seed")
    if seed is None:
        seed = generate_seed()
    world_cfg["random_seed"] = seed
    return seed


def _prepare_model_dir(args: argparse.Namespace, fallback_tag: str) -> tuple[Path, str]:
    root = Path("model")
    root.mkdir(exist_ok=True)

    explicit_name = _sanitize_name(args.model_name) if args.model_name else ""

    if args.model_dir:
        candidate = Path(str(args.model_dir).replace(" ", "_"))
        model_name = explicit_name or _sanitize_name(candidate.name) or fallback_tag
    else:
        raw_name = args.model_name or ""
        if not raw_name:
            try:
                raw_name = input("请输入本次训练的模型名称(留空则使用时间戳): ")
            except EOFError:
                raw_name = ""
        sanitized = _sanitize_name(raw_name) if raw_name else ""
        model_name = sanitized or fallback_tag
        candidate = root / model_name

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        candidate = root / fallback_tag
        candidate.mkdir(parents=True, exist_ok=True)
        model_name = _sanitize_name(candidate.name) or fallback_tag

    return candidate, model_name


def _clone_hidden(hidden):
    if isinstance(hidden, tuple):
        return tuple(item.detach().clone() for item in hidden)
    if hidden is None:
        return None
    return hidden.detach().clone()


def _current_hidden(env: ProtoLifeEnv):
    memory = env.agent_batch.state.get("memory")
    if memory is None:
        return None
    if env.agent_batch.use_lstm:
        return memory, env.agent_batch.state["memory_cell"]
    return memory


def _forward_policy_inputs(
    policy: torch.nn.Module,
    patch: torch.Tensor | None,
    agent_features: torch.Tensor | None,
    agent_obs: torch.Tensor | None,
    hidden,
    active_mask: torch.Tensor | None = None,
    active_indices: torch.Tensor | None = None,
    brain_indices: list[int] | None = None,
):
    if getattr(policy, "use_cnn", False):
        if getattr(policy, "cnn_independent", False):
            return policy(
                patch,
                hidden,
                agent_features,
                active_mask=active_mask,
                active_indices=active_indices,
                brain_indices=brain_indices,
            )
        return policy(patch, hidden, agent_features)
    logits, values = policy(agent_obs)
    return logits, values, None


def _reshape_policy_outputs(
    logits: torch.Tensor, values: torch.Tensor, population_shape: torch.Size
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        logits.reshape(*population_shape, -1),
        values.reshape(*population_shape),
    )


def _behavior_log_prob_and_entropy(
    logits: torch.Tensor, actions: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    if epsilon > 0:
        probabilities = (1.0 - epsilon) * probabilities + epsilon / logits.size(-1)
        log_probs = torch.log(probabilities.clamp_min(1e-8))
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return selected, entropy


def _mask_hidden(hidden, continuation: torch.Tensor):
    if hidden is None:
        return None
    mask = continuation.unsqueeze(-1)
    if isinstance(hidden, tuple):
        return tuple(item * mask.to(dtype=item.dtype) for item in hidden)
    return hidden * mask.to(dtype=hidden.dtype)


def _transition_continuation(
    valid: torch.Tensor,
    next_alive: torch.Tensor,
    dones: torch.Tensor,
    reproduction_events: list[tuple[int, int, int]],
) -> torch.Tensor:
    """切断死亡以及新生命复用旧槽位处的回报和隐藏状态。"""

    born_mask = torch.zeros_like(valid)
    for env_idx, _, child_idx in reproduction_events:
        born_mask[env_idx, child_idx] = True
    return valid & next_alive & ~dones.reshape(valid.shape) & ~born_mask


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continuation: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算截断 n-step rollout 上的 GAE 与 bootstrap return。"""

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(bootstrap_values)
    next_values = bootstrap_values
    for index in range(rewards.size(0) - 1, -1, -1):
        can_continue = continuation[index].to(dtype=rewards.dtype)
        delta = rewards[index] + gamma * next_values * can_continue - values[index]
        next_advantage = delta + gamma * gae_lambda * can_continue * next_advantage
        advantages[index] = next_advantage
        next_values = values[index]
    returns = advantages + values
    return advantages, returns


def _normalize_valid_advantages(
    advantages: torch.Tensor, valid: torch.Tensor, independent: bool
) -> torch.Tensor:
    normalized = torch.zeros_like(advantages)
    if independent:
        flat_advantages = advantages.reshape(advantages.size(0), -1)
        flat_valid = valid.reshape(valid.size(0), -1)
        mask = flat_valid.to(flat_advantages.dtype)
        counts = mask.sum(dim=0)
        means = (flat_advantages * mask).sum(dim=0) / counts.clamp_min(1)
        centered = flat_advantages - means
        variances = (centered.square() * mask).sum(dim=0) / counts.clamp_min(1)
        standardized = centered / (torch.sqrt(variances) + 1e-8)
        # 单样本大脑沿用原 advantage，保持旧行为；无样本位置归零。
        per_brain = torch.where(counts.unsqueeze(0) > 1, standardized, flat_advantages)
        return (per_brain * mask).reshape_as(advantages)

    mask = valid.to(advantages.dtype)
    count = mask.sum()
    if int(count.item()) == 0:
        return normalized
    mean = (advantages * mask).sum() / count
    variance = ((advantages - mean).square() * mask).sum() / count
    standardized = (advantages - mean) / (torch.sqrt(variance) + 1e-8)
    values = torch.where(count > 1, standardized, advantages)
    return values * mask


def _ppo_clipped_surrogate(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_range: float
) -> torch.Tensor:
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return torch.minimum(unclipped, clipped)


def _masked_reduce(
    values: torch.Tensor, valid: torch.Tensor, independent: bool
) -> torch.Tensor:
    if independent:
        flat_values = values.reshape(values.size(0), -1)
        flat_valid = valid.reshape(valid.size(0), -1)
        sample_counts = flat_valid.sum(dim=0)
        per_brain = (flat_values * flat_valid.to(values.dtype)).sum(dim=0)
        per_brain = per_brain / sample_counts.clamp_min(1).to(values.dtype)
        return (per_brain * (sample_counts > 0).to(values.dtype)).sum()
    mask = valid.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _clear_inactive_brain_gradients(
    policy: torch.nn.Module, valid: torch.Tensor
) -> None:
    if not getattr(policy, "cnn_independent", False):
        return
    if getattr(policy, "sparse_active_forward", False):
        # 稀疏前向根本不会把无效大脑接入计算图，其梯度天然为 None。
        return
    active_brains = valid.any(dim=0).reshape(-1)
    for brain_index, is_active in enumerate(active_brains.tolist()):
        if is_active:
            continue
        for parameter in policy.brains[brain_index].parameters():
            parameter.grad = None


def _ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: list[dict],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_range: float,
    ppo_epochs: int,
    minibatch_steps: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    epsilon_greedy: float,
) -> dict[str, float]:
    independent = bool(getattr(policy, "cnn_independent", False))
    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    update_count = 0
    chunk_starts = list(range(0, len(rollout), minibatch_steps))

    for _ in range(ppo_epochs):
        chunk_order = torch.randperm(len(chunk_starts)).tolist()
        for chunk_position in chunk_order:
            start = chunk_starts[chunk_position]
            end = min(start + minibatch_steps, len(rollout))
            hidden = _clone_hidden(rollout[start]["hidden"])
            new_log_probs = []
            new_values = []
            entropies = []

            for time_index in range(start, end):
                sample = rollout[time_index]
                logits, values, new_hidden = _forward_policy_inputs(
                    policy,
                    sample["patch"],
                    sample["agent_features"],
                    sample["agent_obs"],
                    hidden,
                    sample["valid"],
                    sample.get("active_indices"),
                    sample.get("brain_indices"),
                )
                logits, values = _reshape_policy_outputs(
                    logits, values, sample["actions"].shape
                )
                behavior_logits = logits + sample["logit_noise"]
                log_probs, entropy = _behavior_log_prob_and_entropy(
                    behavior_logits, sample["actions"], epsilon_greedy
                )
                new_log_probs.append(log_probs)
                new_values.append(values)
                entropies.append(entropy)
                hidden = _mask_hidden(new_hidden, sample["continuation"])

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            entropies = torch.stack(entropies)
            old_log_probs = torch.stack(
                [rollout[index]["old_log_probs"] for index in range(start, end)]
            )
            old_values = torch.stack(
                [rollout[index]["old_values"] for index in range(start, end)]
            )
            valid = torch.stack(
                [rollout[index]["valid"] for index in range(start, end)]
            )
            if not bool(valid.any().item()):
                continue
            chunk_advantages = advantages[start:end]
            chunk_returns = returns[start:end]

            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            surrogate = _ppo_clipped_surrogate(ratio, chunk_advantages, clip_range)
            policy_loss = -_masked_reduce(surrogate, valid, independent)

            clipped_values = old_values + torch.clamp(
                new_values - old_values, -clip_range, clip_range
            )
            value_error = (new_values - chunk_returns).square()
            clipped_value_error = (clipped_values - chunk_returns).square()
            value_loss = 0.5 * _masked_reduce(
                torch.maximum(value_error, clipped_value_error), valid, independent
            )
            entropy = _masked_reduce(entropies, valid, independent)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _clear_inactive_brain_gradients(policy, valid)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metric_mask = valid.to(log_ratio.dtype)
                metric_count = metric_mask.sum().clamp_min(1)
                metric_sums["policy_loss"] += float(policy_loss.item())
                metric_sums["value_loss"] += float(value_loss.item())
                metric_sums["entropy"] += float(entropy.item())
                metric_sums["approx_kl"] += float(
                    ((-log_ratio * metric_mask).sum() / metric_count).item()
                )
                metric_sums["clip_fraction"] += float(
                    (
                        (((ratio - 1.0).abs() > clip_range).to(ratio.dtype) * metric_mask).sum()
                        / metric_count
                    ).item()
                )
            update_count += 1

    if update_count == 0:
        return metric_sums
    return {key: value / update_count for key, value in metric_sums.items()}


def _save_ppo_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env: ProtoLifeEnv,
    checkpoint_dir: Path,
    step: int,
    *,
    n_steps: int,
    gamma: float,
    gae_lambda: float,
    ppo_clip: float,
    filename_prefix: str = "",
    min_remaining_bytes: int = 2 * 1024**3,
    size_safety_factor: float = 1.10,
) -> dict[str, Path]:
    return save_split_checkpoint(
        checkpoint_dir,
        step,
        env_state=env.export_state(),
        policy_state_dict=policy.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        meta={
            "step": step,
            "algorithm": "ppo_gae_n_step",
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ppo_clip": ppo_clip,
        },
        filename_prefix=filename_prefix,
        min_remaining_bytes=min_remaining_bytes,
        size_safety_factor=size_safety_factor,
    )

def main() -> None:
    global _ACTIVE_PROFILER
    args = parse_args()
    started_at = datetime.now()
    run_tag = started_at.strftime("%Y%m%d_%H%M%S")
    model_dir, model_name = _prepare_model_dir(args, run_tag)
    log_dir = model_dir / "log"
    checkpoint_dir = model_dir / (args.checkpoint_dir or "checkpoint")
    artifact_run_id = secrets.token_hex(6)
    artifact_filename_prefix = f"{INVALID_ARTIFACT_MARKER}{artifact_run_id}_"
    _SIMULATION_CONTEXT.clear()
    _SIMULATION_CONTEXT.update(
        {
            "started_at": started_at,
            "start_step": 0,
            "total_steps": 0,
            "run_steps": 0,
            "model_name": model_name,
            "resumed": False,
            "resume_from": None,
            "count_simulation": True,
            "artifacts": [],
            "artifacts_finalized": False,
            "sequence": None,
            "simulation_count_path": str(SIMULATION_COUNT_PATH),
            "invalid_artifact_prefix": artifact_filename_prefix,
            "stop_reason": "完成配置的训练步数",
            "finalized": False,
        }
    )
    console_log_path: Path | None = None
    if args.profiler is None:
        console_log_path = log_dir / f"{artifact_filename_prefix}console.log"
        _start_console_capture(console_log_path)
        _register_run_artifacts(console_log_path)

    model_config_path = model_dir / f"{model_name}.yaml"
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = model_config_path

    config, default_config = load_config(str(config_path))
    existing_model_config = {}
    if model_config_path.exists():
        existing_model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}

    config = _merge_dict(existing_model_config, config)
    resolved_seed = _resolve_random_seed(config, default_config)
    merged_config = _merge_dict(default_config, config)
    merged_config.setdefault("world", {})["random_seed"] = resolved_seed
    save_config_with_comments(model_config_path, merged_config)
    print(f"模型目录: {model_dir.resolve()}")
    print(f"配置文件: {config_path.resolve()}")
    print(f"Checkpoint 目录: {checkpoint_dir.resolve()}")
    if console_log_path is not None:
        print(f"控制台日志: {console_log_path.resolve()}")
    else:
        print("Profiler 模式：不保存训练 log、有效模拟计数或 checkpoint。")

    profiler_session = None
    if args.profiler is not None:
        # 延迟导入：普通训练不会初始化或导入 Kineto profiler 模块。
        from protolife.profiler import create_profiler

        profiler_config_path = Path(args.profiler)
        profiler_session = create_profiler(
            profiler_config_path,
            model_name=model_name,
            metadata={
                "model_name": model_name,
                "model_dir": str(model_dir.resolve()),
                "training_config": str(config_path.resolve()),
                "profiler_config": str(profiler_config_path.resolve()),
                "checkpoint_dir": str(checkpoint_dir.resolve()),
            },
        )
        _SIMULATION_CONTEXT["count_simulation"] = False
        print(f"PyTorch Profiler 配置: {profiler_config_path.resolve()}")
        print(f"Profiler 测试步数: {profiler_session.total_steps}")

    set_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    env = ProtoLifeEnv(config, default_config)
    policy = build_policy(config, default_config, obs_dim=env.observation_dim, patch_shape=env.patch_shape).to(env.device)
    learning_rate = float(
        get_cfg(config, default_config, "training", "learning_rate", TRAIN_DEFAULTS["training"]["learning_rate"])
    )
    max_grad_norm = float(
        get_cfg(config, default_config, "training", "max_grad_norm", TRAIN_DEFAULTS["training"]["max_grad_norm"])
    )
    if learning_rate <= 0:
        raise ValueError("training.learning_rate 必须大于 0")
    try:
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=learning_rate, fused=env.device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if getattr(policy, "cnn_independent", False):
        print(f"已启用 {policy.brain_count} 个完全独立的 CNN 个体大脑")

    logger = None
    replay_logging_enabled = profiler_session is None
    if get_cfg(config, default_config, "logging", "save_dir", True) and replay_logging_enabled:
        from protolife.logger import ExperimentLogger

        log_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(
            save_dir=log_dir,
            snapshot_interval=get_cfg(config, default_config, "logging", "snapshot_interval", 50),
            env_index=get_cfg(config, default_config, "logging", "env_index", 0),
            run_tag=artifact_run_id,
            metadata={
                "height": env.height,
                "width": env.width,
                "agent_marker_size": env.agent_marker_size,
                "num_envs": env.agent_batch.num_envs,
                "agents_per_env": env.agent_batch.agents_per_env,
                "initial_agents_per_env": env.initial_agents_per_env,
                "max_agents_per_env": env.max_agents_per_env,
                "run_name": model_dir.name,
                "run_tag": run_tag,
                "map_file": env.map_file,
                "food_lifetime": getattr(env, "food_lifetime", None),
                "toxin_lifetime": getattr(env, "toxin_lifetime", None),
                "console_log": str(console_log_path),
                "use_health": env.use_health,
            },
            buffer_on_gpu=get_cfg(
                config, default_config, "logging", "snapshot_gpu_stage", False
            ),
            flush_interval=get_cfg(
                config, default_config, "logging", "snapshot_flush_interval", 8
            ),
            filename_prefix=INVALID_ARTIFACT_MARKER,
        )
        _register_run_artifacts(logger.map_log, logger.agent_log)
    elif profiler_session is not None and not replay_logging_enabled:
        print("Profiler 模式强制关闭 replay 日志，避免 I/O 干扰性能测量。")

    checkpoint_dir = Path(checkpoint_dir)
    save_interval = int(
        args.save_interval
        or get_cfg(config, default_config, "training", "save_interval", 100)
    )
    checkpoint_min_free_gb = float(
        get_cfg(config, default_config, "training", "checkpoint_min_free_gb", 2.0)
    )
    checkpoint_size_safety_factor = float(
        get_cfg(
            config,
            default_config,
            "training",
            "checkpoint_size_safety_factor",
            1.10,
        )
    )
    checkpoint_min_remaining_bytes = int(checkpoint_min_free_gb * 1024**3)
    print_actions = bool(get_cfg(config, default_config, "training", "print_actions", False))
    entropy_coef = float(
        get_cfg(config, default_config, "training", "entropy_coef", 0.01)
    )
    n_steps = int(
        get_cfg(config, default_config, "training", "n_steps", TRAIN_DEFAULTS["training"]["n_steps"])
    )
    gamma = float(
        get_cfg(config, default_config, "training", "gamma", TRAIN_DEFAULTS["training"]["gamma"])
    )
    gae_lambda = float(
        get_cfg(
            config, default_config, "training", "gae_lambda", TRAIN_DEFAULTS["training"]["gae_lambda"]
        )
    )
    ppo_clip = float(
        get_cfg(config, default_config, "training", "ppo_clip", TRAIN_DEFAULTS["training"]["ppo_clip"])
    )
    ppo_epochs = int(
        get_cfg(config, default_config, "training", "ppo_epochs", TRAIN_DEFAULTS["training"]["ppo_epochs"])
    )
    ppo_minibatch_steps = int(
        get_cfg(
            config,
            default_config,
            "training",
            "ppo_minibatch_steps",
            TRAIN_DEFAULTS["training"]["ppo_minibatch_steps"],
        )
    )
    value_coef = float(
        get_cfg(config, default_config, "training", "value_coef", TRAIN_DEFAULTS["training"]["value_coef"])
    )
    normalize_advantages = bool(
        get_cfg(
            config,
            default_config,
            "training",
            "normalize_advantages",
            TRAIN_DEFAULTS["training"]["normalize_advantages"],
        )
    )
    policy_mutation_std = float(
        get_cfg(config, default_config, "reproduction", "policy_mutation_std", 0.01)
    )
    print_interval = args.print_interval or get_cfg(
        config, default_config, "training", "print_interval", TRAIN_DEFAULTS["training"]["print_interval"]
    )

    action_noise_cfg = (
        config.get("training", {}).get("action_noise")
        or default_config.get("training", {}).get("action_noise")
        or TRAIN_DEFAULTS["training"].get("action_noise")
        or {}
    )
    gaussian_noise_std = float(action_noise_cfg.get("gaussian_std", 0.0) or 0.0)
    epsilon_greedy = float(action_noise_cfg.get("epsilon", 0.0) or 0.0)

    if save_interval <= 0:
        raise ValueError("training.save_interval 必须大于 0")
    if checkpoint_min_free_gb < 0:
        raise ValueError("training.checkpoint_min_free_gb 不能为负数")
    if checkpoint_size_safety_factor < 1.0:
        raise ValueError("training.checkpoint_size_safety_factor 不能小于 1")
    if n_steps <= 0:
        raise ValueError("training.n_steps 必须大于 0")
    if not 0 < gamma <= 1:
        raise ValueError("training.gamma 必须在 (0, 1] 范围内")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("training.gae_lambda 必须在 [0, 1] 范围内")
    if not 0 < ppo_clip < 1:
        raise ValueError("training.ppo_clip 必须在 (0, 1) 范围内")
    if ppo_epochs <= 0 or ppo_minibatch_steps <= 0:
        raise ValueError("ppo_epochs 和 ppo_minibatch_steps 必须大于 0")
    if value_coef < 0 or entropy_coef < 0:
        raise ValueError("value_coef 和 entropy_coef 不能为负数")
    if gaussian_noise_std < 0 or not 0 <= epsilon_greedy <= 1:
        raise ValueError("action_noise 参数超出有效范围")

    if logger:
        logger.metadata.update(
            {
                "algorithm": "ppo_gae_n_step",
                "n_steps": n_steps,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "ppo_clip": ppo_clip,
                "ppo_epochs": ppo_epochs,
                "ppo_minibatch_steps": ppo_minibatch_steps,
            }
        )

    start_step = 0
    auto_resume = bool(get_cfg(config, default_config, "training", "auto_resume", True))
    latest_checkpoint = (
        find_latest_checkpoint(checkpoint_dir)
        if auto_resume and not args.fresh
        else None
    )
    checkpoint_path = Path(args.resume_from) if args.resume_from else latest_checkpoint
    resolved_checkpoint_path = (
        str(checkpoint_path.resolve()) if checkpoint_path is not None else None
    )

    if checkpoint_path:
        print(f"使用的 checkpoint: {checkpoint_path.resolve()}")
        env_state, policy_state, optim_state, meta = load_checkpoint(checkpoint_path, map_location=env.device)
        has_env_rng_state = "random_generator_state" in env_state
        global_rng_state = meta.pop("_rng_state", None)
        env.load_state(env_state)
        skipped = _load_state_dict_lenient(policy, policy_state)
        checkpoint_algorithm = meta.get("algorithm")
        if optim_state and not skipped and checkpoint_algorithm == "ppo_gae_n_step":
            optimizer.load_state_dict(optim_state)
            # optimizer checkpoint 会保存旧学习率；续训时以当前 YAML 配置为准。
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        elif optim_state and checkpoint_algorithm != "ppo_gae_n_step":
            print("检测到旧版单步训练 checkpoint，将保留模型/环境并重置 Adam 状态。")
        start_step = int(meta.get("step", 0))
        env.step_count = start_step
        obs = env._build_observations()
        rng_fully_restored = restore_rng_state(global_rng_state)
        if not has_env_rng_state or not rng_fully_restored:
            print(
                "[警告] checkpoint 缺少部分 RNG 状态或当前设备拓扑不同；"
                "本次续训可正常进行，但随机序列不能与保存点完全衔接。"
            )
        else:
            print("已恢复 Python、NumPy、PyTorch CPU/CUDA 与环境 RNG 状态。")
        print(f"从 checkpoint 恢复，起始 step={start_step}，来源: {checkpoint_path}")
    else:
        print("未恢复 checkpoint，将开始新的训练。")
        obs = env.reset()
        initial_model = args.load_model or get_cfg(
            config, default_config, "training", "initial_model", None
        )
        if initial_model:
            initial_model_path = Path(str(initial_model))
            if not initial_model_path.exists():
                model_relative_path = model_dir / initial_model_path
                if model_relative_path.exists():
                    initial_model_path = model_relative_path
                else:
                    raise FileNotFoundError(f"找不到初始模型: {initial_model}")
            _load_initial_model(policy, initial_model_path, env.device)

    _SIMULATION_CONTEXT.update(
        {
            "start_step": start_step,
            "total_steps": start_step,
            "run_steps": 0,
            "resumed": checkpoint_path is not None,
            "resume_from": resolved_checkpoint_path,
        }
    )

    if logger:
        logger.metadata["start_step"] = start_step
        logger.metadata["resumed_from_step"] = start_step
        logger.metadata["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
        logger.step_counter = start_step
        logger._write_header()

    if profiler_session is not None:
        profiler_session.metadata.update(
            {
                "device": str(env.device),
                "start_step": start_step,
                "resume_from": resolved_checkpoint_path,
                "num_envs": env.agent_batch.num_envs,
                "agents_per_env": env.agent_batch.agents_per_env,
                "n_steps": n_steps,
                "ppo_epochs": ppo_epochs,
                "ppo_minibatch_steps": ppo_minibatch_steps,
            }
        )
        profiler_session.start()
        _ACTIVE_PROFILER = profiler_session
        print(f"Profiler 报告目录: {profiler_session.output_dir.resolve()}")

    total_steps = start_step
    run_steps = 0
    rollout_steps = (
        profiler_session.total_steps
        if profiler_session is not None
        else int(get_cfg(config, default_config, "training", "rollout_steps", 128))
    )
    save_checkpoints_enabled = profiler_session is None
    next_checkpoint_step = (
        (total_steps // save_interval + 1) * save_interval
        if save_checkpoints_enabled
        else total_steps + rollout_steps + 1
    )
    last_checkpoint_step = start_step if checkpoint_path else None
    last_print_time = time.perf_counter()
    last_print_step = total_steps
    final_reward_mean = None
    stop_training = False
    stop_reason = "完成配置的训练步数"
    extinction_survivor = None

    while run_steps < rollout_steps and not stop_training:
        rollout: list[dict] = []
        pending_reproduction_events: list[tuple[int, int, int]] = []
        final_survivor = None
        all_dead = False
        rollout_limit = min(
            n_steps,
            rollout_steps - run_steps,
            next_checkpoint_step - total_steps,
        )

        for rollout_index in range(rollout_limit):
            valid = obs["alive"].detach().clone()
            active_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
            brain_indices = active_indices.detach().cpu().tolist()
            has_valid_agents = bool(brain_indices)
            if not has_valid_agents and env.use_death:
                print("所有个体在本轮开始前已经死亡，训练停止。")
                stop_reason = "开始 rollout 时已无存活个体"
                _SIMULATION_CONTEXT["stop_reason"] = stop_reason
                stop_training = True
                break
            last_survivor = _select_last_survivor(env) if has_valid_agents else None

            hidden = _clone_hidden(_current_hidden(env))
            patch = (
                obs["patch"].detach().clone()
                if getattr(policy, "use_cnn", False)
                else None
            )
            agent_features = (
                obs["agent_features"].detach().clone()
                if getattr(policy, "use_cnn", False)
                else None
            )
            agent_obs = (
                None
                if getattr(policy, "use_cnn", False)
                else obs["agent_obs"].detach().clone()
            )

            with torch.no_grad():
                logits, values, new_hidden = _forward_policy_inputs(
                    policy,
                    patch,
                    agent_features,
                    agent_obs,
                    hidden,
                    valid,
                    active_indices,
                    brain_indices,
                )
                logits, values = _reshape_policy_outputs(
                    logits, values, valid.shape
                )
                if gaussian_noise_std > 0:
                    logit_noise = torch.randn_like(logits) * gaussian_noise_std
                else:
                    logit_noise = torch.zeros_like(logits)
                behavior_logits = logits + logit_noise
                actions = torch.distributions.Categorical(
                    logits=behavior_logits
                ).sample()
                if epsilon_greedy > 0:
                    random_actions = torch.randint(
                        low=0,
                        high=len(action_space),
                        size=actions.shape,
                        device=actions.device,
                    )
                    greedy_mask = (
                        torch.rand(actions.shape, device=actions.device)
                        < epsilon_greedy
                    )
                    actions = torch.where(greedy_mask, random_actions, actions)
                old_log_probs, _ = _behavior_log_prob_and_entropy(
                    behavior_logits, actions, epsilon_greedy
                )

            if isinstance(new_hidden, tuple):
                env.agent_batch.state["memory"] = new_hidden[0].detach().clone()
                env.agent_batch.state["memory_cell"] = new_hidden[1].detach().clone()
            elif new_hidden is not None:
                env.agent_batch.state["memory"] = new_hidden.detach().clone()

            if print_actions:
                action_ids = actions.reshape(-1).cpu().tolist()
                names = [action_space.get(int(action), str(int(action))) for action in action_ids]
                print(f"[step={total_steps}] actions: {action_ids} -> {names}")

            step_result = env.step(actions)
            rewards = step_result.rewards.reshape(valid.shape).detach()
            next_obs = step_result.observations
            continuation = _transition_continuation(
                valid,
                next_obs["alive"].detach(),
                step_result.dones,
                step_result.reproduction_events,
            )

            rollout.append(
                {
                    "patch": patch,
                    "agent_features": agent_features,
                    "agent_obs": agent_obs,
                    "hidden": hidden,
                    "actions": actions.detach(),
                    "old_log_probs": old_log_probs.detach(),
                    "old_values": values.detach(),
                    "rewards": rewards,
                    "valid": valid,
                    "active_indices": active_indices,
                    "brain_indices": brain_indices,
                    "continuation": continuation,
                    "logit_noise": logit_noise.detach(),
                }
            )

            final_reward_mean = (
                float(rewards[valid].mean().item()) if has_valid_agents else 0.0
            )
            obs = next_obs
            total_steps += 1
            run_steps += 1
            _SIMULATION_CONTEXT["total_steps"] = total_steps
            _SIMULATION_CONTEXT["run_steps"] = run_steps

            if logger:
                logger.maybe_log(
                    env.map_state, env.agent_batch.export_state(), step=total_steps
                )

            all_dead = env.use_death and not bool(obs["alive"].any().item())
            profile_rollout_boundary = (
                rollout_index + 1 >= rollout_limit
                or all_dead
                or bool(step_result.reproduction_events)
            )
            if profiler_session is not None and not profile_rollout_boundary:
                profiler_session.step()
            if all_dead:
                final_survivor = last_survivor
                stop_reason = "所有个体死亡"
                _SIMULATION_CONTEXT["stop_reason"] = stop_reason
                stop_training = True
                break

            if step_result.reproduction_events:
                pending_reproduction_events.extend(step_result.reproduction_events)
                # 继承会修改策略参数；先结算当前 rollout，保证 PPO behavior policy 固定。
                break

        if not rollout:
            break

        with torch.no_grad():
            if all_dead:
                bootstrap_values = torch.zeros_like(rollout[-1]["old_values"])
            else:
                bootstrap_hidden = _clone_hidden(_current_hidden(env))
                bootstrap_active_indices = torch.nonzero(
                    obs["alive"].reshape(-1), as_tuple=False
                ).flatten()
                bootstrap_brain_indices = (
                    bootstrap_active_indices.detach().cpu().tolist()
                )
                bootstrap_logits, bootstrap_values, _ = _forward_policy_inputs(
                    policy,
                    obs["patch"] if getattr(policy, "use_cnn", False) else None,
                    obs["agent_features"] if getattr(policy, "use_cnn", False) else None,
                    None if getattr(policy, "use_cnn", False) else obs["agent_obs"],
                    bootstrap_hidden,
                    obs["alive"],
                    bootstrap_active_indices,
                    bootstrap_brain_indices,
                )
                _, bootstrap_values = _reshape_policy_outputs(
                    bootstrap_logits, bootstrap_values, obs["alive"].shape
                )

        rollout_rewards = torch.stack([sample["rewards"] for sample in rollout])
        rollout_values = torch.stack([sample["old_values"] for sample in rollout])
        rollout_continuation = torch.stack(
            [sample["continuation"] for sample in rollout]
        )
        rollout_valid = torch.stack([sample["valid"] for sample in rollout])
        advantages, returns = _compute_gae(
            rollout_rewards,
            rollout_values,
            rollout_continuation,
            bootstrap_values.detach(),
            gamma,
            gae_lambda,
        )
        advantages = advantages.detach()
        returns = returns.detach()
        if normalize_advantages:
            advantages = _normalize_valid_advantages(
                advantages,
                rollout_valid,
                bool(getattr(policy, "cnn_independent", False)),
            )

        ppo_metrics = _ppo_update(
            policy,
            optimizer,
            rollout,
            advantages,
            returns,
            clip_range=ppo_clip,
            ppo_epochs=ppo_epochs,
            minibatch_steps=ppo_minibatch_steps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            epsilon_greedy=epsilon_greedy,
        )

        if pending_reproduction_events and hasattr(policy, "inherit_policy_head"):
            for env_idx, parent_idx, child_idx in pending_reproduction_events:
                inherited_parameters = policy.inherit_policy_head(
                    env_idx, parent_idx, child_idx, policy_mutation_std
                )
                for parameter in inherited_parameters:
                    optimizer.state.pop(parameter, None)

        # 每个 rollout 的最后一个 profiler step 延后到 PPO 更新之后，确保反向传播
        # 和 optimizer.step 不会落在 active 窗口之外。
        if profiler_session is not None:
            profiler_session.step()

        if print_interval > 0 and (
            total_steps - last_print_step >= print_interval or stop_training
        ):
            now = time.perf_counter()
            step_delta = total_steps - last_print_step
            time_delta = now - last_print_time
            if time_delta > 0 and step_delta > 0:
                steps_per_sec = step_delta / time_delta
                step_rate_text = (
                    f"{steps_per_sec:.2f} step/s"
                    if steps_per_sec >= 1
                    else f"{(time_delta / step_delta):.2f} s/step"
                )
                agents_processed = (
                    step_delta
                    * env.agent_batch.num_envs
                    * env.agent_batch.agents_per_env
                )
                model_speed_text = f"{agents_processed / time_delta:.2f} agents/s"
            else:
                step_rate_text = "n/a"
                model_speed_text = "n/a"
            current_agent_count = int(obs["alive"].sum().item())
            print(
                f"steps:{total_steps} agent_count:{current_agent_count} "
                f"reward_mean:{final_reward_mean:.6f} "
                f"policy_loss:{ppo_metrics['policy_loss']:.6f} "
                f"value_loss:{ppo_metrics['value_loss']:.6f} "
                f"entropy:{ppo_metrics['entropy']:.6f} "
                f"kl:{ppo_metrics['approx_kl']:.6f} "
                f"clip_fraction:{ppo_metrics['clip_fraction']:.4f} "
                f"step_rate:{step_rate_text} model_speed:{model_speed_text}"
            )
            last_print_time = now
            last_print_step = total_steps

        if total_steps >= next_checkpoint_step:
            try:
                checkpoint_paths = _save_ppo_checkpoint(
                    policy,
                    optimizer,
                    env,
                    checkpoint_dir,
                    total_steps,
                    n_steps=n_steps,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    ppo_clip=ppo_clip,
                    filename_prefix=artifact_filename_prefix,
                    min_remaining_bytes=checkpoint_min_remaining_bytes,
                    size_safety_factor=checkpoint_size_safety_factor,
                )
            except InsufficientCheckpointSpaceError as exc:
                stop_reason = "checkpoint 磁盘空间不足，未写入并停止训练"
                _SIMULATION_CONTEXT["stop_reason"] = stop_reason
                save_checkpoints_enabled = False
                stop_training = True
                print(f"[停止] {exc}")
                break
            _register_run_artifacts(*checkpoint_paths.values())
            last_checkpoint_step = total_steps
            print(f"已保存 checkpoint 至 {checkpoint_dir}, step={total_steps}")
            while next_checkpoint_step <= total_steps:
                next_checkpoint_step += save_interval
        if all_dead:
            extinction_survivor = final_survivor
            print("所有个体已死亡，训练停止。")
            break

    if profiler_session is not None:
        profiler_session.metadata.update(
            {
                "end_step": total_steps,
                "run_steps": run_steps,
                "final_agent_count": int(obs["alive"].sum().item()),
                "stop_reason": stop_reason,
            }
        )
    profiler_output_dir = _stop_active_profiler()
    if profiler_output_dir is not None:
        print(f"Profiler 报告已生成: {profiler_output_dir.resolve()}")

    if (
        save_checkpoints_enabled
        and run_steps > 0
        and last_checkpoint_step != total_steps
    ):
        try:
            checkpoint_paths = _save_ppo_checkpoint(
                policy,
                optimizer,
                env,
                checkpoint_dir,
                total_steps,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ppo_clip=ppo_clip,
                filename_prefix=artifact_filename_prefix,
                min_remaining_bytes=checkpoint_min_remaining_bytes,
                size_safety_factor=checkpoint_size_safety_factor,
            )
        except InsufficientCheckpointSpaceError as exc:
            stop_reason = "checkpoint 磁盘空间不足，最终存档未写入"
            _SIMULATION_CONTEXT["stop_reason"] = stop_reason
            print(f"[停止] {exc}")
        else:
            _register_run_artifacts(*checkpoint_paths.values())
            last_checkpoint_step = total_steps
            print(f"已保存最终 checkpoint 至 {checkpoint_dir}, step={total_steps}")

    if (
        extinction_survivor is not None
        and save_checkpoints_enabled
        and last_checkpoint_step == total_steps
    ):
        try:
            survivor_path = _save_last_survivor_model(
                policy,
                checkpoint_dir,
                extinction_survivor,
                total_steps,
                filename_prefix=artifact_filename_prefix,
                min_remaining_bytes=checkpoint_min_remaining_bytes,
                size_safety_factor=checkpoint_size_safety_factor,
            )
        except InsufficientCheckpointSpaceError as exc:
            print(f"[警告] 完整 checkpoint 已保存，但最后存活模型空间不足: {exc}")
        else:
            _register_run_artifacts(survivor_path)
            print(f"最后存活个体模型已保存至: {survivor_path}")

    if logger:
        logger.flush()

    effective_count = _finalize_simulation_context(stop_reason)
    if effective_count is not None:
        print(
            f"本次模拟已计入 {SIMULATION_COUNT_PATH}，"
            f"累计有效模拟: {effective_count}"
        )

    if final_reward_mean is not None:
        print("训练完成，最终奖励均值:", final_reward_mean)

    _stop_console_capture()
    finalized_artifacts = _finalize_run_artifacts()
    if effective_count is not None:
        print(
            f"已将 {len(finalized_artifacts)} 个训练产物改名为 "
            f"#{effective_count}_..."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断训练。")
        count = _finalize_simulation_context("用户中断")
        if count is not None:
            print(f"累计有效模拟: {count}")
    except Exception as exc:
        _finalize_simulation_context(f"异常停止: {type(exc).__name__}")
        traceback.print_exc()
        raise SystemExit(1) from exc
    finally:
        try:
            _stop_active_profiler()
        finally:
            try:
                _stop_console_capture()
            finally:
                _finalize_run_artifacts()
