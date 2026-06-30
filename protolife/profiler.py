"""按需启用的 PyTorch Profiler 控制器。

本模块不会被常规训练路径导入。训练脚本只应在显式收到 profiler CLI
参数后延迟导入 :func:`create_profiler`，从而让未启用时没有 Kineto 初始化
成本，热循环里也只多一个 ``profiler is not None`` 判断。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Mapping
import warnings

import torch
import yaml


_INTENSITY_PRESETS: dict[str, dict[str, bool]] = {
    # light 适合先定位 CPU/CUDA 大类耗时；额外采集开销最低。
    "light": {
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "with_modules": False,
    },
    # balanced 增加输入 shape 和内存信息，通常足以定位主要算子。
    "balanced": {
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
        "with_flops": False,
        "with_modules": True,
    },
    # deep 采集调用栈/FLOPs；只建议用于很短的 active 窗口。
    "deep": {
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": True,
        "with_flops": True,
        "with_modules": True,
    },
}


@dataclass(frozen=True)
class ProfilerConfig:
    """Profiler 调度、采集强度与报告参数。"""

    test_steps: int = 64
    skip_first: int = 8
    wait: int = 2
    warmup: int = 2
    active: int = 52
    repeat: int = 1
    intensity: str = "light"
    activities: tuple[str, ...] = ("cpu", "cuda")
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    output_dir: str = "profiler_reports"
    sort_by: str = "self_cuda_time_total"
    row_limit: int = 50
    chrome_trace: bool = True
    operator_table: bool = True
    export_stacks: bool = False

    @property
    def minimum_steps_for_report(self) -> int:
        """产生第一个完整 trace 所需的最少 ``profiler.step()`` 次数。"""

        return self.skip_first + self.wait + self.warmup + self.active

    @property
    def scheduled_steps(self) -> int:
        """所有 repeat 调度周期结束所需的步数。"""

        return self.skip_first + (self.wait + self.warmup + self.active) * self.repeat

    @property
    def total_steps(self) -> int:
        """本次 profiler 专用训练应运行的环境步数。"""

        return self.test_steps


def _require_int(section: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"profiler.{key} 必须是 >= {minimum} 的整数，当前为 {value!r}")
    return value


def _require_bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"profiler.{key} 必须是 true/false，当前为 {value!r}")
    return value


def load_profiler_config(path: str | Path) -> ProfilerConfig:
    """读取并严格校验 ``PyTorch_Profiler.yaml``。"""

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Profiler YAML 顶层必须是映射")
    section = raw.get("profiler", raw)
    if not isinstance(section, dict):
        raise ValueError("Profiler YAML 的 profiler 节必须是映射")

    intensity = str(section.get("intensity", "light")).lower()
    if intensity not in _INTENSITY_PRESETS:
        choices = ", ".join(_INTENSITY_PRESETS)
        raise ValueError(f"profiler.intensity 必须是 {choices} 之一")
    preset = _INTENSITY_PRESETS[intensity]

    activity_names = section.get("activities", ["cpu", "cuda"])
    if not isinstance(activity_names, list) or not activity_names:
        raise ValueError("profiler.activities 必须是非空列表")
    activities = tuple(str(item).lower() for item in activity_names)
    unknown = sorted(set(activities) - {"cpu", "cuda"})
    if unknown:
        raise ValueError(f"profiler.activities 包含未知项: {unknown}")

    report = raw.get("report", {})
    if not isinstance(report, dict):
        raise ValueError("Profiler YAML 的 report 节必须是映射")

    config = ProfilerConfig(
        test_steps=_require_int(section, "test_steps", 64, minimum=1),
        skip_first=_require_int(section, "skip_first", 8, minimum=0),
        wait=_require_int(section, "wait", 2, minimum=0),
        warmup=_require_int(section, "warmup", 2, minimum=0),
        active=_require_int(section, "active", 8, minimum=1),
        repeat=_require_int(section, "repeat", 1, minimum=1),
        intensity=intensity,
        activities=activities,
        record_shapes=_require_bool(
            section, "record_shapes", preset["record_shapes"]
        ),
        profile_memory=_require_bool(
            section, "profile_memory", preset["profile_memory"]
        ),
        with_stack=_require_bool(section, "with_stack", preset["with_stack"]),
        with_flops=_require_bool(section, "with_flops", preset["with_flops"]),
        with_modules=_require_bool(section, "with_modules", preset["with_modules"]),
        output_dir=str(report.get("output_dir", "profiler_reports")),
        sort_by=str(report.get("sort_by", "self_cuda_time_total")),
        row_limit=_require_int(report, "row_limit", 50, minimum=1),
        chrome_trace=_require_bool(report, "chrome_trace", True),
        operator_table=_require_bool(report, "operator_table", True),
        export_stacks=_require_bool(report, "export_stacks", False),
    )
    if config.test_steps < config.minimum_steps_for_report:
        raise ValueError(
            "profiler.test_steps 太小，至少需要 "
            f"skip_first + wait + warmup + active = {config.minimum_steps_for_report} 步"
        )
    if config.test_steps < config.scheduled_steps:
        warnings.warn(
            "profiler.test_steps 小于完整 repeat 调度长度；最后一些周期不会生成报告",
            RuntimeWarning,
            stacklevel=2,
        )
    return config


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    return cleaned or "run"


class TrainingProfiler:
    """封装 PyTorch Profiler 生命周期和逐周期报告输出。"""

    def __init__(
        self,
        config: ProfilerConfig,
        *,
        model_name: str,
        report_root: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.config = config
        self.metadata = dict(metadata or {})
        root = Path(report_root) if report_root is not None else Path(config.output_dir)
        stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
        self.output_dir = root / f"{_safe_name(model_name)}_{stamp}"
        self._profiler: Any | None = None
        self._trace_index = 0
        self._steps = 0
        self._activities: list[Any] = []
        self._has_cuda = False
        self._manifest: list[dict[str, Any]] = []
        self._started_at: float | None = None
        self._elapsed_seconds: float | None = None

    @property
    def total_steps(self) -> int:
        """本次性能测试应覆盖的环境步数，供训练脚本覆盖 rollout_steps。"""

        return self.config.total_steps

    def _resolve_activities(self) -> list[Any]:
        activities: list[Any] = []
        if "cpu" in self.config.activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in self.config.activities:
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
                self._has_cuda = True
            else:
                warnings.warn(
                    "Profiler 配置请求 CUDA，但当前 PyTorch 无可用 CUDA；将仅采集 CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not activities:
            raise RuntimeError("当前环境没有 Profiler YAML 请求的可用 activity")
        return activities

    def start(self) -> "TrainingProfiler":
        """启动 Kineto；应在模型、checkpoint 和环境初始化完成后调用。"""

        if self._profiler is not None:
            raise RuntimeError("Profiler 已经启动")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._activities = self._resolve_activities()
        schedule = torch.profiler.schedule(
            wait=self.config.wait,
            warmup=self.config.warmup,
            active=self.config.active,
            repeat=self.config.repeat,
            skip_first=self.config.skip_first,
        )
        self._profiler = torch.profiler.profile(
            activities=self._activities,
            schedule=schedule,
            on_trace_ready=self._write_trace_report,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        )
        self._write_metadata(status="running")
        self._started_at = time.perf_counter()
        self._profiler.start()
        return self

    def step(self) -> None:
        """通知 profiler 一个环境训练 step 已完成。"""

        if self._profiler is None:
            raise RuntimeError("Profiler 尚未启动")
        self._steps += 1
        self._profiler.step()

    def stop(self) -> None:
        """停止采集并刷新报告；可安全重复调用。"""

        if self._profiler is None:
            return
        profiler = self._profiler
        self._profiler = None
        try:
            profiler.stop()
        finally:
            if self._started_at is not None:
                self._elapsed_seconds = time.perf_counter() - self._started_at
            self._write_manifest()
            self._write_run_summary()
            self._write_metadata(status="complete")

    close = stop

    def __enter__(self) -> "TrainingProfiler":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def _report_sort_key(self) -> str:
        if self.config.sort_by == "self_cuda_time_total" and not self._has_cuda:
            return "self_cpu_time_total"
        return self.config.sort_by

    def _write_trace_report(self, prof: Any) -> None:
        self._trace_index += 1
        prefix = f"cycle_{self._trace_index:03d}"
        entry: dict[str, Any] = {
            "cycle": self._trace_index,
            "step": self._steps,
        }

        if self.config.chrome_trace:
            trace_path = self.output_dir / f"{prefix}_trace.json"
            prof.export_chrome_trace(str(trace_path))
            entry["chrome_trace"] = trace_path.name

        if self.config.operator_table:
            table_path = self.output_dir / f"{prefix}_operators.txt"
            table = prof.key_averages(
                group_by_input_shape=self.config.record_shapes,
                group_by_stack_n=5 if self.config.with_stack else 0,
            ).table(
                sort_by=self._report_sort_key(),
                row_limit=self.config.row_limit,
            )
            table_path.write_text(table + "\n", encoding="utf-8")
            entry["operator_table"] = table_path.name
            summary_path = self.output_dir / "summary.txt"
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"=== Profiler cycle {self._trace_index}, "
                    f"step {self._steps}, sort={self._report_sort_key()} ===\n"
                )
                handle.write(table)
                handle.write("\n\n")
            entry["summary"] = summary_path.name

        if self.config.export_stacks and self.config.with_stack:
            stack_files = []
            metrics = ["self_cpu_time_total"]
            if self._has_cuda:
                metrics.append("self_cuda_time_total")
            for metric in metrics:
                stack_path = self.output_dir / f"{prefix}_{metric}_stacks.txt"
                prof.export_stacks(str(stack_path), metric=metric)
                stack_files.append(stack_path.name)
            entry["stacks"] = stack_files

        self._manifest.append(entry)
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.output_dir / "manifest.json").write_text(
            json.dumps({"reports": self._manifest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_run_summary(self) -> None:
        summary_path = self.output_dir / "summary.txt"
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write("=== Profiler run summary ===\n")
            handle.write(f"steps_observed: {self._steps}\n")
            handle.write(f"elapsed_seconds: {self._elapsed_seconds}\n")
            rate = (
                self._steps / self._elapsed_seconds
                if self._elapsed_seconds and self._elapsed_seconds > 0
                else None
            )
            handle.write(f"observed_steps_per_second: {rate}\n")
            handle.write(f"completed_report_cycles: {len(self._manifest)}\n")
            if not self._manifest:
                handle.write(
                    "note: 运行在首个 active 周期完成前结束，没有可导出的算子周期。\n"
                )
            handle.write("\n")

    def _write_metadata(self, *, status: str) -> None:
        payload = {
            "status": status,
            "steps_observed": self._steps,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "config": asdict(self.config),
            "run": self.metadata,
            "reports": self._manifest,
            "elapsed_seconds": self._elapsed_seconds,
            "observed_steps_per_second": (
                self._steps / self._elapsed_seconds
                if self._elapsed_seconds and self._elapsed_seconds > 0
                else None
            ),
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def create_profiler(
    config_path: str | Path,
    *,
    model_name: str,
    report_root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingProfiler:
    """从 YAML 创建尚未启动的 Profiler 控制器。"""

    return TrainingProfiler(
        load_profiler_config(config_path),
        model_name=model_name,
        report_root=report_root,
        metadata=metadata,
    )
