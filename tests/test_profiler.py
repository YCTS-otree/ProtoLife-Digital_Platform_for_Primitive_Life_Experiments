from datetime import datetime
import json
from pathlib import Path
import tempfile
import warnings

import pytest
import torch

from protolife.profiler import TrainingProfiler, load_profiler_config


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_profiler_config_uses_intensity_and_explicit_overrides():
    with tempfile.TemporaryDirectory() as raw_dir:
        path = _write_config(
            Path(raw_dir) / "profiler.yaml",
            """
profiler:
  test_steps: 12
  skip_first: 1
  wait: 1
  warmup: 1
  active: 2
  repeat: 2
  intensity: deep
  activities: [cpu]
  with_stack: false
report:
  output_dir: reports
  row_limit: 12
""",
        )
        config = load_profiler_config(path)

    assert config.test_steps == 12
    assert config.record_shapes is True
    assert config.profile_memory is True
    assert config.with_flops is True
    assert config.with_stack is False
    assert config.activities == ("cpu",)
    assert config.row_limit == 12
    assert config.minimum_steps_for_report == 5
    assert config.scheduled_steps == 9
    assert config.total_steps == 12


def test_profiler_config_rejects_a_test_too_short_for_a_trace():
    with tempfile.TemporaryDirectory() as raw_dir:
        path = _write_config(
            Path(raw_dir) / "profiler.yaml",
            """
profiler:
  test_steps: 3
  skip_first: 1
  wait: 1
  warmup: 1
  active: 1
  repeat: 1
  activities: [cpu]
""",
        )
        with pytest.raises(ValueError, match="test_steps"):
            load_profiler_config(path)


def test_cpu_profiler_writes_trace_table_and_metadata():
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        config_path = _write_config(
            root / "profiler.yaml",
            """
profiler:
  test_steps: 3
  skip_first: 0
  wait: 0
  warmup: 1
  active: 2
  repeat: 1
  intensity: light
  activities: [cpu]
report:
  output_dir: ignored
  row_limit: 10
  chrome_trace: true
  operator_table: true
""",
        )
        config = load_profiler_config(config_path)
        profiler = TrainingProfiler(
            config,
            model_name="test/model",
            report_root=root / "reports",
            metadata={"training_config": "demo.yaml"},
            timestamp=datetime(2026, 6, 30, 12, 0, 0),
        )
        assert profiler.total_steps == 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profiler.start()
            for _ in range(config.test_steps):
                tensor = torch.ones(8, 8)
                _ = tensor @ tensor
                profiler.step()
            profiler.stop()

        output = root / "reports" / "test_model_20260630_120000"
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "complete"
        assert metadata["steps_observed"] == 3
        assert metadata["elapsed_seconds"] > 0
        assert metadata["observed_steps_per_second"] > 0
        assert metadata["run"]["training_config"] == "demo.yaml"
        assert manifest["reports"]
        report = manifest["reports"][0]
        assert (output / report["chrome_trace"]).is_file()
        table = (output / report["operator_table"]).read_text(encoding="utf-8")
        assert "aten::mm" in table
        assert "Profiler run summary" in (output / "summary.txt").read_text(
            encoding="utf-8"
        )
