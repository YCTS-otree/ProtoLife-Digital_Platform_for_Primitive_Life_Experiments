import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tempfile

from scripts.train_phase0 import _record_valid_simulation


def _record(path: Path) -> int | None:
    return _record_valid_simulation(
        started_at=datetime(2026, 6, 30, 9, 8, 7),
        start_step=4096,
        total_steps=6144,
        run_steps=2048,
        stop_reason="用户中断",
        model_name="test_CNN_NS_TBPTT",
        resumed=True,
        resume_from="K:/checkpoints/model_step_4096.pth",
        path=path,
    )


def test_legacy_txt_manual_cumulative_correction_is_migrated():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        legacy_path = directory / "Simulation_counting.txt"
        json_path = directory / "Simulation_counting.json"
        legacy_path.write_text(
            "有效模拟 #1 | 累计有效模拟=1\n"
            "有效模拟 #2 | 累计有效模拟=16\n",
            encoding="utf-8",
        )

        assert _record(json_path) == 17

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["effective_simulation_count"] == 17
        assert data["legacy_migration"]["detected_count"] == 16
        assert [record["sequence"] for record in data["simulations"]] == [1, 2, 17]
        assert data["simulations"][0]["migrated_from_legacy"] is True
        assert data["simulations"][-1]["model_name"] == "test_CNN_NS_TBPTT"
        assert data["simulations"][-1]["resume_from"].endswith("model_step_4096.pth")
        # Migration is non-destructive, so historical evidence is retained.
        assert legacy_path.exists()

        # Once created, JSON is authoritative; later legacy edits cannot reset
        # or unexpectedly jump the sequence.
        data["effective_simulation_count"] = 30
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        legacy_path.write_text("有效模拟 #100 | 累计有效模拟=100\n", encoding="utf-8")
        assert _record(json_path) == 31


def test_manual_json_count_is_authoritative_for_next_sequence():
    with tempfile.TemporaryDirectory() as raw_dir:
        path = Path(raw_dir) / "Simulation_counting.json"
        assert _record(path) == 1
        data = json.loads(path.read_text(encoding="utf-8"))
        data["effective_simulation_count"] = 25
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        assert _record(path) == 26
        updated = json.loads(path.read_text(encoding="utf-8"))
        assert updated["effective_simulation_count"] == 26
        assert updated["simulations"][-1]["sequence"] == 26


def test_invalid_run_does_not_create_json_or_migrate_legacy_file():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        json_path = directory / "Simulation_counting.json"
        (directory / "Simulation_counting.txt").write_text(
            "有效模拟 #16 | 累计有效模拟=16\n", encoding="utf-8"
        )
        result = _record_valid_simulation(
            started_at=datetime(2026, 6, 30),
            start_step=0,
            total_steps=1024,
            run_steps=1024,
            stop_reason="完成",
            path=json_path,
        )
        assert result is None
        assert not json_path.exists()


def test_parallel_valid_runs_receive_unique_sequence_numbers():
    with tempfile.TemporaryDirectory() as raw_dir:
        path = Path(raw_dir) / "Simulation_counting.json"
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: _record(path), range(2)))

        assert sorted(results) == [1, 2]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["effective_simulation_count"] == 2
        assert sorted(item["sequence"] for item in data["simulations"]) == [1, 2]


def test_manually_renamed_legacy_txt_is_still_detected():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        renamed = directory / "Simulation_counting#.txt"
        renamed.write_text("有效模拟 #19 | 累计有效模拟=19\n", encoding="utf-8")

        assert _record(directory / "Simulation_counting.json") == 20
