import json
from datetime import datetime
from pathlib import Path
import tempfile
import time

from protolife.replay import ReplayReader, _select_pairs
from scripts.plot_agent_count_v1 import view_reaches_latest
from scripts.train_phase0 import _record_valid_simulation


def write_log_pair(directory: Path, tag: str, steps: list[int]):
    map_path = directory / f"{tag}_map.log"
    agent_path = directory / f"{tag}_agents.jsonl"
    header = json.dumps({"meta": {"height": 1, "width": 1}})
    map_path.write_text(
        header + "\n" + "\n".join("00" for _ in steps) + "\n",
        encoding="utf-8",
    )
    records = [header] + [
        json.dumps({"step": step, "agent_count": 0, "agents": []})
        for step in steps
    ]
    agent_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return map_path, agent_path


def test_replay_can_start_from_selected_step_and_choose_latest_pair():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        older = write_log_pair(directory, "older", [1, 5, 9])
        time.sleep(0.02)
        newer = write_log_pair(directory, "newer", [2, 6, 10])

        frames = list(ReplayReader(*older).iter_frames(start_step=5))
        assert [frame[1]["step"] for frame in frames] == [5, 9]
        assert _select_pairs([newer, older], use_latest=True) == [newer]


def test_right_edge_reenables_follow_mode():
    assert view_reaches_latest((80.0, 100.0), 100)
    assert view_reaches_latest((80.0, 99.5), 100)
    assert not view_reaches_latest((50.0, 70.0), 100)


def test_simulation_counter_only_records_runs_over_1024_steps():
    with tempfile.TemporaryDirectory() as raw_dir:
        path = Path(raw_dir) / "Simulation_counting.json"
        started_at = datetime(2026, 6, 29, 12, 30, 0)
        invalid = _record_valid_simulation(
            started_at=started_at,
            start_step=0,
            total_steps=1024,
            run_steps=1024,
            stop_reason="完成",
            path=path,
        )
        first = _record_valid_simulation(
            started_at=started_at,
            start_step=100,
            total_steps=1125,
            run_steps=1025,
            stop_reason="用户中断",
            model_name="test_CNN",
            resumed=True,
            resume_from="C:/checkpoints/full_step_100.pt",
            path=path,
        )
        second = _record_valid_simulation(
            started_at=started_at,
            start_step=0,
            total_steps=2048,
            run_steps=2048,
            stop_reason="所有个体死亡",
            path=path,
        )

        data = json.loads(path.read_text(encoding="utf-8"))
        assert invalid is None
        assert first == 1
        assert second == 2
        assert data["effective_simulation_count"] == 2
        assert len(data["simulations"]) == 2
        first_record, second_record = data["simulations"]
        assert first_record["model_name"] == "test_CNN"
        assert first_record["training_type"] == "resume"
        assert first_record["resume_from"] == "C:/checkpoints/full_step_100.pt"
        assert second_record["stop_reason"] == "所有个体死亡"
