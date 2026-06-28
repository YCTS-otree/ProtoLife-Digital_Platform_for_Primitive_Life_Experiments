import json

import pytest

torch = pytest.importorskip("torch")

from protolife.logger import ExperimentLogger


def test_logger_records_real_step_start_and_population_count(tmp_path):
    logger = ExperimentLogger(
        save_dir=tmp_path,
        snapshot_interval=1,
        run_tag="test",
        metadata={"height": 2, "width": 2},
        start_step=100,
    )
    map_state = torch.zeros(1, 2, 2, dtype=torch.int64)
    agents = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]]]
    )

    logger.maybe_log(map_state, agents, step=105)

    lines = logger.agent_log.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])["meta"]
    record = json.loads(lines[1])
    assert header["start_step"] == 100
    assert record["step"] == 105
    assert record["agent_count"] == 1
