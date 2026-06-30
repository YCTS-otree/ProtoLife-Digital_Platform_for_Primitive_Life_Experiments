from pathlib import Path

from scripts.plot_agent_count_v1 import LogData, combine_logs


def test_overlapping_steps_from_different_files_are_preserved():
    newer = LogData(
        path=Path("newer.jsonl"),
        steps=[2, 3],
        counts=[20, 30],
        meta={},
        created_at_ns=200,
        input_order=0,
    )
    older = LogData(
        path=Path("older.jsonl"),
        steps=[1, 2],
        counts=[10, 11],
        meta={},
        created_at_ns=100,
        input_order=1,
    )

    steps, counts, ordered = combine_logs([newer, older])

    assert steps == [1, 2, 2, 3]
    assert counts == [10, 20, 11, 30]
    assert [log.path.name for log in ordered] == ["newer.jsonl", "older.jsonl"]
