import json
from pathlib import Path
import tempfile

import torch

from protolife.checkpoint import find_latest_checkpoint, save_split_checkpoint
from scripts.clean_invalid_training_files import clean_invalid_training_files
from scripts import train_phase0


def _split_state():
    generator = torch.Generator().manual_seed(7)
    return {
        "value": torch.tensor([1.0]),
        "random_generator_state": generator.get_state(),
    }


def test_invalid_checkpoint_is_ignored_and_numbered_checkpoint_is_loadable():
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        invalid = save_split_checkpoint(
            root,
            64,
            _split_state(),
            {"weight": torch.tensor([1.0])},
            {"state": {}, "param_groups": []},
            {"step": 64},
            filename_prefix="无效_20260630_",
        )
        assert find_latest_checkpoint(root) is None

        numbered = save_split_checkpoint(
            root,
            32,
            _split_state(),
            {"weight": torch.tensor([2.0])},
            {"state": {}, "param_groups": []},
            {"step": 32},
            filename_prefix="#20_20260630_",
        )
        assert find_latest_checkpoint(root) == numbered["env"]
        assert all(path.exists() for path in invalid.values())


def test_valid_run_renames_registered_artifacts_and_updates_json():
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        count_path = root / "Simulation_counting.json"
        artifact = root / "无效_20260630_console.log"
        artifact.write_text("log", encoding="utf-8")
        count_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "effective_simulation_count": 20,
                    "simulations": [{"sequence": 20}],
                }
            ),
            encoding="utf-8",
        )

        train_phase0._SIMULATION_CONTEXT.clear()
        train_phase0._SIMULATION_CONTEXT.update(
            {
                "sequence": 20,
                "artifacts": [str(artifact)],
                "artifacts_finalized": False,
                "simulation_count_path": str(count_path),
                "invalid_artifact_prefix": "无效_20260630_",
            }
        )
        try:
            finalized = train_phase0._finalize_run_artifacts()
        finally:
            train_phase0._SIMULATION_CONTEXT.clear()

        assert [item.name for item in finalized] == ["#20_console.log"]
        data = json.loads(count_path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 2
        assert data["simulations"][0]["artifacts"][0].endswith(
            "#20_console.log"
        )


def test_cleaner_deletes_only_invalid_marked_files():
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        nested = root / "model" / "log"
        nested.mkdir(parents=True)
        invalid = nested / "无效_20260630_console.log"
        valid = nested / "#20_20260630_console.log"
        unrelated = nested / "notes.txt"
        invalid.write_text("x", encoding="utf-8")
        valid.write_text("x", encoding="utf-8")
        unrelated.write_text("x", encoding="utf-8")

        removed = clean_invalid_training_files(root)

        assert removed == [invalid]
        assert not invalid.exists()
        assert valid.exists()
        assert unrelated.exists()
