from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from protolife.checkpoint import (
    InsufficientCheckpointSpaceError,
    ensure_checkpoint_space,
    find_latest_checkpoint,
    capture_rng_state,
    load_checkpoint,
    migrate_legacy_checkpoint,
    save_checkpoint,
    save_split_checkpoint,
    restore_rng_state,
)


def _states():
    generator = torch.Generator().manual_seed(99)
    env_state = {
        "energy": torch.tensor([[3.0, 2.0]]),
        "random_generator_state": generator.get_state(),
    }
    policy_state = {"weight": torch.tensor([1.5])}
    optimizer_state = {"state": {}, "param_groups": [{"lr": 0.001}]}
    meta = {"step": 64, "algorithm": "ppo_gae_n_step"}
    return env_state, policy_state, optimizer_state, meta


def test_split_checkpoint_round_trip_has_no_full_file():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        env_state, policy_state, optimizer_state, meta = _states()
        paths = save_split_checkpoint(
            directory,
            64,
            env_state,
            policy_state,
            optimizer_state,
            meta,
        )

        assert all(path.exists() for path in paths.values())
        assert not (directory / "full_step_64.pt").exists()
        loaded_env, loaded_policy, loaded_optim, loaded_meta = load_checkpoint(
            paths["model"]
        )
        assert torch.equal(loaded_env["energy"], env_state["energy"])
        assert torch.equal(loaded_policy["weight"], policy_state["weight"])
        assert loaded_optim["param_groups"][0]["lr"] == 0.001
        assert loaded_meta["step"] == 64
        assert loaded_meta["checkpoint_format"] == "split_v2_rng"
        assert find_latest_checkpoint(directory) == paths["env"]


def test_legacy_checkpoint_loads_and_migrates_before_deletion():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        legacy_path = directory / "full_step_64.pt"
        env_state, policy_state, optimizer_state, meta = _states()
        save_checkpoint(
            legacy_path,
            env_state,
            policy_state,
            optimizer_state,
            meta,
        )

        loaded_env, loaded_policy, _, loaded_meta = load_checkpoint(legacy_path)
        assert torch.equal(loaded_env["energy"], env_state["energy"])
        assert torch.equal(loaded_policy["weight"], policy_state["weight"])
        assert loaded_meta["step"] == 64

        paths = migrate_legacy_checkpoint(legacy_path, delete_full=True)
        assert not legacy_path.exists()
        assert all(path.exists() for path in paths.values())
        migrated_env, migrated_policy, _, migrated_meta = load_checkpoint(paths["env"])
        assert torch.equal(migrated_env["energy"], env_state["energy"])
        assert torch.equal(migrated_policy["weight"], policy_state["weight"])
        assert migrated_meta["step"] == 64


def test_migration_refuses_to_delete_when_existing_split_file_differs():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        legacy_path = directory / "full_step_64.pt"
        env_state, policy_state, optimizer_state, meta = _states()
        save_checkpoint(
            legacy_path,
            env_state,
            policy_state,
            optimizer_state,
            meta,
        )
        torch.save({"weight": torch.tensor([-99.0])}, directory / "model_step_64.pth")

        try:
            migrate_legacy_checkpoint(legacy_path, delete_full=True)
        except ValueError as error:
            assert "模型" in str(error)
        else:
            raise AssertionError("不一致的三件套不应通过迁移校验")

        assert legacy_path.exists()


def test_latest_checkpoint_skips_truncated_torch_zip():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        env_state, policy_state, optimizer_state, _ = _states()
        valid = directory / "full_step_32.pt"
        broken = directory / "full_step_64.pt"
        save_checkpoint(
            valid,
            env_state,
            policy_state,
            optimizer_state,
            {"step": 32},
        )
        save_checkpoint(
            broken,
            env_state,
            policy_state,
            optimizer_state,
            {"step": 64},
        )
        with broken.open("r+b") as handle:
            handle.truncate(max(4, broken.stat().st_size - 128))

        assert find_latest_checkpoint(directory) == valid


def test_global_rng_state_round_trip():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    assert restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


def test_checkpoint_space_preflight_fails_before_writing_files():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        payload = {"large": torch.zeros(1024, dtype=torch.float32)}
        try:
            ensure_checkpoint_space(
                directory,
                payload,
                min_remaining_bytes=1024,
                size_safety_factor=1.0,
                free_bytes=100,
            )
        except InsufficientCheckpointSpaceError as error:
            assert error.free_bytes == 100
            assert error.required_bytes > 100
        else:
            raise AssertionError("空间不足时应在写盘前拒绝 checkpoint")
        assert list(directory.iterdir()) == []


def test_split_checkpoint_failure_never_publishes_partial_final_files():
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        env_state, policy_state, optimizer_state, meta = _states()
        original_save = torch.save
        calls = 0

        def fail_on_second_save(payload, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated disk write failure")
            return original_save(payload, destination)

        torch.save = fail_on_second_save
        try:
            try:
                save_split_checkpoint(
                    directory,
                    64,
                    env_state,
                    policy_state,
                    optimizer_state,
                    meta,
                    min_remaining_bytes=0,
                )
            except OSError:
                pass
            else:
                raise AssertionError("模拟写入失败应向上报告")
        finally:
            torch.save = original_save

        assert not list(directory.glob("*step_64.pth"))
        assert not list(directory.glob("*.tmp"))
