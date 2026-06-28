import pytest

torch = pytest.importorskip("torch")

from protolife.env import ENV_DEFAULTS
from protolife.policy import build_policy
from scripts.train_phase0 import _load_initial_model, _save_last_survivor_model


def _make_policy():
    config = {
        "training": {"num_envs": 1},
        "agents": {"per_env": 3},
        "model": {
            "use_cnn": True,
            "cnn_independent": True,
            "cnn_channels": [4],
            "cnn_feature_dim": 8,
            "cnn_pooling": True,
            "cnn_pool_size": [2, 2],
            "rnn_hidden_dim": 6,
        },
    }
    return build_policy(config, ENV_DEFAULTS, patch_shape=(6, 5, 5))


def test_last_survivor_brain_can_initialize_every_agent(tmp_path):
    source = _make_policy()
    with torch.no_grad():
        for parameter in source.brains[1].parameters():
            parameter.fill_(0.125)
    saved = _save_last_survivor_model(
        source,
        tmp_path,
        {"env_idx": 0, "agent_idx": 1, "energy": 10.0, "health": 20.0},
        step=99,
    )

    target = _make_policy()
    _load_initial_model(target, saved, torch.device("cpu"))

    expected = source.brains[1].state_dict()
    for brain in target.brains:
        actual = brain.state_dict()
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(actual[key], expected[key])
