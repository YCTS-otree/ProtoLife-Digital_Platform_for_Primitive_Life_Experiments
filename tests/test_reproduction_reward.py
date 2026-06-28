import pytest

torch = pytest.importorskip("torch")

from protolife.env import ENV_DEFAULTS, ProtoLifeEnv


def _make_env():
    config = {
        "world": {
            "height": 4,
            "width": 4,
            "food_density": 0.0,
            "toxin_density": 0.0,
        },
        "training": {"num_envs": 1},
        "agents": {
            "per_env": 2,
            "base_energy": 100.0,
            "base_metabolism_cost": 0.0,
            "move_cost": 0.0,
            "reproduction_energy_threshold": 50.0,
            "child_energy_fraction": 0.3,
        },
        "model": {"observation_radius": 0},
        "features": {
            "use_death": True,
            "use_health": False,
            "use_reproduction": True,
            "use_combat": False,
            "use_communication": False,
        },
        "action_rewards": {
            "STAY": 0.0,
            "MOVE": 0.0,
            "EAT": 0.0,
            "ATTACK": 0.0,
            "COMMUNICATE": 0.0,
            "BUILD": 0.0,
            "REMOVE": 0.0,
            "REPRODUCE": 0.2,
        },
        "action_energy_costs": {"REPRODUCE": 0.0},
        "reproduction": {"success_probability": 1.0},
        "rewards": {
            "enable_proximity_reward": False,
            "energy_reward_at_fit": 0.0,
            "energy_reward_at_extreme": 0.0,
            "energy_reward_at_max": 0.0,
            "survival_reward": 0.0,
        },
    }
    env = ProtoLifeEnv(config, default_config=ENV_DEFAULTS)
    env.reset()
    return env


def test_failed_reproduction_has_no_action_reward():
    env = _make_env()

    result = env.step(torch.tensor([10, 0], device=env.device))

    assert result.rewards[0].item() == pytest.approx(0.0)


def test_successful_reproduction_receives_action_reward():
    env = _make_env()
    env.agent_batch.state["energy"][0, 1] = 0.0
    env.agent_batch.state["memory"][0, 1].fill_(1.0)

    result = env.step(torch.tensor([10, 0], device=env.device))

    assert result.rewards[0].item() == pytest.approx(0.2)
    assert env.agent_batch.state["energy"][0, 1].item() > 0.0
    assert result.reproduction_events == [(0, 0, 1)]
    assert torch.count_nonzero(env.agent_batch.state["memory"][0, 1]).item() == 0
