import pytest

torch = pytest.importorskip("torch")

from protolife.env import ENV_DEFAULTS, ProtoLifeEnv


def _make_env(success_probability):
    config = {
        "world": {"height": 4, "width": 4, "food_density": 0.0, "toxin_density": 0.0},
        "training": {"num_envs": 1},
        "agents": {
            "per_env": 2,
            "max_per_env": 4,
            "base_energy": 100.0,
            "base_metabolism_cost": 0.0,
            "move_cost": 0.0,
            "reproduction_energy_threshold": 50.0,
            "child_energy_fraction": 0.1,
        },
        "model": {"observation_radius": 0},
        "features": {
            "use_death": True,
            "use_health": False,
            "use_reproduction": True,
            "use_combat": False,
            "use_communication": False,
        },
        "reproduction": {"success_probability": success_probability},
        "action_rewards": {"STAY": 0.0, "REPRODUCE": 0.2},
        "action_energy_costs": {"REPRODUCE": 0.0},
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


def _population(env):
    return int(env._alive_mask().sum().item())


def test_population_can_grow_from_initial_count_to_configured_maximum():
    env = _make_env(success_probability=1.0)
    assert _population(env) == 2

    first = env.step(torch.tensor([10, 0, 0, 0], device=env.device))
    assert _population(env) == 3
    assert first.rewards[0].item() == pytest.approx(0.2)

    second = env.step(torch.tensor([10, 0, 0, 0], device=env.device))
    assert _population(env) == 4
    assert second.rewards[0].item() == pytest.approx(0.2)

    third = env.step(torch.tensor([10, 0, 0, 0], device=env.device))
    assert _population(env) == 4
    assert third.rewards[0].item() == pytest.approx(0.0)


def test_failed_reproduction_does_not_create_child_or_reward():
    env = _make_env(success_probability=0.0)

    result = env.step(torch.tensor([10, 0, 0, 0], device=env.device))

    assert _population(env) == 2
    assert result.rewards[0].item() == pytest.approx(0.0)
