import pytest

torch = pytest.importorskip("torch")

from protolife.env import ENV_DEFAULTS, ProtoLifeEnv


def _make_env():
    config = {
        "world": {"height": 4, "width": 4, "food_density": 0.0, "toxin_density": 0.0},
        "training": {"num_envs": 1},
        "agents": {
            "per_env": 1,
            "max_per_env": 2,
            "base_energy": 100.0,
            "base_health": 100.0,
            "base_metabolism_cost": 0.0,
            "move_cost": 0.0,
        },
        "model": {"observation_radius": 0},
        "features": {
            "use_death": True,
            "use_health": True,
            "use_reproduction": False,
            "use_combat": True,
            "use_communication": True,
        },
        "combat": {"damage": 10.0, "radius": 2.0, "decay": "none"},
        "action_rewards": {"STAY": 0.0, "ATTACK": 0.0, "COMMUNICATE": 0.0},
        "action_energy_costs": {"ATTACK": 0.0, "COMMUNICATE": 0.0},
        "rewards": {
            "enable_proximity_reward": False,
            "energy_reward_at_fit": 0.0,
            "energy_reward_at_extreme": 0.0,
            "energy_reward_at_max": 0.0,
            "health_reward_at_max": 0.0,
            "health_reward_at_zero": 0.0,
            "health_recovery_per_step": 0.0,
            "health_decay_min": 0.0,
            "survival_reward": 0.0,
        },
    }
    env = ProtoLifeEnv(config, default_config=ENV_DEFAULTS)
    env.reset()
    env.agent_batch.state["x"][0] = torch.tensor([1, 2], device=env.device)
    env.agent_batch.state["y"][0] = torch.tensor([1, 1], device=env.device)
    return env


def test_inactive_capacity_slot_cannot_act_or_become_an_attack_target():
    env = _make_env()

    env.step(torch.tensor([6, 6], device=env.device))

    assert env.agent_batch.state["health"][0, 0].item() == pytest.approx(100.0)
    assert env.agent_batch.state["health"][0, 1].item() == pytest.approx(0.0)
    assert env.agent_batch.state["energy"][0, 1].item() == pytest.approx(0.0)
    assert env.agent_batch.state["age"][0, 1].item() == 0
    assert torch.count_nonzero(env.agent_batch.state["comm"][0, 1]).item() == 0


def test_attack_damage_flows_from_attacker_row_to_live_target_column():
    env = _make_env()
    env.agent_batch.state["energy"][0, 1] = 100.0
    env.agent_batch.state["health"][0, 1] = 100.0

    env.step(torch.tensor([6, 0], device=env.device))

    assert env.agent_batch.state["health"][0, 0].item() == pytest.approx(100.0)
    assert env.agent_batch.state["health"][0, 1].item() == pytest.approx(90.0)
