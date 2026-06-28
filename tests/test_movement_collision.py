import pytest

torch = pytest.importorskip("torch")

from protolife.env import BIT_TERRAIN_0, ENV_DEFAULTS, ProtoLifeEnv


def _make_env(agents=1):
    config = {
        "world": {
            "height": 4,
            "width": 4,
            "food_density": 0.0,
            "toxin_density": 0.0,
        },
        "training": {"num_envs": 1},
        "agents": {
            "per_env": agents,
            "base_energy": 100.0,
            "base_metabolism_cost": 0.0,
            "move_cost": 0.0,
        },
        "model": {"observation_radius": 0},
        "features": {
            "use_death": False,
            "use_health": False,
            "use_reproduction": False,
            "use_combat": False,
            "use_communication": False,
        },
        "action_rewards": {"MOVE": 0.5},
        "action_energy_costs": {"MOVE": 0.0},
        "rewards": {
            "enable_proximity_reward": False,
            "energy_reward_at_fit": 0.0,
            "energy_reward_at_extreme": 0.0,
            "energy_reward_at_max": 0.0,
            "survival_reward": 0.0,
            "move_collision_penalty": -0.03,
        },
    }
    env = ProtoLifeEnv(config, default_config=ENV_DEFAULTS)
    env.reset()
    return env


def test_failed_move_replaces_move_reward_with_configured_penalty():
    env = _make_env()
    env.agent_batch.state["x"][0, 0] = 0
    env.agent_batch.state["y"][0, 0] = 0

    result = env.step(torch.tensor([3], device=env.device))  # LEFT 越界

    assert result.rewards[0].item() == pytest.approx(-0.03)
    assert env.agent_batch.state["x"][0, 0].item() == 0


def test_successful_move_keeps_move_reward():
    env = _make_env()
    env.agent_batch.state["x"][0, 0] = 1
    env.agent_batch.state["y"][0, 0] = 1

    result = env.step(torch.tensor([4], device=env.device))  # RIGHT

    assert result.rewards[0].item() == pytest.approx(0.5)
    assert env.agent_batch.state["x"][0, 0].item() == 2


def test_agents_inside_walls_are_relocated_to_walkable_cells():
    env = _make_env(agents=4)
    env.map_state.fill_(BIT_TERRAIN_0)
    env.map_state[0, 2, 3] = 0
    env.agent_batch.state["x"].zero_()
    env.agent_batch.state["y"].zero_()

    env._relocate_agents_inside_walls()

    assert torch.all(env.agent_batch.state["x"] == 3)
    assert torch.all(env.agent_batch.state["y"] == 2)
