import torch

from protolife.encoding import BIT_FOOD
from protolife.env import ENV_DEFAULTS, ProtoLifeEnv


def make_food_env(*, food_lifetime: int = 0, proximity: bool = True) -> ProtoLifeEnv:
    config = {
        "world": {
            "height": 5,
            "width": 5,
            "map_file": None,
            "food_density": 0.0,
            "toxin_density": 0.0,
            "food_respawn_interval": 0,
            "food_lifetime": food_lifetime,
        },
        "training": {"num_envs": 1},
        "agents": {
            "per_env": 1,
            "max_per_env": 1,
            "base_energy": 50.0,
            "base_health": 100.0,
            "base_metabolism_cost": 0.0,
            "move_cost": 0.0,
        },
        "model": {"observation_radius": 2},
        "features": {
            "use_death": False,
            "use_health": False,
            "use_reproduction": False,
            "use_combat": False,
            "use_communication": False,
            "use_terraforming": False,
        },
        "action_rewards": {"STAY": 0.0},
        "action_energy_costs": {"STAY": 0.0},
        "rewards": {
            "enable_proximity_reward": proximity,
            "see_food_reward": 0.1,
            "stand_on_food_reward": 0.3,
            "vision_decay_mode": "linear",
            "vision_decay_coefficient": 0.0,
            "energy_reward_at_fit": 0.0,
            "energy_reward_at_extreme": 0.0,
            "energy_reward_at_max": 0.0,
            "survival_reward": 0.0,
        },
    }
    env = ProtoLifeEnv(config, default_config=ENV_DEFAULTS)
    env.reset()
    env.map_state.zero_()
    env.food_age.zero_()
    env.agent_batch.state["x"][0, 0] = 2
    env.agent_batch.state["y"][0, 0] = 2
    return env


def test_visible_food_rewards_accumulate_by_quantity():
    one_food = make_food_env()
    one_food.map_state[0, 2, 1] |= BIT_FOOD
    one_reward = one_food.step(torch.tensor([0], device=one_food.device)).rewards[0]

    two_food = make_food_env()
    two_food.map_state[0, 2, 1] |= BIT_FOOD
    two_food.map_state[0, 2, 3] |= BIT_FOOD
    two_reward = two_food.step(torch.tensor([0], device=two_food.device)).rewards[0]

    assert torch.isclose(one_reward.cpu(), torch.tensor(0.1))
    assert torch.isclose(two_reward.cpu(), torch.tensor(0.2))


def test_food_disappears_when_lifetime_is_reached():
    env = make_food_env(food_lifetime=2, proximity=False)
    env.map_state[0, 0, 0] |= BIT_FOOD
    action = torch.tensor([0], device=env.device)

    env.step(action)
    assert bool((env.map_state[0, 0, 0] & BIT_FOOD).item())
    assert env.food_age[0, 0, 0].item() == 1

    env.step(action)
    assert not bool((env.map_state[0, 0, 0] & BIT_FOOD).item())
    assert env.food_age[0, 0, 0].item() == 0


def test_food_age_round_trips_through_checkpoint_state():
    env = make_food_env(food_lifetime=10, proximity=False)
    env.map_state[0, 0, 0] |= BIT_FOOD
    env.food_age[0, 0, 0] = 7
    state = env.export_state()

    restored = make_food_env(food_lifetime=10, proximity=False)
    restored.load_state(state)

    assert restored.food_age[0, 0, 0].item() == 7


def test_environment_generator_continues_after_checkpoint_restore():
    env = make_food_env(food_lifetime=10, proximity=False)
    state = env.export_state()
    expected = torch.rand(8, device=env.device, generator=env.random_generator)

    restored = make_food_env(food_lifetime=10, proximity=False)
    restored.load_state(state)
    actual = torch.rand(8, device=restored.device, generator=restored.random_generator)

    assert torch.equal(actual.cpu(), expected.cpu())
