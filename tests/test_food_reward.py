import torch

from protolife.env import ProtoLifeEnv


def make_env(rewards: dict) -> ProtoLifeEnv:
    env = ProtoLifeEnv.__new__(ProtoLifeEnv)
    env.config = {"rewards": rewards}
    env.default_config = {}
    env.energy_max = 100.0
    return env


def test_linear_food_reward_decreases_with_energy():
    env = make_env(
        {
            "food_reward_min": 0.1,
            "food_reward_max": 1.0,
            "food_reward_mode": "linear",
            "food_reward_coefficient": 4.0,
        }
    )

    rewards = env._compute_food_reward(torch.tensor([0.0, 50.0, 100.0]))

    assert torch.allclose(rewards, torch.tensor([1.0, 0.55, 0.1]))


def test_log_food_reward_drops_sharply_near_full_energy():
    env = make_env(
        {
            "food_reward_min": 0.1,
            "food_reward_max": 1.0,
            "food_reward_mode": "log",
            "food_reward_coefficient": 4.0,
        }
    )

    rewards = env._compute_food_reward(torch.tensor([0.0, 90.0, 100.0]))

    assert rewards[0].item() == 1.0
    assert rewards[1].item() > 0.19
    assert torch.isclose(rewards[2], torch.tensor(0.1))


def test_legacy_fixed_food_reward_is_preserved():
    env = make_env({"food_reward": 0.6})

    rewards = env._compute_food_reward(torch.tensor([0.0, 50.0, 100.0]))

    assert torch.allclose(rewards, torch.full((3,), 0.6))
