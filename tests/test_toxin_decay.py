import pytest

torch = pytest.importorskip("torch")

from protolife.env import BIT_TOXIN, ENV_DEFAULTS, ProtoLifeEnv


def _make_env_config(**overrides):
    config = {
        "world": {
            "height": 4,
            "width": 4,
            "food_density": 0.0,
            "toxin_density": 1.0,
            "toxin_lifetime": 0,
        },
        "training": {"num_envs": 1},
        "agents": {"per_env": 1},
        "model": {"observation_radius": 0},
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    return config


def test_reset_scatter_sets_initial_toxins_and_age():
    config = _make_env_config(world={"toxin_lifetime": 5})
    env = ProtoLifeEnv(config, default_config=ENV_DEFAULTS)

    env.reset()

    toxin_mask = (env.map_state & BIT_TOXIN).bool()
    assert toxin_mask.any(), "reset 应当按密度撒下初始毒素"
    assert torch.all(env.toxin_age[toxin_mask] == 0), "毒素初始年龄需归零"


def test_toxins_decay_after_lifetime():
    config = _make_env_config(world={"toxin_lifetime": 1})
    env = ProtoLifeEnv(config, default_config=ENV_DEFAULTS)
    env.reset()

    initial_toxins = (env.map_state & BIT_TOXIN).sum().item()
    assert initial_toxins > 0, "需要有初始毒素才能测试衰减"

    actions = torch.zeros(env.agent_batch.num_envs * env.agent_batch.agents_per_env, dtype=torch.int64)
    env.step(actions)

    remaining_toxins = (env.map_state & BIT_TOXIN).sum().item()
    assert remaining_toxins == 0, "毒素寿命耗尽后应被清除"
