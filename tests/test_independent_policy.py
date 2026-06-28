import pytest

torch = pytest.importorskip("torch")

from protolife.env import ENV_DEFAULTS
from protolife.policy import IndependentCNNPolicies, build_policy


def _build_independent_policy(pooling=True):
    config = {
        "training": {"num_envs": 1},
        "agents": {"per_env": 2},
        "model": {
            "use_cnn": True,
            "cnn_independent": True,
            "cnn_channels": [4],
            "cnn_feature_dim": 8,
            "cnn_pooling": pooling,
            "cnn_pool_size": [2, 2],
            "rnn_hidden_dim": 6,
            "rnn_type": "gru",
        },
    }
    return build_policy(config, ENV_DEFAULTS, patch_shape=(6, 5, 5))


def test_each_agent_has_a_distinct_cnn_brain():
    policy = _build_independent_policy()

    assert isinstance(policy, IndependentCNNPolicies)
    assert policy.brain_count == 2
    assert policy.brains[0] is not policy.brains[1]
    assert (
        policy.brains[0].cnn[0].weight.data_ptr()
        != policy.brains[1].cnn[0].weight.data_ptr()
    )


def test_pooling_and_agent_state_are_connected_to_each_brain():
    policy = _build_independent_policy()
    patches = torch.zeros(1, 2, 6, 5, 5)
    features = torch.tensor([[[1.0, 0.5, 0.1], [0.2, 0.8, 0.9]]])
    hidden = torch.zeros(1, 2, 6)

    logits, values, new_hidden = policy(patches, hidden, features)

    assert policy.brains[0].pool.output_size == (2, 2)
    assert policy.brains[0].rnn.input_size == 11  # 8 维视觉 + 能量/健康/年龄
    assert logits.shape == (1, 2, 11)
    assert values.shape == (1, 2, 1)
    assert new_hidden.shape == (1, 2, 6)


def test_vectorized_independent_forward_matches_each_brain_separately():
    policy = _build_independent_policy()
    patches = torch.randn(1, 2, 6, 5, 5)
    features = torch.rand(1, 2, 3)
    hidden = torch.randn(1, 2, 6)

    logits, values, new_hidden = policy(patches, hidden, features)
    expected_logits = []
    expected_values = []
    expected_hidden = []
    for index, brain in enumerate(policy.brains):
        one_logits, one_value, one_hidden = brain(
            patches[:, index : index + 1],
            hidden[:, index : index + 1],
            features[:, index : index + 1],
        )
        expected_logits.append(one_logits)
        expected_values.append(one_value)
        expected_hidden.append(one_hidden)

    assert torch.allclose(logits, torch.cat(expected_logits, dim=1), atol=1e-6)
    assert torch.allclose(values, torch.cat(expected_values, dim=1), atol=1e-6)
    assert torch.allclose(new_hidden, torch.cat(expected_hidden, dim=1), atol=1e-6)


def test_environment_memory_reset_cannot_modify_forward_graph_storage():
    policy = _build_independent_policy()
    patches = torch.randn(1, 2, 6, 5, 5)
    features = torch.rand(1, 2, 3)
    hidden = torch.zeros(1, 2, 6)

    logits, values, new_hidden = policy(patches, hidden, features)
    environment_memory = new_hidden.detach().clone()
    environment_memory[0, 1].zero_()  # 模拟子代出生时清空槽位记忆
    (logits.sum() + values.sum()).backward()

    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_child_inherits_parent_policy_head_with_optional_mutation():
    policy = _build_independent_policy()
    parent = policy.brains[0].policy_head
    child = policy.brains[1].policy_head
    with torch.no_grad():
        parent.weight.fill_(0.25)
        parent.bias.fill_(-0.1)
        child.weight.zero_()
        child.bias.zero_()

    policy.inherit_policy_head(0, 0, 1, mutation_std=0.0)
    assert torch.equal(parent.weight, child.weight)
    assert torch.equal(parent.bias, child.bias)

    policy.inherit_policy_head(0, 0, 1, mutation_std=0.01)
    assert not torch.equal(parent.weight, child.weight)
