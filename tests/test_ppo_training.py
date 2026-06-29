import torch

from protolife.policy import MLPPolicy
from scripts.train_phase0 import (
    _behavior_log_prob_and_entropy,
    _compute_gae,
    _mask_hidden,
    _ppo_clipped_surrogate,
    _ppo_update,
    _transition_continuation,
)


def test_gae_uses_n_step_bootstrap_return():
    rewards = torch.tensor([[[1.0]], [[1.0]]])
    values = torch.tensor([[[0.5]], [[0.25]]])
    continuation = torch.ones_like(rewards, dtype=torch.bool)
    bootstrap = torch.tensor([[0.75]])

    advantages, returns = _compute_gae(
        rewards, values, continuation, bootstrap, gamma=1.0, gae_lambda=1.0
    )

    assert torch.allclose(advantages.flatten(), torch.tensor([2.25, 1.5]))
    assert torch.allclose(returns.flatten(), torch.tensor([2.75, 1.75]))


def test_newborn_slot_never_bootstraps_the_previous_life():
    valid = torch.tensor([[True, True]])
    next_alive = torch.tensor([[True, True]])
    dones = torch.tensor([False, False])

    continuation = _transition_continuation(
        valid, next_alive, dones, reproduction_events=[(0, 0, 1)]
    )

    assert torch.equal(continuation, torch.tensor([[True, False]]))

def test_gae_stops_at_death_or_birth_boundary():
    rewards = torch.tensor([[[1.0]], [[1.0]]])
    values = torch.tensor([[[0.5]], [[0.25]]])
    continuation = torch.tensor([[[False]], [[True]]])
    bootstrap = torch.tensor([[0.75]])

    advantages, returns = _compute_gae(
        rewards, values, continuation, bootstrap, gamma=1.0, gae_lambda=1.0
    )

    assert torch.allclose(advantages.flatten(), torch.tensor([0.5, 1.5]))
    assert torch.allclose(returns.flatten(), torch.tensor([1.0, 1.75]))


def test_ppo_surrogate_clips_both_advantage_directions():
    ratio = torch.tensor([2.0, 0.5])
    advantages = torch.tensor([1.0, -1.0])

    surrogate = _ppo_clipped_surrogate(ratio, advantages, clip_range=0.2)

    assert torch.allclose(surrogate, torch.tensor([1.2, -0.8]))


def test_extreme_negative_advantage_keeps_ppo_correction_gradient():
    log_ratio = torch.tensor(25.0, requires_grad=True)
    ratio = torch.exp(log_ratio)
    surrogate = _ppo_clipped_surrogate(
        ratio, torch.tensor(-1.0), clip_range=0.2
    )

    (-surrogate).backward()

    assert log_ratio.grad is not None
    assert log_ratio.grad.item() > 0

def test_epsilon_behavior_log_prob_matches_mixture_distribution():
    logits = torch.log(torch.tensor([[[4.0, 1.0]]]))
    actions = torch.tensor([[0]])

    log_prob, entropy = _behavior_log_prob_and_entropy(logits, actions, epsilon=0.3)

    expected_probabilities = torch.tensor([0.71, 0.29])
    expected_entropy = -(
        expected_probabilities * torch.log(expected_probabilities)
    ).sum()
    assert torch.allclose(log_prob.squeeze(), torch.log(expected_probabilities[0]))
    assert torch.allclose(entropy.squeeze(), expected_entropy)


def test_hidden_state_is_cut_at_identity_boundary():
    hidden = torch.ones(1, 2, 3)
    continuation = torch.tensor([[True, False]])

    masked = _mask_hidden(hidden, continuation)

    assert torch.equal(masked[0, 0], torch.ones(3))
    assert torch.equal(masked[0, 1], torch.zeros(3))


def test_ppo_update_changes_policy_with_valid_rollout():
    torch.manual_seed(7)
    policy = MLPPolicy(obs_dim=3, hidden_dim=8, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rollout = []
    for _ in range(4):
        observation = torch.randn(1, 3)
        with torch.no_grad():
            logits, values = policy(observation)
            logits = logits.reshape(1, 1, 2)
            values = values.reshape(1, 1)
            actions = torch.distributions.Categorical(logits=logits).sample()
            old_log_probs, _ = _behavior_log_prob_and_entropy(
                logits, actions, epsilon=0.0
            )
        rollout.append(
            {
                "patch": None,
                "agent_features": None,
                "agent_obs": observation,
                "hidden": None,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "old_values": values,
                "valid": torch.ones(1, 1, dtype=torch.bool),
                "continuation": torch.ones(1, 1, dtype=torch.bool),
                "logit_noise": torch.zeros_like(logits),
            }
        )

    before = [parameter.detach().clone() for parameter in policy.parameters()]
    advantages = torch.ones(4, 1, 1)
    returns = torch.stack([sample["old_values"] for sample in rollout]) + 1.0

    metrics = _ppo_update(
        policy,
        optimizer,
        rollout,
        advantages,
        returns,
        clip_range=0.2,
        ppo_epochs=2,
        minibatch_steps=2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        epsilon_greedy=0.0,
    )

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters())
    )
    assert metrics["value_loss"] > 0
