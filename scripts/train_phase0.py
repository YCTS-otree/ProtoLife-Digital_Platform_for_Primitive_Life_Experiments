"""Phase0 训练脚本占位。

示例用法：
    python -m scripts.train_phase0 --config config/phase0_survival.yaml
"""
from __future__ import annotations

import argparse
import copy
import re
import time
from pathlib import Path

import torch
import yaml

from protolife.config_loader import load_config, save_config_with_comments
from protolife.checkpoint import load_checkpoint, save_checkpoint
from protolife.env import ProtoLifeEnv
from protolife.policy import action_space, build_policy
from protolife.utils.seed_utils import generate_seed, set_seed


TRAIN_DEFAULTS = {
    "world": {"random_seed": 0},
    "training": {
        "checkpoint_dir": "checkpoints",
        "initial_model": None,
        "auto_resume": True,
        "rollout_steps": 128,
        "n_steps": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "ppo_epochs": 4,
        "ppo_minibatch_steps": 4,
        "value_coef": 0.5,
        "normalize_advantages": True,
        "save_interval": 100,
        "entropy_coef": 0.01,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "action_noise": {"gaussian_std": 0.0, "epsilon": 0.0},
        "print_actions": False,
        "print_interval": 128,
    },
    "logging": {"save_dir": "runs/default", "snapshot_interval": 50},
}


def _load_state_dict_lenient(model: torch.nn.Module, state_dict: dict) -> bool:
    """加载 state_dict，自动跳过尺寸不匹配的权重并返回是否有跳过。"""

    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    dropped_keys = []
    for key, tensor in state_dict.items():
        if key not in current:
            continue
        if tensor.shape != current[key].shape:
            dropped_keys.append(key)
            continue
        compatible[key] = tensor

    model.load_state_dict(compatible, strict=False)
    missing_keys = [key for key in current if key not in compatible]
    if dropped_keys:
        print(
            "[警告] 以下权重尺寸不匹配，已跳过加载（可能是 observation_radius 调整导致）：",
            dropped_keys,
        )
    if missing_keys:
        print(f"[警告] checkpoint 缺少 {len(missing_keys)} 个当前模型参数，使用新初始化值")
    return bool(dropped_keys or missing_keys)


def _extract_model_state(payload: dict) -> dict:
    """兼容单脑模型、完整策略模型以及完整 checkpoint。"""

    for key in ("brain_state_dict", "policy_state_dict", "model_state_dict", "state_dict"):
        state = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(state, dict):
            return state
    return payload


def _load_initial_model(policy: torch.nn.Module, path: Path, device: torch.device) -> None:
    """将一个单个体模型复制到所有独立大脑，或载入完整策略模型。"""

    payload = torch.load(path, map_location=device)
    state = _extract_model_state(payload)
    if not isinstance(state, dict):
        raise ValueError(f"初始模型格式不受支持: {path}")

    is_independent = getattr(policy, "cnn_independent", False)
    is_full_independent_state = any(str(key).startswith("brains.") for key in state)
    if is_independent and not is_full_independent_state:
        skipped = False
        for brain in policy.brains:
            skipped = _load_state_dict_lenient(brain, state) or skipped
        if skipped:
            print("[警告] 初始单脑模型有部分参数尺寸不匹配，已跳过")
        print(f"已将初始单脑模型复制到 {policy.brain_count} 个个体: {path}")
        return

    _load_state_dict_lenient(policy, state)
    print(f"已加载完整初始模型: {path}")


def _select_last_survivor(env: ProtoLifeEnv) -> dict | None:
    """选择当前仍存活且生命状态最高的个体，供灭绝时保存。"""

    energy = env.agent_batch.state["energy"]
    health = env.agent_batch.state["health"]
    alive = energy > 0
    if env.use_health:
        alive = alive & (health > 0)
    if not alive.any():
        return None

    score = energy / max(env.energy_max, 1e-6)
    if env.use_health:
        score = score + health / max(env.health_max, 1e-6)
    score = torch.where(alive, score, torch.full_like(score, float("-inf")))
    flat_index = int(score.reshape(-1).argmax().item())
    agent_count = env.agent_batch.agents_per_env
    env_idx, agent_idx = divmod(flat_index, agent_count)
    return {
        "env_idx": env_idx,
        "agent_idx": agent_idx,
        "energy": float(energy[env_idx, agent_idx].item()),
        "health": float(health[env_idx, agent_idx].item()),
    }


def _save_last_survivor_model(
    policy: torch.nn.Module,
    checkpoint_dir: Path,
    survivor: dict,
    step: int,
) -> Path:
    """保存灭绝前最后存活个体的完整独立大脑。"""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / "last_survivor.pth"
    payload = {
        "step": int(step),
        "env_idx": survivor["env_idx"],
        "agent_idx": survivor["agent_idx"],
        "energy_before_final_step": survivor["energy"],
        "health_before_final_step": survivor["health"],
    }
    if getattr(policy, "cnn_independent", False):
        brain_index = survivor["env_idx"] * policy.agents_per_env + survivor["agent_idx"]
        payload["brain_state_dict"] = policy.brains[brain_index].state_dict()
    else:
        payload["policy_state_dict"] = policy.state_dict()
    torch.save(payload, destination)
    return destination


def get_cfg(config: dict, default_config: dict, section: str, key: str, fallback):
    return config.get(section, {}).get(
        key,
        default_config.get(section, {}).get(key, TRAIN_DEFAULTS.get(section, {}).get(key, fallback)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife Phase0 训练")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径，不提供时会使用 model/<name>/<name>.yaml",
    )
    parser.add_argument("--save-interval", type=int, default=None, help="多少步保存一次模型/存档")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint 输出目录")
    parser.add_argument("--model-dir", type=str, default=None, help="模型目录，支持指向已有模型继续训练")
    parser.add_argument("--model-name", type=str, default=None, help="模型名称，若缺省则交互式输入")
    parser.add_argument("--resume-from", type=str, default=None, help="从完整 checkpoint 继续推演")
    parser.add_argument(
        "--fresh", action="store_true", help="忽略已有 checkpoint，从初始模型或随机参数开始"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="无 checkpoint 时使用的初始模型；单脑模型会复制给所有个体",
    )
    parser.add_argument("--print-interval", type=int, default=None, help="训练日志打印间隔")
    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w.-]", "_", name)
    return name


def _merge_dict(default: dict, override: dict) -> dict:
    result = copy.deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_random_seed(config: dict, default_config: dict) -> int:
    world_cfg = config.setdefault("world", {})
    seed = world_cfg.get("random_seed")
    if seed is None:
        seed = default_config.get("world", {}).get("random_seed")
    if seed is None:
        seed = generate_seed()
    world_cfg["random_seed"] = seed
    return seed


def _prepare_model_dir(args: argparse.Namespace, fallback_tag: str) -> tuple[Path, str]:
    root = Path("model")
    root.mkdir(exist_ok=True)

    explicit_name = _sanitize_name(args.model_name) if args.model_name else ""

    if args.model_dir:
        candidate = Path(str(args.model_dir).replace(" ", "_"))
        model_name = explicit_name or _sanitize_name(candidate.name) or fallback_tag
    else:
        raw_name = args.model_name or ""
        if not raw_name:
            try:
                raw_name = input("请输入本次训练的模型名称(留空则使用时间戳): ")
            except EOFError:
                raw_name = ""
        sanitized = _sanitize_name(raw_name) if raw_name else ""
        model_name = sanitized or fallback_tag
        candidate = root / model_name

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        candidate = root / fallback_tag
        candidate.mkdir(parents=True, exist_ok=True)
        model_name = _sanitize_name(candidate.name) or fallback_tag

    return candidate, model_name


def _find_latest_full_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("full_step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _clone_hidden(hidden):
    if isinstance(hidden, tuple):
        return tuple(item.detach().clone() for item in hidden)
    if hidden is None:
        return None
    return hidden.detach().clone()


def _current_hidden(env: ProtoLifeEnv):
    memory = env.agent_batch.state.get("memory")
    if memory is None:
        return None
    if env.agent_batch.use_lstm:
        return memory, env.agent_batch.state["memory_cell"]
    return memory


def _forward_policy_inputs(
    policy: torch.nn.Module,
    patch: torch.Tensor | None,
    agent_features: torch.Tensor | None,
    agent_obs: torch.Tensor | None,
    hidden,
):
    if getattr(policy, "use_cnn", False):
        return policy(patch, hidden, agent_features)
    logits, values = policy(agent_obs)
    return logits, values, None


def _reshape_policy_outputs(
    logits: torch.Tensor, values: torch.Tensor, population_shape: torch.Size
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        logits.reshape(*population_shape, -1),
        values.reshape(*population_shape),
    )


def _behavior_log_prob_and_entropy(
    logits: torch.Tensor, actions: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    if epsilon > 0:
        probabilities = (1.0 - epsilon) * probabilities + epsilon / logits.size(-1)
        log_probs = torch.log(probabilities.clamp_min(1e-8))
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return selected, entropy


def _mask_hidden(hidden, continuation: torch.Tensor):
    if hidden is None:
        return None
    mask = continuation.unsqueeze(-1)
    if isinstance(hidden, tuple):
        return tuple(item * mask.to(dtype=item.dtype) for item in hidden)
    return hidden * mask.to(dtype=hidden.dtype)


def _transition_continuation(
    valid: torch.Tensor,
    next_alive: torch.Tensor,
    dones: torch.Tensor,
    reproduction_events: list[tuple[int, int, int]],
) -> torch.Tensor:
    """切断死亡以及新生命复用旧槽位处的回报和隐藏状态。"""

    born_mask = torch.zeros_like(valid)
    for env_idx, _, child_idx in reproduction_events:
        born_mask[env_idx, child_idx] = True
    return valid & next_alive & ~dones.reshape(valid.shape) & ~born_mask


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continuation: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算截断 n-step rollout 上的 GAE 与 bootstrap return。"""

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(bootstrap_values)
    next_values = bootstrap_values
    for index in range(rewards.size(0) - 1, -1, -1):
        can_continue = continuation[index].to(dtype=rewards.dtype)
        delta = rewards[index] + gamma * next_values * can_continue - values[index]
        next_advantage = delta + gamma * gae_lambda * can_continue * next_advantage
        advantages[index] = next_advantage
        next_values = values[index]
    returns = advantages + values
    return advantages, returns


def _normalize_valid_advantages(
    advantages: torch.Tensor, valid: torch.Tensor, independent: bool
) -> torch.Tensor:
    normalized = torch.zeros_like(advantages)
    if independent:
        flat_advantages = advantages.reshape(advantages.size(0), -1)
        flat_valid = valid.reshape(valid.size(0), -1)
        flat_normalized = normalized.reshape(normalized.size(0), -1)
        for brain_index in range(flat_advantages.size(1)):
            brain_mask = flat_valid[:, brain_index]
            brain_values = flat_advantages[:, brain_index][brain_mask]
            if brain_values.numel() == 0:
                continue
            if brain_values.numel() > 1:
                brain_values = (
                    (brain_values - brain_values.mean())
                    / (brain_values.std(unbiased=False) + 1e-8)
                )
            flat_normalized[brain_mask, brain_index] = brain_values
        return normalized

    valid_values = advantages[valid]
    if valid_values.numel() == 0:
        return normalized
    if valid_values.numel() > 1:
        valid_values = (valid_values - valid_values.mean()) / (
            valid_values.std(unbiased=False) + 1e-8
        )
    normalized[valid] = valid_values
    return normalized


def _ppo_clipped_surrogate(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_range: float
) -> torch.Tensor:
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return torch.minimum(unclipped, clipped)


def _masked_reduce(
    values: torch.Tensor, valid: torch.Tensor, independent: bool
) -> torch.Tensor:
    if independent:
        flat_values = values.reshape(values.size(0), -1)
        flat_valid = valid.reshape(valid.size(0), -1)
        sample_counts = flat_valid.sum(dim=0)
        active_brains = sample_counts > 0
        per_brain = (flat_values * flat_valid.to(values.dtype)).sum(dim=0)
        per_brain = per_brain / sample_counts.clamp_min(1).to(values.dtype)
        return per_brain[active_brains].sum()
    return values[valid].mean()


def _clear_inactive_brain_gradients(
    policy: torch.nn.Module, valid: torch.Tensor
) -> None:
    if not getattr(policy, "cnn_independent", False):
        return
    active_brains = valid.any(dim=0).reshape(-1)
    for brain_index, is_active in enumerate(active_brains.tolist()):
        if is_active:
            continue
        for parameter in policy.brains[brain_index].parameters():
            parameter.grad = None


def _ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: list[dict],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_range: float,
    ppo_epochs: int,
    minibatch_steps: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    epsilon_greedy: float,
) -> dict[str, float]:
    independent = bool(getattr(policy, "cnn_independent", False))
    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    update_count = 0
    chunk_starts = list(range(0, len(rollout), minibatch_steps))

    for _ in range(ppo_epochs):
        chunk_order = torch.randperm(len(chunk_starts)).tolist()
        for chunk_position in chunk_order:
            start = chunk_starts[chunk_position]
            end = min(start + minibatch_steps, len(rollout))
            hidden = _clone_hidden(rollout[start]["hidden"])
            new_log_probs = []
            new_values = []
            entropies = []

            for time_index in range(start, end):
                sample = rollout[time_index]
                logits, values, new_hidden = _forward_policy_inputs(
                    policy,
                    sample["patch"],
                    sample["agent_features"],
                    sample["agent_obs"],
                    hidden,
                )
                logits, values = _reshape_policy_outputs(
                    logits, values, sample["actions"].shape
                )
                behavior_logits = logits + sample["logit_noise"]
                log_probs, entropy = _behavior_log_prob_and_entropy(
                    behavior_logits, sample["actions"], epsilon_greedy
                )
                new_log_probs.append(log_probs)
                new_values.append(values)
                entropies.append(entropy)
                hidden = _mask_hidden(new_hidden, sample["continuation"])

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            entropies = torch.stack(entropies)
            old_log_probs = torch.stack(
                [rollout[index]["old_log_probs"] for index in range(start, end)]
            )
            old_values = torch.stack(
                [rollout[index]["old_values"] for index in range(start, end)]
            )
            valid = torch.stack(
                [rollout[index]["valid"] for index in range(start, end)]
            )
            if not bool(valid.any().item()):
                continue
            chunk_advantages = advantages[start:end]
            chunk_returns = returns[start:end]

            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            surrogate = _ppo_clipped_surrogate(ratio, chunk_advantages, clip_range)
            policy_loss = -_masked_reduce(surrogate, valid, independent)

            clipped_values = old_values + torch.clamp(
                new_values - old_values, -clip_range, clip_range
            )
            value_error = (new_values - chunk_returns).square()
            clipped_value_error = (clipped_values - chunk_returns).square()
            value_loss = 0.5 * _masked_reduce(
                torch.maximum(value_error, clipped_value_error), valid, independent
            )
            entropy = _masked_reduce(entropies, valid, independent)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _clear_inactive_brain_gradients(policy, valid)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                valid_log_ratio = log_ratio[valid]
                metric_sums["policy_loss"] += float(policy_loss.item())
                metric_sums["value_loss"] += float(value_loss.item())
                metric_sums["entropy"] += float(entropy.item())
                metric_sums["approx_kl"] += float((-valid_log_ratio).mean().item())
                metric_sums["clip_fraction"] += float(
                    ((ratio[valid] - 1.0).abs() > clip_range).float().mean().item()
                )
            update_count += 1

    if update_count == 0:
        return metric_sums
    return {key: value / update_count for key, value in metric_sums.items()}


def _save_ppo_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env: ProtoLifeEnv,
    checkpoint_dir: Path,
    step: int,
    *,
    n_steps: int,
    gamma: float,
    gae_lambda: float,
    ppo_clip: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_dir / f"model_step_{step}.pth")
    torch.save(optimizer.state_dict(), checkpoint_dir / f"optim_step_{step}.pth")
    save_checkpoint(
        checkpoint_dir / f"full_step_{step}.pt",
        env_state=env.export_state(),
        policy_state_dict=policy.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        meta={
            "step": step,
            "algorithm": "ppo_gae_n_step",
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ppo_clip": ppo_clip,
        },
    )

def main() -> None:
    args = parse_args()
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    model_dir, model_name = _prepare_model_dir(args, run_tag)
    log_dir = model_dir / "log"
    checkpoint_dir = model_dir / (args.checkpoint_dir or "checkpoint")

    model_config_path = model_dir / f"{model_name}.yaml"
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = model_config_path

    config, default_config = load_config(str(config_path))
    existing_model_config = {}
    if model_config_path.exists():
        existing_model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}

    config = _merge_dict(existing_model_config, config)
    resolved_seed = _resolve_random_seed(config, default_config)
    merged_config = _merge_dict(default_config, config)
    merged_config.setdefault("world", {})["random_seed"] = resolved_seed
    save_config_with_comments(model_config_path, merged_config)
    print(f"模型目录: {model_dir.resolve()}")
    print(f"配置文件: {config_path.resolve()}")
    print(f"Checkpoint 目录: {checkpoint_dir.resolve()}")
    set_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    env = ProtoLifeEnv(config, default_config)
    policy = build_policy(config, default_config, obs_dim=env.observation_dim, patch_shape=env.patch_shape).to(env.device)
    learning_rate = float(
        get_cfg(config, default_config, "training", "learning_rate", TRAIN_DEFAULTS["training"]["learning_rate"])
    )
    max_grad_norm = float(
        get_cfg(config, default_config, "training", "max_grad_norm", TRAIN_DEFAULTS["training"]["max_grad_norm"])
    )
    if learning_rate <= 0:
        raise ValueError("training.learning_rate 必须大于 0")
    try:
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=learning_rate, fused=env.device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if getattr(policy, "cnn_independent", False):
        print(f"已启用 {policy.brain_count} 个完全独立的 CNN 个体大脑")

    logger = None
    if get_cfg(config, default_config, "logging", "save_dir", True):
        from protolife.logger import ExperimentLogger

        log_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(
            save_dir=log_dir,
            snapshot_interval=get_cfg(config, default_config, "logging", "snapshot_interval", 50),
            env_index=get_cfg(config, default_config, "logging", "env_index", 0),
            run_tag=run_tag,
            metadata={
                "height": env.height,
                "width": env.width,
                "agent_marker_size": env.agent_marker_size,
                "num_envs": env.agent_batch.num_envs,
                "agents_per_env": env.agent_batch.agents_per_env,
                "initial_agents_per_env": env.initial_agents_per_env,
                "max_agents_per_env": env.max_agents_per_env,
                "run_name": model_dir.name,
                "run_tag": run_tag,
                "map_file": env.map_file,
                "toxin_lifetime": getattr(env, "toxin_lifetime", None),
                "use_health": env.use_health,
            },
            buffer_on_gpu=get_cfg(
                config, default_config, "logging", "snapshot_gpu_stage", False
            ),
            flush_interval=get_cfg(
                config, default_config, "logging", "snapshot_flush_interval", 8
            ),
        )

    checkpoint_dir = Path(checkpoint_dir)
    save_interval = int(
        args.save_interval
        or get_cfg(config, default_config, "training", "save_interval", 100)
    )
    print_actions = bool(get_cfg(config, default_config, "training", "print_actions", False))
    entropy_coef = float(
        get_cfg(config, default_config, "training", "entropy_coef", 0.01)
    )
    n_steps = int(
        get_cfg(config, default_config, "training", "n_steps", TRAIN_DEFAULTS["training"]["n_steps"])
    )
    gamma = float(
        get_cfg(config, default_config, "training", "gamma", TRAIN_DEFAULTS["training"]["gamma"])
    )
    gae_lambda = float(
        get_cfg(
            config, default_config, "training", "gae_lambda", TRAIN_DEFAULTS["training"]["gae_lambda"]
        )
    )
    ppo_clip = float(
        get_cfg(config, default_config, "training", "ppo_clip", TRAIN_DEFAULTS["training"]["ppo_clip"])
    )
    ppo_epochs = int(
        get_cfg(config, default_config, "training", "ppo_epochs", TRAIN_DEFAULTS["training"]["ppo_epochs"])
    )
    ppo_minibatch_steps = int(
        get_cfg(
            config,
            default_config,
            "training",
            "ppo_minibatch_steps",
            TRAIN_DEFAULTS["training"]["ppo_minibatch_steps"],
        )
    )
    value_coef = float(
        get_cfg(config, default_config, "training", "value_coef", TRAIN_DEFAULTS["training"]["value_coef"])
    )
    normalize_advantages = bool(
        get_cfg(
            config,
            default_config,
            "training",
            "normalize_advantages",
            TRAIN_DEFAULTS["training"]["normalize_advantages"],
        )
    )
    policy_mutation_std = float(
        get_cfg(config, default_config, "reproduction", "policy_mutation_std", 0.01)
    )
    print_interval = args.print_interval or get_cfg(
        config, default_config, "training", "print_interval", TRAIN_DEFAULTS["training"]["print_interval"]
    )

    action_noise_cfg = (
        config.get("training", {}).get("action_noise")
        or default_config.get("training", {}).get("action_noise")
        or TRAIN_DEFAULTS["training"].get("action_noise")
        or {}
    )
    gaussian_noise_std = float(action_noise_cfg.get("gaussian_std", 0.0) or 0.0)
    epsilon_greedy = float(action_noise_cfg.get("epsilon", 0.0) or 0.0)

    if save_interval <= 0:
        raise ValueError("training.save_interval 必须大于 0")
    if n_steps <= 0:
        raise ValueError("training.n_steps 必须大于 0")
    if not 0 < gamma <= 1:
        raise ValueError("training.gamma 必须在 (0, 1] 范围内")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("training.gae_lambda 必须在 [0, 1] 范围内")
    if not 0 < ppo_clip < 1:
        raise ValueError("training.ppo_clip 必须在 (0, 1) 范围内")
    if ppo_epochs <= 0 or ppo_minibatch_steps <= 0:
        raise ValueError("ppo_epochs 和 ppo_minibatch_steps 必须大于 0")
    if value_coef < 0 or entropy_coef < 0:
        raise ValueError("value_coef 和 entropy_coef 不能为负数")
    if gaussian_noise_std < 0 or not 0 <= epsilon_greedy <= 1:
        raise ValueError("action_noise 参数超出有效范围")

    if logger:
        logger.metadata.update(
            {
                "algorithm": "ppo_gae_n_step",
                "n_steps": n_steps,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "ppo_clip": ppo_clip,
                "ppo_epochs": ppo_epochs,
                "ppo_minibatch_steps": ppo_minibatch_steps,
            }
        )

    start_step = 0
    auto_resume = bool(get_cfg(config, default_config, "training", "auto_resume", True))
    latest_checkpoint = (
        _find_latest_full_checkpoint(checkpoint_dir)
        if auto_resume and not args.fresh
        else None
    )
    checkpoint_path = Path(args.resume_from) if args.resume_from else latest_checkpoint

    if checkpoint_path:
        print(f"使用的 checkpoint: {checkpoint_path.resolve()}")
        env_state, policy_state, optim_state, meta = load_checkpoint(checkpoint_path, map_location=env.device)
        env.load_state(env_state)
        skipped = _load_state_dict_lenient(policy, policy_state)
        checkpoint_algorithm = meta.get("algorithm")
        if optim_state and not skipped and checkpoint_algorithm == "ppo_gae_n_step":
            optimizer.load_state_dict(optim_state)
            # optimizer checkpoint 会保存旧学习率；续训时以当前 YAML 配置为准。
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        elif optim_state and checkpoint_algorithm != "ppo_gae_n_step":
            print("检测到旧版单步训练 checkpoint，将保留模型/环境并重置 Adam 状态。")
        start_step = int(meta.get("step", 0))
        env.step_count = start_step
        obs = env._build_observations()
        print(f"从 checkpoint 恢复，起始 step={start_step}，来源: {checkpoint_path}")
    else:
        print("未恢复 checkpoint，将开始新的训练。")
        obs = env.reset()
        initial_model = args.load_model or get_cfg(
            config, default_config, "training", "initial_model", None
        )
        if initial_model:
            initial_model_path = Path(str(initial_model))
            if not initial_model_path.exists():
                model_relative_path = model_dir / initial_model_path
                if model_relative_path.exists():
                    initial_model_path = model_relative_path
                else:
                    raise FileNotFoundError(f"找不到初始模型: {initial_model}")
            _load_initial_model(policy, initial_model_path, env.device)

    if logger:
        logger.metadata["start_step"] = start_step
        logger.metadata["resumed_from_step"] = start_step
        logger.metadata["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
        logger.step_counter = start_step
        logger._write_header()

    total_steps = start_step
    run_steps = 0
    rollout_steps = int(
        get_cfg(config, default_config, "training", "rollout_steps", 128)
    )
    next_checkpoint_step = (total_steps // save_interval + 1) * save_interval
    last_checkpoint_step = start_step if checkpoint_path else None
    last_print_time = time.perf_counter()
    last_print_step = total_steps
    final_reward_mean = None
    stop_training = False

    while run_steps < rollout_steps and not stop_training:
        rollout: list[dict] = []
        pending_reproduction_events: list[tuple[int, int, int]] = []
        final_survivor = None
        all_dead = False
        rollout_limit = min(
            n_steps,
            rollout_steps - run_steps,
            next_checkpoint_step - total_steps,
        )

        for _ in range(rollout_limit):
            valid = obs["alive"].detach().clone()
            last_survivor = _select_last_survivor(env)
            if last_survivor is None and env.use_death:
                print("所有个体在本轮开始前已经死亡，训练停止。")
                stop_training = True
                break

            hidden = _clone_hidden(_current_hidden(env))
            patch = (
                obs["patch"].detach().clone()
                if getattr(policy, "use_cnn", False)
                else None
            )
            agent_features = (
                obs["agent_features"].detach().clone()
                if getattr(policy, "use_cnn", False)
                else None
            )
            agent_obs = (
                None
                if getattr(policy, "use_cnn", False)
                else obs["agent_obs"].detach().clone()
            )

            with torch.no_grad():
                logits, values, new_hidden = _forward_policy_inputs(
                    policy, patch, agent_features, agent_obs, hidden
                )
                logits, values = _reshape_policy_outputs(
                    logits, values, valid.shape
                )
                if gaussian_noise_std > 0:
                    logit_noise = torch.randn_like(logits) * gaussian_noise_std
                else:
                    logit_noise = torch.zeros_like(logits)
                behavior_logits = logits + logit_noise
                actions = torch.distributions.Categorical(
                    logits=behavior_logits
                ).sample()
                if epsilon_greedy > 0:
                    random_actions = torch.randint(
                        low=0,
                        high=len(action_space),
                        size=actions.shape,
                        device=actions.device,
                    )
                    greedy_mask = (
                        torch.rand(actions.shape, device=actions.device)
                        < epsilon_greedy
                    )
                    actions = torch.where(greedy_mask, random_actions, actions)
                old_log_probs, _ = _behavior_log_prob_and_entropy(
                    behavior_logits, actions, epsilon_greedy
                )

            if isinstance(new_hidden, tuple):
                env.agent_batch.state["memory"] = new_hidden[0].detach().clone()
                env.agent_batch.state["memory_cell"] = new_hidden[1].detach().clone()
            elif new_hidden is not None:
                env.agent_batch.state["memory"] = new_hidden.detach().clone()

            if print_actions:
                action_ids = actions.reshape(-1).cpu().tolist()
                names = [action_space.get(int(action), str(int(action))) for action in action_ids]
                print(f"[step={total_steps}] actions: {action_ids} -> {names}")

            step_result = env.step(actions)
            rewards = step_result.rewards.reshape(valid.shape).detach()
            next_obs = step_result.observations
            continuation = _transition_continuation(
                valid,
                next_obs["alive"].detach(),
                step_result.dones,
                step_result.reproduction_events,
            )

            rollout.append(
                {
                    "patch": patch,
                    "agent_features": agent_features,
                    "agent_obs": agent_obs,
                    "hidden": hidden,
                    "actions": actions.detach(),
                    "old_log_probs": old_log_probs.detach(),
                    "old_values": values.detach(),
                    "rewards": rewards,
                    "valid": valid,
                    "continuation": continuation,
                    "logit_noise": logit_noise.detach(),
                }
            )

            final_reward_mean = float(rewards[valid].mean().item()) if valid.any() else 0.0
            obs = next_obs
            total_steps += 1
            run_steps += 1

            if logger:
                logger.maybe_log(
                    env.map_state, env.agent_batch.export_state(), step=total_steps
                )

            all_dead = env.use_death and not bool(obs["alive"].any().item())
            if all_dead:
                final_survivor = last_survivor
                stop_training = True
                break

            if step_result.reproduction_events:
                pending_reproduction_events.extend(step_result.reproduction_events)
                # 继承会修改策略参数；先结算当前 rollout，保证 PPO behavior policy 固定。
                break

        if not rollout:
            break

        with torch.no_grad():
            if all_dead:
                bootstrap_values = torch.zeros_like(rollout[-1]["old_values"])
            else:
                bootstrap_hidden = _clone_hidden(_current_hidden(env))
                bootstrap_logits, bootstrap_values, _ = _forward_policy_inputs(
                    policy,
                    obs["patch"] if getattr(policy, "use_cnn", False) else None,
                    obs["agent_features"] if getattr(policy, "use_cnn", False) else None,
                    None if getattr(policy, "use_cnn", False) else obs["agent_obs"],
                    bootstrap_hidden,
                )
                _, bootstrap_values = _reshape_policy_outputs(
                    bootstrap_logits, bootstrap_values, obs["alive"].shape
                )

        rollout_rewards = torch.stack([sample["rewards"] for sample in rollout])
        rollout_values = torch.stack([sample["old_values"] for sample in rollout])
        rollout_continuation = torch.stack(
            [sample["continuation"] for sample in rollout]
        )
        rollout_valid = torch.stack([sample["valid"] for sample in rollout])
        advantages, returns = _compute_gae(
            rollout_rewards,
            rollout_values,
            rollout_continuation,
            bootstrap_values.detach(),
            gamma,
            gae_lambda,
        )
        advantages = advantages.detach()
        returns = returns.detach()
        if normalize_advantages:
            advantages = _normalize_valid_advantages(
                advantages,
                rollout_valid,
                bool(getattr(policy, "cnn_independent", False)),
            )

        ppo_metrics = _ppo_update(
            policy,
            optimizer,
            rollout,
            advantages,
            returns,
            clip_range=ppo_clip,
            ppo_epochs=ppo_epochs,
            minibatch_steps=ppo_minibatch_steps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            epsilon_greedy=epsilon_greedy,
        )

        if pending_reproduction_events and hasattr(policy, "inherit_policy_head"):
            for env_idx, parent_idx, child_idx in pending_reproduction_events:
                inherited_parameters = policy.inherit_policy_head(
                    env_idx, parent_idx, child_idx, policy_mutation_std
                )
                for parameter in inherited_parameters:
                    optimizer.state.pop(parameter, None)

        if print_interval > 0 and (
            total_steps - last_print_step >= print_interval or stop_training
        ):
            now = time.perf_counter()
            step_delta = total_steps - last_print_step
            time_delta = now - last_print_time
            if time_delta > 0 and step_delta > 0:
                steps_per_sec = step_delta / time_delta
                step_rate_text = (
                    f"{steps_per_sec:.2f} step/s"
                    if steps_per_sec >= 1
                    else f"{(time_delta / step_delta):.2f} s/step"
                )
                agents_processed = (
                    step_delta
                    * env.agent_batch.num_envs
                    * env.agent_batch.agents_per_env
                )
                model_speed_text = f"{agents_processed / time_delta:.2f} agents/s"
            else:
                step_rate_text = "n/a"
                model_speed_text = "n/a"
            print(
                f"steps:{total_steps} reward_mean:{final_reward_mean:.6f} "
                f"policy_loss:{ppo_metrics['policy_loss']:.6f} "
                f"value_loss:{ppo_metrics['value_loss']:.6f} "
                f"entropy:{ppo_metrics['entropy']:.6f} "
                f"kl:{ppo_metrics['approx_kl']:.6f} "
                f"clip_fraction:{ppo_metrics['clip_fraction']:.4f} "
                f"step_rate:{step_rate_text} model_speed:{model_speed_text}"
            )
            last_print_time = now
            last_print_step = total_steps

        if total_steps >= next_checkpoint_step:
            _save_ppo_checkpoint(
                policy,
                optimizer,
                env,
                checkpoint_dir,
                total_steps,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ppo_clip=ppo_clip,
            )
            last_checkpoint_step = total_steps
            print(f"已保存 checkpoint 至 {checkpoint_dir}, step={total_steps}")
            while next_checkpoint_step <= total_steps:
                next_checkpoint_step += save_interval
        if all_dead:
            survivor_path = _save_last_survivor_model(
                policy, checkpoint_dir, final_survivor, total_steps
            )
            print(
                f"所有个体已死亡，训练停止；最后存活个体模型已保存至: {survivor_path}"
            )
            break

    if run_steps > 0 and last_checkpoint_step != total_steps:
        _save_ppo_checkpoint(
            policy,
            optimizer,
            env,
            checkpoint_dir,
            total_steps,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_clip=ppo_clip,
        )
        print(f"已保存最终 checkpoint 至 {checkpoint_dir}, step={total_steps}")

    if logger:
        logger.flush()

    if final_reward_mean is not None:
        print("训练完成，最终奖励均值:", final_reward_mean)


if __name__ == "__main__":
    main()
