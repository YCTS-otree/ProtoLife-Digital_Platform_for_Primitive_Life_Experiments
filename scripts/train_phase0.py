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
        "save_interval": 100,
        "entropy_coef": 0.01,
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
    try:
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=1e-3, fused=env.device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
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
    save_interval = args.save_interval or get_cfg(config, default_config, "training", "save_interval", 100)
    print_actions = get_cfg(config, default_config, "training", "print_actions", False)
    entropy_coef = get_cfg(config, default_config, "training", "entropy_coef", 0.01)
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
        if optim_state and not skipped:
            optimizer.load_state_dict(optim_state)
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
    rollout_steps = get_cfg(config, default_config, "training", "rollout_steps", 128)
    last_print_time = time.perf_counter()
    last_print_step = total_steps
    final_reward_mean = None
    for step in range(rollout_steps):
        alive_for_loss = obs["alive"].reshape(-1)
        last_survivor = _select_last_survivor(env)
        if last_survivor is None and env.use_death:
            print("所有个体在本轮开始前已经死亡，训练停止。")
            break

        if getattr(policy, "use_cnn", False):
            patches = obs["patch"]
            memory_state = env.agent_batch.state.get("memory")
            if env.agent_batch.use_lstm:
                memory_cell = env.agent_batch.state.get("memory_cell")
                hidden = (memory_state, memory_cell) if memory_state is not None else None
            else:
                hidden = memory_state
            logits, values, new_hidden = policy(
                patches, hidden, obs["agent_features"]
            )
            if isinstance(new_hidden, tuple):
                # clone 解除与当前计算图的共享存储，环境可安全重置新生槽位记忆。
                env.agent_batch.state["memory"] = new_hidden[0].detach().clone()
                env.agent_batch.state["memory_cell"] = new_hidden[1].detach().clone()
            else:
                # detach 不复制存储；繁殖时的 zero_ 会破坏反向图，因此必须 clone。
                env.agent_batch.state["memory"] = new_hidden.detach().clone()
        else:
            flat_obs = obs["agent_obs"]
            logits, values = policy(flat_obs)

        if gaussian_noise_std > 0:
            logits = logits + torch.randn_like(logits) * gaussian_noise_std

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        if epsilon_greedy > 0:
            random_actions = torch.randint(
                low=0, high=len(action_space), size=actions.shape, device=actions.device
            )
            greedy_mask = torch.rand(actions.shape, device=actions.device, dtype=torch.float32) < epsilon_greedy
            actions = torch.where(greedy_mask, random_actions, actions)

        if print_actions:
            action_ids = actions.reshape(-1).cpu().tolist()
            names = [action_space.get(int(a), str(int(a))) for a in action_ids]
            print(f"[step={total_steps}] actions: {action_ids} -> {names}")

        step_result = env.step(actions)
        rewards = step_result.rewards
        final_reward_mean = rewards.mean().item()

        log_probs = dist.log_prob(actions).view(-1)[alive_for_loss]
        predicted_values = values.view(-1)[alive_for_loss]
        active_rewards = rewards.detach()[alive_for_loss]
        active_entropy = dist.entropy().view(-1)[alive_for_loss]
        if getattr(policy, "cnn_independent", False):
            # 每套大脑只产生一个样本，使用求和避免梯度被个体数量额外缩小。
            policy_loss = -(active_rewards * log_probs).sum()
            value_loss = 0.5 * ((predicted_values - active_rewards) ** 2).sum()
            entropy = active_entropy.sum()
        else:
            policy_loss = -(active_rewards * log_probs).mean()
            value_loss = 0.5 * ((predicted_values - active_rewards) ** 2).mean()
            entropy = active_entropy.mean()
        entropy_loss = -entropy_coef * entropy
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step_result.reproduction_events and hasattr(policy, "inherit_policy_head"):
            for env_idx, parent_idx, child_idx in step_result.reproduction_events:
                inherited_parameters = policy.inherit_policy_head(
                    env_idx, parent_idx, child_idx, policy_mutation_std
                )
                for parameter in inherited_parameters:
                    optimizer.state.pop(parameter, None)

        obs = step_result.observations
        total_steps += 1

        if print_interval > 0 and total_steps % print_interval == 0:
            now = time.perf_counter()
            step_delta = total_steps - last_print_step
            time_delta = now - last_print_time
            if time_delta > 0 and step_delta > 0:
                steps_per_sec = step_delta / time_delta
                if steps_per_sec >= 1:
                    step_rate_text = f"{steps_per_sec:.2f} step/s"
                else:
                    step_rate_text = f"{(time_delta / step_delta):.2f} s/step"
                agents_processed = step_delta * env.agent_batch.num_envs * env.agent_batch.agents_per_env
                model_speed = agents_processed / time_delta
                model_speed_text = f"{model_speed:.2f} agents/s"
            else:
                step_rate_text = "n/a"
                model_speed_text = "n/a"

            print(
                f"steps:{total_steps}  rewards:{rewards}  step_rate:{step_rate_text}  model_speed:{model_speed_text}"
            )
            last_print_time = now
            last_print_step = total_steps

        if logger:
            logger.maybe_log(
                env.map_state, env.agent_batch.export_state(), step=total_steps
            )

        all_dead = env.use_death and bool(step_result.dones.all().item())
        if all_dead:
            survivor_path = _save_last_survivor_model(
                policy, checkpoint_dir, last_survivor, total_steps
            )
            print(f"所有个体已死亡，训练停止；最后存活个体模型已保存至: {survivor_path}")
            break

        if total_steps % save_interval == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), checkpoint_dir / f"model_step_{total_steps}.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir / f"optim_step_{total_steps}.pth")
            save_checkpoint(
                checkpoint_dir / f"full_step_{total_steps}.pt",
                env_state=env.export_state(),
                policy_state_dict=policy.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                meta={"step": total_steps},
            )
            print(f"已保存 checkpoint 至 {checkpoint_dir}, step={total_steps}")

    if logger:
        logger.flush()

    if final_reward_mean is not None:
        print("训练完成，最终奖励均值:", final_reward_mean)


if __name__ == "__main__":
    main()
