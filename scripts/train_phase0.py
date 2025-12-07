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

from protolife.config_loader import load_config
from protolife.checkpoint import load_checkpoint, save_checkpoint
from protolife.env import ProtoLifeEnv
from protolife.policy import action_space, build_policy
from protolife.utils.seed_utils import set_seed


TRAIN_DEFAULTS = {
    "world": {"random_seed": 0},
    "training": {
        "checkpoint_dir": "checkpoints",
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
    if dropped_keys:
        print(
            "[警告] 以下权重尺寸不匹配，已跳过加载（可能是 observation_radius 调整导致）：",
            dropped_keys,
        )
    return bool(dropped_keys)


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
    parser.add_argument("--load-model", type=str, default=None, help="仅加载模型权重")
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
    merged_config = _merge_dict(default_config, config)
    model_config_path.write_text(yaml.safe_dump(merged_config, allow_unicode=True), encoding="utf-8")
    print(f"模型目录: {model_dir.resolve()}")
    print(f"配置文件: {config_path.resolve()}")
    print(f"Checkpoint 目录: {checkpoint_dir.resolve()}")
    set_seed(get_cfg(config, default_config, "world", "random_seed", 0))

    env = ProtoLifeEnv(config, default_config)
    policy = build_policy(config, default_config, obs_dim=env.observation_dim).to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

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
                "num_envs": env.agent_batch.num_envs,
                "agents_per_env": env.agent_batch.agents_per_env,
                "run_name": model_dir.name,
                "run_tag": run_tag,
                "map_file": env.map_file,
                "toxin_lifetime": getattr(env, "toxin_lifetime", None),
            },
        )

    checkpoint_dir = Path(checkpoint_dir)
    save_interval = args.save_interval or get_cfg(config, default_config, "training", "save_interval", 100)
    print_actions = get_cfg(config, default_config, "training", "print_actions", False)
    entropy_coef = get_cfg(config, default_config, "training", "entropy_coef", 0.01)
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
    latest_checkpoint = _find_latest_full_checkpoint(checkpoint_dir)
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
    else:
        checkpoint_path = latest_checkpoint

    if checkpoint_path:
        print(f"使用的 checkpoint: {checkpoint_path.resolve()}")
        env_state, policy_state, optim_state, meta = load_checkpoint(checkpoint_path, map_location=env.device)
        env.load_state(env_state)
        skipped = _load_state_dict_lenient(policy, policy_state)
        if optim_state and not skipped:
            optimizer.load_state_dict(optim_state)
        start_step = int(meta.get("step", 0))
        obs = env._build_observations()
        print(f"从 checkpoint 恢复，起始 step={start_step}，来源: {checkpoint_path}")
    else:
        print("未找到现有 checkpoint，将从头开始训练。")
        obs = env.reset()
        if args.load_model:
            state = torch.load(args.load_model, map_location=env.device)
            _load_state_dict_lenient(policy, state)
            print(f"仅加载模型参数：{args.load_model}")

    total_steps = start_step
    rollout_steps = get_cfg(config, default_config, "training", "rollout_steps", 128)
    last_print_time = time.perf_counter()
    last_print_step = total_steps
    for step in range(rollout_steps):
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
            action_ids = actions.cpu().tolist()
            names = [action_space.get(int(a), str(int(a))) for a in action_ids]
            print(f"[step={total_steps}] actions: {action_ids} -> {names}")

        step_result = env.step(actions)
        rewards = step_result.rewards

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        policy_loss = -(rewards.detach() * log_probs).mean()
        value_loss = 0.5 * ((values.squeeze(-1) - rewards.detach()) ** 2).mean()
        entropy_loss = -entropy_coef * entropy
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            logger.maybe_log(env.map_state, env.agent_batch.export_state())

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

    print("训练完成，最终奖励均值:", step_result.rewards.mean().item())


if __name__ == "__main__":
    main()
