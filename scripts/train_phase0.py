"""Phase0 训练脚本占位。

示例用法：
    python -m scripts.train_phase0 --config config/phase0_survival.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

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


def get_cfg(config: dict, default_config: dict, section: str, key: str, fallback):
    return config.get(section, {}).get(
        key,
        default_config.get(section, {}).get(key, TRAIN_DEFAULTS.get(section, {}).get(key, fallback)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife Phase0 训练")
    parser.add_argument("--config", type=str, default="config/phase0_survival.yaml", help="配置文件路径")
    parser.add_argument("--save-interval", type=int, default=None, help="多少步保存一次模型/存档")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint 输出目录")
    parser.add_argument("--resume-from", type=str, default=None, help="从完整 checkpoint 继续推演")
    parser.add_argument("--load-model", type=str, default=None, help="仅加载模型权重")
    parser.add_argument("--print-interval", type=int, default=None, help="训练日志打印间隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, default_config = load_config(args.config)
    set_seed(get_cfg(config, default_config, "world", "random_seed", 0))

    env = ProtoLifeEnv(config, default_config)
    policy = build_policy(config, default_config, obs_dim=env.observation_dim).to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    logger = None
    if get_cfg(config, default_config, "logging", "save_dir", None):
        from protolife.logger import ExperimentLogger

        logger = ExperimentLogger(
            save_dir=get_cfg(config, default_config, "logging", "save_dir", "runs/default"),
            snapshot_interval=get_cfg(config, default_config, "logging", "snapshot_interval", 50),
            env_index=get_cfg(config, default_config, "logging", "env_index", 0),
        )

    checkpoint_dir = Path(
        args.checkpoint_dir
        or get_cfg(config, default_config, "training", "checkpoint_dir", "checkpoints")
    )
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
    if args.resume_from:
        env_state, policy_state, optim_state, meta = load_checkpoint(Path(args.resume_from), map_location=env.device)
        env.load_state(env_state)
        policy.load_state_dict(policy_state)
        if optim_state:
            optimizer.load_state_dict(optim_state)
        start_step = int(meta.get("step", 0))
        obs = env._build_observations()
        print(f"从 checkpoint 恢复，起始 step={start_step}")
    else:
        obs = env.reset()
        if args.load_model:
            state = torch.load(args.load_model, map_location=env.device)
            policy.load_state_dict(state)
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
