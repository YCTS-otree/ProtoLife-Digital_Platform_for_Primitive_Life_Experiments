"""Phase0 训练脚本占位。

示例用法：
    python scripts/train_phase0.py --config config/phase0_survival.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protolife.config_loader import load_config
from protolife.checkpoint import load_checkpoint, save_checkpoint
from protolife.env import ProtoLifeEnv
from protolife.policy import build_policy
from protolife.utils.seed_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife Phase0 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/phase0_survival.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="多少步保存一次模型/存档（覆盖 config.training.save_interval）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="checkpoint 输出目录（覆盖 config.training.checkpoint_dir）",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从完整 checkpoint 继续推演",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="仅加载模型权重",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 读取用户配置 + 默认配置
    config = load_config(args.config)
    default_config = load_config("config/default.yaml")

    # ---- world / 随机种子：config > default_config > 0 ----
    world_cfg = config.get("world", {})
    default_world_cfg = default_config.get("world", {})
    random_seed = world_cfg.get(
        "random_seed",
        default_world_cfg.get("random_seed", 0),
    )
    set_seed(random_seed)

    # 2. 构建环境和策略
    env = ProtoLifeEnv(config)
    policy = build_policy(config).to(env.device)

    # ---- training 配置：config.training 优先，其次 default.training，最后硬编码 ----
    training_cfg = config.get("training", {})
    default_training_cfg = default_config.get("training", {})

    lr = training_cfg.get("lr", default_training_cfg.get("lr", 1e-3))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    checkpoint_dir = Path(
        args.checkpoint_dir
        or training_cfg.get(
            "checkpoint_dir",
            default_training_cfg.get("checkpoint_dir", "checkpoints"),
        )
    )

    save_interval = args.save_interval or training_cfg.get(
        "save_interval",
        default_training_cfg.get("save_interval", 100),
    )

    rollout_steps = training_cfg.get(
        "rollout_steps",
        default_training_cfg.get("rollout_steps", 100),
    )

    gamma = training_cfg.get(
        "gamma",
        default_training_cfg.get("gamma", 0.99),
    )

    value_loss_coef = training_cfg.get(
        "value_loss_coef",
        default_training_cfg.get("value_loss_coef", 0.5),
    )

    debug_interval = training_cfg.get(
        "debug_interval",
        default_training_cfg.get("debug_interval", 512),
    )

    # 3. checkpoint / 初始化观测
    start_step = 0
    if args.resume_from:
        env_state, policy_state, optim_state, meta = load_checkpoint(
            Path(args.resume_from),
            map_location=env.device,
        )
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

    # 4. Actor-Critic 训练循环（一步 TD）
    total_steps = start_step

    for step in range(rollout_steps):
        # 4.1 当前观测 -> [batch, obs_dim]
        obs_agents = obs["agents"].to(env.device)  # [num_envs, num_agents, obs_dim]
        flat_obs = obs_agents.view(-1, obs_agents.shape[-1])

        # 4.2 策略前向：logits + state value
        logits, values = policy(flat_obs)  # values: [batch] 或 [batch, 1]

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()                 # [batch]
        log_probs = dist.log_prob(actions)      # [batch]

        # 4.3 环境一步
        step_result = env.step(actions)
        rewards = step_result.rewards.to(env.device).view(-1)  # [batch]

        if debug_interval and (total_steps % debug_interval == 0):
            print(
                f"[debug] step={total_steps}, "
                f"reward_mean={rewards.mean().item():.6f}"
            )

        # 4.4 计算 next_value（用于 advantage）
        next_obs_agents = step_result.observations["agents"].to(env.device)
        next_flat_obs = next_obs_agents.view(-1, next_obs_agents.shape[-1])
        with torch.no_grad():
            _, next_values = policy(next_flat_obs)  # [batch] 或 [batch, 1]

        # 保证 values / next_values 为 [batch]
        values = values.view(-1)
        next_values = next_values.view(-1)

        # 4.5 一步 advantage: A = r + gamma * V_next - V_now
        advantage = rewards + gamma * next_values - values

        # 4.6 策略损失 & 值函数损失
        #     注意：advantage 只是用来加权 log_prob，不需要反向
        policy_loss = -(advantage.detach() * log_probs).mean()
        value_loss = 0.5 * advantage.pow(2).mean()
        loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4.7 准备下一步
        obs = step_result.observations
        total_steps += 1

        # 4.8 checkpoint 保存
        if total_steps % save_interval == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                policy.state_dict(),
                checkpoint_dir / f"model_step_{total_steps}.pth",
            )
            torch.save(
                optimizer.state_dict(),
                checkpoint_dir / f"optim_step_{total_steps}.pth",
            )
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
