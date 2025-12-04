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
    parser.add_argument("--config", type=str, default="config/phase0_survival.yaml", help="配置文件路径")
    parser.add_argument("--save-interval", type=int, default=None, help="多少步保存一次模型/存档")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint 输出目录")
    parser.add_argument("--resume-from", type=str, default=None, help="从完整 checkpoint 继续推演")
    parser.add_argument("--load-model", type=str, default=None, help="仅加载模型权重")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("world", {}).get("random_seed", 0))

    env = ProtoLifeEnv(config)
    policy = build_policy(config, obs_dim=env.observation_dim).to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    logger = None
    if config.get("logging", {}).get("save_dir"):
        from protolife.logger import ExperimentLogger

        logger = ExperimentLogger(
            save_dir=config["logging"].get("save_dir", "runs/default"),
            snapshot_interval=config["logging"].get("snapshot_interval", 50),
        )

    checkpoint_dir = Path(
        args.checkpoint_dir
        or config.get("training", {}).get("checkpoint_dir", "checkpoints")
    )
    save_interval = args.save_interval or config.get("training", {}).get("save_interval", 100)

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

    flat_obs = obs["agent_obs"]
    logits, values = policy(flat_obs)
    actions = torch.distributions.Categorical(logits=logits).sample()

    total_steps = start_step
    rollout_steps = config.get("training", {}).get("rollout_steps", 128)
    for step in range(rollout_steps):
        step_result = env.step(actions)
        loss = -step_result.rewards.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = step_result.observations
        flat_obs = obs["agent_obs"]
        logits, values = policy(flat_obs)
        actions = torch.distributions.Categorical(logits=logits).sample()
        total_steps += 1

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
