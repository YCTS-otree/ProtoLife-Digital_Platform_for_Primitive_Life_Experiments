"""Phase0 训练脚本占位。

示例用法：
    python scripts/train_phase0.py --config config/phase0_survival.yaml
"""
from __future__ import annotations

import argparse

import torch

from protolife.config_loader import load_config
from protolife.env import ProtoLifeEnv
from protolife.policy import build_policy
from protolife.utils.seed_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife Phase0 训练")
    parser.add_argument("--config", type=str, default="config/phase0_survival.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("world", {}).get("random_seed", 0))

    env = ProtoLifeEnv(config)
    policy = build_policy(config).to(env.device)

    obs = env.reset()
    flat_obs = obs["agents"].view(-1, obs["agents"].shape[-1])
    logits, values = policy(flat_obs)
    actions = torch.argmax(logits, dim=-1)
    step_result = env.step(actions)

    print("初步观测形状:", {k: v.shape for k, v in obs.items()})
    print("动作 logits 形状:", logits.shape)
    print("奖励示例均值:", step_result.rewards.mean().item())


if __name__ == "__main__":
    main()
