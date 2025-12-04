"""读取日志并调用 matplotlib 回放。"""
from __future__ import annotations

import argparse

from protolife.replay import playback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife 实验回放")
    parser.add_argument("--log-dir", required=True, help="包含 map.log/agents.jsonl 的目录")
    parser.add_argument("--height", type=int, required=True, help="地图高度")
    parser.add_argument("--width", type=int, required=True, help="地图宽度")
    parser.add_argument("--interval", type=float, default=0.2, help="帧间隔（秒）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playback(args.log_dir, height=args.height, width=args.width, interval=args.interval)


if __name__ == "__main__":
    main()
