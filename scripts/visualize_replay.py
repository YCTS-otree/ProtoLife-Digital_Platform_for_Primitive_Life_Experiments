"""读取日志并调用 matplotlib 回放。"""
from __future__ import annotations

import argparse

from protolife.replay import playback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProtoLife 实验回放")
    parser.add_argument("--log-target", required=True, help="日志所在目录或模型目录")
    parser.add_argument("--interval", type=float, default=0.2, help="帧间隔（秒）")
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="从指定 step 开始播放（默认从头播放）",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="直接使用创建时间最新的一组日志；未传入时保留交互选择",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playback(
        args.log_target,
        interval=args.interval,
        start_step=args.start_step,
        use_latest=args.latest,
    )


if __name__ == "__main__":
    main()
