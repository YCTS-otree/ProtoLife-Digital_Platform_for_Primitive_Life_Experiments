"""将旧 full checkpoint 转为无重复数据的三文件格式。"""

from __future__ import annotations

import argparse
from pathlib import Path

from protolife.checkpoint import migrate_legacy_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="迁移 full_step_*.pt；校验成功后可删除 full 文件以释放空间"
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="一个或多个 full_step_*.pt 文件或包含这些文件的目录",
    )
    parser.add_argument(
        "--delete-full",
        action="store_true",
        help="迁移并校验成功后删除旧 full_step_*.pt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="用 full 文件内容覆盖已存在的 env/model/optim 三件套",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索目标目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅列出将处理的 full 文件",
    )
    return parser.parse_args()


def _collect_legacy_checkpoints(targets: list[Path], recursive: bool) -> list[Path]:
    found: set[Path] = set()
    for target in targets:
        if target.is_dir():
            iterator = (
                target.rglob("*full_step_*.pt")
                if recursive
                else target.glob("*full_step_*.pt")
            )
            found.update(item.resolve() for item in iterator if item.is_file())
        elif (
            target.is_file()
            and "full_step_" in target.name
            and target.suffix == ".pt"
        ):
            found.add(target.resolve())
        elif not target.exists():
            raise FileNotFoundError(target)
        else:
            raise ValueError(f"目标不是 full_step_*.pt 或目录: {target}")
    return sorted(found)


def main() -> None:
    args = parse_args()
    checkpoints = _collect_legacy_checkpoints(args.targets, args.recursive)
    if not checkpoints:
        print("没有找到 full_step_*.pt")
        return

    for checkpoint in checkpoints:
        if args.dry_run:
            print(f"将迁移: {checkpoint}")
            continue
        paths = migrate_legacy_checkpoint(
            checkpoint,
            delete_full=args.delete_full,
            overwrite=args.overwrite,
        )
        action = "已迁移并删除 full" if args.delete_full else "已迁移并保留 full"
        print(f"{action}: {checkpoint}")
        print(
            "  "
            + " | ".join(f"{name}={path.name}" for name, path in paths.items())
        )


if __name__ == "__main__":
    main()
