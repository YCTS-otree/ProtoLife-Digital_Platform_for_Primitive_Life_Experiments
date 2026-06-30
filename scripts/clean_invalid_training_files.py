"""递归删除训练遗留的 ``无效_`` 标记文件。"""

from __future__ import annotations

import argparse
from pathlib import Path


INVALID_MARKER = "无效_"


def find_invalid_training_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name.startswith(INVALID_MARKER)
    )


def clean_invalid_training_files(root: Path, *, dry_run: bool = False) -> list[Path]:
    targets = find_invalid_training_files(root)
    if not dry_run:
        for target in targets:
            target.unlink()
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清除 model 目录下文件名以‘无效_’开头的训练产物"
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("model"),
        help="扫描根目录，默认 model；也可传外部 checkpoint 根目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出，不删除",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = clean_invalid_training_files(args.root, dry_run=args.dry_run)
    action = "将删除" if args.dry_run else "已删除"
    for target in targets:
        print(f"{action}: {target}")
    print(f"{action} {len(targets)} 个无效训练文件。")


if __name__ == "__main__":
    main()
