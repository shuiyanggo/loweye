"""把演示数据复位到「零工单」状态，便于录屏与答辩演示从头开始。

它只删自己产出的运行时数据（工单流水、追溯快照、上传件、报告），
不碰知识库索引（data/index、data/processed）与演示图（data/assets）：
索引重建要几秒到几十秒，演示前没必要重跑。

用法：
    .venv\\Scripts\\python.exe tools\\reset_demo.py --dry-run    # 先看要删什么
    .venv\\Scripts\\python.exe tools\\reset_demo.py --yes        # 真删
    .venv\\Scripts\\python.exe tools\\reset_demo.py --yes --with-index   # 连知识库索引一起重建
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loweye.config import load_config  # noqa: E402


def _targets(cfg) -> list[Path]:
    out = cfg.path("out")
    uploads = cfg.path("uploads")
    targets = [
        out / "work_orders.jsonl",
        out / "traces",
        uploads,
        out / "reports",
    ]
    # 每次分析/测试都会落一份报告（HTML + PDF），最容易堆积，一并纳入
    targets.extend(sorted(out.glob("report_*.pdf")))
    targets.extend(sorted(out.glob("report_*.html")))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="确认执行（不加则只做预演）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要删除的内容")
    ap.add_argument("--with-index", action="store_true", help="连知识库索引目录一起清掉")
    args = ap.parse_args()

    cfg = load_config()
    targets = _targets(cfg)
    if args.with_index:
        targets.extend([cfg.path("index_dir"), cfg.path("data_processed")])

    do_it = args.yes and not args.dry_run
    print(f"项目根：{ROOT}")
    print("模式：" + ("执行删除" if do_it else "预演（不删任何东西）"))
    print()

    total = 0
    for path in targets:
        if not path.exists():
            print(f"  - 跳过（不存在）：{path}")
            continue
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            print(f"  - 目录 {path}：{len(files)} 个文件 / {size / 1024:.0f} KB")
            total += len(files)
            if do_it:
                shutil.rmtree(path)
        else:
            print(f"  - 文件 {path}：{path.stat().st_size / 1024:.1f} KB")
            total += 1
            if do_it:
                path.unlink()

    print()
    if do_it:
        cfg.path("uploads").mkdir(parents=True, exist_ok=True)
        print(f"已清理 {total} 个文件。下次启动服务即为零工单状态。")
        if args.with_index:
            print("索引已删除，请重建：python -m loweye.cli.build_kb")
    else:
        print(f"预演：共 {total} 个文件将被删除。加 --yes 才会真的删。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
