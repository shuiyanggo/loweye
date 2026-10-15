"""把用到的 Lucide 图标抓下来，拼成一个本地 SVG sprite。

为什么要单独做这一步：
全局前端规范要求「不要用 emoji，统一使用开源 UI 图标」。图标如果用 CDN 引用，
答辩现场一旦断网或校园网抖动，界面就会掉一片方块。所以图标在开发期一次性抓取、
落到 web/vendor/icons.svg，运行时只读本地文件，断网照常显示。

用法：
    python tools/fetch_icons.py            # 抓取并写 sprite
    python tools/fetch_icons.py --list     # 只打印图标清单

只依赖标准库 + httpx（已在 requirements.txt 里）。
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "vendor" / "icons.svg"

# 版本锁定：图标库升级会改名，锁住版本才能保证重跑结果一致
LUCIDE_VERSION = "0.469.0"
BASE = f"https://cdn.jsdelivr.net/npm/lucide-static@{LUCIDE_VERSION}/icons"

# 需要的图标 + 同义名兜底（lucide 改过名，例如 alert-triangle → triangle-alert）
ICONS: dict[str, list[str]] = {
    "activity": [],
    "triangle-alert": ["alert-triangle"],
    "badge-check": ["check-circle-2", "circle-check"],
    "bar-chart-3": ["chart-column"],
    "book-open": [],
    "check": [],
    "chevron-right": [],
    "clipboard-list": [],
    "clock": [],
    "cpu": [],
    "database": [],
    "download": [],
    "file-text": [],
    "gauge": [],
    "history": [],
    "image": ["image-icon"],
    "layers": [],
    "layout-dashboard": [],
    "loader-circle": ["loader-2"],
    "play": [],
    "refresh-cw": [],
    "route": [],
    "search": [],
    "send": [],
    "server": [],
    "shield-check": [],
    "sparkles": [],
    "target": [],
    "upload": [],
    "user-check": [],
    "workflow": [],
    "x": [],
    "zap": [],
    "radio": [],
    "map-pin": [],
    "scan-line": [],
    "list-checks": [],
    "box": [],
    "circle-slash": ["ban"],
    "cloud-off": ["cloud-off-icon"],
    "arrow-right": [],
    "shield-off": ["shield-x"],
    "audio-lines": ["audio-lines-icon", "align-left"],
}

_VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"', re.I)
_INNER_RE = re.compile(r"<svg[^>]*>(.*)</svg>", re.I | re.S)


def fetch(name: str) -> str | None:
    url = f"{BASE}/{name}.svg"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - 单个图标失败不应中断整批
        print(f"  ! {name}: {type(exc).__name__}: {exc}")
        return None


def to_symbol(name: str, svg: str) -> str | None:
    vb = _VIEWBOX_RE.search(svg)
    inner = _INNER_RE.search(svg)
    if not vb or not inner:
        return None
    body = inner.group(1).strip()
    # <use> 引用的 symbol 不能带 id 冲突的写法，顺便统一成 currentColor
    body = body.replace('stroke="currentColor"', 'stroke="currentColor"')
    return (
        f'  <symbol id="i-{name}" viewBox="{vb.group(1)}" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">\n'
        f"    {body}\n"
        f"  </symbol>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for k, v in ICONS.items():
            print(f"{k}{'  (兜底: ' + ', '.join(v) + ')' if v else ''}")
        return 0

    symbols: list[str] = []
    missing: list[str] = []
    for name, aliases in ICONS.items():
        svg = fetch(name)
        used = name
        if svg is None:
            for alt in aliases:
                svg = fetch(alt)
                if svg is not None:
                    used = alt
                    break
        if svg is None:
            missing.append(name)
            continue
        symbol = to_symbol(name, svg)
        if symbol is None:
            missing.append(name)
            continue
        symbols.append(symbol)
        print(f"  √ {name}" + (f"（用 {used} 兜底）" if used != name else ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<!-- Lucide v{LUCIDE_VERSION} 图标子集，由 tools/fetch_icons.py 生成，勿手工编辑 -->\n"
        '<svg xmlns="http://www.w3.org/2000/svg" style="display:none">\n'
        + "\n".join(symbols)
        + "\n</svg>\n",
        encoding="utf-8",
    )
    print(f"\n写入 {OUT}（{len(symbols)} 个图标，{OUT.stat().st_size} 字节）")
    if missing:
        print("缺失：" + "、".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
