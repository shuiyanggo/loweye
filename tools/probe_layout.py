"""量窄屏版式：起一个 headless Edge，用 CDP 读关键元素的实际盒子。

headless 截图只能看出「像不像对」，量不出「谁把谁挤出了屏幕」。这个探针
直接向浏览器要 getBoundingClientRect 与 computed style，用来判断窄屏下
是元素太宽、还是文档整体横向溢出。

用法：
    python -m tools.probe_layout --url http://127.0.0.1:8848/ --width 390 --height 844
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

SELECTORS = [
    ".topbar",
    ".topbar-right",
    ".status-pill",
    "#health-text",
    "#btn-selfcheck",
    ".tabs",
    ".tab-ink",
    "#view-console",
    ".rail",
    ".stage-col",
    ".pipeline",
    ".pnode",
    ".stage-meta",
    ".meta-cell",
    ".frame",
    ".findings",
    ".foot",
]

PROBE_JS = """
(() => {
  const box = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return {
      x: Math.round(r.x), y: Math.round(r.y),
      w: Math.round(r.width), h: Math.round(r.height),
      right: Math.round(r.right), bottom: Math.round(r.bottom),
      display: cs.display, position: cs.position,
      overflowX: cs.overflowX, whiteSpace: cs.whiteSpace,
    };
  };
  const out = {
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    docScrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
    elements: {},
  };
  for (const sel of %s) out.elements[sel] = box(sel);
  return JSON.stringify(out);
})()
"""


def find_edge() -> str:
    for cand in EDGE_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise SystemExit("没找到 msedge.exe")


async def cdp_eval(
    ws_url: str,
    expression: str,
    *,
    wait: float,
    width: int,
    height: int,
    shot: Path | None = None,
) -> str:
    """连上 CDP 后先强制设备尺寸，再取测量值（可选：同尺寸截图）。

    必须走 Emulation.setDeviceMetricsOverride：Windows 上浏览器窗口有最小宽度
    （实测约 500px），直接开 --window-size=390 时布局仍按 ~504px 计算，
    `--screenshot` 只是把左上角 390x844 裁出来 —— 该被切掉的元素看起来"正常"，
    窄屏溢出会被完全掩盖。要可信的窄屏结论就得走 CDP。
    """
    import base64

    async with websockets.connect(ws_url, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "id": 0,
            "method": "Emulation.setDeviceMetricsOverride",
            "params": {
                "width": width,
                "height": height,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        }))
        await ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True, "awaitPromise": False},
        }))
        if shot is not None:
            await ws.send(json.dumps({
                "id": 2,
                "method": "Page.captureScreenshot",
                "params": {"format": "png", "captureBeyondViewport": False},
            }))

        value, deadline = "", time.time() + wait
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=wait)
            msg = json.loads(raw)
            if msg.get("id") == 1:
                value = msg["result"]["result"].get("value", "")
                if shot is None:
                    return value
            elif msg.get("id") == 2:
                shot.parent.mkdir(parents=True, exist_ok=True)
                shot.write_bytes(base64.b64decode(msg["result"]["data"]))
                return value
        raise SystemExit("等 CDP 结果超时")


def page_ws(port: int, *, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception as exc:  # 浏览器还没起来
            last = str(exc)
        time.sleep(0.4)
    raise SystemExit(f"没拿到 CDP 页面目标：{last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8848/")
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--height", type=int, default=844)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--user-data-dir", default=None)
    ap.add_argument("--settle", type=float, default=3.5, help="等页面自己加载完的秒数")
    ap.add_argument("--shot", default=None, help="同时在强制尺寸下截图到该路径")
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[1]
    profile = Path(args.user_data_dir) if args.user_data_dir else project / "out" / "edge-probe"
    profile.mkdir(parents=True, exist_ok=True)
    errlog = (project / "out" / "probe_layout.stderr.log").open("w", encoding="utf-8")

    edge = find_edge()
    proc = subprocess.Popen(
        [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--force-prefers-reduced-motion",
            f"--remote-debugging-port={args.port}",
            f"--user-data-dir={profile}",
            f"--window-size={args.width},{args.height}",
            args.url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=errlog,
    )
    try:
        ws_url = page_ws(args.port)
        time.sleep(args.settle)
        raw = asyncio.run(cdp_eval(
            ws_url, PROBE_JS % json.dumps(SELECTORS),
            wait=20, width=args.width, height=args.height,
            shot=Path(args.shot) if args.shot else None,
        ))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        errlog.close()

    data = json.loads(raw)
    print(f"视口 {data['innerWidth']}x{data['innerHeight']}  "
          f"文档 scrollWidth={data['docScrollWidth']}  body scrollWidth={data['bodyScrollWidth']}")
    if data["docScrollWidth"] > data["innerWidth"]:
        print(f"  !! 文档横向溢出 {data['docScrollWidth'] - data['innerWidth']}px（有元素把页面撑宽了）")
    print(f"{'选择器':<18}{'x':>6}{'宽':>7}{'右边界':>8}  display / overflow")
    for sel in SELECTORS:
        b = data["elements"].get(sel)
        if b is None:
            print(f"{sel:<18}{'—':>6}{'—':>7}{'—':>8}  未渲染")
            continue
        flag = "  << 超出视口" if b["right"] > data["innerWidth"] + 0.5 else ""
        print(f"{sel:<18}{b['x']:>6}{b['w']:>7}{b['right']:>8}  {b['display']}/{b['overflowX']}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
