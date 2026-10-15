"""把 Web 控制台的真实交互录成演示视频。

这不是人手录屏，而是用 CDP 驱动真实浏览器把演示脚本走一遍、逐步截图，再用 ffmpeg
合成 MP4。好处是可重跑、可回归：界面改动后重跑一次就知道演示还走不走得通，
而且每一帧都是真实渲染结果，不是拼出来的效果图。

用法：
    python -m tools.record_demo --url http://127.0.0.1:8848/ --out out/demo_screencast.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import websockets

PROJECT = Path(__file__).resolve().parents[1]
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]

# (步骤说明, URL 后缀或 None=不跳转直接用当前页面, 点击选择器, 等待秒数)
STEPS: list[tuple[str, str | None, str | None, float]] = [
    ("① 打开控制台 3 份文档 68 个条款块", "", None, 2.0),
    ("② 选一张巡检图 走完整链路", "?image=demo_tower_01", None, 6.0),
    ("③ P1 绝缘子自爆 转人工复核 P2 金具锈蚀 自动派单", None, None, 2.5),
    ("④ 每条处置动作都带条款号 点开看原文", None, ".finding-card .cite-chip", 3.0),
    ("⑤ 知识库检索 越界问题直接拒答", "?view=kb&q=" + urllib.parse.quote("绝缘子自爆怎么处理"), None, 4.0),
    ("⑥ 越界提问 系统拒绝作答而不是编造", "?view=kb&q=" + urllib.parse.quote("今天股市行情怎么样"), None, 3.5),
    ("⑦ 工单队列 可确认派单或退回", "?view=orders", None, 3.0),
    ("⑧ 追溯 每次分析的完整链路留痕", "?view=traces", None, 3.0),
    ("⑨ 一键全链路自检", "", "#btn-selfcheck", 4.5),
]


def find_edge() -> str:
    for cand in EDGE_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise SystemExit("没找到 msedge.exe")


def page_ws(port: int, *, timeout: float = 25.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise SystemExit("没拿到 CDP 页面目标")


class Session:
    """极简 CDP 会话：够用就好，不引第三方驱动。"""

    def __init__(self, ws) -> None:
        self._ws = ws
        self._id = 0

    async def send(self, method: str, **params):
        self._id += 1
        msg_id = self._id
        await self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=60)
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method} 失败：{msg['error']}")
                return msg.get("result", {})

    async def evaluate(self, expression: str):
        res = await self.send("Runtime.evaluate", expression=expression, returnByValue=True)
        return res.get("result", {}).get("value")

    async def shot(self, path: Path) -> None:
        res = await self.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(res["data"]))


def caption_frame(png: Path, text: str, *, width: int, height: int, index: int) -> Path:
    """在截图底部压一条说明条，让视频不用旁白也看得懂。"""
    from PIL import Image, ImageDraw, ImageFont

    font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    if font_path is None:
        return png
    img = Image.open(png).convert("RGB")
    bar = 56
    canvas = Image.new("RGB", (img.width, img.height + bar), (20, 24, 29))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, 24)
    draw.rectangle([0, img.height, canvas.width, canvas.height], fill=(20, 24, 29))
    draw.text((22, img.height + 14), text, font=font, fill=(255, 255, 255))
    draw.text((canvas.width - 120, img.height + 18), f"{index:02d}", font=ImageFont.truetype(font_path, 18),
              fill=(120, 132, 148))
    out = png.with_name(f"cap_{png.name}")
    canvas.save(out)
    return out


async def run(args) -> int:
    edge = find_edge()
    profile = PROJECT / "out" / "edge-demo"
    profile.mkdir(parents=True, exist_ok=True)
    frames_dir = PROJECT / "out" / "demo_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    errlog = (PROJECT / "out" / "record_demo.stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            edge, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--force-prefers-reduced-motion", f"--remote-debugging-port={args.port}",
            f"--user-data-dir={profile}", f"--window-size={args.width},{args.height}",
            args.url,
        ],
        stdout=subprocess.DEVNULL, stderr=errlog,
    )

    frames: list[tuple[Path, str]] = []
    try:
        ws_url = page_ws(args.port)
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            s = Session(ws)
            await s.send("Page.enable")
            await s.send("Runtime.enable")
            await s.send(
                "Emulation.setDeviceMetricsOverride",
                width=args.width, height=args.height, deviceScaleFactor=1, mobile=False,
            )
            time.sleep(2.0)

            for idx, (text, suffix, click_sel, wait_s) in enumerate(STEPS, start=1):
                target = args.url.rstrip("/") + "/" + suffix if suffix else args.url
                await s.send("Page.navigate", url=target)
                time.sleep(wait_s)
                if click_sel:
                    clicked = await s.evaluate(
                        "(sel => { const el = document.querySelector(sel);"
                        " if (!el) return 'miss'; el.click(); return 'ok'; })"
                        f"({json.dumps(click_sel)})"
                    )
                    print(f"  步骤 {idx} 点击 {click_sel} → {clicked}", flush=True)
                    time.sleep(1.6)
                raw = frames_dir / f"raw_{idx:02d}.png"
                await s.shot(raw)
                frames.append((raw, text))
                print(f"  步骤 {idx} 已截图：{text}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        errlog.close()

    if not frames:
        raise SystemExit("一帧都没截到")

    captioned = [
        caption_frame(png, text, width=args.width, height=args.height, index=i)
        for i, (png, text) in enumerate(frames, start=1)
    ]

    ffmpeg = PROJECT / "tools" / "bin" / "ffmpeg.exe"
    if not ffmpeg.exists():
        raise SystemExit("缺 tools/bin/ffmpeg.exe（用 pip install imageio-ffmpeg 后复制过来）")

    # 每步停留 hold 秒：用 concat 的方式给每帧不同时长，避免整片节奏太快
    list_file = frames_dir / "frames.txt"
    hold = args.hold
    lines = []
    for png in captioned:
        lines.append(f"file '{png.as_posix()}'")
        lines.append(f"duration {hold}")
    lines.append(f"file '{captioned[-1].as_posix()}'")  # concat 末帧要重复一次
    list_file.write_text("\n".join(lines), encoding="utf-8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vf", f"fps={args.fps},format=yuv420p", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-movflags", "+faststart", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit("ffmpeg 合成失败")
    size_mb = out.stat().st_size / 1024 / 1024
    duration = hold * len(captioned)
    print(f"完成：{out}（{size_mb:.1f} MB，约 {duration:.0f} 秒，{len(captioned)} 个镜头）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8848/")
    ap.add_argument("--out", default=str(PROJECT / "out" / "demo_screencast.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--port", type=int, default=9344)
    ap.add_argument("--hold", type=float, default=3.2, help="每个镜头停留秒数")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
