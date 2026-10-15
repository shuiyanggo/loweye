"""工具层：工单落库、群机器人通知、天气、报告导出。

每个工具都遵守同一条纪律：**失败必须留下痕迹**。
通知发不出去、天气取不到，都不许静默跳过 —— 编排层会把失败记进
degradations，前端直接在界面上显示「本步降级」，这才是可观测的系统。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from ..config import Config
from ..schemas import PipelineResult, WorkOrder

_SEQ_WIDTH = 4


@dataclass
class ToolResult:
    ok: bool
    detail: str
    degraded: bool = False


@dataclass
class ToolBox:
    cfg: Config
    routes: dict

    def __post_init__(self) -> None:
        self.out_dir = self.cfg.path("out")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.orders_path = self.out_dir / "work_orders.jsonl"
        self.traces_dir = self.out_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._seq = self._load_seq()
        self.results: list[dict] = []

    # -------------------------------------------------------- 工单

    def _load_seq(self) -> int:
        if not self.orders_path.exists():
            return 0
        n = 0
        with self.orders_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n

    def next_order_id(self) -> str:
        self._seq += 1
        day = datetime.now().strftime("%Y%m%d")
        return f"WO-{day}-{self._seq:0{_SEQ_WIDTH}d}"

    def create_work_order(self, order: WorkOrder) -> ToolResult:
        """工单落库（JSONL 追加写，天然可审计、可 grep、可导入 Excel）。"""
        if not self.cfg.get("tools.workorder.enabled", True):
            return ToolResult(ok=False, detail="工单工具已禁用", degraded=True)
        try:
            with self.orders_path.open("a", encoding="utf-8") as fh:
                fh.write(order.model_dump_json() + "\n")
            return ToolResult(ok=True, detail=f"已写入 {self.orders_path.name}")
        except Exception as exc:
            return ToolResult(ok=False, detail=f"工单落库失败：{type(exc).__name__}: {exc}", degraded=True)

    def list_work_orders(self, limit: int = 50) -> list[dict]:
        if not self.orders_path.exists():
            return []
        rows: list[dict] = []
        with self.orders_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows[-limit:][::-1]

    def update_work_order(self, order_id: str, patch: dict) -> dict | None:
        """人工复核回写：把某条工单的状态/动作改掉，行级重写 JSONL。"""
        if not self.orders_path.exists():
            return None
        rows: list[dict] = []
        updated: dict | None = None
        with self.orders_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("order_id") == order_id:
                    row.update(patch)
                    updated = row
                rows.append(row)
        if updated is None:
            return None
        with self.orders_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return updated

    # -------------------------------------------------------- 通知

    def notify(self, order: WorkOrder) -> ToolResult:
        if not self.cfg.get("tools.notify.enabled", False):
            return ToolResult(ok=False, detail="群机器人通知未启用（config: tools.notify.enabled=false）", degraded=False)
        url = Config.env(str(self.cfg.get("tools.notify.webhook_url_env", "LOWEYE_NOTIFY_WEBHOOK")))
        if not url:
            return ToolResult(ok=False, detail="未设置 webhook 环境变量", degraded=True)
        text = (
            f"【{order.severity}】{order.title}\n"
            f"责任班组：{order.assignee_role}｜时限：{order.sla_hours}h\n"
            f"动作：{'；'.join(order.actions[:3])}\n"
            f"工单号：{order.order_id}"
        )
        try:
            resp = httpx.post(
                url,
                json={"msgtype": "text", "text": {"content": text}},
                timeout=float(self.cfg.get("tools.notify.timeout_s", 5)),
            )
            resp.raise_for_status()
            return ToolResult(ok=True, detail="已推送群机器人")
        except Exception as exc:
            return ToolResult(ok=False, detail=f"通知发送失败：{type(exc).__name__}: {exc}", degraded=True)

    # -------------------------------------------------------- 天气

    def weather(self, scene: str = "power") -> tuple[dict | None, ToolResult]:
        """作业窗口提示：大风/降雨会影响无人机可飞性，属于派单前的现实约束。"""
        if not self.cfg.get("tools.weather.enabled", False):
            return None, ToolResult(ok=False, detail="天气工具未启用（config: tools.weather.enabled=false）")
        key = Config.env(str(self.cfg.get("tools.weather.api_key_env", "LOWEYE_WEATHER_KEY")))
        if not key:
            return None, ToolResult(ok=False, detail="未设置天气 API Key", degraded=True)
        try:
            resp = httpx.get(
                "https://devapi.qweather.com/v7/weather/now",
                params={"location": "101010100", "key": key},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return {"text": data.get("now", {}).get("text"), "windScale": data.get("now", {}).get("windScale")}, ToolResult(
                ok=True, detail="天气已获取"
            )
        except Exception as exc:
            return None, ToolResult(ok=False, detail=f"天气获取失败：{type(exc).__name__}: {exc}", degraded=True)

    # -------------------------------------------------------- 归档

    def archive(self, result: PipelineResult) -> ToolResult:
        try:
            path = self.traces_dir / f"{result.trace_id}.json"
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return ToolResult(ok=True, detail=f"已归档 {path.name}")
        except Exception as exc:
            return ToolResult(ok=False, detail=f"归档失败：{type(exc).__name__}: {exc}", degraded=True)

    # -------------------------------------------------------- 报告

    def render_report(self, result: PipelineResult, *, fmt: str = "pdf") -> tuple[Path | None, ToolResult]:
        if not self.cfg.get("tools.report.enabled", True):
            return None, ToolResult(ok=False, detail="报告工具未启用")
        fmt = (fmt or "").strip().lower()
        if fmt not in ("html", "pdf"):
            # 不能静默按 PDF 出：调用方写错格式却拿到 ok=True，会把错误藏到交付物里
            return None, ToolResult(ok=False, detail=f"不支持的报告格式：{fmt or '(空)'}，可选 html 或 pdf")
        tpl_path = self.cfg.abs_path(self.cfg.get("tools.report.template", "config/report_template.html"))
        if not tpl_path.exists():
            return None, ToolResult(ok=False, detail=f"报告模板不存在：{tpl_path}", degraded=True)
        try:
            from jinja2 import Template

            html = Template(tpl_path.read_text(encoding="utf-8")).render(
                r=result, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as exc:
            return None, ToolResult(ok=False, detail=f"报告渲染失败：{type(exc).__name__}: {exc}", degraded=True)

        html_path = self.out_dir / f"report_{result.trace_id}.html"
        html_path.write_text(html, encoding="utf-8")
        if fmt == "html":
            return html_path, ToolResult(ok=True, detail=f"已生成 {html_path.name}")

        pdf_path = self.out_dir / f"report_{result.trace_id}.pdf"
        # DocumentWriter 直出的 PDF 会把中文回退字体整包塞进去，一份两页报告 3.6MB。
        # 因此先写到临时文件，再用 garbage=4 + deflate 重存到最终路径（实测 3.6MB → 1.7MB）。
        # 注意：不能对 DocumentWriter 刚写完的那个文件做原地替换，Windows 上它的句柄
        # 要等对象被回收才释放，会抛 PermissionError。
        raw_path = self.out_dir / f".report_{result.trace_id}.raw.pdf"
        try:
            import pymupdf

            story = pymupdf.Story(html=html)
            writer = pymupdf.DocumentWriter(str(raw_path))
            mediabox = pymupdf.paper_rect("a4")
            where = mediabox + (36, 36, -36, -36)
            more = 1
            pages = 0
            while more and pages < 40:
                dev = writer.begin_page(mediabox)
                more, _ = story.place(where)
                story.draw(dev)
                writer.end_page()
                pages += 1
            writer.close()
            del writer

            compact_note = ""
            try:
                doc = pymupdf.open(str(raw_path))
                doc.save(str(pdf_path), garbage=4, deflate=True, clean=True)
                doc.close()
                compact_note = f"，压缩至 {pdf_path.stat().st_size // 1024}KB"
            except Exception as exc:
                # 压缩失败不影响可用性：直接把原始产物改名交付，并把原因说清楚
                os.replace(raw_path, pdf_path)
                compact_note = f"，未压缩（{type(exc).__name__}）"
            finally:
                if raw_path.exists():
                    try:
                        raw_path.unlink()
                    except OSError:
                        pass
            return pdf_path, ToolResult(ok=True, detail=f"已生成 {pdf_path.name}（{pages} 页{compact_note}）")
        except Exception as exc:
            return html_path, ToolResult(
                ok=True,
                detail=f"PDF 生成失败（{type(exc).__name__}: {exc}），已回退输出 HTML：{html_path.name}",
                degraded=True,
            )

    # -------------------------------------------------------- 汇总

    def snapshot(self) -> list[dict]:
        return list(self.results)

    def record(self, tool: str, result: ToolResult) -> None:
        self.results.append(
            {"tool": tool, "ok": result.ok, "degraded": result.degraded, "detail": result.detail, "at": time.time()}
        )
