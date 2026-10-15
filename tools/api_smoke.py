"""服务端自检脚本：把 FastAPI 暴露的接口逐个调一遍，结果打到控制台。

用途有两个：
1. 开发时回归 —— 改完接口跑一次，确认没有接口悄悄坏掉；
2. 答辩前现场检查 —— 一条命令给出「服务是否可用」的证据，而不是靠肉眼点页面。

用法：
    python tools/api_smoke.py [base_url]
默认 base_url = http://127.0.0.1:8848
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8848"
FAILS: list[str] = []


def show(title: str, ok: bool, detail: str) -> None:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {title} — {detail}")
    if not ok:
        FAILS.append(title)


def brief(data, limit: int = 220) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=120.0)

    r = client.get("/api/health")
    body = r.json()
    show("GET /api/health", r.status_code == 200 and body.get("status") == "ok", brief(body))

    r = client.get("/api/kb/stats")
    body = r.json()
    show("GET /api/kb/stats", r.status_code == 200, brief(body))

    r = client.get("/api/assets")
    body = r.json()
    names = [a["image_id"] for a in body.get("items", [])]
    show("GET /api/assets", r.status_code == 200 and len(names) > 0, f"{len(names)} 张：{names}")

    r = client.post("/api/search", json={"query": "绝缘子自爆怎么处理", "top_k": 3})
    body = r.json()
    hits = body.get("hits", [])
    top = hits[0] if hits else {}
    show(
        "POST /api/search（应有命中）",
        r.status_code == 200 and body.get("covered") is True and len(hits) > 0,
        f"covered={body.get('covered')} top={top.get('clause', {}).get('clause_no')} "
        f"score={body.get('top_score')} elapsed={body.get('elapsed_ms')}ms",
    )

    r = client.post("/api/search", json={"query": "今天股市行情怎么样", "top_k": 3})
    body = r.json()
    show(
        "POST /api/search（应拒答）",
        r.status_code == 200 and body.get("covered") is False,
        f"covered={body.get('covered')} top_score={body.get('top_score')}",
    )

    started = time.perf_counter()
    r = client.post("/api/analyze", json={"image_id": "demo_tower_01"})
    ms = (time.perf_counter() - started) * 1000
    body = r.json()
    ok = r.status_code == 200 and len(body.get("work_orders", [])) > 0
    show(
        "POST /api/analyze",
        ok,
        f"{len(body.get('diagnoses', []))} 条诊断 / {len(body.get('work_orders', []))} 张工单，"
        f"trace={body.get('trace_id')}，端到端 {body.get('total_ms')}ms（含 HTTP {ms:.0f}ms）",
    )
    trace_id = body.get("trace_id")

    r = client.get(f"/api/traces/{trace_id}")
    body = r.json()
    show("GET /api/traces/{id}", r.status_code == 200 and body.get("trace_id") == trace_id, brief(body, 120))

    r = client.get("/api/traces")
    body = r.json()
    show("GET /api/traces", r.status_code == 200 and body.get("count", 0) > 0, f"{body.get('count')} 条历史")

    r = client.get("/api/orders")
    body = r.json()
    orders = body.get("items", [])
    show("GET /api/orders", r.status_code == 200 and len(orders) > 0, f"{len(orders)} 张工单")

    if orders:
        order_id = orders[0]["order_id"]
        r = client.post(
            f"/api/orders/{order_id}/review",
            json={"decision": "approve", "reviewer": "自检脚本", "note": "接口连通性自检"},
        )
        body = r.json()
        show(
            "POST /api/orders/{id}/review",
            r.status_code == 200 and body.get("order_id") == order_id,
            f"decision={body.get('decision')} status={body.get('status')} reviewer={body.get('reviewer')}",
        )

    for fmt in ("html", "pdf"):
        r = client.get(f"/api/report/{trace_id}", params={"fmt": fmt})
        show(
            f"GET /api/report?fmt={fmt}",
            r.status_code == 200 and len(r.content) > 1000,
            f"{r.headers.get('content-type')} {len(r.content)} bytes",
        )

    r = client.post("/api/selfcheck")
    body = r.json()
    steps = body.get("steps", [])
    show(
        "POST /api/selfcheck",
        r.status_code == 200 and body.get("ok") is True,
        f"ok={body.get('ok')}，" + "，".join(f"{s['step']}={'√' if s.get('ok') else '×'}" for s in steps),
    )

    client.close()
    print()
    if FAILS:
        print(f"自检未通过 {len(FAILS)} 项：" + "、".join(FAILS))
        return 1
    print("全部接口自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
