"""接口层测试（FastAPI TestClient，不起网络端口）。

守的是「接口是否稳定」这条评分点：状态码、字段名、契约一致性。
其中 review 那一组是回归测试——曾经因为 order_id 设成必填，
前端只传 decision 就吃到 422。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from loweye import api as api_module
from loweye.api import app
from loweye.config import load_config

CFG = load_config()

# 与 test_pipeline 同理：接口测试也会 POST /api/analyze、/api/selfcheck 真落盘，
# 必须改道到临时目录，别把测试工单灌进演示数据。
# 接口侧的管道是懒加载的，get_pipeline() 自己调 load_config()（loweye/api.py:38-49），
# 所以这里要替换模块里的 load_config，而不是只改 CFG 对象。
_TMP = tempfile.TemporaryDirectory(prefix="loweye-api-test-")
_TMP_ROOT = Path(_TMP.name)
CFG.as_dict().setdefault("paths", {})
CFG.as_dict()["paths"]["out"] = str(_TMP_ROOT / "out")
CFG.as_dict()["paths"]["uploads"] = str(_TMP_ROOT / "uploads")
api_module.load_config = lambda *args, **kwargs: CFG

client = TestClient(app)


class HealthAndStatsTest(unittest.TestCase):
    def test_health_initializes_pipeline(self) -> None:
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["status"], {"ok", "degraded"})
        self.assertGreater(body["kb_chunks"], 0, "健康检查不应返回空知识库")
        self.assertGreater(body["kb_docs"], 0)
        for key in ("detector", "generator", "embedder", "reranker", "bm25"):
            self.assertIn(key, body["backends"])

    def test_kb_stats(self) -> None:
        r = client.get("/api/kb/stats")
        self.assertEqual(r.status_code, 200)
        stats = r.json()["stats"]
        for key in ("chunks", "docs", "bm25_backend", "embedder", "reranker", "dim"):
            self.assertIn(key, stats, f"统计缺字段 {key}")
        self.assertGreaterEqual(stats["chunks"], 60, "条款块太少，切分可能退化了")
        self.assertEqual(stats["docs"], 3)
        self.assertGreater(stats["dim"], 0)


class StartupConfigPlumbingTest(unittest.TestCase):
    """`serve --config X` 必须一路管到流水线。

    回归点：get_pipeline() 曾经自己调无参 load_config()，于是
    `serve --config config/app.cplid_gpu.yaml` 表面起来了，实际跑的还是默认配置的
    sidecar 后端 —— 接口不报错，只是结果全空，最难查的那种错。
    """

    def setUp(self) -> None:
        from fastapi import HTTPException  # noqa: F401  (供断言用)

        self._backup_state = dict(api_module._STATE)
        self._backup_load = api_module.load_config

    def tearDown(self) -> None:
        api_module.load_config = self._backup_load
        api_module._STATE.clear()
        api_module._STATE.update(self._backup_state)

    def test_config_path_reaches_load_config(self) -> None:
        seen: dict = {}

        def spy(path=None):
            seen["path"] = path
            raise RuntimeError("到此为止：本用例只验参数传递，不真建流水线")

        api_module._STATE.update({"pipeline": None, "config_path": None, "error": None})
        api_module.load_config = spy
        api_module.use_config("config/app.cplid_gpu.yaml")
        with self.assertRaises(Exception):
            api_module.get_pipeline()
        self.assertEqual(seen.get("path"), "config/app.cplid_gpu.yaml")

    def test_switching_config_after_init_is_rejected(self) -> None:
        api_module._STATE.update({"pipeline": object(), "config_path": None})
        with self.assertRaises(RuntimeError):
            api_module.use_config("config/app.bge.yaml")


class SearchContractTest(unittest.TestCase):
    def test_hit_shape(self) -> None:
        r = client.post("/api/search", json={"query": "绝缘子自爆怎么处理", "top_k": 3})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["covered"])
        hit = body["hits"][0]
        for key in ("rank", "clause", "bm25", "vector", "rrf", "rerank"):
            self.assertIn(key, hit, f"命中结构缺字段 {key}")
        for key in ("chunk_id", "doc_id", "doc_title", "clause_no", "hierarchy_path", "text"):
            self.assertIn(key, hit["clause"])

    def test_refusal_contract(self) -> None:
        """拒答契约：covered=False，且**没有任何一条命中达到覆盖阈值**。

        接口刻意保留 hits（四路分数摊开 = 可解释性），因此不能断言 hits 为空；
        必须断言的不变量是「拒答时不存在高分引用」——否则前端会把噪声条款
        当成处置依据展示出来。
        """
        threshold = float(CFG.require("retrieval.min_relevance"))
        r = client.post("/api/search", json={"query": "今天股市行情怎么样"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["covered"])
        self.assertLess(body["top_score"], threshold)
        for hit in body["hits"]:
            self.assertLess(hit["rerank"], threshold, f"拒答里混进高分条款：{hit['clause']['clause_no']}")

    def test_empty_and_blank_query_rejected(self) -> None:
        """空查询是「没问问题」，不是「知识库未覆盖」，必须在入口 422。"""
        for bad in ("", "   ", "\n\t"):
            r = client.post("/api/search", json={"query": bad})
            self.assertEqual(r.status_code, 422, f"query={bad!r} 没被拒绝")

    def test_top_k_bounds(self) -> None:
        self.assertEqual(client.post("/api/search", json={"query": "绝缘子", "top_k": 0}).status_code, 422)
        self.assertEqual(client.post("/api/search", json={"query": "绝缘子", "top_k": 99}).status_code, 422)


class AnalyzeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.body = client.post("/api/analyze", json={"image_id": "demo_tower_01"}).json()

    def test_result_shape(self) -> None:
        for key in (
            "trace_id", "image_id", "image_url", "scene", "detect",
            "diagnoses", "work_orders", "degradations", "timings", "retrievals", "total_ms",
        ):
            self.assertIn(key, self.body, f"结果缺字段 {key}")

    def test_image_url_points_at_static_mount(self) -> None:
        """前端拿这个 URL 直接贴到 <img>，必须能被静态挂载服务到。"""
        url = self.body["image_url"]
        self.assertTrue(url.startswith("/assets/"), url)
        self.assertEqual(client.get(url).status_code, 200)

    def test_retrievals_keyed_by_class(self) -> None:
        ret = self.body["retrievals"]
        self.assertIn("insulator_broken", ret)
        for cls, trace in ret.items():
            for key in ("query", "covered", "top_score", "elapsed_ms", "hits"):
                self.assertIn(key, trace, f"retrievals.{cls} 缺 {key}")

    def test_bboxes_inside_image(self) -> None:
        for det in self.body["detect"]["detections"]:
            b = det["bbox"]
            self.assertLess(b["x1"], b["x2"])
            self.assertLess(b["y1"], b["y2"])
            self.assertGreaterEqual(b["x1"], 0)
            self.assertLessEqual(b["x2"], 1600)

    def test_unknown_image_404(self) -> None:
        r = client.post("/api/analyze", json={"image_id": "不存在的图"})
        self.assertEqual(r.status_code, 404)


class OrderReviewContractTest(unittest.TestCase):
    def test_review_without_order_id_in_body(self) -> None:
        """回归：body 里不传 order_id 也必须 200（order_id 以路径为准）。"""
        result = client.post("/api/analyze", json={"image_id": "demo_fire_01"}).json()
        order = result["work_orders"][0]
        r = client.post(
            f"/api/orders/{order['order_id']}/review",
            json={"decision": "approve", "reviewer": "单测", "note": "回归测试"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("status", body, "响应应含 ReviewResponse 字段")
        self.assertIn("order", body, "响应应带回更新后的工单")
        self.assertEqual(body["order"]["order_id"], order["order_id"])

    def test_review_unknown_order_404(self) -> None:
        r = client.post("/api/orders/WO-不存在/review", json={"decision": "approve"})
        self.assertEqual(r.status_code, 404)

    def test_orders_list_shape(self) -> None:
        r = client.get("/api/orders")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("items", body)
        self.assertEqual(body["count"], len(body["items"]))


class TraceAndReportTest(unittest.TestCase):
    def test_trace_roundtrip_and_reports(self) -> None:
        created = client.post("/api/analyze", json={"image_id": "demo_channel_01"}).json()
        tid = created["trace_id"]

        got = client.get(f"/api/traces/{tid}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["trace_id"], tid)

        self.assertEqual(client.get("/api/traces").status_code, 200)
        self.assertEqual(client.get(f"/api/traces/{tid}").status_code, 200)

        html = client.get(f"/api/report/{tid}", params={"fmt": "html"})
        self.assertEqual(html.status_code, 200)
        self.assertIn("text/html", html.headers["content-type"])

        pdf = client.get(f"/api/report/{tid}", params={"fmt": "pdf"})
        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.content[:5], b"%PDF-")

    def test_report_unknown_trace_404(self) -> None:
        self.assertEqual(client.get("/api/report/T-nope").status_code, 404)


class SelfcheckTest(unittest.TestCase):
    def test_selfcheck_all_green(self) -> None:
        r = client.post("/api/selfcheck")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        failed = [s for s in body["steps"] if not s["ok"]]
        self.assertTrue(body["ok"], f"自检未通过：{failed}")
        self.assertGreaterEqual(len(body["steps"]), 5, "自检步骤太少，覆盖不足")


class StaticMountTest(unittest.TestCase):
    def test_console_and_assets_served(self) -> None:
        """静态挂载顺序曾出错（/ 抢在 /assets 前面，图片全 404），这里守住。"""
        for path in ("/", "/app.js", "/styles.css", "/vendor/icons.svg"):
            self.assertEqual(client.get(path).status_code, 200, path)
        assets = client.get("/api/assets").json()
        self.assertGreaterEqual(assets["count"], 4)
        for item in assets["items"]:
            self.assertEqual(client.get(item["url"]).status_code, 200, item["url"])


if __name__ == "__main__":
    unittest.main()
