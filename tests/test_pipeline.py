"""端到端链路测试：识别 → 诊断 → 分级 → 派单 → 归档 → 出报告。

这是最重要的一组测试：前面全是零件，这里测的是「整条链路真的能跑出可执行结果」。
注意区分两类结论——
  本文件断言的是**链路的可达性与一致性**（跑得通、字段齐、引用带条款号、越界会拒答）；
  它不是**模型精度**结论（检测后端是 sidecar 合成标注 / ONNX 权重另配，精度由 M6 评估脚本单独报）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from loweye.agent.pipeline import Pipeline
from loweye.config import load_config
from loweye.llm.base import DiagnosisContext
from loweye.llm.template import TemplateGenerator, condition_prefix
from loweye.schemas import Clause, ScoredClause
from loweye.text import is_numbered_clause

CFG = load_config()

# 这组测试会真跑链路并落盘（工单 JSONL、追溯、报告），必须把 out 与 uploads 改道到临时目录。
# 改成临时目录之前，每跑一次测试就往演示数据里灌十几条工单——演示台上那批残留工单就是这么来的。
# Config.path() 对绝对路径原样返回（loweye/config.py:50-54），所以只改这两项即可，
# assets / index_dir / web 等仍按项目根解析。
_TMP = tempfile.TemporaryDirectory(prefix="loweye-test-")
_TMP_ROOT = Path(_TMP.name)
CFG.as_dict().setdefault("paths", {})
CFG.as_dict()["paths"]["out"] = str(_TMP_ROOT / "out")
CFG.as_dict()["paths"]["uploads"] = str(_TMP_ROOT / "uploads")

ASSETS: Path = CFG.path("assets")


class PipelineChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = Pipeline.from_config(CFG, build_kb=True)

    def test_demo_assets_exist(self) -> None:
        for name in ("demo_tower_01", "demo_channel_01", "demo_field_01", "demo_fire_01"):
            self.assertTrue((ASSETS / f"{name}.png").exists(), f"缺演示图 {name}.png")

    def test_run_produces_full_trace(self) -> None:
        r = self.pipe.run(ASSETS / "demo_tower_01.png", image_id="demo_tower_01")
        self.assertTrue(r.trace_id.startswith("T"))
        self.assertEqual(r.scene, "power")
        self.assertEqual(len(r.detect.detections), 2)
        self.assertEqual(len(r.diagnoses), 2)
        self.assertEqual(len(r.work_orders), 2)
        self.assertGreater(r.total_ms, 0)
        self.assertEqual([t.stage for t in r.timings], ["detect", "diagnose", "grade_dispatch", "archive"])

    def test_p1_defect_goes_to_human_review(self) -> None:
        r = self.pipe.run(ASSETS / "demo_tower_01.png", image_id="demo_tower_01")
        by_cls = {d.cls: d for d in r.diagnoses}
        self.assertIn("insulator_broken", by_cls)
        self.assertEqual(by_cls["insulator_broken"].severity, "P1")
        order = next(o for o in r.work_orders if o.severity == "P1")
        self.assertEqual(order.route, "human_review")
        self.assertTrue(order.review_reasons, "P1 工单必须写明为什么要人工确认")
        self.assertLessEqual(order.sla_hours, 24)

    def test_every_action_is_numbered_and_cited(self) -> None:
        """可溯源的硬要求：给出的每一条动作都要能指回具体条款。"""
        r = self.pipe.run(ASSETS / "demo_field_01.png", image_id="demo_field_01")
        for d in r.diagnoses:
            if not d.grounded:
                continue
            self.assertTrue(d.actions, f"{d.cls} 判定为可溯源却没给出动作")
            self.assertTrue(d.citations, f"{d.cls} 有动作但没有引用条款")
            for c in d.citations:
                self.assertTrue(is_numbered_clause(c.clause_no), f"引用到非条款号：{c.clause_no}")
                self.assertTrue(c.quote.strip(), "引用没有原文片段")

    def test_citations_match_the_clause_text(self) -> None:
        """引用片段必须真的来自知识库，不能是模型自己编的句子。"""
        r = self.pipe.run(ASSETS / "demo_channel_01.png", image_id="demo_channel_01")
        corpus = {c.chunk_id: c for c in self.pipe.kb.clauses}
        for d in r.diagnoses:
            for c in d.citations:
                self.assertIn(c.chunk_id, corpus, f"引用了不存在的 chunk：{c.chunk_id}")
                self.assertEqual(corpus[c.chunk_id].clause_no, c.clause_no)
                head = c.quote[:12]
                self.assertIn(head, corpus[c.chunk_id].text, "引用片段与知识库原文对不上")

    def test_agriculture_scene_routing(self) -> None:
        r = self.pipe.run(ASSETS / "demo_field_01.png", image_id="demo_field_01")
        self.assertEqual(r.scene, "agriculture")
        for o in r.work_orders:
            self.assertNotEqual(o.assignee_role, "输电运检班")

    def test_fire_scene_is_p1_emergency(self) -> None:
        r = self.pipe.run(ASSETS / "demo_fire_01.png", image_id="demo_fire_01")
        self.assertEqual(r.scene, "emergency")
        self.assertEqual(r.work_orders[0].severity, "P1")

    def test_trace_store_records_run(self) -> None:
        r = self.pipe.run(ASSETS / "demo_tower_01.png", image_id="demo_tower_01")
        self.assertIsNotNone(self.pipe.traces.get(r.trace_id))
        self.assertGreaterEqual(len(self.pipe.traces), 1)

    def test_conditional_branch_not_turned_into_action(self) -> None:
        """条件分支不得变成处置动作。

        演示导则 3.1 是「金具锈蚀」小节，其中 3.1.3 讲的是裂纹/变形/磨损，
        若被当成本次动作，工单就会写着「按危急缺陷流程处理，立即申请停电更换」——
        对一处锈蚀来说那是一次不必要的停电。这一条就是防它回归。
        """
        r = self.pipe.run(ASSETS / "demo_tower_01.png", image_id="demo_tower_01")
        corrosion = next((d for d in r.diagnoses if d.cls == "fitting_corrosion"), None)
        if corrosion is None:
            self.skipTest("本次演示图未检出金具锈蚀")
        for action in corrosion.actions:
            self.assertNotIn("裂纹", action, f"把别的情形的条件分支当成了动作：{action}")
            self.assertNotIn("危急缺陷流程", action, f"越权动作：{action}")

    def test_response_level_clause_is_not_an_action(self) -> None:
        """应急预案的「一级／二级响应」是分级判据，不是处置动作。

        它含「响应」二字，早期正则把里面的「应」当成了情态词，
        于是工单动作里出现「二级响应：发现烟雾但未确认明火……」。
        """
        r = self.pipe.run(ASSETS / "demo_fire_01.png", image_id="demo_fire_01")
        fire = next(d for d in r.diagnoses if d.cls == "fire_smoke")
        for action in fire.actions:
            self.assertNotIn("级响应", action, f"分级判据被当成动作：{action}")

    def test_reference_citations_are_separated_from_basis(self) -> None:
        """检索命中但未抽成动作的条款要以 role=reference 保留，且不算依据。"""
        r = self.pipe.run(ASSETS / "demo_fire_01.png", image_id="demo_fire_01")
        fire = next(d for d in r.diagnoses if d.cls == "fire_smoke")
        roles = {c.role for c in fire.citations}
        self.assertIn("basis", roles, "火情诊断没有任何依据条款")
        self.assertIn("reference", roles, "分级判据条款被整条丢掉了，应当作为参考条款保留")
        self.assertTrue(fire.grounded, "有依据条款时 grounded 应为真")
        # 参考条款不能算成依据：这里按角色分开断言，防止以后把两者混成一锅
        basis = [c for c in fire.citations if c.role == "basis"]
        refs = [c for c in fire.citations if c.role == "reference"]
        self.assertTrue(all(c.chunk_id not in {b.chunk_id for b in basis} for c in refs))
        self.assertIn("1.2.2", {c.clause_no for c in refs}, "一级／二级响应条款应作为参考条款出现")

    def test_scene_scope_limits_documents(self) -> None:
        """输电场景的检索不得命中《植保作业指引》——各专业规程各管一段。"""
        r = self.pipe.run(ASSETS / "demo_tower_01.png", image_id="demo_tower_01")
        self.assertEqual(r.scene, "power")
        for d in r.diagnoses:
            for c in d.citations:
                self.assertNotIn("植保", c.doc_title, f"输电缺陷引用了农业文档：{c.doc_title}")
        for trace in r.retrievals.values():
            self.assertEqual(trace.scope_docs, ["demo_line_guide"])


class ConditionalBranchTest(unittest.TestCase):
    """生成器侧的条件分支判定（不依赖检索结果，确定可复现）。"""

    def _ctx(self, text: str, *, cls: str = "fitting_corrosion") -> DiagnosisContext:
        clause = Clause(
            chunk_id=f"{cls}#0001",
            doc_id="demo",
            doc_title="演示导则",
            clause_no="3.1.3",
            hierarchy_path="第三章 金具与导地线 › 3.1 金具锈蚀",
            page=1,
            text=text,
            source_version="demo",
            source_date="2026-10-12",
        )
        scored = ScoredClause(clause=clause, bm25=1.0, vector=0.5, rrf=0.03, rerank=0.5)
        return DiagnosisContext(
            cls=cls,
            label="金具锈蚀",
            severity="P2",
            n_detections=1,
            confidence=0.81,
            clauses=[scored],
            keywords=["金具", "锈蚀", "腐蚀"],
        )

    def test_condition_prefix_extracted(self) -> None:
        self.assertEqual(
            condition_prefix("金具出现裂纹、变形或明显磨损的，应按照危急缺陷流程处理。"),
            "裂纹、变形或明显磨损",
        )
        # 没有条件引导词 / 没有逗号断开的都不算条件句
        self.assertIsNone(condition_prefix("锈蚀部位应在 72 小时内安排除锈防腐处理。"))
        self.assertIsNone(condition_prefix("明火扑灭后，应继续监视 24 小时。"))

    def test_other_condition_is_skipped_and_named(self) -> None:
        gen = TemplateGenerator()
        out = gen.generate(
            self._ctx("金具出现裂纹、变形或明显磨损的，应按照危急缺陷流程处理，立即申请停电更换。")
        )
        # 不能把别的情形的条件分支当成处置动作
        self.assertEqual(len(out.actions), 1)
        self.assertIn("条件分支", out.actions[0])
        self.assertNotIn("停电更换", out.actions[0])
        # 结论里要点名是哪一条、什么条件，让人能自己翻原文
        self.assertIn("3.1.3", out.finding)
        self.assertIn("条件分支", out.finding)

    def test_same_condition_is_kept(self) -> None:
        """条件说的就是本类别时，正常抽成动作。"""
        gen = TemplateGenerator()
        out = gen.generate(
            self._ctx("金具出现锈蚀、镀锌层脱落的，应安排线路检修班组进行除锈防腐处理。")
        )
        self.assertEqual(len(out.actions), 1)
        self.assertIn("除锈", out.actions[0])
        self.assertIn("〔3.1.3〕", out.actions[0])

    def test_insulator_condition_kept_for_its_own_class(self) -> None:
        ctx = self._ctx("发现绝缘子自爆后，应立即上报线路运维管理部门。", cls="insulator_broken")
        ctx.label = "绝缘子自爆（伞裙破损）"
        ctx.keywords = ["绝缘子", "自爆", "破损", "伞裙", "零值"]
        out = TemplateGenerator().generate(ctx)
        self.assertEqual(len(out.actions), 1)
        self.assertIn("上报", out.actions[0])

    def test_consequence_condition_is_skipped(self) -> None:
        """「异物已导致放电痕迹或灼伤的，应……停电清除」是后果分支，不是当前动作。"""
        ctx = self._ctx(
            "异物已导致放电痕迹或灼伤的，应按照危急缺陷流程处理，立即申请停电清除。", cls="foreign_object"
        )
        ctx.label = "异物悬挂"
        ctx.keywords = ["异物", "悬挂", "风筝", "塑料", "鸟巢"]
        out = TemplateGenerator().generate(ctx)
        self.assertEqual(len(out.actions), 1, "后果分支应退化为兜底说明而不是动作")
        self.assertNotIn("停电清除", out.actions[0])
        self.assertIn("条件分支", out.actions[0])

    def test_retrieved_but_unused_clause_becomes_reference(self) -> None:
        gen = TemplateGenerator()
        ctx = self._ctx("锈蚀部位应在 72 小时内安排除锈防腐处理。")
        out = gen.generate(ctx)
        self.assertEqual(len(out.citation_ids), 1)
        self.assertEqual(out.reference_ids, [], "全部条款都抽成了动作，不该有参考条款")

    def test_scope_statement_never_becomes_reference(self) -> None:
        """适用范围/术语定义这类元信息条款连参考列都不进（否则是拿文档头充数）。"""
        ctx = self._ctx("本导则适用于架空线路无人机巡检缺陷判定与处置。")
        # 再挂一条真正的处置条款，这样才走「有动作 + 有参考」的正常分支
        extra = Clause(
            chunk_id="fitting_corrosion#0002",
            doc_id="demo",
            doc_title="演示导则",
            clause_no="3.1.2",
            hierarchy_path="第三章 金具与导地线 › 3.1 金具锈蚀",
            page=1,
            text="锈蚀部位应在 72 小时内安排线路检修班组进行除锈防腐处理。",
            source_version="demo",
            source_date="2026-10-12",
        )
        ctx.clauses.append(ScoredClause(clause=extra, bm25=1.0, vector=0.5, rrf=0.03, rerank=0.4))
        out = TemplateGenerator().generate(ctx)
        self.assertEqual(out.citation_ids, ["fitting_corrosion#0002"])
        self.assertEqual(out.reference_ids, [], "适用范围不该作为参考条款出现")

    def test_response_word_is_not_a_modal(self) -> None:
        """「响应」「供应」「相应」里的「应」不是情态词。"""
        from loweye.llm.template import is_actionable

        self.assertFalse(is_actionable("二级响应：发现烟雾但未确认明火，或明火距线路走廊 500 米以外。"))
        self.assertFalse(is_actionable("物资供应由后勤部门统一负责，按季度盘点。"))
        self.assertTrue(is_actionable("发现绝缘子自爆后，应立即上报线路运维管理部门。"))
        self.assertTrue(is_actionable("锈蚀部位应在 72 小时内安排除锈防腐处理。"))


class RefusalAndGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = Pipeline.from_config(CFG, build_kb=True)

    def test_in_scope_query_is_covered(self) -> None:
        resp = self.pipe.kb.search("绝缘子自爆怎么处理", top_k=3)
        self.assertTrue(resp.covered)
        self.assertGreaterEqual(len(resp.hits), 1)
        self.assertEqual(resp.hits[0].clause.clause_no, "2.1.1")

    def test_out_of_scope_query_refused(self) -> None:
        """越界问题必须被判为「知识库未覆盖」。

        检索层刻意保留 hits（四路分数摊开是可解释性的一部分，UI 会把这些条款
        标注为「低于覆盖阈值 不作为处置依据」），所以这里断言的是**阈值不变量**：
        covered=False 时 top_score 必须低于配置阈值，且每一条 hit 的重排分也都低于阈值。
        这样即使以后调阈值或换检索后端，也不会出现「拒答却带着高分引用」。
        """
        threshold = float(CFG.require("retrieval.min_relevance"))
        for q in ("今天股市行情怎么样", "怎么做红烧肉"):
            resp = self.pipe.kb.search(q, top_k=3)
            self.assertFalse(resp.covered, f"越界问题「{q}」没有被拒答")
            self.assertLess(resp.top_score, threshold, f"越界问题「{q}」的 top_score 达到阈值")
            for hit in resp.hits:
                self.assertLess(hit.rerank, threshold, f"拒答结果里混进了高分条款：{q} → {hit.clause.clause_no}")

    def test_hits_always_carry_clause_number(self) -> None:
        """索引层开了 require_clause_no，检索结果不应出现"无编号"占位块。"""
        for q in ("绝缘子自爆", "树障超限", "导地线断股", "火灾 浓烟"):
            for hit in self.pipe.kb.search(q, top_k=5).hits:
                self.assertTrue(is_numbered_clause(hit.clause.clause_no), f"{q} → {hit.clause.clause_no}")

    def test_scores_are_monotonic(self) -> None:
        hits = self.pipe.kb.search("树障 净空距离", top_k=5).hits
        scores = [h.rerank for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True), "命中没有按重排分降序")


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = Pipeline.from_config(CFG, build_kb=True)
        cls.result = cls.pipe.run(ASSETS / "demo_tower_01.png", image_id="demo_tower_01")

    def test_html_report(self) -> None:
        path, tool = self.pipe.toolbox.render_report(self.result, fmt="html")
        self.assertTrue(tool.ok, tool.detail)
        self.assertIsNotNone(path)
        html = Path(path).read_text(encoding="utf-8")
        self.assertIn(self.result.trace_id, html)
        self.assertIn("绝缘子自爆", html)
        self.assertIn("2.1.", html, "报告里没有条款号")
        self.assertNotIn("{{", html, "模板变量没渲染干净")

    def test_pdf_report(self) -> None:
        path, tool = self.pipe.toolbox.render_report(self.result, fmt="pdf")
        self.assertTrue(tool.ok, tool.detail)
        pdf = Path(path)
        self.assertTrue(pdf.exists())
        self.assertGreater(pdf.stat().st_size, 20_000, "PDF 太小，可能是空文档")
        self.assertEqual(pdf.read_bytes()[:5], b"%PDF-", "不是合法 PDF")

    def test_unknown_format_degrades_gracefully(self) -> None:
        path, tool = self.pipe.toolbox.render_report(self.result, fmt="doc")
        self.assertFalse(tool.ok)
        self.assertFalse(tool.detail == "", "降级时必须给出原因，不能静默失败")


if __name__ == "__main__":
    unittest.main()
