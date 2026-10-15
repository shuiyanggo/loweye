"""Tests for tools/write_model_report.py.

The report is a delivery document, so the properties that matter are: every number
comes from the artifact bytes, a missing artifact is disclosed rather than guessed,
and two runs over the same inputs are byte-identical.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_model_report import (MISSING, PARITY_TOLERANCE, TASKS,  # noqa: E402
                                      build_document)

STAMP = "2026-01-01T00:00:00"


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_fixture(directory: Path, *, trainer_map=0.111111, independent_map=0.123457):
    """Two tasks with complete, internally consistent evidence."""
    for index, spec in enumerate(TASKS):
        task, dataset = spec["task"], spec["dataset"]
        classes = ["alpha", "beta"]
        write_json(directory / f"{task}_fcos_metadata.json", {
            "task": task, "classes": classes, "architecture": "torchvision FCOS ResNet50 FPN",
            "implementation_license": "BSD-3-Clause", "smoke": False,
            "selection": {"split": "val", "criterion": "COCO map", "epoch": 7},
            "manifest_sha256": "a" * 64,
            "val_metrics": {"map": 0.2 + index * 0.01, "map50": 0.3 + index * 0.01},
            "test_metrics": {"map": trainer_map, "map50": 0.5 + index * 0.01,
                             "operating_point": {"overall": {"recall": 0.7, "false_positive_fraction": 0.8,
                                                             "false_alarms_per_image": 4.0}}},
            "test_status": "evaluated once",
            "dependencies": {"torch": "2.8.0", "onnxruntime": "1.31.0"},
        })
        write_json(directory / f"{task}_test.json", {
            "signature": {"task": task}, "checkpoint_sha256": "b" * 64,
            "metrics": {"map": trainer_map}, "evaluations": 1,
        })
        write_json(directory / f"{task}_onnx_eval.json", {
            "artifact": {"onnx_bytes": 1234, "onnx_sha256": "c" * 64},
            "runtime": {"onnxruntime": "1.31.0", "session_providers": ["CPUExecutionProvider"]},
            "protocol": {"images": 100 + index, "classes": len(classes)},
            "coco": {"map": independent_map, "map50": 0.51},
            "per_class": {"alpha": {"map": 0.4, "map50": 0.5, "gt": 10, "evaluable": True},
                          "beta": {"map": None, "map50": None, "gt": 0, "evaluable": False}},
            "operating_point": {"overall": {"recall": 0.71, "false_alarms_per_image": 4.2}},
            "counts": {"ground_truth_boxes": 10, "detections_at_floor": 300,
                       "non_evaluable_classes": ["beta"]},
            "preprocessing_parity": {"checked_images": 8, "max_abs_difference": 0.0},
            "timing": {"seconds": 12.5, "seconds_per_image": 0.125},
        })
        write_json(directory / f"{task}_run.json", {
            "signature": {"epochs": 24, "batch": 4, "lr": 0.0002, "workers": 8, "seed": 42,
                          "limit_train": None, "limit_val": None, "manifest_sha256": "a" * 64},
            "smoke": False,
        })
        write_json(directory / f"{task}_manifest.json", {
            "dataset": dataset, "classes": classes,
            "statistics": {"train": {"images": 90, "empty_images": 1, "source_groups": 90,
                                     "objects": {"alpha": 3, "beta": 0}},
                           "val": {"images": 10, "empty_images": 0, "source_groups": 10,
                                   "objects": {"alpha": 1, "beta": 0}}},
            "annotation_formats": {f"VOC_{index}based_inclusive": 100},
            "voc_class_id_base": index, "split_strategy": "official test preserved",
            "seed": 42, "dedup": {"removed_count": 0},
            "source_group_audit": {"cross_split_groups": []},
            "quarantine": {"enabled": True, "count": 3, "by_original_split": {"train": 3},
                           "by_reason": {"box extends beyond image": 3}},
            "class_coverage": {"policy": "strict_all_classes_observed", "missing_ids": [2],
                               "missing_names": ["beta"]},
        })
        write_json(directory / f"{dataset}_source_license.json", {
            "dataset": dataset, "source": f"https://example.invalid/{dataset}",
            "archive_bytes": 1000, "sha256": "d" * 64,
        })
        write_json(directory / f"{dataset}_transport.json", {"archive_bytes": 1000, "sha256": "d" * 64})
        artifact = directory / f"{task}_fcos_best.pth"
        artifact.write_bytes(b"weights")
        write_json(directory / f"{task}_recovery.json", [
            {"remote": "/root/" + artifact.name, "local": str(artifact), "sha256": "e" * 64}])
        (directory / f"{task}_epochs.jsonl").write_text(
            "\n".join(json.dumps(row) for row in [
                {"epoch": 1, "loss": 1.5, "map": 0.01, "map50": 0.02, "seconds": 480.0},
                {"epoch": 2, "loss": 1.2, "map": 0.05, "map50": 0.09, "seconds": 470.0},
            ]) + "\n", encoding="utf-8")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        build_fixture(self.directory)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self):
        return build_document(self.directory, generated_at=STAMP)

    def test_numbers_come_from_the_artifacts(self):
        document, payload = self.build()
        self.assertIn("0.111111", document)      # trainer receipt
        self.assertIn("0.123457", document)      # independent ONNX report
        self.assertIn("100", document)           # fire_smoke test images
        self.assertIn("101", document)           # crop_pest test images
        self.assertIn("2026-01-01T00:00:00", document)
        self.assertEqual([item["task"] for item in payload["tasks"]], ["fire_smoke", "crop_pest"])
        self.assertEqual(payload["gaps"], [])
        self.assertEqual(payload["tasks"][0]["best_epoch"], 2)
        self.assertEqual(payload["tasks"][0]["best_val_map"], 0.05)

    def test_independent_implementations_are_compared_not_merged(self):
        _, payload = self.build()
        item = payload["tasks"][0]
        self.assertAlmostEqual(item["map_absolute_delta"], abs(0.111111 - 0.123457), places=6)
        self.assertEqual(item["parity_warning"], item["map_absolute_delta"] > PARITY_TOLERANCE)
        document, _ = self.build()
        self.assertIn("训练器自测 mAP", document)
        self.assertIn("独立 ONNX 复评 mAP", document)

    def test_agreement_below_tolerance_is_not_a_warning(self):
        build_fixture(self.directory, trainer_map=0.100000, independent_map=0.100400)
        _, payload = self.build()
        self.assertLess(payload["tasks"][0]["map_absolute_delta"], PARITY_TOLERANCE)
        self.assertFalse(payload["tasks"][0]["parity_warning"])

    def test_missing_artifact_is_disclosed_never_guessed(self):
        (self.directory / "fire_smoke_onnx_eval.json").unlink()
        document, payload = self.build()
        self.assertIn("fire_smoke:missing:fire_smoke_onnx_eval.json", payload["gaps"])
        self.assertIsNone(payload["tasks"][0]["independent_onnx_map"])
        self.assertIn(MISSING, document)
        # the surviving trainer receipt must still be reported
        self.assertIn("0.111111", document)
        self.assertEqual(payload["tasks"][1]["independent_onnx_map"], 0.123457)

    def test_missing_manifest_does_not_invent_split_counts(self):
        (self.directory / "crop_pest_manifest.json").unlink()
        document, payload = self.build()
        self.assertIn("crop_pest:missing:crop_pest_manifest.json", payload["gaps"])
        self.assertIsNone(payload["tasks"][1]["class_coverage_policy"])
        self.assertIn("无法给出划分计数", document)

    def test_output_is_deterministic(self):
        first, first_payload = self.build()
        second, second_payload = self.build()
        self.assertEqual(first, second)
        self.assertEqual(first_payload, second_payload)

    def test_limitations_report_missing_classes_from_the_manifest(self):
        document, _ = self.build()
        self.assertIn("`strict_all_classes_observed`", document)
        self.assertIn("[2]", document)
        self.assertIn("无法训练、也无法评估", document)


class AbsentArtifactRenderingTests(unittest.TestCase):
    """A missing input must render as the missing marker, never as a plausible value.

    Found by generating the report from real (partially complete) artifacts: with no
    manifest, "类别数" rendered as `0`, "标注格式" as `{}` and the source-group audit as
    a quoted marker string, so an absent input looked like a measured zero.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        build_fixture(self.directory)

    def tearDown(self):
        self._tmp.cleanup()

    def drop(self, *names):
        for name in names:
            (self.directory / name).unlink()

    def test_missing_manifest_renders_marker_instead_of_zero(self):
        self.drop("crop_pest_manifest.json")
        document, _ = build_document(self.directory, STAMP)
        self.assertNotIn("| 类别数 | 0 |", document)
        self.assertIn(f"| 类别数 | {MISSING} |", document)
        self.assertIn(f"| 标注格式 | {MISSING} |", document)
        self.assertIn(f"| 跨划分同源分组（应空） | {MISSING} |", document)

    def test_missing_evidence_is_not_rendered_as_an_empty_json_object(self):
        self.drop("crop_pest_fcos_metadata.json", "crop_pest_onnx_eval.json")
        document, _ = build_document(self.directory, STAMP)
        self.assertIn(f"| torch / onnxruntime | {MISSING} |", document)
        self.assertIn(f"| 执行提供者 | {MISSING} |", document)
        self.assertIn(f"| 预处理一致性（与训练器数据集类逐元素比较） | {MISSING} |", document)
        self.assertNotIn("| 可评估类别数 | 0 |", document)
        self.assertIn(f"| 可评估类别数 | {MISSING} |", document)
        self.assertIn(f"| 不可评估类别数 | {MISSING} |", document)

    def test_missing_marker_is_never_wrapped_in_quotes(self):
        self.drop("crop_pest_manifest.json", "crop_pest_onnx_eval.json")
        document, _ = build_document(self.directory, STAMP)
        self.assertNotIn(f'"{MISSING}"', document)

    def test_limitations_do_not_claim_a_policy_when_the_manifest_is_absent(self):
        self.drop("crop_pest_manifest.json")
        document, _ = build_document(self.directory, STAMP)
        self.assertNotIn("`None`", document)
        self.assertIn(f"类别覆盖策略：{MISSING}", document)

    def test_sidecar_carries_parity_warnings(self):
        # the fixture's trainer/independent maps differ by ~0.0123 > PARITY_TOLERANCE,
        # so both tasks must be flagged; the point is that the sidecar reports it.
        _, payload = build_document(self.directory, STAMP)
        self.assertEqual(payload["parity_warnings"], [spec["task"] for spec in TASKS])
        self.assertEqual(payload["parity_tolerance"], PARITY_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
