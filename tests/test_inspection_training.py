"""CPU-only pure-function tests: no torch import, GPU, training or network."""
from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "gpu" / "train_inspection_fcos.py"
spec = importlib.util.spec_from_file_location("inspection_training_under_test", SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def sample(name, group, labels=None):
    labels = [1] if labels is None else labels
    return {"image": str((SCRIPT.parent / f"{name}.jpg").resolve()), "source_group": group,
            "boxes": [[0, 0, 10, 10] for _ in labels], "labels": labels}


def manifest():
    return {"classes": ["fire", "smoke"], "train": [sample("a", "train-scene")],
            "val": [sample("b", "val-scene")], "test": []}


class ParityHelperTests(unittest.TestCase):
    """The ONNX parity verdict rests on these two helpers, so pin their semantics.

    The end-to-end effect: a real export defect must fail the verdict, while a near-tied
    score field -- which makes the top-k selection non-unique -- must not be reported as
    the model disagreeing with itself.
    """

    def test_score_separation_uses_nearest_neighbour(self):
        sep = training.score_separation([0.50, 0.40, 0.10])
        self.assertEqual(len(sep), 3)
        self.assertAlmostEqual(sep[0], 0.10)
        self.assertAlmostEqual(sep[1], 0.10)
        self.assertAlmostEqual(sep[2], 0.30)
        self.assertTrue(all(value > 0 for value in sep))
        self.assertEqual(len(training.score_separation([0.5])), 1)

    def test_match_is_independent_of_order(self):
        candidate = [[10, 10, 20, 20], [100, 100, 110, 110]]
        reference = [[100, 100, 110, 110], [10, 10, 20, 20]]
        pairs = training.match_detections_by_box(candidate, [1, 1], reference, [1, 1], 1.0)
        self.assertEqual(pairs, [(0, 1), (1, 0)])

    def test_match_respects_class_and_tolerance(self):
        boxes = [[10, 10, 20, 20]]
        self.assertEqual(training.match_detections_by_box(boxes, [2], boxes, [1], 1.0), [])
        self.assertEqual(training.match_detections_by_box(boxes, [1], [[90, 90, 99, 99]], [1], 1.0), [])
        self.assertEqual(training.match_detections_by_box(boxes, [1], [[10.4, 10.4, 20, 20]], [1], 1.0), [(0, 0)])

    def test_each_reference_is_used_once(self):
        candidate = [[10, 10, 20, 20], [10.2, 10.2, 20, 20]]
        reference = [[10, 10, 20, 20]]
        self.assertEqual(training.match_detections_by_box(candidate, [1, 1], reference, [1], 1.0), [(0, 0)])

    def test_mismatched_lengths_are_rejected(self):
        boxes = [[10, 10, 20, 20]]
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            training.match_detections_by_box(boxes, [1, 1], boxes, [1], 1.0)
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            training.match_detections_by_box(boxes, [1], boxes, [1, 1], 1.0)


class ManifestTests(unittest.TestCase):
    def test_valid_and_empty_annotations(self):
        data = manifest()
        data["train"].append(sample("negative", "train-scene", []))
        validated = training.validate_manifest(data, check_files=False)
        self.assertEqual(training.split_statistics(validated["train"], validated["classes"]),
                         {"images": 2, "empty_images": 1, "source_groups": 1,
                          "objects": {"fire": 1, "smoke": 0}})
        self.assertEqual(validated["train"][1]["boxes"], [])

    def test_cross_split_group_leakage_before_limits(self):
        data = manifest()
        data["val"].append(sample("leaked", "train-scene"))
        with self.assertRaisesRegex(ValueError, "source_group leakage"):
            training.validate_manifest(data, check_files=False)

    def test_duplicate_image(self):
        data = manifest()
        data["val"][0]["image"] = data["train"][0]["image"]
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            training.validate_manifest(data, check_files=False)

    def test_bad_classes(self):
        for classes in ([], ["x", "x"], [""], [1], "fire"):
            data = manifest()
            data["classes"] = classes
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                training.validate_manifest(data, check_files=False)

    def test_bad_labels_boxes_and_paths(self):
        variants = ({"labels": [0]}, {"labels": [3]}, {"labels": [True]},
                    {"labels": [1.0]}, {"boxes": []}, {"boxes": [[0, 0, 0, 3]]},
                    {"boxes": [[-1, 0, 4, 3]]}, {"boxes": [[0, 0, float("nan"), 3]]},
                    {"image": "relative.jpg"}, {"source_group": ""})
        for update in variants:
            data = manifest()
            data["train"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                training.validate_manifest(data, check_files=False)

    def test_missing_files_and_empty_train(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            training.validate_manifest(manifest())
        data = manifest()
        data["train"] = []
        with self.assertRaises(ValueError):
            training.validate_manifest(data, check_files=False)

    def test_preparation_failure_report_is_named(self):
        """A failure report at the --output path must not read as a malformed manifest."""
        for errors, expected in (([{"message": "line 1: box extends beyond image"}], "box extends beyond image"),
                                 ([], "no error details recorded"),
                                 (["plain string error"], "plain string error")):
            data = {"status": "error", "error_count": 340, "usable_for_training": False, "errors": errors}
            with self.subTest(errors=errors), self.assertRaisesRegex(ValueError, "preparation failure report") as ctx:
                training.validate_manifest(data, check_files=False)
            self.assertIn(expected, str(ctx.exception))
            self.assertIn("error_count=340", str(ctx.exception))

    def test_seeded_limit_not_mutating_or_dropping_negatives(self):
        items = [sample(str(i), "scene", [] if i % 2 else [1]) for i in range(10)]
        original = copy.deepcopy(items)
        self.assertEqual(training.select_samples(items, 3, 7), training.select_samples(items, 3, 7))
        self.assertEqual(len(training.select_samples(items, 1, 7)), 1)
        self.assertEqual(training.select_samples(items, 0, 7), items)
        self.assertEqual(training.select_samples(items, 100, 7), items)
        self.assertEqual(items, original)
        self.assertEqual(training.select_samples([items[1]], 1, 7)[0]["boxes"], [])
        with self.assertRaises(ValueError):
            training.select_samples(items, -1, 7)


class MatchingTests(unittest.TestCase):
    def match(self, boxes, labels, scores, gt_boxes=None, gt_labels=None):
        return training.greedy_counts({"boxes": boxes, "labels": labels, "scores": scores},
                                      {"boxes": [[0, 0, 10, 10]] if gt_boxes is None else gt_boxes,
                                       "labels": [1] if gt_labels is None else gt_labels}, 2)

    def test_wrong_class_cannot_match(self):
        result = self.match([[0, 0, 10, 10]], [2], [.9])
        self.assertEqual(result[1], {"gt": 1, "tp": 0, "fp": 0})
        self.assertEqual(result[2], {"gt": 0, "tp": 0, "fp": 1})

    def test_duplicates_are_false_positives(self):
        result = self.match([[0, 0, 10, 10]] * 2, [1, 1], [.4, .9])
        self.assertEqual(result[1], {"gt": 1, "tp": 1, "fp": 1})

    def test_score_order_precedes_greedy_iou(self):
        # High-confidence ambiguous prediction consumes GT0; lower prediction
        # cannot reuse it. This distinguishes score-order from GT-order matching.
        result = self.match([[0, 0, 15, 10], [0, 0, 10, 10]], [1, 1], [.9, .8],
                            [[0, 0, 10, 10], [8, 0, 18, 10]], [1, 1])
        self.assertEqual(result[1], {"gt": 2, "tp": 1, "fp": 1})

    def test_thresholds_inclusive(self):
        result = self.match([[0, 0, 5, 10], [0, 0, 10, 10]], [1, 1], [.35, .349])
        self.assertEqual(result[1], {"gt": 1, "tp": 1, "fp": 0})

    def test_same_class_matching_ignores_closer_wrong_class(self):
        result = self.match([[0, 0, 10, 10]], [1], [.9],
                            [[0, 0, 10, 10], [0, 0, 9, 10]], [2, 1])
        self.assertEqual(result[1]["tp"], 1)
        self.assertEqual(result[2]["tp"], 0)

    def test_empty_gt_null_rates_preserve_false_positives(self):
        counts = self.match([[0, 0, 10, 10]], [1], [.9], [], [])
        result = training.summarize_counts(counts, ["fire", "smoke"], 1)
        self.assertEqual(result["overall"]["fp"], 1)
        self.assertIsNone(result["overall"]["recall"])
        self.assertIsNone(result["overall"]["false_positive_fraction"])
        self.assertIsNone(result["per_class"]["smoke"]["false_alarms_per_image"])

    def test_missed_gt_and_rates(self):
        result = training.summarize_counts({1: {"gt": 2, "tp": 1, "fp": 1},
                                           2: {"gt": 0, "tp": 0, "fp": 0}}, ["fire", "smoke"], 4)
        self.assertEqual(result["overall"]["recall"], .5)
        self.assertEqual(result["overall"]["false_positive_fraction"], .5)
        self.assertEqual(result["overall"]["false_alarms_per_image"], .25)
        self.assertEqual(result["overall"]["fn"], 1)

    def test_iou(self):
        self.assertEqual(training.box_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(training.box_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertEqual(training.box_iou_xyxy([0, 0, 0, 0], [0, 0, 0, 0]), 0)


class ResumeTests(unittest.TestCase):
    def test_coco_backbone_freezing_policy(self):
        for name in ("layer2.0.conv1.weight", "layer3.1.conv2.weight", "layer4.0.downsample.0.weight"):
            self.assertTrue(training.backbone_parameter_trainable(name))
        for name in ("conv1.weight", "bn1.weight", "layer1.0.conv1.weight"):
            self.assertFalse(training.backbone_parameter_trainable(name))


class GeometryTests(unittest.TestCase):
    def test_rounded_actual_xy_scaling(self):
        geometry = training.letterbox_geometry(7, 3, 10)
        self.assertEqual(geometry[:4], (10, 4, 0, 3))
        self.assertEqual(training.transform_boxes([[0, 0, 7, 3]], 7, 3, 10), [[0, 3, 10, 7]])

    def test_flip_and_empty_boxes(self):
        self.assertEqual(training.transform_boxes([[0, 0, 4, 4]], 8, 4, 8, True), [[4, 2, 8, 6]])
        self.assertEqual(training.transform_boxes([], 8, 4, 8, True), [])

    def test_out_of_bounds_and_zero_size(self):
        with self.assertRaises(ValueError):
            training.transform_boxes([[0, 0, 9, 4]], 8, 4, 8)
        with self.assertRaises(ValueError):
            training.letterbox_geometry(0, 4, 8)


if __name__ == "__main__":
    unittest.main()
