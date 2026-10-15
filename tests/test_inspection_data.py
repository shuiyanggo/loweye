"""Local synthetic CPU tests; no real dataset, GPU, SSH or download."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_inspection_data.py"
SPEC = importlib.util.spec_from_file_location("inspection_data", SCRIPT)
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def item(name, sha, group=None):
    return {"image": name, "sha256": sha, "source_group": group or sha,
            "boxes": [[0, 0, 10, 10]], "labels": [1]}


class PureTests(unittest.TestCase):
    def test_yolo_pixels_and_one_based(self):
        boxes, labels = DATA.convert_yolo_labels("0 .5 .5 .2 .4\n1 .5 .5 1 1\n", 200, 100, 2)
        self.assertEqual(boxes, [[80., 30., 120., 70.], [0., 0., 200., 100.]])
        self.assertEqual(labels, [1, 2])
        self.assertEqual(DATA.convert_yolo_labels(" \n", 100, 100, 2), ([], []))

    def test_yolo_rejects_bad_labels_and_boxes(self):
        invalid = ["2 .5 .5 .1 .1", "-1 .5 .5 .1 .1", "1.0 .5 .5 .1 .1", "0 .5 .5 0 .2",
                   "0 nan .5 .2 .2", "0 .5 .5 inf .2", "0 1.1 .5 .2 .2", "0 .1 .1 .5 .5", "0 .5 .5 .2"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                DATA.convert_yolo_labels(text, 100, 100, 2)

    def test_voc_official_index_and_inclusive_pixels(self):
        text = '<annotation><size><width>10</width><height>20</height></size><object><name>102</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>'
        self.assertEqual(DATA.convert_voc_labels(text, 10, 20, [str(i) for i in range(102)]),
                         ([[0., 1., 10., 20.]], [102]))
        with self.assertRaises(ValueError):
            DATA.convert_voc_labels(text.replace('<name>102', '<name>103'), 10, 20, [str(i) for i in range(102)])
        with self.assertRaises(ValueError):
            DATA.convert_voc_labels(text, 11, 20, [str(i) for i in range(102)])

    def test_dedup_priority_and_audit_pure(self):
        splits = {"train": [item("train_a", "a"), item("train_b", "b")],
                  "val": [item("val_a", "a")], "test": [item("test_a", "a")]}
        original = copy.deepcopy(splits)
        final, audit = DATA.deduplicate_splits(splits)
        self.assertEqual(splits, original)
        self.assertEqual([x["image"] for x in final["train"]], ["train_b"])
        self.assertEqual(final["val"], [])
        self.assertEqual(len(audit), 2)
        self.assertTrue(all(x["kept_split"] == "test" for x in audit))
        splits["train"][0]["labels"] = [2]
        _, audit = DATA.deduplicate_splits(splits)
        self.assertTrue(any(x["annotation_conflict"] for x in audit))

    def test_hash_group_partition_is_deterministic(self):
        samples = [item(str(i), str(i), f"group{i // 2}") for i in range(40)]
        original = copy.deepcopy(samples)
        splits = DATA.grouped_partition(samples, (.7, .15, .15))
        self.assertEqual(splits, DATA.grouped_partition(list(reversed(samples)), (.7, .15, .15)))
        self.assertEqual(samples, original)
        self.assertEqual(sum(map(len, splits)), 40)
        group_sets = [{i["source_group"] for i in s} for s in splits]
        self.assertTrue(all(not group_sets[a] & group_sets[b] for a in range(3) for b in range(a)))
        with self.assertRaises(ValueError):
            DATA.grouped_partition(samples[:2], (.7, .15, .15))

    def test_group_inference_conservative(self):
        root = Path(".").resolve()
        self.assertEqual(DATA.infer_source_group(root / "clip_frame0001.jpg", root, "abc"), "filename-video:clip")
        self.assertEqual(DATA.infer_source_group(root / "clip_0001.jpg", root, "abc"), "sha256:abc")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        # Explicit D: project-local temp area, never system C: temp.
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent / "tests")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, split, name, label="0 .5 .5 .5 .5", layout="images_first", color=0):
        if layout == "images_first":
            folder = self.root / "images" / split
            labels = self.root / "labels" / split
        else:
            folder = self.root / split / "images"
            labels = self.root / split / "labels"
        folder.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        image = folder / (name + ".png")
        Image.new("RGB", (20, 10), (color, 30, 40)).save(image)
        if label is not None:
            (labels / (name + ".txt")).write_text(label, encoding="utf-8")
        return image

    def yaml(self, names="[fire, smoke]", splits=""):
        (self.root / "data.yaml").write_text(f"names: {names}\nnc: 2\n" + splits, encoding="utf-8")

    def populate(self, splits=("train", "val", "test"), layout="images_first"):
        for k, split in enumerate(splits):
            for i in range(10):
                self.image(split, f"img{k}_{i}", f"{i % 2} .5 .5 .5 .5", layout, k * 50 + i)

    def test_yaml_order_official_split_and_negative(self):
        self.yaml()
        self.populate()
        self.image("train", "negative", "", color=200)
        manifest = DATA.build_manifest(self.root, "dfire")
        self.assertEqual(manifest["classes"], ["fire", "smoke"])
        self.assertEqual(manifest["statistics"]["train"]["empty_images"], 1)
        self.assertEqual(manifest["split_strategy"], "official split preserved")
        self.assertTrue(Path(manifest["train"][0]["image"]).is_absolute())
        self.assertTrue(manifest["limitations"])
        self.assertIn("license_reference", manifest["license"])

    def test_train_images_layout_train_test_preserves_test(self):
        self.populate(("train", "test"), "split_first")
        manifest = DATA.build_manifest(self.root, "dfire", ["smoke", "fire"])
        self.assertEqual(len(manifest["test"]), 10)
        self.assertEqual(len(manifest["train"]) + len(manifest["val"]), 10)
        self.assertTrue(all(i["original_split"] == "train" for i in manifest["val"]))

    def test_unsplit_fixed_seed(self):
        self.populate(("",))
        first = DATA.build_manifest(self.root, "dfire", ["smoke", "fire"])
        second = DATA.build_manifest(self.root, "dfire", ["smoke", "fire"])
        self.assertEqual(first, second)
        self.assertEqual([len(first[s]) for s in DATA.SPLITS], [7, 2, 1])
        self.assertIn("no official split", first["split_strategy"])

    def test_classes_no_guessing(self):
        self.populate()
        with self.assertRaisesRegex(DATA.DataValidationError, "explicit --classes"):
            DATA.build_manifest(self.root, "dfire")
        self.yaml()
        with self.assertRaisesRegex(DATA.DataValidationError, "conflicts"):
            DATA.build_manifest(self.root, "dfire", ["smoke", "fire"])

    def test_missing_corrupt_and_bad_annotations_reported(self):
        self.yaml()
        self.populate()
        missing = self.image("train", "missing", None, color=201)
        corrupt = self.image("val", "corrupt", color=202)
        corrupt.write_bytes(b"not an image")
        self.image("test", "bad_label", "2 .5 .5 .5 .5", color=203)
        with self.assertRaises(DATA.DataValidationError) as context:
            DATA.build_manifest(self.root, "dfire")
        self.assertEqual(len(context.exception.errors), 3)
        self.assertTrue(any(str(missing) in e for e in context.exception.errors))

    def test_cli_failure_writes_nontraining_error_report(self):
        self.populate()
        output = self.root / "manifest.json"
        self.assertEqual(DATA.main(["--root", str(self.root), "--dataset", "dfire", "--output", str(output)]), 2)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "error")

    def test_too_few_classes_and_empty_split_fail(self):
        self.yaml()
        for split in DATA.SPLITS:
            self.image(split, split, color=DATA.SPLITS.index(split))
        with self.assertRaisesRegex(DATA.DataValidationError, "too few observed classes"):
            DATA.build_manifest(self.root, "dfire")
        (self.root / "labels" / "test" / "test.txt").write_text("1 .5 .5 .5 .5", encoding="utf-8")
        # Exact cross-split image bytes remove train and val.
        (self.root / "images" / "train" / "train.png").write_bytes((self.root / "images" / "test" / "test.png").read_bytes())
        with self.assertRaisesRegex(DATA.DataValidationError, "contain samples"):
            DATA.build_manifest(self.root, "dfire")

    def test_ip102_numbered_classes_voc_and_imagesets(self):
        (self.root / "classes.txt").write_text("\n".join(f"{i + 1} pest {i + 1}" for i in range(102)), encoding="utf-8")
        images, annotations = self.root / "Images", self.root / "Annotations"
        images.mkdir(); annotations.mkdir()
        for i in range(102):
            stem = f"IP{i:09d}"
            Image.new("RGB", (20, 10), (i, 0, 0)).save(images / f"{stem}.png")
            (annotations / f"{stem}.xml").write_text(f'<annotation><object><name>{i + 1}</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>', encoding="utf-8")
        main = self.root / "ImageSets" / "Main"
        main.mkdir(parents=True)
        (main / "trainval.txt").write_text("\n".join(f"IP{i:09d}" for i in range(90)), encoding="utf-8")
        (main / "test.txt").write_text("\n".join(f"IP{i:09d}" for i in range(90, 102)), encoding="utf-8")
        manifest = DATA.build_manifest(self.root, "ip102")
        self.assertEqual(len(manifest["classes"]), 102)
        self.assertEqual(manifest["classes"][0], "pest 1")
        self.assertEqual(len(manifest["test"]), 12)
        self.assertEqual(manifest["annotation_formats"], {"VOC_1based_inclusive": 102})

    def test_standard_voc2007_jpegimages_layout(self):
        self.test_ip102_numbered_classes_voc_and_imagesets()
        # Nested standard VOC layout with mixed-case JPEGImages directory.
        voc = self.root / "VOC2007"
        voc.mkdir()
        (self.root / "Images").rename(voc / "JpEgImAgEs")
        (self.root / "Annotations").rename(voc / "Annotations")
        (self.root / "ImageSets").rename(voc / "ImageSets")
        manifest = DATA.build_manifest(self.root, "ip102")
        self.assertEqual(len(manifest["test"]), 12)
        self.assertEqual(sum(len(manifest[s]) for s in DATA.SPLITS), 102)
        self.assertTrue(all("JpEgImAgEs" in i["image"] for s in DATA.SPLITS for i in manifest[s]))
        self.assertTrue(all(i["original_split"] == "test" for i in manifest["test"]))
        self.assertEqual(manifest["annotation_formats"], {"VOC_1based_inclusive": 102})

    def test_voc_root_trainval_test_lists_preserve_official_test(self):
        self.test_ip102_numbered_classes_voc_and_imagesets()
        (self.root / "Images").rename(self.root / "JPEGImages")
        directory = self.root / "ImageSets" / "Main"
        (directory / "trainval.txt").rename(self.root / "trainval.txt")
        (directory / "test.txt").rename(self.root / "test.txt")
        manifest = DATA.build_manifest(self.root, "ip102")
        self.assertEqual(len(manifest["test"]), 12)
        self.assertEqual({Path(i["image"]).stem for i in manifest["test"]}, {f"IP{i:09d}" for i in range(90, 102)})
        self.assertTrue(all(i["original_split"] == "train" for i in manifest["val"]))
        self.assertIn("official test preserved", manifest["split_strategy"])
        # Unknown TXT in Annotations cannot silently substitute for VOC XML.
        annotation = self.root / "Annotations" / "IP000000101.xml"
        annotation.rename(annotation.with_suffix(".txt"))
        with self.assertRaisesRegex(DATA.DataValidationError, "Missing or ambiguous annotation"):
            DATA.build_manifest(self.root, "ip102")

    def test_ip102_requires_102_classes_and_zero_based_yolo(self):
        (self.root / "classes.txt").write_text("\n".join(f"pest{i}" for i in range(102)), encoding="utf-8")
        for i in range(102):
            self.image("", f"pest{i}", f"{i} .5 .5 .5 .5", color=i)
        manifest = DATA.build_manifest(self.root, "ip102")
        self.assertEqual({label for split in DATA.SPLITS for sample in manifest[split] for label in sample["labels"]}, set(range(1, 103)))
        (self.root / "labels" / "pest101.txt").write_text("102 .5 .5 .5 .5", encoding="utf-8")
        with self.assertRaisesRegex(DATA.DataValidationError, "outside 0..101"):
            DATA.build_manifest(self.root, "ip102")

    def test_group_leakage_audit_and_separate_nonofficial_cli_manifest(self):
        self.yaml(); self.populate()
        for k, split in enumerate(DATA.SPLITS):
            self.image(split, f"video_frame{k}", color=220 + k)
        output = self.root / "official.json"
        self.assertEqual(DATA.main(["--root", str(self.root), "--dataset", "dfire", "--output", str(output)]), 0)
        original = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(original["source_group_audit"]["cross_split_groups"]), 1)
        safe = json.loads((self.root / "official_group_safe_manifest.json").read_text(encoding="utf-8"))
        self.assertIn("NONOFFICIAL", safe["split_strategy"])
        self.assertTrue(safe["official_split_changes"])
        groups = [{i["source_group"] for i in safe[s]} for s in DATA.SPLITS]
        self.assertTrue(all(not groups[a] & groups[b] for a in range(3) for b in range(a)))

    def test_yaml_image_lists_override_folder_split(self):
        self.populate(("",))
        for split, indices in (("train", range(6)), ("val", range(6, 8)), ("test", range(8, 10))):
            (self.root / f"{split}.txt").write_text("\n".join(f"images/img0_{i}.png" for i in indices), encoding="utf-8")
        self.yaml(splits="train: train.txt\nval: val.txt\ntest: test.txt\n")
        manifest = DATA.build_manifest(self.root, "dfire")
        self.assertEqual([len(manifest[s]) for s in DATA.SPLITS], [6, 2, 2])


if __name__ == "__main__":
    unittest.main()
