import json
import os
import random
import tempfile
import unittest

import numpy as np

from map_evaluator import Evaluator, iou_matrix, load_coco


def gt(*boxes, cat="a", **kw):
    return [dict(bbox=list(b), category=cat, **kw) for b in boxes]


def dt(*items, cat="a"):
    return [dict(bbox=list(b), score=s, category=cat) for b, s in items]


BOX = (10, 10, 100, 100)


def naive_ap(gts, dts, iou_thr):
    """An independent, slow AP at one threshold for one class, with no ignored boxes."""
    n = sum(len(v) for v in gts.values())
    flat = sorted(((d["score"], i, img, d) for img, ds in dts.items() for i, d in enumerate(ds)), key=lambda x: -x[0])
    per_image = {img: sorted(ds, key=lambda d: -d["score"]) for img, ds in dts.items()}
    outcome = {}
    for img, ds in per_image.items():
        taken = set()
        for d in ds:
            best, who = min(iou_thr, 1 - 1e-10), -1
            for j, g in enumerate(gts.get(img, [])):
                if j in taken:
                    continue
                v = float(iou_matrix([d["bbox"]], [g["bbox"]])[0, 0])
                if v >= best:
                    best, who = v, j
            if who >= 0:
                taken.add(who)
            outcome[id(d)] = who >= 0
    tp = fp = 0
    rec, pre = [], []
    for s, _, img, d in sorted(flat, key=lambda x: -x[0]):
        if outcome[id(d)]:
            tp += 1
        else:
            fp += 1
        rec.append(tp / n)
        pre.append(tp / (tp + fp))
    total = 0.0
    for r in np.linspace(0, 1, 101):
        total += max([p for p, q in zip(pre, rec) if q >= r], default=0.0)
    return total / 101


class TestIou(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0], 1)
        self.assertAlmostEqual(iou_matrix([[0, 0, 10, 10]], [[5, 0, 10, 10]])[0, 0], 50 / 150)
        self.assertEqual(iou_matrix([[0, 0, 10, 10]], [[20, 20, 5, 5]])[0, 0], 0)

    def test_crowd_uses_detection_area(self):
        self.assertAlmostEqual(iou_matrix([[0, 0, 10, 10]], [[0, 0, 100, 100]], [True])[0, 0], 1)

    def test_empty(self):
        self.assertEqual(iou_matrix([], [[0, 0, 1, 1]]).shape, (0, 1))

    def test_zero_area_gives_zero_not_nan(self):
        self.assertEqual(iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0], 0)


class TestAp(unittest.TestCase):
    def test_perfect(self):
        r = Evaluator({1: gt(BOX)}, {1: dt((BOX, 0.9))}).evaluate()
        self.assertAlmostEqual(r["AP"], 1)
        self.assertAlmostEqual(r["AR100"], 1)

    def test_no_detections(self):
        self.assertEqual(Evaluator({1: gt(BOX)}, {}).evaluate()["AP"], 0)

    def test_missing_category_is_minus_one(self):
        self.assertEqual(Evaluator({}, {}).evaluate()["AP"], -1)

    def test_iou_just_below_threshold_fails(self):
        d = dt(((10, 10, 100, 60), 0.9))         # IoU 0.6
        r = Evaluator({1: gt(BOX)}, {1: d})
        self.assertAlmostEqual(r.mean(r.ap(0.6)), 1)
        self.assertAlmostEqual(r.mean(r.ap(0.65)), 0)
        self.assertAlmostEqual(r.evaluate()["AP"], 0.3)       # passes 0.50, 0.55 and 0.60 of the ten thresholds

    def test_false_positive_with_higher_score_hurts(self):
        far = (500, 500, 50, 50)
        hi = Evaluator({1: gt(BOX)}, {1: dt((far, 0.9), (BOX, 0.5))}).evaluate()["AP50"]
        lo = Evaluator({1: gt(BOX)}, {1: dt((far, 0.1), (BOX, 0.5))}).evaluate()["AP50"]
        self.assertAlmostEqual(lo, 1)
        self.assertAlmostEqual(hi, 0.5)

    def test_duplicate_counts_against_you(self):
        r = Evaluator({1: gt(BOX, (300, 300, 50, 50))}, {1: dt((BOX, 0.9), (BOX, 0.8), ((300, 300, 50, 50), 0.7))})
        self.assertAlmostEqual(r.evaluate()["AP50"], (51 + 50 * 2 / 3) / 101, places=6)

    def test_interpolation_uses_envelope(self):
        a, b, c = (0, 0, 50, 50), (100, 0, 50, 50), (500, 500, 9, 9)
        r = Evaluator({1: gt(a, b)}, {1: dt((a, 0.9), (c, 0.8), (b, 0.7))}).evaluate()
        self.assertAlmostEqual(r["AP50"], (51 * 1 + 50 * 2 / 3) / 101, places=6)

    def test_mean_over_classes(self):
        g = {1: gt(BOX, cat="a") + gt((300, 300, 60, 60), cat="b")}
        d = {1: dt((BOX, 0.9), cat="a")}
        r = Evaluator(g, d).evaluate()
        self.assertAlmostEqual(r["AP50"], 0.5)
        self.assertEqual(r["per_class"], {"a": 1.0, "b": 0.0})

    def test_detection_of_wrong_class_is_not_a_match(self):
        self.assertEqual(Evaluator({1: gt(BOX, cat="a")}, {1: dt((BOX, 0.9), cat="b")}).evaluate()["per_class"]["a"], 0)

    def test_max_detections(self):
        g = gt(*[(i * 60, 0, 50, 50) for i in range(5)])
        d = dt(*[((i * 60, 0, 50, 50), 0.9 - i / 10) for i in range(5)])
        r = Evaluator({1: g}, {1: d}).evaluate()
        self.assertAlmostEqual(r["AR1"], 0.2)
        self.assertAlmostEqual(r["AR100"], 1)

    def test_against_naive_implementation(self):
        rng = random.Random(1)
        for _ in range(40):
            gts, dts = {}, {}
            for img in range(5):
                boxes = [(rng.randint(0, 300), rng.randint(0, 300), rng.randint(20, 80), rng.randint(20, 80)) for _ in range(rng.randint(0, 6))]
                gts[img] = gt(*boxes)
                ds = []
                for b in boxes:
                    if rng.random() < 0.8:
                        j = [v + rng.randint(-8, 8) for v in b[:2]] + list(b[2:])
                        ds.append((tuple(j), round(rng.random(), 4)))
                for _ in range(rng.randint(0, 3)):
                    ds.append(((rng.randint(0, 300), rng.randint(0, 300), 40, 40), round(rng.random(), 4)))
                dts[img] = dt(*ds)
            if not any(gts.values()):
                continue
            e = Evaluator(gts, dts, thresholds=[0.5, 0.75])
            for t in (0.5, 0.75):
                self.assertAlmostEqual(e.mean(e.ap(t)), naive_ap(gts, dts, t), places=8)


class TestIgnored(unittest.TestCase):
    def test_detection_on_crowd_is_neither_right_nor_wrong(self):
        g = gt(BOX) + gt((300, 300, 200, 200), iscrowd=1)
        with_d = Evaluator({1: g}, {1: dt(((320, 320, 50, 50), 0.99), (BOX, 0.5))}).evaluate()
        self.assertAlmostEqual(with_d["AP50"], 1)

    def test_crowd_can_absorb_many_detections(self):
        g = gt(BOX) + gt((300, 300, 200, 200), iscrowd=1)
        d = dt(((320, 320, 50, 50), 0.99), ((340, 340, 50, 50), 0.98), (BOX, 0.5))
        self.assertAlmostEqual(Evaluator({1: g}, {1: d}).evaluate()["AP50"], 1)

    def test_only_crowd_means_not_evaluated(self):
        self.assertEqual(Evaluator({1: gt(BOX, iscrowd=1)}, {}).evaluate()["AP"], -1)

    def test_size_ranges(self):
        small, large = (0, 0, 20, 20), (100, 100, 200, 200)
        e = Evaluator({1: gt(small, large)}, {1: dt((small, 0.9))}).evaluate()
        self.assertAlmostEqual(e["APs"], 1)
        self.assertAlmostEqual(e["APl"], 0)
        self.assertEqual(e["APm"], -1)

    def test_area_field_is_used_for_size(self):
        g = [dict(bbox=[0, 0, 100, 100], category="a", area=100.0)]
        self.assertAlmostEqual(Evaluator({1: g}, {1: dt(((0, 0, 100, 100), 0.9))}).evaluate()["APs"], 1)

    def test_unmatched_detection_outside_size_range_is_ignored(self):
        large = (0, 0, 200, 200)
        g = gt(large)
        d = dt(((500, 500, 10, 10), 0.99), (large, 0.5))
        self.assertAlmostEqual(Evaluator({1: g}, {1: d}).evaluate()["APl"], 1)


class TestErrors(unittest.TestCase):
    def test_each_kind(self):
        a = (0, 0, 100, 100)
        g = {1: gt(a, cat="a")}
        cases = [
            (dt((a, 0.9), cat="a"), "tp"),
            (dt((a, 0.9), cat="b"), "cls"),
            (dt(((50, 0, 100, 100), 0.9), cat="a"), "loc"),
            (dt(((50, 0, 100, 100), 0.9), cat="b"), "both"),
            (dt(((900, 900, 10, 10), 0.9), cat="a"), "bkg"),
        ]
        for d, kind in cases:
            e = Evaluator(g, {1: d}, categories=["a", "b"]).errors()
            self.assertEqual(e[kind], 1, kind)

    def test_duplicate_and_miss(self):
        a, b = (0, 0, 100, 100), (300, 0, 100, 100)
        e = Evaluator({1: gt(a, b)}, {1: dt((a, 0.9), (a, 0.8))}).errors()
        self.assertEqual((e["tp"], e["dupe"], e["miss"]), (1, 1, 1))

    def test_confusion(self):
        a, b = (0, 0, 100, 100), (300, 0, 100, 100)
        g = {1: gt(a, cat="x") + gt(b, cat="y")}
        d = {1: dt((a, 0.9), cat="y") + dt(((900, 0, 50, 50), 0.9), cat="x")}
        names, m = Evaluator(g, d).confusion()
        self.assertEqual(names, ["x", "y", "background"])
        self.assertEqual(m.tolist(), [[0, 1, 0], [0, 0, 1], [1, 0, 0]])


class TestFiles(unittest.TestCase):
    def test_coco_files(self):
        d = tempfile.mkdtemp()
        a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        with open(a, "w") as f:
            json.dump({"images": [{"id": 1}, {"id": 2}], "categories": [{"id": 3, "name": "cell"}], "annotations": [{"image_id": 1, "category_id": 3, "bbox": [0, 0, 50, 50], "area": 2500, "iscrowd": 0}]}, f)
        with open(b, "w") as f:
            json.dump([{"image_id": 1, "category_id": 3, "bbox": [0, 0, 50, 50], "score": 0.9}], f)
        r = Evaluator(*load_coco(a, b)).evaluate()
        self.assertAlmostEqual(r["AP"], 1)
        self.assertEqual(list(r["per_class"]), ["cell"])

    def test_pr_curve_shape(self):
        e = Evaluator({1: gt(BOX)}, {1: dt((BOX, 0.9))})
        r, p = e.pr_curve("a")
        self.assertEqual((len(r), p.min()), (101, 1.0))
        self.assertTrue(np.all(np.diff(e.pr_curve("a")[1]) <= 0))


if __name__ == "__main__":
    unittest.main()
