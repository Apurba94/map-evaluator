"""Compare all twelve COCO numbers with pycocotools. Skipped if it is not installed."""
import contextlib
import io
import random
import unittest

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None

from map_evaluator import Evaluator

KEYS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")


def dataset(rng, images=30, cats=3):
    anns, results, gt, dt = [], [], {}, {}
    for i in range(1, images + 1):
        gt[i], dt[i] = [], []
        for _ in range(rng.randint(0, 8)):
            scale = rng.choice([15, 60, 180])
            w, h = rng.randint(5, scale), rng.randint(5, scale)
            x, y = rng.randint(0, 500), rng.randint(0, 500)
            c = rng.randint(1, cats)
            crowd = int(rng.random() < 0.1)
            box = [x, y, w, h]
            area = float(w * h)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c, "bbox": box, "area": area, "iscrowd": crowd})
            gt[i].append({"bbox": box, "category": c, "iscrowd": crowd, "area": area})
            for _ in range(rng.choice([0, 1, 1, 1, 2])):
                d = [x + rng.randint(-6, 6), y + rng.randint(-6, 6), max(1, w + rng.randint(-6, 6)), max(1, h + rng.randint(-6, 6))]
                cat = c if rng.random() < 0.9 else rng.randint(1, cats)
                score = round(rng.random(), 5)
                results.append({"image_id": i, "category_id": cat, "bbox": d, "score": score})
                dt[i].append({"bbox": d, "category": cat, "score": score})
        for _ in range(rng.randint(0, 3)):
            d = [rng.randint(0, 500), rng.randint(0, 500), rng.randint(5, 120), rng.randint(5, 120)]
            cat = rng.randint(1, cats)
            score = round(rng.random(), 5)
            results.append({"image_id": i, "category_id": cat, "bbox": d, "score": score})
            dt[i].append({"bbox": d, "category": cat, "score": score})
    coco = {"images": [{"id": i} for i in range(1, images + 1)], "categories": [{"id": c} for c in range(1, cats + 1)], "annotations": anns}
    return coco, results, gt, dt


def reference(coco_dict, results):
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = coco_dict
        c.createIndex()
        e = COCOeval(c, c.loadRes(results), "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    return dict(zip(KEYS, e.stats))


@unittest.skipIf(COCO is None, "pycocotools is not installed")
class TestSameNumbers(unittest.TestCase):
    def test_random_datasets(self):
        rng = random.Random(11)
        for round_ in range(25):
            coco, results, gt, dt = dataset(rng)
            want = reference(coco, results)
            got = Evaluator(gt, dt, categories=[1, 2, 3]).evaluate()
            for k in KEYS:
                self.assertAlmostEqual(got[k], want[k], places=9, msg=f"{k} in round {round_}")


if __name__ == "__main__":
    unittest.main()
