"""COCO-style detection evaluation, written from the published definition.

Boxes are [x, y, width, height] in pixels. For each image and class, detections are
matched to ground truth in order of falling score. Crowd boxes and boxes outside the
size range being measured are "ignored": a detection on them counts neither way.
"""
import json
from collections import defaultdict

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)
RECALL_POINTS = np.linspace(0, 1, 101)
AREAS = {"all": (0, 1e10), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
MAX_DETS = (1, 10, 100)


def iou_matrix(dets, gts, crowd=None):
    """IoU of every detection with every ground-truth box. For a crowd box the union is the
    detection's own area, so a detection inside a crowd region scores high."""
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)))
    d = np.asarray(dets, float)
    g = np.asarray(gts, float)
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    w = np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, 0, None], g[None, :, 0])
    h = np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, 1, None], g[None, :, 1])
    inter = np.clip(w, 0, None) * np.clip(h, 0, None)
    da, ga = d[:, 2] * d[:, 3], g[:, 2] * g[:, 3]
    union = da[:, None] + ga[None] - inter
    if crowd is not None:
        c = np.asarray(crowd, bool)
        union = np.where(c[None], da[:, None], union)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match_image(dets, gts, thresholds, area_range, max_det):
    """Match one image and class. Returns arrays over thresholds (T) and detections (D):
    matched T×D, ignored T×D, plus scores and the number of ground truths that count."""
    dets = sorted(dets, key=lambda d: -d["score"])[:max_det]
    gt_ignore = np.array([bool(g.get("iscrowd")) or bool(g.get("ignore")) or not (area_range[0] <= g["area"] <= area_range[1]) for g in gts], bool)
    order = np.argsort(gt_ignore, kind="stable")          # real boxes are tried before ignored ones
    gts = [gts[i] for i in order]
    gt_ignore = gt_ignore[order]
    crowd = [bool(g.get("iscrowd")) for g in gts]
    ious = iou_matrix([d["bbox"] for d in dets], [g["bbox"] for g in gts], crowd)
    T, D, G = len(thresholds), len(dets), len(gts)
    matched = np.zeros((T, D), bool)
    ignored = np.zeros((T, D), bool)
    taken = np.zeros((T, G), bool)
    for t, thr in enumerate(thresholds):
        for d in range(D):
            best = min(thr, 1 - 1e-10)
            m = -1
            for g in range(G):
                if taken[t, g] and not crowd[g]:
                    continue
                if m > -1 and not gt_ignore[m] and gt_ignore[g]:
                    break                                 # a real match is never swapped for an ignored one
                if ious[d, g] < best:
                    continue
                best = ious[d, g]
                m = g
            if m > -1:
                taken[t, m] = True
                matched[t, d] = True
                ignored[t, d] = gt_ignore[m]
    areas = np.array([d["bbox"][2] * d["bbox"][3] for d in dets], float)
    outside = (areas < area_range[0]) | (areas > area_range[1])
    ignored = ignored | (~matched & outside[None])
    return matched, ignored, np.array([d["score"] for d in dets], float), int((~gt_ignore).sum())


def accumulate(records, thresholds):
    """Turn matches for one class and area into AP (over thresholds) and recall."""
    T = len(thresholds)
    records = [r for r in records if r is not None]
    npig = sum(r[3] for r in records)
    if npig == 0:
        return np.full(T, np.nan), np.full(T, np.nan)
    if not any(r[2].size for r in records):
        return np.zeros(T), np.zeros(T)
    scores = np.concatenate([r[2] for r in records])
    order = np.argsort(-scores, kind="mergesort")
    matched = np.concatenate([r[0] for r in records], axis=1)[:, order]
    ignored = np.concatenate([r[1] for r in records], axis=1)[:, order]
    tp = np.cumsum(matched & ~ignored, axis=1, dtype=float)
    fp = np.cumsum(~matched & ~ignored, axis=1, dtype=float)
    ap, recall = np.zeros(T), np.zeros(T)
    for t in range(T):
        rc = tp[t] / npig
        pr = tp[t] / np.maximum(tp[t] + fp[t], np.spacing(1))
        recall[t] = rc[-1] if rc.size else 0
        env = np.maximum.accumulate(pr[::-1])[::-1]       # precision can only fall as recall rises
        idx = np.searchsorted(rc, RECALL_POINTS, side="left")
        got = np.zeros(RECALL_POINTS.size)
        good = idx < env.size
        got[good] = env[idx[good]]
        ap[t] = got.mean()
    return ap, recall


class Evaluator:
    """gt: {image id: [{"bbox", "category", optional "iscrowd", "ignore", "area"}]}
    dt: {image id: [{"bbox", "category", "score"}]}"""

    def __init__(self, gt, dt, categories=None, thresholds=IOU_THRESHOLDS):
        self.thresholds = np.asarray(thresholds, float)
        self.images = sorted(set(gt) | set(dt), key=str)
        self.gt, self.dt = defaultdict(list), defaultdict(list)
        cats = set()
        for img, boxes in gt.items():
            for g in boxes:
                g = dict(g)
                g.setdefault("area", g["bbox"][2] * g["bbox"][3])
                self.gt[(img, g["category"])].append(g)
                cats.add(g["category"])
        for img, boxes in dt.items():
            for d in boxes:
                self.dt[(img, d["category"])].append(d)
                cats.add(d["category"])
        self.categories = sorted(categories if categories is not None else cats, key=str)
        self.cache = {}

    def evaluate(self):
        """Return the twelve standard COCO numbers, plus per-class AP."""
        out = {
            "AP": self.mean(self.ap()), "AP50": self.mean(self.ap(0.5)), "AP75": self.mean(self.ap(0.75)),
            "APs": self.mean(self.ap(area="small")), "APm": self.mean(self.ap(area="medium")), "APl": self.mean(self.ap(area="large")),
            "AR1": self.mean(self.ar(max_det=1)), "AR10": self.mean(self.ar(max_det=10)), "AR100": self.mean(self.ar()),
            "ARs": self.mean(self.ar(area="small")), "ARm": self.mean(self.ar(area="medium")), "ARl": self.mean(self.ar(area="large")),
        }
        out["per_class"] = {c: self.mean(self.ap(cat=c)) for c in self.categories}
        return out

    @staticmethod
    def mean(values):
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else -1.0

    def table(self, cat, area, max_det):
        key = (cat, area, max_det)
        if key not in self.cache:
            rec = []
            for img in self.images:
                g, d = self.gt.get((img, cat), []), self.dt.get((img, cat), [])
                if g or d:
                    rec.append(match_image(d, g, self.thresholds, AREAS[area], max_det))
            self.cache[key] = accumulate(rec, self.thresholds)
        return self.cache[key]

    def ap(self, iou=None, area="all", max_det=100, cat=None):
        cats = self.categories if cat is None else [cat]
        rows = np.array([self.table(c, area, max_det)[0] for c in cats]).reshape(len(cats), len(self.thresholds))
        return rows if iou is None else rows[:, self._t(iou)]

    def ar(self, iou=None, area="all", max_det=100, cat=None):
        cats = self.categories if cat is None else [cat]
        rows = np.array([self.table(c, area, max_det)[1] for c in cats]).reshape(len(cats), len(self.thresholds))
        return rows if iou is None else rows[:, self._t(iou)]

    def _t(self, iou):
        return int(np.argmin(np.abs(self.thresholds - iou)))

    def pr_curve(self, cat, iou=0.5):
        """Precision at each of the 101 recall points, for plotting."""
        rec = []
        for img in self.images:
            g, d = self.gt.get((img, cat), []), self.dt.get((img, cat), [])
            if g or d:
                rec.append(match_image(d, g, np.array([iou]), AREAS["all"], 100))
        rec = [r for r in rec if r[2].size]
        n = sum(r[3] for r in [match_image(self.dt.get((i, cat), []), self.gt.get((i, cat), []), np.array([iou]), AREAS["all"], 100) for i in self.images])
        if not rec or n == 0:
            return RECALL_POINTS, np.zeros(101)
        scores = np.concatenate([r[2] for r in rec])
        order = np.argsort(-scores, kind="mergesort")
        m = np.concatenate([r[0] for r in rec], axis=1)[0, order]
        ig = np.concatenate([r[1] for r in rec], axis=1)[0, order]
        tp, fp = np.cumsum(m & ~ig, dtype=float), np.cumsum(~m & ~ig, dtype=float)
        rc = tp / n
        env = np.maximum.accumulate((tp / np.maximum(tp + fp, 1e-12))[::-1])[::-1]
        idx = np.searchsorted(rc, RECALL_POINTS, side="left")
        p = np.zeros(101)
        good = idx < env.size
        p[good] = env[idx[good]]
        return RECALL_POINTS, p

    def errors(self, foreground=0.5, background=0.1):
        """Explain each false positive and missed box, in the spirit of the TIDE toolbox.

        cls: right place, wrong class. loc: right class, poor overlap. both: wrong class and
        poor overlap. dupe: a second detection on a found box. bkg: nothing there. miss: a
        ground-truth box nobody found.
        """
        counts = dict(tp=0, cls=0, loc=0, both=0, dupe=0, bkg=0, miss=0)
        for img in self.images:
            gts = [g for c in self.categories for g in self.gt.get((img, c), []) if not g.get("iscrowd") and not g.get("ignore")]
            dts = sorted((d for c in self.categories for d in self.dt.get((img, c), [])), key=lambda d: -d["score"])
            ious = iou_matrix([d["bbox"] for d in dts], [g["bbox"] for g in gts])
            found = [False] * len(gts)
            for i, d in enumerate(dts):
                row = ious[i] if len(gts) else np.zeros(0)
                same = [j for j, g in enumerate(gts) if g["category"] == d["category"]]
                free = [j for j in same if not found[j] and row[j] >= foreground]
                if free:
                    found[max(free, key=lambda j: row[j])] = True
                    counts["tp"] += 1
                    continue
                if any(row[j] >= foreground for j in same):
                    counts["dupe"] += 1
                elif row.size and row.max() >= foreground:
                    counts["cls"] += 1
                elif any(row[j] >= background for j in same):
                    counts["loc"] += 1
                elif row.size and row.max() >= background:
                    counts["both"] += 1
                else:
                    counts["bkg"] += 1
            counts["miss"] += found.count(False)
        return counts

    def confusion(self, iou=0.5, score=0.25):
        """A confusion matrix with a final row and column for background."""
        names = list(self.categories) + ["background"]
        idx = {c: i for i, c in enumerate(names)}
        m = np.zeros((len(names), len(names)), int)
        bg = len(names) - 1
        for img in self.images:
            gts = [g for c in self.categories for g in self.gt.get((img, c), [])]
            dts = sorted((d for c in self.categories for d in self.dt.get((img, c), []) if d["score"] >= score), key=lambda d: -d["score"])
            ious = iou_matrix([d["bbox"] for d in dts], [g["bbox"] for g in gts])
            used = set()
            for i, d in enumerate(dts):
                best, bj = iou, -1
                for j in range(len(gts)):
                    if j not in used and ious[i, j] >= best:
                        best, bj = ious[i, j], j
                if bj >= 0:
                    used.add(bj)
                    m[idx[gts[bj]["category"]], idx[d["category"]]] += 1
                else:
                    m[bg, idx[d["category"]]] += 1
            for j, g in enumerate(gts):
                if j not in used:
                    m[idx[g["category"]], bg] += 1
        return names, m


def load_coco(gt_path, dt_path):
    """Read a COCO annotation file and a COCO results file."""
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(dt_path, encoding="utf-8") as f:
        results = json.load(f)
    gt = {im["id"]: [] for im in data["images"]}
    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    for a in data["annotations"]:
        gt[a["image_id"]].append({"bbox": a["bbox"], "category": names.get(a["category_id"], a["category_id"]), "iscrowd": a.get("iscrowd", 0), "area": a.get("area", a["bbox"][2] * a["bbox"][3])})
    dt = defaultdict(list)
    for r in results:
        dt[r["image_id"]].append({"bbox": r["bbox"], "category": names.get(r["category_id"], r["category_id"]), "score": r["score"]})
    return gt, dict(dt)


def report(result):
    lines = [f"{k:<6} {result[k]:.4f}" for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")]
    lines.append("per class:")
    lines += [f"  {str(c):<20} {v:.4f}" for c, v in result["per_class"].items()]
    return "\n".join(lines)
