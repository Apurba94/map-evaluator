"""Evaluate detections: python -m map_evaluator annotations.json results.json"""
import sys

from . import Evaluator, load_coco, report


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    gt, dt = load_coco(argv[1], argv[2])
    e = Evaluator(gt, dt)
    print(report(e.evaluate()))
    print("errors:", e.errors())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
