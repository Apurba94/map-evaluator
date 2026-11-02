# mAP Evaluator

Object detection metrics written from the published COCO definition, in about 300 lines of readable NumPy. Beyond the usual numbers, it tells you why a detector loses points.

```python
from map_evaluator import Evaluator, report

gt = {"img1": [{"bbox": [10, 10, 100, 80], "category": "RBC"}]}          # [x, y, width, height]
dt = {"img1": [{"bbox": [12, 11, 98, 80], "category": "RBC", "score": 0.93}]}

e = Evaluator(gt, dt)
print(report(e.evaluate()))
print(e.errors())
```

For COCO files:

```
python -m map_evaluator annotations.json results.json
```

## What you get

- **The twelve COCO numbers:** AP, AP50, AP75, AP and AR by object size, and AR at 1, 10 and 100 detections
- **Per-class AP** and precision-recall curves
- **Error analysis:** each false positive is labelled as a wrong class, poor localisation, both, a duplicate or background, and each missed box is counted. This is in the spirit of the TIDE toolbox
- **A confusion matrix** with a background row and column

The rules follow COCO: detections are matched in score order, crowd regions absorb detections without penalty, 101-point interpolation is used, and classes with no ground truth are left out of the mean.

## Why not pycocotools?

pycocotools is the standard, so cite it for final numbers in a paper. This project exists for when you want to read the logic, step through it, install nothing compiled, or see the error breakdown beside the score.

## How it is tested

When pycocotools is installed, `tests/test_against_pycocotools.py` compares all twelve numbers on 25 random datasets with crowd regions, several classes and every size range. They agree to 9 decimal places. Without pycocotools that test is skipped.

AP is also checked by hand-worked examples and against a separate, slow implementation on 40 random datasets, and the tests cover crowd handling, size ranges and every error type.

## Limits

It handles boxes only, not masks or keypoints. It is much slower than the compiled pycocotools on large datasets such as full COCO.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
