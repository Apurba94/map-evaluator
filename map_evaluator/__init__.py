"""mAP Evaluator: COCO-style object detection metrics and error analysis."""
from .core import AREAS, IOU_THRESHOLDS, Evaluator, iou_matrix, load_coco, report

__all__ = ["Evaluator", "iou_matrix", "load_coco", "report", "IOU_THRESHOLDS", "AREAS"]
