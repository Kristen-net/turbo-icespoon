"""下游任务增益指标测试 (P1-1): mAP 实现 / 标签解析 / 增益计算."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from icewave.eval.downstream import (
    _average_precision,
    _iou_xyxy,
    _load_yolo_labels,
    downstream_gain,
    evaluate_set,
)

H, W = 96, 128
# 所有测试图的 GT 框 (中心 1/4 区域, 由 YOLO 标签 0 0.5 0.5 0.25 0.25 生成)
GT_BOX = (48.0, 36.0, 80.0, 60.0)


class _Detector:
    """桩检测器: 按调用序号循环输出预设检测 (evaluate_set 逐图顺序调用)."""

    def __init__(self, sequence: list[list[dict]]):
        self.sequence = sequence
        self._i = 0

    def detect(self, img_bgr, conf=None):
        dets = self.sequence[self._i % len(self.sequence)]
        self._i += 1
        return dets


def _det(box=GT_BOX, cls_id=0, conf=0.9):
    return {"class": "obj", "cls_id": cls_id, "bbox": box, "conf": conf}


def _make_images_and_labels(tmp: Path, n: int = 3, labeled: int | None = None):
    """图像 + YOLO 标签 (前 labeled 张有标签; 默认全部有)."""
    img_dir = tmp / "imgs"
    labels_dir = tmp / "labels"
    img_dir.mkdir(parents=True), labels_dir.mkdir()
    for i in range(n):
        stem = f"im{i:03d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"),
                    np.full((H, W, 3), 128, np.uint8))
        if labeled is None or i < labeled:
            (labels_dir / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    return img_dir, labels_dir


class TestIoU:
    def test_identical(self):
        assert _iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # 交集 50 / 并集 150 = 1/3
        assert _iou_xyxy((0, 0, 10, 10), (0, 5, 10, 15)) == pytest.approx(1 / 3)

    def test_degenerate_box(self):
        assert _iou_xyxy((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        assert _average_precision([1.0], [1.0]) == pytest.approx(1.0)

    def test_half_recall_full_precision(self):
        # 召回到 0.5 后精度仍 1.0 → AP = 0.5
        assert _average_precision([0.5], [1.0]) == pytest.approx(0.5)


class TestLoadLabels:
    def test_yolo_to_absolute(self, tmp_path):
        txt = tmp_path / "a.txt"
        txt.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        gts = _load_yolo_labels(txt, W, H)
        assert len(gts) == 1
        cls, box = gts[0]
        assert cls == 0
        assert box == pytest.approx(GT_BOX)

    def test_missing_file(self, tmp_path):
        assert _load_yolo_labels(tmp_path / "nope.txt", W, H) == []

    def test_malformed_lines_skipped(self, tmp_path):
        txt = tmp_path / "a.txt"
        txt.write_text("bad line\n0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        assert len(_load_yolo_labels(txt, W, H)) == 1


class TestEvaluateSet:
    def test_perfect_detector_map_one(self, tmp_path):
        img_dir, labels_dir = _make_images_and_labels(tmp_path)
        res = evaluate_set(_Detector([[_det()]]),
                           sorted(img_dir.glob("*.png")), labels_dir)
        assert res["mAP"] == pytest.approx(1.0)
        assert res["n_images"] == 3

    def test_all_wrong_reduces_map_to_zero(self, tmp_path):
        img_dir, labels_dir = _make_images_and_labels(tmp_path)
        bad = _det(box=(0, 0, 10, 10))  # IoU=0 → 全 FP
        res = evaluate_set(_Detector([[bad]]),
                           sorted(img_dir.glob("*.png")), labels_dir)
        assert res["mAP"] == pytest.approx(0.0)

    def test_fp_in_gt_empty_image_penalized(self, tmp_path):
        """无 GT 图像中的检测必须计为 FP, 且在召回未饱和时拉低 AP.

        注: all-point 插值 AP 是 PR 曲线下面积, 当最后一个检测是 FP 且
        召回率已达 1.0 时, AP 数学上不会下降 (与 COCO/VOC 一致)。故此处
        让 FP 置信度**高于**一个 TP, 使 FP 插入 TP 之间, 从而降低 AP。
        """
        img_dir, labels_dir = _make_images_and_labels(tmp_path, n=3, labeled=2)

        class _FPInterleavedDetector:
            def __init__(self):
                self._i = 0

            def detect(self, img, conf=None):
                i, self._i = self._i, self._i + 1
                if i == 0:      # 高置信 TP
                    return [_det(conf=0.9)]
                if i == 1:      # 低置信 TP
                    return [_det(conf=0.4)]
                return [_det(conf=0.6)]  # 无 GT 图: 中置信 FP, 插在两者之间

        with_fp = evaluate_set(_FPInterleavedDetector(),
                               sorted(img_dir.glob("*.png")), labels_dir)
        # FP 插在 TP 之间: 在 recall=0.5 处精度 = 1/2 → AP < 1.0
        assert 0.0 < with_fp["mAP"] < 1.0

    def test_no_gt_no_detections_empty(self, tmp_path):
        img_dir, labels_dir = _make_images_and_labels(tmp_path, n=2, labeled=0)
        res = evaluate_set(_Detector([[]]),
                           sorted(img_dir.glob("*.png")), labels_dir)
        assert res.get("n_empty") is True


class TestDownstreamGain:
    def test_delta_positive_when_dehaze_helps(self, tmp_path):
        """hazy 集: 坏框; dehazed/clear 集: 精确框 → ΔmAP > 0, gap=0."""
        img_dir, labels_dir = _make_images_and_labels(tmp_path, n=3)
        bad = _det(box=(10, 10, 40, 30), conf=0.9)

        class _GainDetector:
            """前 n_hazy 次调用 (hazy 集) 输出坏框, 之后输出精确框."""

            def __init__(self, n_hazy):
                self.n_hazy, self._i = n_hazy, 0

            def detect(self, img, conf=None):
                i, self._i = self._i, self._i + 1
                return [bad] if i < self.n_hazy else [_det()]

        res = downstream_gain(_GainDetector(3), img_dir, img_dir, labels_dir,
                              clear_dir=img_dir)
        assert res["hazy"]["mAP"] == pytest.approx(0.0)
        assert res["dehazed"]["mAP"] == pytest.approx(1.0)
        assert res["delta_mAP_dehazed_minus_hazy"] == pytest.approx(1.0)
        assert res["gap_clear_minus_dehazed"] == pytest.approx(0.0, abs=1e-9)

    def test_without_clear_dir(self, tmp_path):
        img_dir, labels_dir = _make_images_and_labels(tmp_path)
        res = downstream_gain(_Detector([[_det()]]), img_dir, img_dir,
                              labels_dir)
        assert "clear" not in res
        assert res["delta_mAP_dehazed_minus_hazy"] == pytest.approx(0.0)

    def test_class_names_mapped(self, tmp_path):
        img_dir, labels_dir = _make_images_and_labels(tmp_path)
        res = evaluate_set(_Detector([[_det()]]),
                           sorted(img_dir.glob("*.png")), labels_dir,
                           class_names=["insulator", "power_line"])
        assert "insulator" in res["AP_per_class"]
