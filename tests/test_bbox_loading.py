"""bbox 标签读取测试 (JOINT_OPTIMIZATION_FRAMEWORK §5.2).

验证 IceAwareDataset 的 return_bboxes 模式:
- 无标签文件 → 空张量 (graph-connected)
- 有标签 → 正确的归一化 xyxy + cls
- 裁剪后坐标调整正确
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from icewave.data.dataset import IceAwareDataset, _load_yolo_labels

H, W = 64, 64


def _make_split_with_labels(root: Path, split: str = "val", n: int = 2):
    """构造带 YOLO 标签的最小数据集目录."""
    base = root / split
    (base / "hazy").mkdir(parents=True)
    (base / "clear").mkdir()
    (base / "ice_mask").mkdir()
    (base / "labels").mkdir()

    for i in range(n):
        bname = f"img_{i:04d}"
        clear = np.full((H, W, 3), 100 + i * 20, np.uint8)
        cv2.imwrite(str(base / "clear" / f"{bname}.png"), clear)
        hazy = np.full((H, W, 3), 200, np.uint8)
        cv2.imwrite(str(base / "hazy" / f"{bname}_haze0.png"), hazy)
        ice = np.zeros((H, W), np.uint8)
        ice[:, :W // 2] = 255
        cv2.imwrite(str(base / "ice_mask" / f"{bname}_ice.png"), ice)
        # YOLO 标签: 1 个全图框 + 1 个左上角小框
        txt = (f"0 0.5 0.5 1.0 1.0\n"    # cls, xc, yc, w, h (全图)
               f"1 0.25 0.25 0.3 0.3\n")  # 左上角小框
        (base / "labels" / f"{bname}.txt").write_text(txt, encoding="utf-8")
    return base


class TestLoadYoloLabels:
    def test_no_file_returns_empty(self, tmp_path):
        bbox, cls = _load_yolo_labels(tmp_path / "nonexistent.txt", 64, 64)
        assert bbox.shape[0] == 0
        assert cls.shape[0] == 0

    def test_parse_correct(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("0 0.5 0.5 1.0 1.0\n1 0.25 0.25 0.5 0.5\n", encoding="utf-8")
        bbox, cls = _load_yolo_labels(txt, 100, 100)
        assert bbox.shape == (2, 4)
        assert cls.tolist() == [0, 1]
        # 第一个框: 全图 (0, 0, 1, 1)
        np.testing.assert_array_almost_equal(bbox[0], [0.0, 0.0, 1.0, 1.0])
        # 第二个框: (0.0, 0.0, 0.5, 0.5)
        np.testing.assert_array_almost_equal(bbox[1], [0.0, 0.0, 0.5, 0.5])

    def test_crop_adjustment(self, tmp_path):
        """裁剪后框坐标应正确调整."""
        txt = tmp_path / "test.txt"
        # 全图框 (0.5, 0.5, 1.0, 1.0) in 100×100
        txt.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        # 裁剪 (y=25, x=25, size=50)
        bbox, cls = _load_yolo_labels(txt, 100, 100, patch_size=50,
                                     crop_y=25, crop_x=25)
        # 框从 (0,0)-(100,100) → 裁剪后 (0,0)-(50,50) → 归一化 (0,0,1,1)
        np.testing.assert_array_almost_equal(bbox[0], [0.0, 0.0, 1.0, 1.0])

    def test_box_outside_crop_filtered(self, tmp_path):
        txt = tmp_path / "test.txt"
        # 右下角小框 (0.75, 0.75, 0.1, 0.1) in 100×100
        txt.write_text("0 0.75 0.75 0.1 0.1\n", encoding="utf-8")
        # 裁剪左上 50×50, 框完全在裁剪区域外
        bbox, cls = _load_yolo_labels(txt, 100, 100, patch_size=50,
                                     crop_y=0, crop_x=0)
        assert bbox.shape[0] == 0  # 被过滤


class TestDatasetBboxMode:
    def test_return_bboxes_shapes(self, tmp_path):
        _make_split_with_labels(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0,
                             return_bboxes=True)
        hazy, clear, ice, bbox, cls = ds[0]
        assert hazy.shape == (3, H, W)
        assert bbox.shape[1] == 4
        assert cls.shape[0] == bbox.shape[0]
        assert bbox.shape[0] == 2  # 两个标签框

    def test_no_labels_returns_empty(self, tmp_path):
        """无 labels 目录 → 空 bbox, 不崩溃."""
        (tmp_path / "ds" / "val" / "hazy").mkdir(parents=True)
        (tmp_path / "ds" / "val" / "clear").mkdir()
        clear = np.full((H, W, 3), 100, np.uint8)
        cv2.imwrite(str(tmp_path / "ds" / "val" / "clear" / "img.png"), clear)
        hazy = np.full((H, W, 3), 200, np.uint8)
        cv2.imwrite(str(tmp_path / "ds" / "val" / "hazy" / "img_haze0.png"), hazy)
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0,
                             return_bboxes=True)
        _, _, _, bbox, cls = ds[0]
        assert bbox.shape[0] == 0
        assert cls.shape[0] == 0

    def test_bbox_with_corridor(self, tmp_path):
        """return_bboxes + corridor_from_ice 同时启用."""
        _make_split_with_labels(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0,
                             return_bboxes=True, corridor_from_ice=True)
        result = ds[0]
        assert len(result) == 6  # hazy, clear, ice, corridor, bbox, cls
        hazy, clear, ice, corridor, bbox, cls = result
        assert corridor.shape == (1, H, W)
        assert bbox.shape[1] == 4

    def test_bbox_range_normalized(self, tmp_path):
        """bbox 坐标归一化到 [0, 1]."""
        _make_split_with_labels(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0,
                             return_bboxes=True)
        _, _, _, bbox, _ = ds[0]
        assert float(bbox.min()) >= 0.0
        assert float(bbox.max()) <= 1.0
