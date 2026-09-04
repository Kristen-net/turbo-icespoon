"""detect.yolo 单元测试 (无须 ultralytics)。

设计原则:
- detect.yolo 是 [detect] extra 的核心, ultralytics ≥ 8.2 是 AGPL 依赖,
  CI 默认不安装。本测试覆盖纯 Python 路径 (filter_ice / 常量 / YOLODetector
  构造) + 通过 monkeypatch 模拟 ultralytics 缺失时 ImportError 的友好提示。
- YOLODetector.draw 为 staticmethod, 不依赖 ultralytics 模型实例, 可直接测。
"""
from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest


# ---- 常量 ----
def test_yolo_classes_contains_expected():
    """类别表至少含 insulator/power_line/ice/tower 四类 (输电线路场景核心)."""
    from icewave.detect.yolo import YOLO_CLASSES
    for cls in ("insulator", "power_line", "ice", "tower"):
        assert cls in YOLO_CLASSES, f"YOLO_CLASSES 缺少核心类别: {cls}"


def test_yolo_colors_cover_all_classes():
    """颜色映射覆盖类别表中所有条目 (保证 draw 不会落到默认灰)。"""
    from icewave.detect.yolo import YOLO_CLASSES, YOLO_COLORS
    for cls in YOLO_CLASSES:
        assert cls in YOLO_COLORS, f"YOLO_COLORS 缺 {cls} 的颜色"
        c = YOLO_COLORS[cls]
        assert len(c) == 3
        assert all(0 <= v <= 255 for v in c), f"{cls} 颜色越界: {c}"


# ---- filter_ice_detections ----
def test_filter_ice_no_lines_drops_all_ice():
    """没有任何 line/insulator 时, 全部 ice 检测应被剔除 (抑制独立误检)."""
    from icewave.detect.yolo import filter_ice_detections
    dets = [
        {"class": "ice", "cls_id": 2, "bbox": (10, 10, 50, 50), "conf": 0.9},
        {"class": "tower", "cls_id": 3, "bbox": (0, 0, 100, 100), "conf": 0.8},
        {"class": "ice", "cls_id": 2, "bbox": (60, 60, 80, 80), "conf": 0.7},
    ]
    out = filter_ice_detections(dets)
    classes_left = [d["class"] for d in out]
    assert "ice" not in classes_left, f"无线路上下文, ice 不应保留: {classes_left}"
    assert "tower" in classes_left


def test_filter_ice_keeps_overlapping():
    """ice 检测与 power_line 框重叠 → 保留。"""
    from icewave.detect.yolo import filter_ice_detections
    dets = [
        {"class": "power_line", "cls_id": 1, "bbox": (10, 10, 100, 30), "conf": 0.9},
        {"class": "ice", "cls_id": 2, "bbox": (40, 18, 70, 28), "conf": 0.85},
    ]
    out = filter_ice_detections(dets)
    assert any(d["class"] == "ice" for d in out), "重叠 ice 应保留"


def test_filter_ice_drops_non_overlapping():
    """ice 检测与 line 不重叠 → 剔除。"""
    from icewave.detect.yolo import filter_ice_detections
    dets = [
        # line 在画面顶部
        {"class": "power_line", "cls_id": 1, "bbox": (10, 10, 100, 30), "conf": 0.9},
        # ice 远离 line, 在画面底部
        {"class": "ice", "cls_id": 2, "bbox": (50, 500, 80, 600), "conf": 0.9},
    ]
    out = filter_ice_detections(dets)
    classes_left = [d["class"] for d in out]
    assert "ice" not in classes_left, f"不相交 ice 应被剔除: {classes_left}"
    assert any(d["class"] == "power_line" for d in out)


def test_filter_ice_preserves_others():
    """其他类别 (insulator/tower) 总是保留, 与 ice 处理正交。"""
    from icewave.detect.yolo import filter_ice_detections
    dets = [
        {"class": "insulator", "cls_id": 0, "bbox": (10, 10, 50, 50), "conf": 0.9},
        {"class": "tower", "cls_id": 3, "bbox": (200, 200, 400, 400), "conf": 0.8},
        {"class": "ice", "cls_id": 2, "bbox": (300, 300, 350, 350), "conf": 0.6},
    ]
    out = filter_ice_detections(dets)
    classes_left = {d["class"] for d in out}
    assert "insulator" in classes_left
    assert "tower" in classes_left
    assert "ice" not in classes_left, "无 line 上下文 ice 被剔除, 但 insulator/tower 必须保留"


# ---- YOLODetector 构造 ----
def test_yolodetector_missing_weights_raises_filenotfound(tmp_path):
    """YOLODetector 构造时若权重文件不存在, 抛 FileNotFoundError 含中英文提示."""
    from icewave.detect.yolo import YOLODetector
    bogus = tmp_path / "no_such.pt"
    with pytest.raises(FileNotFoundError) as exc:
        YOLODetector(weights=bogus)
    msg = str(exc.value)
    assert str(bogus) in msg
    assert "download_weights" in msg, "应提示可使用 download_weights.py 修复"


def test_import_yolo_missing_emits_friendly_message(monkeypatch):
    """ultralytics 未装时, _import_yolo 应抛出 ImportError 并提示安装命令
    (而非裸 ModuleNotFoundError), 且含 AGPL 风险提示。"""
    # 把 ultralytics 从 sys.modules 移除并阻止再次 import
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # 重新导入 yolo 模块避免其顶部 from 链缓存
    sys.modules.pop("icewave.detect.yolo", None)
    mod = importlib.import_module("icewave.detect.yolo")

    with pytest.raises(ImportError) as exc:
        mod._import_yolo()
    msg = str(exc.value)
    assert "ultralytics" in msg
    assert "AGPL" in msg or "agpl" in msg.lower(), "应有 AGPL 许可提示"
    assert "pip install" in msg, "应有安装命令提示"


# ---- draw ----
def test_yolodetector_draw_returns_same_shape_and_no_overflow():
    """YOLODetector.draw 在画框 / 文字时不应越界, 输出形状与输入一致."""
    from icewave.detect.yolo import YOLODetector

    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    dets = [
        {"class": "insulator", "cls_id": 0, "bbox": (50, 50, 120, 100), "conf": 0.92},
        {"class": "ice", "cls_id": 2, "bbox": (300, 400, 580, 460), "conf": 0.83},
    ]
    annotated = YOLODetector.draw(img, dets)
    assert annotated.shape == img.shape
    # draw 必须在图像上至少画了一些像素 (矩形/文字填充)
    assert np.any(annotated != img), "draw 后图像应有可见变化"


def test_yolodetector_draw_unknown_class_uses_gray():
    """未知类别仍能绘制 (fallback 灰色)."""
    from icewave.detect.yolo import YOLODetector

    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    dets = [{"class": "unknown_obj", "cls_id": 99, "bbox": (10, 10, 50, 50), "conf": 0.5}]
    annotated = YOLODetector.draw(img, dets)
    assert annotated.shape == img.shape
    assert np.any(annotated != img)
