"""测试 DetectorWrapper (§5.1 Stage-B): 创建, 前向, 特征提取, 梯度反传."""

from __future__ import annotations

import pytest
import torch

from icewave.models.detector_wrapper import DetectorWrapper


class TestDetectorWrapperStub:
    """Stub 模式测试 (无 ultralytics 依赖)."""

    def test_creation(self):
        det = DetectorWrapper()
        assert det._is_stub is True
        assert det.num_classes == 4

    def test_creation_custom_classes(self):
        det = DetectorWrapper(num_classes=10)
        assert det.num_classes == 10

    def test_forward_returns_list_of_dicts(self):
        det = DetectorWrapper()
        img = torch.randn(2, 3, 64, 64)
        out = det(img)
        assert isinstance(out, list)
        assert len(out) == 3  # P3, P4, P5
        for i, s in enumerate(out):
            assert "cls" in s
            assert "box" in s
            assert "obj" in s
            assert s["scale"] == i
            assert s["cls"].shape[0] == 2  # batch
            assert s["obj"].shape[0] == 2

    def test_forward_output_shapes(self):
        det = DetectorWrapper(num_classes=4)
        img = torch.randn(1, 3, 64, 64)
        out = det(img)
        for s in out:
            B, N, C = s["cls"].shape
            assert B == 1
            assert C == 4  # num_classes
            assert s["box"].shape == (B, N, 4)
            assert s["obj"].shape == (B, N)

    def test_backbone_frozen(self):
        det = DetectorWrapper(freeze_backbone=True)
        for p in det.backbone.parameters():
            assert p.requires_grad is False

    def test_backbone_unfrozen(self):
        det = DetectorWrapper(freeze_backbone=False)
        for p in det.backbone.parameters():
            assert p.requires_grad is True

    def test_trainable_parameters_excludes_backbone(self):
        det = DetectorWrapper(freeze_backbone=True)
        trainable = det.trainable_parameters
        for p in trainable:
            assert p.requires_grad is True
        # 确保 trainable 中不包含 backbone 参数
        backbone_params = set(id(p) for p in det.backbone.parameters())
        train_params = set(id(p) for p in trainable)
        assert not backbone_params & train_params

    def test_extract_neck_features(self):
        det = DetectorWrapper()
        img = torch.randn(1, 3, 64, 64)
        feats = det.extract_neck_features(img)
        assert len(feats) == 3
        for f in feats:
            assert f.dim() == 4  # (B, C, H, W)
            assert f.shape[0] == 1

    def test_gradient_to_neck(self):
        det = DetectorWrapper(freeze_backbone=True)
        img = torch.randn(1, 3, 64, 64)
        out = det(img)
        loss = sum(s["obj"].mean() for s in out)
        loss.backward()
        # neck 参数应有梯度
        neck_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in det.neck.parameters()
        )
        assert neck_has_grad

    def test_no_gradient_to_backbone(self):
        det = DetectorWrapper(freeze_backbone=True)
        img = torch.randn(1, 3, 64, 64)
        out = det(img)
        loss = sum(s["obj"].mean() for s in out)
        loss.backward()
        # backbone 参数不应有梯度
        for p in det.backbone.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_different_input_sizes(self):
        det = DetectorWrapper()
        for size in [32, 64, 128]:
            img = torch.randn(1, 3, size, size)
            out = det(img)
            assert len(out) == 3


class TestDetectorWrapperReal:
    """真实 YOLOv8 测试 (需要 ultralytics + 权重文件)."""

    def test_real_yolo_creation(self, tmp_path):
        pytest.importorskip("ultralytics")
        # 需要真实权重文件, 跳过如果不存在
        weights = tmp_path / "yolov8n.pt"
        if not weights.exists():
            pytest.skip("需要 YOLOv8 权重文件来测试真实模式")
        det = DetectorWrapper(weights=str(weights), num_classes=80)
        assert det._is_stub is False
        img = torch.randn(1, 3, 640, 640)
        feats = det.extract_neck_features(img)
        assert len(feats) == 3
