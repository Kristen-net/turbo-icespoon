"""测试检测感知损失 (§3.2-§3.4): BoxFeaturePreservationLoss(detector),
DetectabilityLoss, BoxAlignLoss, UncertaintyWeighting(K=5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from icewave.losses.detect import (
    BoxFeaturePreservationLoss,
    BoxAlignLoss,
    DetectabilityLoss,
    UncertaintyWeighting,
)
from icewave.models.detector_wrapper import DetectorWrapper


# ---------------------------------------------------------------------------
# BoxFeaturePreservationLoss (detector 模式)
# ---------------------------------------------------------------------------
class TestBoxFeatureLossDetector:
    def _make_inputs(self, B=1, C=3, H=64, W=64):
        pred = torch.rand(B, C, H, W, requires_grad=True)
        clear = torch.rand(B, C, H, W)
        bboxes = torch.tensor([[0.1, 0.1, 0.4, 0.4],
                               [0.5, 0.5, 0.8, 0.8]], dtype=torch.float32)
        return pred, clear, bboxes

    def test_detector_mode_zero_when_perfect(self):
        pred, clear, bboxes = self._make_inputs()
        det = DetectorWrapper()  # stub mode
        loss_fn = BoxFeaturePreservationLoss(feature_type="detector")
        loss = loss_fn(pred, clear, bboxes, detector=det)
        assert float(loss) == pytest.approx(0.0, abs=1e-3)

    def test_detector_mode_positive_when_different(self):
        pred, clear, bboxes = self._make_inputs()
        pred2 = torch.rand_like(pred, requires_grad=True)
        det = DetectorWrapper()
        loss_fn = BoxFeaturePreservationLoss(feature_type="detector")
        loss = loss_fn(pred2, clear, bboxes, detector=det)
        assert float(loss) > 0

    def test_detector_mode_gradient_flows(self):
        pred, clear, bboxes = self._make_inputs()
        pred.requires_grad_(True)
        det = DetectorWrapper()
        loss_fn = BoxFeaturePreservationLoss(feature_type="detector")
        loss = loss_fn(pred, clear, bboxes, detector=det)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_zero_bboxes(self):
        pred, clear, _ = self._make_inputs()
        bboxes = torch.zeros(0, 4)
        det = DetectorWrapper()
        loss_fn = BoxFeaturePreservationLoss(feature_type="detector")
        loss = loss_fn(pred, clear, bboxes, detector=det)
        assert float(loss) == pytest.approx(0.0, abs=1e-8)

    def test_requires_detector(self):
        pred, clear, bboxes = self._make_inputs()
        loss_fn = BoxFeaturePreservationLoss(feature_type="detector")
        with pytest.raises(ValueError, match="detector"):
            loss_fn(pred, clear, bboxes)


# ---------------------------------------------------------------------------
# DetectabilityLoss
# ---------------------------------------------------------------------------
class TestDetectabilityLoss:
    def _make_det_out(self, B=1, N=100, C=4):
        """造 DetectorWrapper.forward 格式的输出."""
        cls = torch.sigmoid(torch.randn(B, N, C))
        obj = torch.sigmoid(torch.randn(B, N))
        return [{
            "cls": cls, "obj": obj, "box": torch.randn(B, N, 4),
            "grid_h": 8, "grid_w": 8, "scale": 0,
        }]

    def test_returns_negative(self):
        det_out = self._make_det_out()
        gt_cls = torch.tensor([[0, 1]], dtype=torch.long)
        loss = DetectabilityLoss(tau_cls=0.01)(det_out, gt_cls)
        assert float(loss) < 0  # 负号 → 最大化置信度

    def test_no_matching_anchors(self):
        """所有 cls 分数都很低 → 退化路径."""
        det_out = [{
            "cls": torch.zeros(1, 50, 4),
            "obj": torch.ones(1, 50) * 0.5,
            "box": torch.zeros(1, 50, 4),
            "grid_h": 8, "grid_w": 8, "scale": 0,
        }]
        gt_cls = torch.tensor([[0]], dtype=torch.long)
        loss = DetectabilityLoss(tau_cls=0.99)(det_out, gt_cls)
        # 退化路径返回 -mean*0.01
        assert float(loss) < 0
        assert float(loss) > -0.02

    def test_gradient_flows(self):
        det_out = self._make_det_out()
        det_out[0]["obj"].requires_grad_(True)
        gt_cls = torch.tensor([[0]], dtype=torch.long)
        loss = DetectabilityLoss(tau_cls=0.01)(det_out, gt_cls)
        loss.backward()
        assert det_out[0]["obj"].grad is not None


# ---------------------------------------------------------------------------
# BoxAlignLoss
# ---------------------------------------------------------------------------
class TestBoxAlignLoss:
    def _make_det_out(self, B=1, N=50, grid_h=8, grid_w=8):
        return [{
            "cls": torch.sigmoid(torch.randn(B, N, 4)),
            "obj": torch.sigmoid(torch.randn(B, N)),
            "box": torch.randn(B, N, 4),
            "grid_h": grid_h, "grid_w": grid_w, "scale": 0,
        }]

    def test_returns_nonneg(self):
        det_out = self._make_det_out()
        gt_bboxes = torch.tensor([[[0.1, 0.1, 0.3, 0.3],
                                   [0.5, 0.5, 0.8, 0.8]]],
                                  dtype=torch.float32)
        gt_cls = torch.tensor([[0, 1]], dtype=torch.long)
        loss = BoxAlignLoss()(det_out, gt_bboxes, gt_cls)
        assert float(loss) >= -1e-6

    def test_zero_gt_returns_zero(self):
        det_out = self._make_det_out()
        gt_bboxes = torch.zeros(1, 0, 4)
        gt_cls = torch.zeros(1, 0, dtype=torch.long)
        loss = BoxAlignLoss()(det_out, gt_bboxes, gt_cls)
        assert float(loss) == pytest.approx(0.0, abs=1e-8)

    def test_gradient_flows(self):
        det_out = self._make_det_out()
        det_out[0]["box"].requires_grad_(True)
        gt_bboxes = torch.tensor([[[0.1, 0.1, 0.3, 0.3]]], dtype=torch.float32)
        gt_cls = torch.tensor([[0]], dtype=torch.long)
        loss = BoxAlignLoss()(det_out, gt_bboxes, gt_cls)
        loss.backward()
        assert det_out[0]["box"].grad is not None


# ---------------------------------------------------------------------------
# UncertaintyWeighting (K=5)
# ---------------------------------------------------------------------------
class TestUncertaintyWeightingK5:
    def test_k5_initial_weights_equal(self):
        uw = UncertaintyWeighting(num_losses=5)
        # 初始 s=0 → exp(-0)=1.0 → 权重均为 1.0
        for s in uw.log_vars:
            assert float(s) == pytest.approx(0.0)

    def test_k5_forward(self):
        uw = UncertaintyWeighting(num_losses=5)
        losses = [torch.tensor(1.0) for _ in range(5)]
        result = uw(losses)
        # 初始: 5*1.0*1.0 + 0.5*0.0 = 5.0
        assert float(result) == pytest.approx(5.0)

    def test_k5_mismatch_raises(self):
        uw = UncertaintyWeighting(num_losses=5)
        with pytest.raises(ValueError, match="不一致"):
            uw([torch.tensor(1.0)] * 4)

    def test_s_clamp(self):
        uw = UncertaintyWeighting(num_losses=5, s_clamp=(-6.0, 6.0))
        # 人为推到极端值
        with torch.no_grad():
            uw.log_vars.fill_(10.0)
        uw([torch.tensor(1.0)] * 5)
        # 应被 clamp 到 6.0
        assert float(uw.log_vars.max()) <= 6.0

    def test_s_init(self):
        uw = UncertaintyWeighting(num_losses=5, s_init=1.0)
        assert float(uw.log_vars[0]) == pytest.approx(1.0)

    def test_gradient_to_s(self):
        uw = UncertaintyWeighting(num_losses=5)
        losses = [torch.tensor(float(v), requires_grad=True)
                   for v in [1, 2, 0.5, 3, 1]]
        result = uw(losses)
        result.backward()
        assert uw.log_vars.grad is not None
        assert not torch.isnan(uw.log_vars.grad).any()
