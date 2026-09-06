"""BoxFeaturePreservationLoss 测试 (JOINT_OPTIMIZATION_FRAMEWORK §3.2).

Stub 版 (D11): 无需 YOLO 检测器, 基于 Sobel + 像素统计.
验证:
- 空框返回 graph-connected 零 (可反传)
- 完美匹配时损失 ≈ 0
- 不匹配时损失 > 0
- 梯度流通
"""

from __future__ import annotations

import pytest
import torch

from icewave.losses.detect import BoxFeaturePreservationLoss, IcePhysicalLoss

torch.manual_seed(0)


def _make(batch=1, h=64, w=64, n_boxes=2):
    pred = torch.rand(batch, 3, h, w)
    clear = torch.rand(batch, 3, h, w)
    bboxes = torch.tensor([
        [0.1, 0.1, 0.4, 0.4],
        [0.5, 0.5, 0.8, 0.8],
    ][:n_boxes], dtype=torch.float32)
    if n_boxes == 0:
        bboxes = torch.zeros(0, 4)
    return pred, clear, bboxes


class TestBoxFeaturePreservationLoss:
    def test_zero_when_perfect(self):
        clear = torch.rand(1, 3, 64, 64)
        bboxes = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
        loss = BoxFeaturePreservationLoss()(clear, clear, bboxes)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_mismatch(self):
        pred, clear, bboxes = _make()
        assert float(BoxFeaturePreservationLoss()(pred, clear, bboxes)) > 0

    def test_empty_boxes_graph_connected(self):
        pred, clear, _ = _make(n_boxes=0)
        loss = BoxFeaturePreservationLoss()(pred.requires_grad_(True), clear,
                                            torch.zeros(0, 4))
        assert loss.item() == 0.0
        loss.backward()
        assert pred.grad is not None

    def test_gradient_flows(self):
        pred, clear, bboxes = _make()
        pred = pred.requires_grad_(True)
        BoxFeaturePreservationLoss()(pred, clear, bboxes).backward()
        assert pred.grad.abs().sum() > 0

    def test_loss_decreases_as_pred_approaches_clear(self):
        clear, _, bboxes = _make()[1], None, _make()[2]
        pred_far = clear + 0.3
        pred_near = clear + 0.05
        fn = BoxFeaturePreservationLoss()
        assert float(fn(pred_far, clear, bboxes)) > float(fn(pred_near, clear, bboxes))

    def test_multiple_boxes(self):
        pred, clear, bboxes = _make(n_boxes=2)
        loss = BoxFeaturePreservationLoss()(pred, clear, bboxes)
        assert loss.dim() == 0  # scalar
        assert float(loss) >= 0.0

    def test_stub_type(self):
        fn = BoxFeaturePreservationLoss(feature_type="stub")
        assert fn.feature_type == "stub"


class TestIcePhysicalLoss:
    def test_zero_when_perfect(self):
        t = torch.rand(1, 1, 32, 32)
        a = torch.rand(1, 1, 32, 32)
        loss = IcePhysicalLoss()(t, t, a, a)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_mismatch(self):
        t_hat = torch.rand(1, 1, 32, 32)
        t_tgt = torch.rand(1, 1, 32, 32)
        a_hat = torch.rand(1, 1, 32, 32)
        a_tgt = torch.rand(1, 1, 32, 32)
        assert float(IcePhysicalLoss()(t_hat, t_tgt, a_hat, a_tgt)) > 0

    def test_gradient_flows(self):
        t_hat = torch.rand(1, 1, 32, 32, requires_grad=True)
        t_tgt = torch.rand(1, 1, 32, 32)
        a_hat = torch.rand(1, 1, 32, 32, requires_grad=True)
        a_tgt = torch.rand(1, 1, 32, 32)
        IcePhysicalLoss()(t_hat, t_tgt, a_hat, a_tgt).backward()
        assert t_hat.grad.abs().sum() > 0
        assert a_hat.grad.abs().sum() > 0

    def test_lambda_weights(self):
        t = torch.rand(1, 1, 32, 32)
        a = torch.rand(1, 1, 32, 32)
        t_tgt = torch.zeros(1, 1, 32, 32)
        a_tgt = torch.zeros(1, 1, 32, 32)
        loss_default = IcePhysicalLoss()(t, t_tgt, a, a_tgt)
        loss_weighted = IcePhysicalLoss(lambda_trans=2.0, lambda_ice=3.0)(t, t_tgt, a, a_tgt)
        assert float(loss_weighted) > float(loss_default)
