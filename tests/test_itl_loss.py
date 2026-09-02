"""ITL 损失修复测试 (P0-1b 的验收标准).

逐条验证 itl.py docstring 声称的三处修复:
1. batch 尺寸无关性 (旧 bug: B=4 与 B=1 数值不同);
2. 空掩码返回 graph-connected 零 (旧 bug: 0.0 常量, 反传报错);
3. region_term='ssim' 真正调用 SSIM (旧 bug: torchmetrics 死代码)。
"""

from __future__ import annotations

import pytest
import torch

from icewave.losses.itl import ITLLoss, ssim_map

torch.manual_seed(0)


def _make(batch: int, h: int = 64, w: int = 64, ice_frac: float = 0.3):
    pred = torch.rand(batch, 3, h, w)
    clear = torch.rand(batch, 3, h, w)
    mask = (torch.rand(batch, 1, h, w) < ice_frac).float()
    return pred, clear, mask


class TestITLFixes:
    def test_batch_size_invariance(self):
        """同一图重复 B 份, 损失值必须不变 (旧实现的 bug)."""
        pred, clear, mask = _make(1)
        loss_fn = ITLLoss()
        r1, b1, t1 = loss_fn(pred, clear, mask)
        r4, b4, t4 = loss_fn(pred.repeat(4, 1, 1, 1), clear.repeat(4, 1, 1, 1),
                             mask.repeat(4, 1, 1, 1))
        torch.testing.assert_close(r1, r4, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(b1, b4, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(t1, t4, rtol=1e-6, atol=1e-7)

    def test_empty_mask_graph_connected(self):
        """空掩码 → 0 且可反传 (旧实现返回 0.0 常量导致 RuntimeError)."""
        pred, clear, _ = _make(1)
        empty = torch.zeros(1, 1, 64, 64)
        loss_fn = ITLLoss()
        r, b, t = loss_fn(pred.requires_grad_(True), clear, empty)
        assert float(t) == 0.0
        t.backward()  # 不应抛 "does not require grad"
        assert pred.grad is not None

    def test_ssim_term_actually_used(self):
        """region_term='ssim' 与 'l1' 在同输入下应给出不同区域项."""
        pred, clear, mask = _make(1)
        r_ssim, _, _ = ITLLoss(region_term="ssim")(pred, clear, mask)
        r_l1, _, _ = ITLLoss(region_term="l1")(pred, clear, mask)
        assert not torch.allclose(r_ssim, r_l1), "SSIM 项疑似未生效 (旧死代码 bug)"

    def test_zero_loss_when_perfect(self):
        """pred == clear → 区域项与边界项均为 0."""
        clear = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        r, b, t = ITLLoss()(clear, clear, mask)
        assert float(r) < 1e-5
        assert float(b) < 1e-4
        assert float(t) < 1e-4

    def test_loss_positive_when_mismatch(self):
        pred, clear, mask = _make(1)
        r, b, t = ITLLoss()(pred, clear, mask)
        assert float(r) > 0
        assert float(b) > 0

    def test_invalid_region_term(self):
        with pytest.raises(ValueError, match="region_term"):
            ITLLoss(region_term="mse")

    def test_gradient_flows(self):
        pred, clear, mask = _make(1)
        pred = pred.requires_grad_(True)
        *_, total = ITLLoss()(pred, clear, mask)
        total.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_ice_region_only(self):
        """冰区外误差不影响区域项 (用纯 L1 项验证空间隔离性).

        注: 默认 region_term='ssim' 时, SSIM 的 11×11 高斯窗会让邻近非冰区
        像素微弱泄漏进冰区 SSIM 值, 这是 SSIM 固有性质, 非 bug。此处用
        'l1' 项严格验证"区域约束只作用于冰区"。
        """
        pred, clear, mask = _make(1, ice_frac=0.3)
        loss_fn = ITLLoss(region_term="l1")
        r_before, _, _ = loss_fn(pred, clear, mask)
        outside = mask.expand(-1, 3, -1, -1) < 0.5
        pred2 = pred.clone()
        pred2[outside] = torch.rand_like(pred2[outside])
        r_after, _, _ = loss_fn(pred2, clear, mask)
        torch.testing.assert_close(r_before, r_after, rtol=1e-5, atol=1e-6)


class TestSSIMMap:
    def test_identical_images(self):
        x = torch.rand(2, 3, 64, 64)
        assert float(ssim_map(x, x).mean()) == pytest.approx(1.0, abs=1e-5)

    def test_different_images_lower(self):
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        assert float(ssim_map(x, y).mean()) < 1.0
