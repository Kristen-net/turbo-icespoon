"""检测感知损失与不确定性加权测试 (P1-1 联合优化框架)."""

from __future__ import annotations

import pytest
import torch

from icewave.losses.detect import CorridorTextureLoss, UncertaintyWeighting

torch.manual_seed(0)


def _make(batch=1, h=64, w=64, corridor_frac=0.25):
    pred = torch.rand(batch, 3, h, w)
    clear = torch.rand(batch, 3, h, w)
    mask = (torch.rand(batch, 1, h, w) < corridor_frac).float()
    return pred, clear, mask


class TestCorridorTextureLoss:
    def test_zero_when_perfect(self):
        clear = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 8:56, 8:56] = 1.0
        loss = CorridorTextureLoss()(clear, clear, mask)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_mismatch(self):
        pred, clear, mask = _make()
        assert float(CorridorTextureLoss()(pred, clear, mask)) > 0

    def test_empty_corridor_graph_connected(self):
        pred, clear, _ = _make()
        empty = torch.zeros(1, 1, 64, 64)
        loss = CorridorTextureLoss()(pred.requires_grad_(True), clear, empty)
        assert loss.item() == 0.0
        loss.backward()  # 旧式 0.0 常量会在此崩溃
        assert pred.grad is not None

    def test_loss_decreases_as_pred_approaches_clear(self):
        clear, _, mask = _make()[1], None, _make()[2]
        pred_far = clear + 0.3
        pred_near = clear + 0.05
        fn = CorridorTextureLoss()
        assert float(fn(pred_far, clear, mask)) > float(fn(pred_near, clear, mask))

    def test_mask_reduces_loss_scope(self):
        """走廊掩码确实约束了损失作用范围: 缩小掩码 → 损失值改变.

        注: 该损失两项均基于 Sobel 梯度 (3×3 邻域卷积), 故不存在
        "走廊外像素完全不影响损失"的性质 —— 走廊边缘的梯度天然受邻域耦合。
        这里仅验证掩码确实参与了计算 (缩小掩码会显著改变损失)。
        """
        pred, clear, mask = _make(corridor_frac=0.4)
        fn = CorridorTextureLoss()
        loss_full = fn(pred, clear, mask)
        mask_tiny = mask.clone()
        mask_tiny[:, :, 24:40, 24:40] = 0.0  # 挖掉大部分走廊
        loss_reduced = fn(pred, clear, mask_tiny)
        assert not torch.allclose(loss_full, loss_reduced)

    def test_gradient_flows(self):
        pred, clear, mask = _make()
        pred = pred.requires_grad_(True)
        CorridorTextureLoss()(pred, clear, mask).backward()
        assert pred.grad.abs().sum() > 0

    def test_contrast_term_scale_invariant(self):
        """对数对比度项对整体亮度缩放近似不变 (eps=1e-3 仅引入微小偏差)."""
        pred, clear, mask = _make()
        # 令逐像素梯度项为零: pred 与 clear 的梯度场相同 → 只考察对比度项
        clear = torch.zeros(1, 3, 64, 64)
        clear[:, :, 10:20, :] = 0.5          # 阶梯边缘 → 非零梯度能量
        pred = clear * 2.0                    # 同结构, 能量 ×2 → log 比值 = log2
        mask = torch.ones(1, 1, 64, 64)
        loss = CorridorTextureLoss(lambda_grad=0.0)(pred, clear, mask)
        assert float(loss) == pytest.approx(0.693, abs=0.02)  # log(2)


class TestUncertaintyWeighting:
    def test_initial_weights_are_one(self):
        """s=0 初始化 → total = Σ L_i + 0."""
        uw = UncertaintyWeighting(num_losses=2)
        l1 = torch.tensor(2.0)
        l2 = torch.tensor(3.0)
        assert uw([l1, l2]).item() == pytest.approx(5.0)

    def test_learnable_log_vars(self):
        uw = UncertaintyWeighting(num_losses=2)
        assert any(p.requires_grad for p in uw.parameters())

    def test_gradient_flows_to_log_vars(self):
        uw = UncertaintyWeighting(num_losses=2)
        total = uw([torch.tensor(1.0), torch.tensor(2.0)])
        total.backward()
        assert uw.log_vars.grad is not None

    def test_large_loss_gets_downweighted(self):
        """优化 s 会自动降低大损失的权重 (Kendall & Gal 机制)."""
        uw = UncertaintyWeighting(num_losses=2)
        opt = torch.optim.SGD(uw.parameters(), lr=0.1)
        big, small = torch.tensor(10.0), torch.tensor(0.1)
        for _ in range(50):
            opt.zero_grad()
            loss = uw([big, small])
            loss.backward()
            opt.step()
        # 大损失对应的 s 应上升 → 权重 exp(-s) 下降
        w = torch.exp(-uw.log_vars).detach()
        assert w[0] < w[1], f"预期大损失权重被压低: {w}"

    def test_count_mismatch_raises(self):
        uw = UncertaintyWeighting(num_losses=3)
        with pytest.raises(ValueError, match="不一致"):
            uw([torch.tensor(1.0), torch.tensor(2.0)])
