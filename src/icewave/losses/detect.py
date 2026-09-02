"""检测感知损失与多任务不确定性加权 (P1-1 联合优化框架组件).

动机
----
旧管线中检测与去雾完全解耦: 去雾损失只看像素重建, 对"去雾结果是否更利于
下游覆冰/部件检测"没有任何约束。本模块提供把检测需求注入去雾训练的两种组件:

1. ``CorridorTextureLoss``: 在任务相关区域 (导线/绝缘子走廊) 约束
   (a) 纹理能量 (梯度幅值均值) 的对数保持 → 尺度不变的对比度保持, 对应
       检测器依赖的局部对比特征;
   (b) 逐像素梯度 L1 → 防止走廊内结构细节丢失。
   两项均无标量随 batch 变化的问题, 空走廊返回 graph-connected 零。

2. ``UncertaintyWeighting``: Kendall & Gal (CVPR'18) 多任务不确定性加权,
   L_total = Σ_i exp(-s_i)·L_i + 0.5·Σ_i s_i, s_i = log σ_i² 可学习,
   替代手工调 λ, 供 joint 模式使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorridorTextureLoss(nn.Module):
    """走廊纹理保持损失 (检测感知去雾约束).

    参数
    ----
    lambda_grad : 逐像素梯度 L1 项权重 (默认 1.0)。
    eps : 对数能量下界, 避免 log(0) (默认 1e-3)。
    """

    def __init__(self, lambda_grad: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.eps = eps
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, img: torch.Tensor) -> torch.Tensor:
        gray = img.mean(dim=1, keepdim=True)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                      self.sobel_x.to(img.dtype), padding=0)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                      self.sobel_y.to(img.dtype), padding=0)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, pred: torch.Tensor, clear: torch.Tensor,
                corridor_mask: torch.Tensor):
        """pred/clear: (B,3,H,W) in [0,1]; corridor_mask: (B,1,H,W) in {0,1}."""
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float()
            clear = clear.float()
            corridor_mask = corridor_mask.float()

            zero = pred.sum() * 0.0

            n_pix = corridor_mask.sum()
            if n_pix < 1:
                return zero

            g_pred = self._gradient(pred)
            g_clear = self._gradient(clear)

            # (a) 尺度不变对比度保持: 逐样本走廊内平均梯度能量的对数差
            m3 = corridor_mask.expand(-1, -1, -1, -1)  # (B,1,H,W) 与梯度同形
            e_pred = (g_pred * m3).sum(dim=(1, 2, 3)) / (
                m3.sum(dim=(1, 2, 3)) + 1e-6
            )
            e_clear = (g_clear * m3).sum(dim=(1, 2, 3)) / (
                m3.sum(dim=(1, 2, 3)) + 1e-6
            )
            contrast_term = (
                (torch.log(self.eps + e_pred) - torch.log(self.eps + e_clear)).abs()
            ).mean()

            # (b) 逐像素梯度保持 (走廊内)
            grad_term = (torch.abs(g_pred - g_clear) * m3).sum() / (n_pix + 1e-6)

            return contrast_term + self.lambda_grad * grad_term


class UncertaintyWeighting(nn.Module):
    """Kendall & Gal 多任务不确定性加权.

    forward(losses: list[Tensor]) → Σ exp(-s_i)·L_i + 0.5·Σ s_i
    s 初始化为 0 (即初始权重 1.0)。注意 s 是可学习参数, 会被记入 checkpoint。
    """

    def __init__(self, num_losses: int = 2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        if len(losses) != self.log_vars.numel():
            raise ValueError(
                f"损失数量 {len(losses)} 与权重数量 {self.log_vars.numel()} 不一致"
            )
        weighted = [torch.exp(-s) * l for s, l in zip(self.log_vars, losses)]
        penalty = 0.5 * self.log_vars.sum()
        return sum(weighted) + penalty
