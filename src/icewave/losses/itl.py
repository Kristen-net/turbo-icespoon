"""ITL: Ice-aware Territory Loss (P0-1b 修复版).

旧实现 (phase5/itl_loss.py) 的三处问题, 本文件全部修复:
1. **文档与实现不符**: docstring 声称 L_region = SSIM(冰区), 实际是加权 L1;
   导入并实例化的 torchmetrics SSIM 从未被调用 (死代码)。
   → 本版默认 ``region_term='ssim'`` 实现真正的冰区 SSIM 项 (与论文表述一致);
   ``region_term='l1'`` 保留旧数值行为 (修复第 3 条 bug 后)。
2. **batch 尺寸依赖 bug**: 旧加权项 ice_weight = mask/n_ice*H*W, 其均值在
   (B,3,H,W) 上取, 导致区域损失随 batch 大小变化 (B=4 与 B=1 数值不同)。
   → 本版对像素与通道取 masked 均值, 与 batch 无关。
3. **空掩码返回 0.0 常量**: 图被截断, 反传时报 "does not require grad"。
   → 空掩码返回 graph-connected 的零张量。

公式 (与论文/README 声明一致):
    L_itl   = λ_region * L_region + λ_boundary * L_boundary
    L_region = mean_{x∈ice}|pred-clear| + w_ssim * (1 - SSIM_masked(pred, clear))
    L_boundary = mean_{x∈boundary} |∇pred - ∇clear|₁     (Sobel 梯度幅值)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def _filter_gauss(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    C = x.shape[1]
    kernel = window.to(x.device, x.dtype).expand(C, 1, -1, -1)
    return F.conv2d(x, kernel, padding=window.shape[0] // 2, groups=C)


def ssim_map(x: torch.Tensor, y: torch.Tensor,
             window: torch.Tensor | None = None) -> torch.Tensor:
    """逐像素 SSIM 图 (标准 11x11/σ=1.5, data_range=1), 模块级复用入口."""
    if window is None:
        window = _gaussian_kernel()
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = _filter_gauss(x, window), _filter_gauss(y, window)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x2 = _filter_gauss(x * x, window) - mu_x2
    sigma_y2 = _filter_gauss(y * y, window) - mu_y2
    sigma_xy = _filter_gauss(x * y, window) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return num / (den + 1e-12)


class ITLLoss(nn.Module):
    """覆冰感知损失.

    参数
    ----
    lambda_region, lambda_boundary : 两项权重 (默认 0.5 / 0.3, 与旧版一致)。
    w_ssim : SSIM 项在 L_region 内的权重 (默认 0.1)。
    region_term : 'ssim' (默认, 论文表述) 或 'l1' (旧代码行为, 修复 batch bug 后)。
    boundary_ksize : 边界带膨胀核大小 (默认 7, 与旧版一致)。
    """

    def __init__(self, lambda_region: float = 0.5, lambda_boundary: float = 0.3,
                 w_ssim: float = 0.1, region_term: str = "ssim",
                 boundary_ksize: int = 7):
        super().__init__()
        if region_term not in ("ssim", "l1"):
            raise ValueError("region_term 必须为 'ssim' 或 'l1'")
        self.lambda_region = lambda_region
        self.lambda_boundary = lambda_boundary
        self.w_ssim = w_ssim
        self.region_term = region_term
        self.boundary_ksize = boundary_ksize

        self.register_buffer("_window", _gaussian_kernel())
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    # ------------------------------------------------------------------
    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ssim_map(x, y, self._window)

    def _gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Sobel 梯度幅值, 输入 (B,3,H,W) 输出 (B,1,H,W)."""
        gray = img.mean(dim=1, keepdim=True)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                      self.sobel_x.to(img.dtype), padding=0)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                      self.sobel_y.to(img.dtype), padding=0)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def _dilate_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        k = kernel_size
        pad = k // 2
        dilated = F.max_pool2d(F.pad(mask, (pad, pad, pad, pad), mode="replicate"), k, 1)
        return dilated

    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, clear: torch.Tensor,
                ice_mask: torch.Tensor):
        """返回 (region_loss, boundary_loss, total_loss), 均为带梯度张量.

        pred/clear: (B,3,H,W) in [0,1]; ice_mask: (B,1,H,W) in {0,1}。
        """
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float()
            clear = clear.float()
            ice_mask = ice_mask.float()

            zero = pred.sum() * 0.0  # graph-connected 零

            # ---- 1. 区域约束 ----
            n_ice = ice_mask.sum()
            if n_ice > 0:
                mask3 = ice_mask.expand(-1, 3, -1, -1)
                masked_l1 = (torch.abs(pred - clear) * mask3).sum() / (
                    n_ice * 3 + 1e-6
                )
                if self.region_term == "ssim":
                    ssim_map = self._ssim_map(pred, clear)
                    masked_ssim = (ssim_map * mask3).sum() / (n_ice * 3 + 1e-6)
                    region = masked_l1 + self.w_ssim * (1.0 - masked_ssim)
                else:  # 'l1': 旧代码行为 (修复 batch 依赖后)
                    region = masked_l1 + self.w_ssim * masked_l1
            else:
                region = zero

            # ---- 2. 边界约束 ----
            dilated = self._dilate_mask(ice_mask, self.boundary_ksize)
            boundary_band = (dilated - ice_mask).clamp(min=0)
            n_boundary = boundary_band.sum()
            if n_boundary > 0:
                pred_grad = self._gradient(pred)
                clear_grad = self._gradient(clear)
                boundary = (torch.abs(pred_grad - clear_grad) * boundary_band).sum() / (
                    n_boundary + 1e-6
                )
            else:
                boundary = zero

            total = self.lambda_region * region + self.lambda_boundary * boundary

        return region, boundary, total
