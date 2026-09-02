"""
ITL: Ice-aware Territory Loss

覆冰感知领地损失, 约束去雾模型在冰区不过度平滑

设计动机:
  通用去雾方法倾向于过度平滑白色覆冰区域,
  导致冰区纹理丢失, 影响下游覆冰检测

双约束:
  1. 区域约束 (Region): 在冰区内加权重建损失,
     强调冰区纹理保持, 防止过度平滑
  2. 边界约束 (Boundary): 在冰/非冰边界处梯度损失,
     保持覆冰边界锐利

L_itl = λ_region * L_region + λ_boundary * L_boundary

L_region = SSIM(pred_ice, clear_ice)  (冰区内SSIM)
L_boundary = ||∇pred - ∇clear||_1 * boundary_mask  (边界梯度L1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class ITLLoss(nn.Module):
    """Ice-aware Territory Loss"""

    def __init__(self, lambda_region=0.5, lambda_boundary=0.3):
        super().__init__()
        self.lambda_region = lambda_region
        self.lambda_boundary = lambda_boundary
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Sobel算子 (边界梯度提取)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, img):
        """计算图像梯度 (单通道)"""
        gray = img.mean(dim=1, keepdim=True)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                       self.sobel_x, padding=0)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                       self.sobel_y, padding=0)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def _dilate_mask(self, mask, kernel_size=5):
        """膨胀掩码 → 边界带"""
        padding = kernel_size // 2
        weight = torch.ones(1, 1, kernel_size, kernel_size,
                            device=mask.device, dtype=mask.dtype)
        dilated = F.conv2d(
            F.pad(mask, (padding, padding, padding, padding), mode="replicate"),
            weight, padding=0
        )
        dilated = (dilated > 0).float()
        return dilated

    def forward(self, pred, clear, ice_mask):
        """
        pred:     (B, 3, H, W) 去雾结果 [0,1]
        clear:    (B, 3, H, W) 清晰图 [0,1]
        ice_mask: (B, 1, H, W) 覆冰掩码 [0,1]

        返回: (region_loss, boundary_loss, total_loss)
        """
        with torch.amp.autocast('cuda', enabled=False):
            pred = pred.float()
            clear = clear.float()
            ice_mask = ice_mask.float()

            B, C, H, W = pred.shape

            # --- 1. 区域约束: 冰区内SSIM损失 ---
            ice_mask_flat = ice_mask.expand(-1, C, -1, -1)

            n_ice = ice_mask.sum() + 1e-6
            n_total = ice_mask.numel()

            if n_ice / n_total > 0.001:
                region_l1 = (torch.abs(pred - clear) * ice_mask_flat).sum() / n_ice / C

                ice_weight = ice_mask_flat / (ice_mask_flat.sum() + 1e-6) * H * W
                weighted_diff = (torch.abs(pred - clear) * ice_weight).mean()
                region_ssim_loss = weighted_diff

                region_loss = region_l1 + 0.1 * region_ssim_loss
            else:
                region_loss = torch.tensor(0.0, device=pred.device)

            # --- 2. 边界约束: 冰/非冰边界梯度保持 ---
            dilated = self._dilate_mask(ice_mask, kernel_size=7)
            boundary = (dilated - ice_mask).clamp(min=0)

            if boundary.sum() > 0:
                pred_grad = self._gradient(pred)
                clear_grad = self._gradient(clear)

                boundary_expanded = boundary.expand(-1, 1, -1, -1)
                n_boundary = boundary.sum() + 1e-6

                boundary_loss = (torch.abs(pred_grad - clear_grad) *
                                 boundary_expanded).sum() / n_boundary
            else:
                boundary_loss = torch.tensor(0.0, device=pred.device)

            total = self.lambda_region * region_loss + self.lambda_boundary * boundary_loss

        return region_loss.item(), boundary_loss.item(), total


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    itl = ITLLoss().to(device)

    B, C, H, W = 2, 3, 192, 192
    pred = torch.rand(B, C, H, W, device=device)
    clear = torch.rand(B, C, H, W, device=device)
    ice_mask = (torch.rand(B, 1, H, W, device=device) > 0.5).float()

    r, b, total = itl(pred, clear, ice_mask)
    print(f"Region loss: {r:.6f}")
    print(f"Boundary loss: {b:.6f}")
    print(f"Total ITL loss: {total:.6f}")
    print("ITL loss test passed!")
