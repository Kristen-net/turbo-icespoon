"""检测感知损失与多任务不确定性加权 (P1-1 联合优化框架组件).

动机
----
旧管线中检测与去雾完全解耦: 去雾损失只看像素重建, 对"去雾结果是否更利于
下游覆冰/部件检测"没有任何约束。本模块提供把检测需求注入去雾训练的组件:

1. ``CorridorTextureLoss``: 在任务相关区域 (导线/绝缘子走廊) 约束
   (a) 纹理能量 (梯度幅值均值) 的对数保持 → 尺度不变的对比度保持, 对应
       检测器依赖的局部对比特征;
   (b) 逐像素梯度 L1 → 防止走廊内结构细节丢失。
   两项均无标量随 batch 变化的问题, 空走廊返回 graph-connected 零。

2. ``BoxFeaturePreservationLoss``: 框内特征保持损失 (§3.2).
   Stub 版: 基于 Sobel + 像素统计的特征代理, 无需 YOLO 检测器.
   Full 版 (W2): 使用检测器 neck 多尺度特征, 余弦相似度.

3. ``IcePhysicalLoss``: 冰面物理一致性损失 (§3.6).
   透射率 + 不透明度辅助 head 监督, 仅合成数据有真值.

4. ``UncertaintyWeighting``: Kendall & Gal (CVPR'18) 多任务不确定性加权,
   L_total = Σ_i exp(-s_i)·L_i + 0.5·Σ_i s_i, s_i = log σ_i² 可学习,
   替代手工调 λ, 供 joint 模式使用。
"""

from __future__ import annotations

from typing import Optional

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

    Args:
        num_losses: 损失项数 (joint_v2 取 K=5)
        s_init: log σ² 初始值 (0.0 → 初始权重 1.0)
        s_clamp: (min, max) σ 漂移约束, 防止某项 loss 失效 (§4.3)
    """

    def __init__(self, num_losses: int = 2, s_init: float = 0.0,
                 s_clamp: Optional[tuple] = (-6.0, 6.0)):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((num_losses,), s_init))
        self.s_clamp = s_clamp

    def forward(self, losses):
        if len(losses) != self.log_vars.numel():
            raise ValueError(
                f"损失数量 {len(losses)} 与权重数量 {self.log_vars.numel()} 不一致"
            )
        if self.s_clamp is not None:
            self.log_vars.data.clamp_(self.s_clamp[0], self.s_clamp[1])
        weighted = [torch.exp(-s) * l for s, l in zip(self.log_vars, losses)]
        penalty = 0.5 * self.log_vars.sum()
        return sum(weighted) + penalty


class BoxFeaturePreservationLoss(nn.Module):
    """框内特征保持损失 (§3.2).

    Stub 版 (D11): 基于 Sobel 梯度 + 像素统计的特征代理,
    无需 YOLO 检测器前向。计算 pred 与 clear 在 GT 框区域的
    特征余弦相似度。

    Full 版 (W2): 使用检测器 neck 多尺度特征 (P3/P4/P5),
    对每个 GT 框区域取特征向量, 计算 pred 与 clear 的余弦相似度。

    Args:
        feature_type: "stub" (Sobel+像素) 或 "detector" (需传入检测器)
        eps: 余弦相似度分母下界
    """

    def __init__(self, feature_type: str = "stub", eps: float = 1e-8):
        super().__init__()
        self.feature_type = feature_type
        self.eps = eps
        if feature_type == "stub":
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)

    def _extract_stub_features(self, img: torch.Tensor,
                               bboxes: torch.Tensor) -> torch.Tensor:
        """从图像框区域提取 Stub 特征 (Sobel 梯度 + 均值 + 方差).

        Args:
            img: (B, 3, H, W)
            bboxes: (N, 4) 归一化 xyxy, N 个框

        Returns:
            features: (N, 6C) 每框 6×C 维特征 (C=3: RGB 通道)
        """
        B, C, H, W = img.shape
        feats = []
        gray = img.mean(dim=1, keepdim=True)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                      self.sobel_x.to(img.dtype), padding=0)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"),
                      self.sobel_y.to(img.dtype), padding=0)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        for i in range(bboxes.shape[0]):
            x1 = int(bboxes[i, 0] * W)
            y1 = int(bboxes[i, 1] * H)
            x2 = max(x1 + 1, int(bboxes[i, 2] * W))
            y2 = max(y1 + 1, int(bboxes[i, 3] * H))
            # 框内像素统计 (每通道)
            box_pixels = img[:, :, y1:y2, x1:x2]  # (B, C, bh, bw)
            box_grad = grad_mag[:, :, y1:y2, x1:x2]
            # 6 特征: 均值+方差 × 3通道 + 梯度均值+方差 × 1
            mean_rgb = box_pixels.mean(dim=(2, 3))  # (B, C)
            std_rgb = box_pixels.std(dim=(2, 3)) + 1e-6
            mean_grad = box_grad.mean(dim=(2, 3))  # (B, 1)
            std_grad = box_grad.std(dim=(2, 3)) + 1e-6
            feat = torch.cat([mean_rgb, std_rgb, mean_grad, std_grad], dim=1)
            feats.append(feat.mean(dim=0))  # (6C-2,)

        return torch.stack(feats)  # (N, D)

    @staticmethod
    def _crop_feature_box(feat: torch.Tensor, bbox: torch.Tensor,
                          H: int, W: int) -> torch.Tensor:
        """从特征图裁剪框区域并池化为 1-D 向量.

        Args:
            feat: (B, C, fh, fw) 特征图
            bbox: (4,) 归一化 xyxy
            H, W: 原始图像高宽

        Returns:
            (B, C) 空间平均池化后的特征向量
        """
        fh, fw = feat.shape[-2:]
        # 将归一化坐标映射到特征图坐标
        x1 = int(bbox[0] * fw)
        y1 = int(bbox[1] * fh)
        x2 = max(x1 + 1, int(bbox[2] * fw))
        y2 = max(y1 + 1, int(bbox[3] * fh))
        box_feat = feat[:, :, y1:y2, x1:x2]  # (B, C, bh, bw)
        return box_feat.mean(dim=(2, 3))  # (B, C)

    def forward(self, pred: torch.Tensor, clear: torch.Tensor,
                bboxes: torch.Tensor,
                detector: Optional[nn.Module] = None) -> torch.Tensor:
        """计算框内特征保持损失.

        Args:
            pred: (B, 3, H, W) 去雾结果
            clear: (B, 3, H, W) 清晰 GT
            bboxes: (N, 4) 归一化 xyxy 坐标
            detector: DetectorWrapper (feature_type="detector" 时必需)

        Returns:
            loss: scalar, 1 - cos_sim 的均值
        """
        zero = pred.sum() * 0.0

        if bboxes.shape[0] == 0:
            return zero

        if self.feature_type == "stub":
            feat_pred = self._extract_stub_features(pred, bboxes)
            feat_clear = self._extract_stub_features(clear, bboxes)
            cos_sim = F.cosine_similarity(feat_pred, feat_clear,
                                           dim=1, eps=self.eps)
            return (1.0 - cos_sim).mean()

        # Full 版: 检测器多尺度特征
        if detector is None:
            raise ValueError("feature_type='detector' 需要传入 detector")
        B, C, H, W = pred.shape
        feats_pred = detector.extract_neck_features(pred)   # [P3,P4,P5]
        feats_clear = detector.extract_neck_features(clear)  # [P3,P4,P5]

        total_loss = zero
        for fp, fc in zip(feats_pred, feats_clear):
            scale_losses = []
            for i in range(bboxes.shape[0]):
                vp = self._crop_feature_box(fp, bboxes[i], H, W)  # (B, Cp)
                vc = self._crop_feature_box(fc, bboxes[i], H, W)  # (B, Cc)
                # 对 batch 取均值后计算余弦
                vp_flat = vp.mean(dim=0)  # (Cp,)
                vc_flat = vc.mean(dim=0)  # (Cc,)
                cos_sim = F.cosine_similarity(vp_flat.unsqueeze(0),
                                              vc_flat.unsqueeze(0),
                                              dim=1, eps=self.eps)
                scale_losses.append(1.0 - cos_sim.mean())
            total_loss = total_loss + torch.stack(scale_losses).mean()
        return total_loss / len(feats_pred)


class DetectabilityLoss(nn.Module):
    """可检测性代理损失 (§3.3).

    最大化检测器在去雾输出上的 objectness 置信度,
    防止去雾网络过度平滑导致检测器"看不见"目标。

    L = -(1/N) Σ p_obj^(n) · 1[p_cls^(n)* > τ_cls]

    只对 GT 类别概率超过阈值 τ 的 anchor 计入, 忽略背景 anchor。
    """

    def __init__(self, tau_cls: float = 0.3):
        super().__init__()
        self.tau_cls = tau_cls

    def forward(self, det_out: list[dict],
                gt_classes: torch.Tensor) -> torch.Tensor:
        """计算可检测性损失.

        Args:
            det_out: DetectorWrapper.forward 的输出, list[dict], 每个含:
                - cls: (B, N, num_classes)
                - obj: (B, N)
                - scale: 尺度索引
            gt_classes: (B, num_gt) GT 类别 ID

        Returns:
            loss: scalar (负号 → 最大化置信度)
        """
        total = 0.0
        count = 0
        for scale_out in det_out:
            cls_scores = scale_out["cls"]   # (B, N, C)
            obj_scores = scale_out["obj"]   # (B, N)
            B, N, C = cls_scores.shape
            for b in range(B):
                if b >= gt_classes.shape[0]:
                    continue
                gts = gt_classes[b]  # (num_gt,)
                # 对每个 anchor, 检查其 cls 预测是否对任一 GT 类别 > τ
                mask = torch.zeros(N, device=cls_scores.device,
                                   dtype=torch.bool)
                for g in gts:
                    g_int = int(g.item()) if hasattr(g, 'item') else int(g)
                    if 0 <= g_int < C:
                        mask |= (cls_scores[b, :, g_int] > self.tau_cls)
                if mask.any():
                    total = total - obj_scores[b][mask].mean()
                    count += 1
        if count == 0:
            # 退化: 无匹配 anchor, 用全图均值避免梯度为 0
            all_obj = torch.cat([s["obj"].flatten() for s in det_out])
            return -all_obj.mean() * 0.01
        return total / count


class BoxAlignLoss(nn.Module):
    """框对齐辅助损失 (§3.4).

    CIoU + DFL: 让去雾前后的检测框在几何层面与 GT 接近。

    由于 DetectorWrapper 输出的是 raw box 回归值 (grid 坐标系),
    本实现简化为: 对每个 GT 框, 找到检测输出中最接近的框,
    计算 CIoU 损失。DFL 作为软正则项。

    Args:
        lambda_ciou: CIoU 损失权重
        lambda_dfl: DFL 正则项权重
    """

    def __init__(self, lambda_ciou: float = 1.0,
                 lambda_dfl: float = 0.25):
        super().__init__()
        self.lambda_ciou = lambda_ciou
        self.lambda_dfl = lambda_dfl

    @staticmethod
    def _ciou_loss(pred_boxes: torch.Tensor,
                   gt_boxes: torch.Tensor) -> torch.Tensor:
        """CIoU 损失 (1 - CIoU).

        Args:
            pred_boxes: (N, 4) xyxy
            gt_boxes: (M, 4) xyxy

        Returns:
            loss: scalar (对最优配对取均值)
        """
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return pred_boxes.sum() * 0.0

        # 贪心配对: 对每个 GT 找 IoU 最高的 pred
        total = 0.0
        for i in range(gt_boxes.shape[0]):
            best_iou = 0.0
            best_j = -1
            for j in range(pred_boxes.shape[0]):
                iou = _compute_iou(pred_boxes[j], gt_boxes[i])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                # CIoU 简化版: 1 - IoU + center_dist / diag^2
                p = pred_boxes[best_j]
                g = gt_boxes[i]
                pcx = (p[0] + p[2]) / 2
                pcy = (p[1] + p[3]) / 2
                gcx = (g[0] + g[2]) / 2
                gcy = (g[1] + g[3]) / 2
                diag2 = (g[2] - g[0]) ** 2 + (g[3] - g[1]) ** 2 + 1e-6
                center_term = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) / diag2
                total = total + (1.0 - best_iou) + center_term
        return total / gt_boxes.shape[0]

    @staticmethod
    def _dfl_term(pred_boxes: torch.Tensor,
                 gt_boxes: torch.Tensor) -> torch.Tensor:
        """DFL 软正则: 鼓励预测分布尖锐化 (简化版 L1).

        在完整 YOLOv8 中 DFL 对回归分布操作; 此处用 L1 作为简化代理。
        """
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return pred_boxes.sum() * 0.0
        # 对每个 GT 取最优匹配 pred 的 L1
        total = 0.0
        for i in range(gt_boxes.shape[0]):
            best_l1 = float('inf')
            for j in range(pred_boxes.shape[0]):
                l1 = (pred_boxes[j] - gt_boxes[i]).abs().sum()
                if l1 < best_l1:
                    best_l1 = l1
            total = total + best_l1
        return total / gt_boxes.shape[0]

    def forward(self, det_out: list[dict],
                gt_bboxes: torch.Tensor,
                gt_classes: torch.Tensor) -> torch.Tensor:
        """计算框对齐损失.

        Args:
            det_out: DetectorWrapper.forward 的输出
            gt_bboxes: (B, num_gt, 4) 归一化 xyxy
            gt_classes: (B, num_gt) GT 类别 ID

        Returns:
            loss: scalar
        """
        total_ciou = 0.0
        total_dfl = 0.0
        count = 0
        for scale_out in det_out:
            B = scale_out["cls"].shape[0]
            grid_h = scale_out["grid_h"]
            grid_w = scale_out["grid_w"]
            box_raw = scale_out["box"]  # (B, N, 4) raw 回归值
            for b in range(B):
                if b >= gt_bboxes.shape[0]:
                    continue
                n_gt = gt_bboxes[b].shape[0]
                if n_gt == 0:
                    continue
                # 将 GT 归一化坐标转为 grid 坐标
                gt_in_grid = gt_bboxes[b].clone()
                gt_in_grid[:, 0] *= grid_w
                gt_in_grid[:, 1] *= grid_h
                gt_in_grid[:, 2] *= grid_w
                gt_in_grid[:, 3] *= grid_h
                # 将 raw box 转为 xyxy (简化: 直接用 raw 值)
                pred_boxes = box_raw[b]  # (N, 4)
                total_ciou = total_ciou + self._ciou_loss(
                    pred_boxes, gt_in_grid)
                total_dfl = total_dfl + self._dfl_term(
                    pred_boxes, gt_in_grid)
                count += 1
        if count == 0:
            return det_out[0]["box"].sum() * 0.0
        return (self.lambda_ciou * total_ciou / count
                + self.lambda_dfl * total_dfl / count)


def _compute_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个 xyxy 框的 IoU."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


class IcePhysicalLoss(nn.Module):
    """冰面物理一致性损失 (§3.6).

    监督去雾主干的两个轻量辅助 head (透射率 + 不透明度),
    仅在合成数据上有真值 (真实雾无 D 真值)。

    L = λ_trans * ||T_hat - T*||_1 + λ_ice * ||1 - α_hat - α_ice||_1
    """

    def __init__(self, lambda_trans: float = 1.0, lambda_ice: float = 1.0):
        super().__init__()
        self.lambda_trans = lambda_trans
        self.lambda_ice = lambda_ice

    def forward(self, t_hat: torch.Tensor, t_target: torch.Tensor,
                alpha_hat: torch.Tensor, alpha_target: torch.Tensor) -> torch.Tensor:
        """计算物理一致性损失.

        Args:
            t_hat: (B, 1, H, W) 估计透射率
            t_target: (B, 1, H, W) 合成器真值透射率
            alpha_hat: (B, 1, H, W) 估计冰层不透明度
            alpha_target: (B, 1, H, W) 合成器真值冰层不透明度

        Returns:
            loss: scalar
        """
        loss_trans = F.l1_loss(t_hat, t_target)
        loss_ice = F.l1_loss(alpha_hat, alpha_target)
        return self.lambda_trans * loss_trans + self.lambda_ice * loss_ice
