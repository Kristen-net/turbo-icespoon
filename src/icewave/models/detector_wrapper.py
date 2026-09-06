"""检测器封装 (JOINT_OPTIMIZATION_FRAMEWORK §5.1 Stage-B).

冻结 YOLOv8 backbone, 微调 neck+head, 暴露:
1. ``extract_neck_features(img)`` → P3/P4/P5 多尺度特征 (供 BoxFeaturePreservationLoss)
2. ``forward(img)`` → raw detection outputs {cls, box, obj} (供 DetectabilityLoss / BoxAlignLoss)

无 ultralytics 时使用 Stub 模式 (简单 CNN), 保证测试可在 CPU 运行。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StubBackbone(nn.Module):
    """简单 CNN 替代 YOLOv8 backbone (测试用)."""

    def __init__(self, in_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1), nn.SiLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1), nn.SiLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, 2, 1), nn.SiLU(),
        )

    def forward(self, x):
        f0 = self.layer0(x)   # /4  → P3
        f1 = self.layer1(f0)  # /8  → P4
        f2 = self.layer2(f1)  # /16 → P5
        return [f0, f1, f2]


class _StubNeck(nn.Module):
    """简单 FPN 替代 YOLOv8 neck+head."""

    def __init__(self, in_chs=(32, 64, 128), num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, 64, 1) for c in in_chs
        ])
        self.heads = nn.ModuleList([
            nn.Conv2d(64, 5 + num_classes, 1) for _ in in_chs
        ])

    def forward(self, feats):
        """feats: [P3, P4, P5] → list of (cls, box, obj) per scale."""
        outs = []
        for i, (f, lat, head) in enumerate(zip(feats, self.lateral, self.heads)):
            x = lat(f)
            raw = head(x)
            B, _, H, W = raw.shape
            raw = raw.permute(0, 2, 3, 1).reshape(B, H * W, 5 + self.num_classes)
            obj = torch.sigmoid(raw[..., 4])
            cls = torch.sigmoid(raw[..., 5:])
            box = raw[..., :4]
            outs.append({"cls": cls, "box": box, "obj": obj,
                         "grid_h": H, "grid_w": W, "scale": i})
        return outs


class DetectorWrapper(nn.Module):
    """冻结 backbone 的检测器封装.

    Args:
        weights: YOLOv8 权重路径 (None → Stub 模式)
        num_classes: 类别数
        freeze_backbone: 冻结 backbone 参数
    """

    def __init__(self, weights: Optional[str] = None,
                 num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self._is_stub = weights is None

        if self._is_stub:
            self.backbone = _StubBackbone(in_ch=3, base_ch=16)
            self.neck = _StubNeck(in_chs=(32, 64, 128),
                                  num_classes=num_classes)
        else:
            try:
                from ultralytics import YOLO
                model = YOLO(weights)
                self.backbone, self.neck = self._split_yolo(model.model.model)
            except ImportError:
                raise ImportError(
                    "加载真实 YOLOv8 需要 ultralytics: "
                    "pip install 'ultralytics>=8.2'"
                )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _split_yolo(yolo_layers):
        """将 YOLOv8 层列表拆分为 backbone + neck+head."""
        raise NotImplementedError("真实 YOLOv8 拆分待 GPU 环境实现")

    def extract_neck_features(self, img: torch.Tensor) -> list[torch.Tensor]:
        """提取 P3/P4/P5 三尺度特征 (供 BoxFeaturePreservationLoss).

        Args:
            img: (B, 3, H, W)

        Returns:
            [P3, P4, P5] 张量列表, 每个形状 (B, C, H/s, W/s)
        """
        with torch.no_grad():
            feats = self.backbone(img)
        # neck 的 lateral 输出 (有梯度)
        return [lat(f) for lat, f in zip(self.neck.lateral, feats)]

    def forward(self, img: torch.Tensor) -> list[dict]:
        """前向传播, 返回多尺度检测输出.

        Returns:
            list[dict], 每个含:
            - cls: (B, N, num_classes) 类别概率
            - box: (B, N, 4) 边界框回归值
            - obj: (B, N) objectness
            - grid_h, grid_w: 网格尺寸
            - scale: 尺度索引
        """
        feats = self.backbone(img)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            feats = [f.detach() for f in feats]
        return self.neck(feats)

    @property
    def trainable_parameters(self):
        """返回可训练参数 (neck+head, 不含 backbone)."""
        return list(self.neck.parameters())
