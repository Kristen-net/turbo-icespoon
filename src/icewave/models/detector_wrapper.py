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


# ---------------------------------------------------------------------------
# YOLOv8 真实模型封装 (基于 hook, 不拆 Sequential)
# ---------------------------------------------------------------------------
class _YOLOv8BackboneAdapter(nn.Module):
    """包装 YOLOv8 层 0-9 (backbone), 输出 [P3, P4, P5].

    YOLOv8 backbone 结构 (yolov8.yaml):
        0: Conv P1/2    1: C2f P2/4    2: Conv P3/8    3: C2f
        4: Conv P4/16  5: C2f         6: Conv P5/32  7: C2f
        8: SPPF
    保存的索引 (供 neck concat): 4 (P3), 6 (P4), 8/9 (P5)
    """

    # backbone 终止层 (0-indexed), backbone = layers[:9+1]
    BACKBONE_END = 9  # SPPF 在 index 9 (如有 SPPF) 或 8

    def __init__(self, yolo_layers):
        super().__init__()
        self.layers = nn.ModuleList(yolo_layers[:self.BACKBONE_END + 1])
        # 保存需要输出的层索引 (P3=4, P4=6, P5=8/9)
        self._p3_idx = 4
        self._p4_idx = 6
        self._p5_idx = min(self.BACKBONE_END, len(self.layers) - 1)

    def forward(self, x):
        saved = {}
        for i, layer in enumerate(self.layers):
            f = layer.f if hasattr(layer, 'f') else -1
            if f != -1:
                if isinstance(f, int):
                    x = saved[f]
                else:
                    x = [saved[j] if j != -1 else x for j in f]
            x = layer(x)
            # 只保存需要的中间结果
            if i in (self._p3_idx, self._p4_idx, self._p5_idx,
                     self._p4_idx - 1, self._p3_idx - 1):
                saved[i] = x
            # 也保存后面可能需要的
            if i >= self._p3_idx - 1:
                saved[i] = x
        return [saved[self._p3_idx], saved[self._p4_idx], saved[self._p5_idx]]


class _YOLOv8NeckAdapter(nn.Module):
    """包装 YOLOv8 层 10-22 (neck+head), 接收 [P3, P4, P5].

    使用 YOLOv8 的路由逻辑 (m.f 字段) 处理 FPN+PAN skip connections。
    最终 Detect 层输出 [cls, box, obj] per scale。
    """

    def __init__(self, yolo_layers, backbone_end: int = 9,
                 num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList(yolo_layers[backbone_end + 1:])
        self._backbone_end = backbone_end
        self._p3_idx = 4
        self._p4_idx = 6
        self._p5_idx = min(backbone_end, len(yolo_layers) - 1)
        # lateral (供 extract_neck_features 使用)
        self._lateral_channels = [256, 512, 1024]  # YOLOv8 P3/P4/P5 通道

    def forward(self, feats):
        """feats: [P3, P4, P5] from backbone, 返回检测输出 list[dict]."""
        saved = {
            self._p3_idx: feats[0],
            self._p4_idx: feats[1],
            self._p5_idx: feats[2],
        }
        x = feats[2]  # 从 P5 开始 (upsample)
        for i, layer in enumerate(self.layers):
            real_idx = self._backbone_end + 1 + i
            f = layer.f if hasattr(layer, 'f') else -1
            if f != -1:
                if isinstance(f, int):
                    x = saved.get(f, x)
                else:
                    x = [saved.get(j, x) if j != -1 else x for j in f]
            x = layer(x)
            saved[real_idx] = x

        # 解析 Detect 输出
        return self._parse_detect_output(x)

    def _parse_detect_output(self, det_out):
        """将 YOLOv8 Detect 层输出转为统一格式 list[dict]."""
        outs = []
        if isinstance(det_out, (list, tuple)):
            for i, scale_out in enumerate(det_out):
                if isinstance(scale_out, torch.Tensor):
                    B = scale_out.shape[0]
                    if scale_out.dim() == 4:
                        B, C, H, W = scale_out.shape
                        scale_out = scale_out.permute(0, 2, 3, 1).reshape(
                            B, H * W, C)
                    n_anchors = scale_out.shape[1]
                    box_dim = 4 + 4 * 4  # 4 + reg_max * 4 (YOLOv8 DFL)
                    if scale_out.shape[-1] >= box_dim + self.num_classes:
                        box_raw = scale_out[..., :4]
                        obj = torch.sigmoid(scale_out[..., box_dim:box_dim + 1].squeeze(-1))
                        cls = torch.sigmoid(scale_out[..., box_dim + 1:])
                    else:
                        box_raw = scale_out[..., :4]
                        obj = torch.sigmoid(scale_out[..., 4])
                        cls = torch.sigmoid(scale_out[..., 5:])
                    outs.append({
                        "cls": cls, "box": box_raw, "obj": obj,
                        "grid_h": int(n_anchors ** 0.5),
                        "grid_w": int(n_anchors ** 0.5),
                        "scale": i,
                    })
        return outs

    @property
    def lateral(self):
        """返回 lateral conv 列表 (供 extract_neck_features 兼容).

        用 1x1 conv 近似: YOLOv8 neck 的第一个 C2f 有降维效果。
        此属性仅为 API 兼容; 真实特征提取用 extract_neck_features。
        """
        if not hasattr(self, '_lateral_mods'):
            self._lateral_mods = nn.ModuleList([
                nn.Conv2d(c, c, 1) for c in self._lateral_channels
            ])
        return self._lateral_mods


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
                yolo_layers = model.model.model
                self.backbone = _YOLOv8BackboneAdapter(yolo_layers)
                self.neck = _YOLOv8NeckAdapter(
                    yolo_layers,
                    backbone_end=self.backbone.BACKBONE_END,
                    num_classes=num_classes,
                )
            except ImportError:
                raise ImportError(
                    "加载真实 YOLOv8 需要 ultralytics: "
                    "pip install 'ultralytics>=8.2'"
                )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

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
        if self._is_stub:
            return [lat(f) for lat, f in zip(self.neck.lateral, feats)]
        # YOLOv8: 直接返回 backbone 特征 (已 detach)
        return feats

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
