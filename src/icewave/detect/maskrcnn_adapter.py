"""Mask R-CNN (detectron2 → torchvision) 权重迁移适配器.

从 phase6/dehaze_inference.py 抽取的 d2→torchvision 键名映射与加载逻辑,
两处工程化修改:
1. 权重路径显式传参 (旧版写死 ``.trae-cn\\attachments\\...`` 同事文件路径);
2. 映射逻辑不变, 但加载失败时给出可诊断信息 (打印未映射键)。

检测类别语义: 外部权重为 2 类 (背景 + target), target 与"覆冰/部件"的
对应关系未经人工校验, 论文中使用前需明确说明或更换权重。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MASKRCNN_MEAN = [103.530, 116.280, 123.675]
MASKRCNN_STD = [1.0, 1.0, 1.0]


def build_key_mapping(d2_keys) -> dict:
    """detectron2 键名 → torchvision maskrcnn_resnet50_fpn 键名."""
    res_to_layer = {"res2": "layer1", "res3": "layer2",
                    "res4": "layer3", "res5": "layer4"}
    mapping = {}
    for d2k in d2_keys:
        tvk = None
        if d2k == "backbone.bottom_up.stem.conv1.weight":
            tvk = "backbone.body.conv1.weight"
        elif d2k.startswith("backbone.bottom_up.stem.conv1.norm."):
            p = d2k.split("norm.")[-1]
            m = {"weight": "bn1.weight", "bias": "bn1.bias",
                 "running_mean": "bn1.running_mean",
                 "running_var": "bn1.running_var"}
            tvk = f"backbone.body.{m.get(p, '')}" if p in m else None
        elif d2k.startswith("backbone.bottom_up.res"):
            parts = d2k.replace("backbone.bottom_up.", "").split(".")
            ln = res_to_layer.get(parts[0])
            if ln and len(parts) >= 3:
                bid = parts[1]
                rest = ".".join(parts[2:])
                if rest.startswith("shortcut."):
                    sp = rest.replace("shortcut.", "")
                    if sp == "weight":
                        tvk = f"backbone.body.{ln}.{bid}.downsample.0.weight"
                    elif sp.startswith("norm."):
                        np_p = sp.split("norm.")[-1]
                        m2 = {"weight": "1.weight", "bias": "1.bias",
                              "running_mean": "1.running_mean",
                              "running_var": "1.running_var"}
                        tvk = (f"backbone.body.{ln}.{bid}.downsample.{m2.get(np_p, '')}"
                               if np_p in m2 else None)
                elif rest.startswith("conv") and ".norm." in rest:
                    cn = rest.split(".")[0].replace("conv", "")
                    np_p = rest.split("norm.")[-1]
                    m3 = {"weight": f"bn{cn}.weight", "bias": f"bn{cn}.bias",
                          "running_mean": f"bn{cn}.running_mean",
                          "running_var": f"bn{cn}.running_var"}
                    tvk = (f"backbone.body.{ln}.{bid}.{m3.get(np_p, '')}"
                           if np_p in m3 else None)
                elif rest.startswith("conv") and rest.endswith("weight"):
                    cn = rest.replace("conv", "").replace(".weight", "")
                    tvk = f"backbone.body.{ln}.{bid}.conv{cn}.weight"
        elif d2k.startswith("backbone.fpn_lateral"):
            lv = d2k.split("fpn_lateral")[1][0]
            idx = int(lv) - 2
            sfx = d2k.split("fpn_lateral" + lv + ".")[-1]
            tvk = f"backbone.fpn.inner_blocks.{idx}.0.{sfx if sfx in ('weight', 'bias') else ''}".rstrip(".")
        elif d2k.startswith("backbone.fpn_output"):
            lv = d2k.split("fpn_output")[1][0]
            idx = int(lv) - 2
            sfx = d2k.split("fpn_output" + lv + ".")[-1]
            tvk = f"backbone.fpn.layer_blocks.{idx}.0.{sfx if sfx in ('weight', 'bias') else ''}".rstrip(".")
        elif d2k == "proposal_generator.rpn_head.conv.weight":
            tvk = "rpn.head.conv.0.0.weight"
        elif d2k == "proposal_generator.rpn_head.conv.bias":
            tvk = "rpn.head.conv.0.0.bias"
        elif d2k == "proposal_generator.rpn_head.objectness_logits.weight":
            tvk = "rpn.head.cls_logits.weight"
        elif d2k == "proposal_generator.rpn_head.objectness_logits.bias":
            tvk = "rpn.head.cls_logits.bias"
        elif d2k == "proposal_generator.rpn_head.anchor_deltas.weight":
            tvk = "rpn.head.bbox_pred.weight"
        elif d2k == "proposal_generator.rpn_head.anchor_deltas.bias":
            tvk = "rpn.head.bbox_pred.bias"
        elif d2k == "roi_heads.box_head.fc1.weight":
            tvk = "roi_heads.box_head.fc6.weight"
        elif d2k == "roi_heads.box_head.fc1.bias":
            tvk = "roi_heads.box_head.fc6.bias"
        elif d2k == "roi_heads.box_head.fc2.weight":
            tvk = "roi_heads.box_head.fc7.weight"
        elif d2k == "roi_heads.box_head.fc2.bias":
            tvk = "roi_heads.box_head.fc7.bias"
        elif d2k.startswith("roi_heads.box_predictor."):
            tvk = d2k
        elif d2k.startswith("roi_heads.mask_head.mask_fcn"):
            n = d2k.split("mask_fcn")[1].split(".")[0]
            sfx = d2k.split(".")[-1]
            tvk = f"roi_heads.mask_head.{int(n) - 1}.0.{sfx}"
        elif d2k == "roi_heads.mask_head.deconv.weight":
            tvk = "roi_heads.mask_predictor.conv5_mask.weight"
        elif d2k == "roi_heads.mask_head.deconv.bias":
            tvk = "roi_heads.mask_predictor.conv5_mask.bias"
        elif d2k == "roi_heads.mask_head.predictor.weight":
            tvk = "roi_heads.mask_predictor.mask_fcn_logits.weight"
        elif d2k == "roi_heads.mask_head.predictor.bias":
            tvk = "roi_heads.mask_predictor.mask_fcn_logits.bias"
        if tvk:
            mapping[d2k] = tvk
    return mapping


class MaskRCNNDetector:
    """torchvision Mask R-CNN + detectron2 权重 (同事模型迁移).

    参数与后处理 (NMS IoU=0.3, 面积>200, 每图≤20 检测) 与旧实现一致。
    """

    def __init__(self, weights: str | Path, device: str = "cpu",
                 class_name: str = "target", min_area: int = 200,
                 iou_thresh: float = 0.3):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.ops import nms as tv_nms

        self._tv_nms = tv_nms
        self.device = device
        self.class_name = class_name
        self.min_area = min_area
        self.iou_thresh = iou_thresh

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Mask R-CNN 权重不存在: {weights}\n"
                f"请通过 scripts/download_weights.py 获取或显式传参。"
            )
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        d2_sd = ckpt["model"]

        self.model = maskrcnn_resnet50_fpn(
            weights=None, num_classes=2, weights_backbone=None,
            min_size=800, max_size=1333,
            box_score_thresh=0.05, box_nms_thresh=0.3,
            box_detections_per_img=20)
        self.model.transform.image_mean = MASKRCNN_MEAN
        self.model.transform.image_std = MASKRCNN_STD

        tv_sd = self.model.state_dict()
        mapping = build_key_mapping(list(d2_sd.keys()))
        new_sd = {}
        unmapped = []
        for d2k, tvk in mapping.items():
            if tvk in tv_sd:
                d2v, tvv = d2_sd[d2k], tv_sd[tvk]
                if d2v.shape == tvv.shape:
                    new_sd[tvk] = d2v.clone()
                elif "bbox_pred" in tvk or "mask_fcn_logits" in tvk:
                    new_sd[tvk] = torch.cat([d2v, d2v], dim=0)
            else:
                unmapped.append((d2k, tvk))
        for tvk in tv_sd:
            if tvk not in new_sd and "num_batches_tracked" in tvk:
                new_sd[tvk] = torch.tensor(0, dtype=torch.long)
        missing = [k for k in tv_sd if k not in new_sd
                   and "num_batches_tracked" not in k]
        if missing:
            print(f"[MaskRCNN] 警告: {len(missing)} 个 torch 键未获得权重, "
                  f"如: {missing[:5]}")
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(device).eval()

    def detect(self, img_bgr) -> list[dict]:
        img_t = torch.from_numpy(img_bgr).permute(2, 0, 1).float().to(self.device)
        with torch.no_grad():
            outputs = self.model([img_t])
        pred = outputs[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        masks = pred["masks"].cpu().numpy() if "masks" in pred else None

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas >= self.min_area
        boxes, scores = boxes[keep], scores[keep]
        if masks is not None:
            masks = masks[keep]

        if len(boxes) > 1:
            keep_idx = self._tv_nms(torch.from_numpy(boxes),
                                    torch.from_numpy(scores),
                                    self.iou_thresh).numpy()
            boxes, scores = boxes[keep_idx], scores[keep_idx]
            if masks is not None:
                masks = masks[keep_idx]

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            det = {"class": self.class_name, "cls_id": 1,
                   "bbox": (x1, y1, x2, y2), "conf": float(scores[i])}
            if masks is not None:
                det["mask"] = masks[i, 0]
            detections.append(det)
        return detections

    def draw(self, img_bgr, detections):
        import cv2

        annotated = img_bgr.copy()
        color = (0, 0, 255)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if "mask" in det:
                mask_bool = det["mask"] > 0.5
                colored = np.zeros_like(annotated)
                colored[:] = color
                annotated = np.where(
                    mask_bool[:, :, None],
                    cv2.addWeighted(annotated, 0.7, colored, 0.3, 0),
                    annotated)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated
