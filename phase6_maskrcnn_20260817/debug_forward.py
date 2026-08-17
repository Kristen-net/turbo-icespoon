"""手动追踪Mask R-CNN前向传播，定位分数全为1.0的原因"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

from maskrcnn_inference import load_detectron2_model, D2_MEAN, D2_STD, DEVICE

model = load_detectron2_model()

# 读取测试图片
img_bgr = cv2.imread(r"D:\dehaze_fusion\my_test\input\real_0002.png")
h, w = img_bgr.shape[:2]
print(f"Image: {w}x{h}, dtype={img_bgr.dtype}")

# 手动预处理: BGR float [0,255] → 减去均值
img_float = img_bgr.astype(np.float32)
img_norm = (img_float - np.array(D2_MEAN, dtype=np.float32)) / np.array(D2_STD, dtype=np.float32)
img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
print(f"Input tensor: shape={img_tensor.shape}, min={img_tensor.min():.2f}, max={img_tensor.max():.2f}, mean={img_tensor.mean():.2f}")

# 1. 运行backbone+FPN
print("\n=== Step 1: Backbone + FPN ===")
with torch.no_grad():
    features = model.backbone(img_tensor)

print(f"Feature maps: {len(features)} levels")
for i, (name, feat) in enumerate(features.items()):
    print(f"  {name}: shape={feat.shape}, min={feat.min():.4f}, max={feat.max():.4f}, mean={feat.mean():.4f}")

# 2. 运行RPN
print("\n=== Step 2: RPN ===")
with torch.no_grad():
    image_list = [img_tensor.squeeze(0)]
    from torchvision.models.detection.image_list import ImageList
    il = ImageList(img_tensor, [(h, w)])

    features_list = [features[f'0'], features[f'1'], features[f'2'], features[f'3'], features[f'pool']]
    if len(features) == 5:
        features_list = list(features.values())

    proposals, rpn_losses = model.rpn(il, features)
    print(f"RPN proposals: {len(proposals[0])} boxes")
    if len(proposals[0]) > 0:
        boxes = proposals[0]
        print(f"  first 5 boxes: {boxes[:5].tolist()}")
        bw = (boxes[:, 2] - boxes[:, 0])
        bh = (boxes[:, 3] - boxes[:, 1])
        print(f"  box widths: min={bw.min():.1f}, max={bw.max():.1f}, mean={bw.mean():.1f}")
        print(f"  box heights: min={bh.min():.1f}, max={bh.max():.1f}, mean={bh.mean():.1f}")

# 3. 运行ROI heads
print("\n=== Step 3: ROI Heads ===")
with torch.no_grad():
    detections, _ = model.roi_heads(features, proposals, il.image_sizes)
    det = detections[0]
    print(f"Detections: {len(det['boxes'])} boxes")
    if len(det['boxes']) > 0:
        scores = det['scores']
        print(f"  scores: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")
        print(f"  unique scores: {torch.unique(scores).numel()}")

        # 检查原始分类器输出 (softmax前)
        # 手动运行box head
        box_features = model.roi_heads.box_roi_align(features, proposals[0], il.image_sizes)
        print(f"\n  ROI features: shape={box_features.shape}")
        print(f"  ROI features: min={box_features.min():.4f}, max={box_features.max():.4f}, mean={box_features.mean():.4f}")

        box_features = box_features.flatten(1)
        box_features = model.roi_heads.box_head(box_features)
        print(f"  Box head output: shape={box_features.shape}")
        print(f"  Box head: min={box_features.min():.4f}, max={box_features.max():.4f}, mean={box_features.mean():.4f}")

        cls_scores = model.roi_heads.box_predictor.cls_score(box_features)
        print(f"\n  Raw cls_scores: shape={cls_scores.shape}")
        print(f"  cls_scores[:, 0] (bg): min={cls_scores[:, 0].min():.4f}, max={cls_scores[:, 0].max():.4f}, mean={cls_scores[:, 0].mean():.4f}")
        print(f"  cls_scores[:, 1] (fg): min={cls_scores[:, 1].min():.4f}, max={cls_scores[:, 1].max():.4f}, mean={cls_scores[:, 1].mean():.4f}")
        print(f"  diff (fg-bg): min={(cls_scores[:, 1]-cls_scores[:, 0]).min():.4f}, max={(cls_scores[:, 1]-cls_scores[:, 0]).max():.4f}")

        # softmax
        probs = F.softmax(cls_scores, dim=-1)
        print(f"\n  Softmax fg probs: min={probs[:, 1].min():.6f}, max={probs[:, 1].max():.6f}")
