"""追踪ROI heads内部，定位分数全为1.0的原因"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

from maskrcnn_inference import load_detectron2_model, D2_MEAN, D2_STD, DEVICE

model = load_detectron2_model()

# 检查RoIHeads的属性名
print("RoIHeads attributes:")
for name, _ in model.roi_heads.named_children():
    print(f"  {name}")

img_bgr = cv2.imread(r"D:\dehaze_fusion\my_test\input\real_0002.png")
h, w = img_bgr.shape[:2]
img_float = img_bgr.astype(np.float32)
img_norm = (img_float - np.array(D2_MEAN, dtype=np.float32)) / np.array(D2_STD, dtype=np.float32)
img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    from torchvision.models.detection.image_list import ImageList
    il = ImageList(img_tensor, [(h, w)])
    features = model.backbone(img_tensor)
    features_list = list(features.values())
    proposals, _ = model.rpn(il, features)

    # 手动运行ROI heads
    roi_pool = model.roi_heads.box_roi_pool
    box_head = model.roi_heads.box_head
    box_predictor = model.roi_heads.box_predictor

    # ROI pool
    roi_features = roi_pool(features, [proposals[0]], il.image_sizes)
    print(f"\nROI features: shape={roi_features.shape}")
    print(f"  min={roi_features.min():.4f}, max={roi_features.max():.4f}, mean={roi_features.mean():.4f}")
    print(f"  std={roi_features.std():.4f}")

    # Flatten and box head
    flat = roi_features.flatten(1)
    print(f"\nFlattened: shape={flat.shape}")
    fc_out = box_head(flat)
    print(f"Box head output: shape={fc_out.shape}")
    print(f"  min={fc_out.min():.4f}, max={fc_out.max():.4f}, mean={fc_out.mean():.4f}")
    print(f"  std={fc_out.std():.4f}")
    print(f"  first 10 values: {fc_out[0][:10].tolist()}")

    # Box predictor
    cls_scores = box_predictor.cls_score(fc_out)
    bbox_pred = box_predictor.bbox_pred(fc_out)
    print(f"\nRaw cls_scores: shape={cls_scores.shape}")
    print(f"  bg (col 0): min={cls_scores[:, 0].min():.4f}, max={cls_scores[:, 0].max():.4f}, mean={cls_scores[:, 0].mean():.4f}")
    print(f"  fg (col 1): min={cls_scores[:, 1].min():.4f}, max={cls_scores[:, 1].max():.4f}, mean={cls_scores[:, 1].mean():.4f}")
    print(f"  first 5 rows:")
    for i in range(min(5, len(cls_scores))):
        print(f"    bg={cls_scores[i, 0]:.4f}, fg={cls_scores[i, 1]:.4f}")

    # Softmax
    probs = F.softmax(cls_scores, dim=-1)
    print(f"\nSoftmax fg probs:")
    print(f"  min={probs[:, 1].min():.8f}, max={probs[:, 1].max():.8f}")
    print(f"  first 5: {probs[:5, 1].tolist()}")

    # 检查fc_out的值是否过大导致softmax饱和
    diff = cls_scores[:, 1] - cls_scores[:, 0]
    print(f"\nScore diff (fg-bg): min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")
    print(f"  When |diff| > 30, softmax will saturate to 1.0 or 0.0")

    # 检查是否有NaN或Inf
    print(f"\nNaN check: fc_out has NaN={torch.isnan(fc_out).any()}, Inf={torch.isinf(fc_out).any()}")
    print(f"NaN check: cls_scores has NaN={torch.isnan(cls_scores).any()}, Inf={torch.isinf(cls_scores).any()}")
