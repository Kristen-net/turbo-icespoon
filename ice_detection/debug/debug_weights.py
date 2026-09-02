"""深入检查权重加载和中间特征"""
import torch
import cv2
import numpy as np

CKPT_PATH = r"c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth"
DEVICE = "cuda"

ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
d2_sd = ckpt['model']

# 检查关键参数值
print("=== 关键参数检查 ===")
for key in ['roi_heads.box_predictor.cls_score.weight',
            'roi_heads.box_predictor.cls_score.bias',
            'roi_heads.box_predictor.bbox_pred.weight',
            'roi_heads.box_predictor.bbox_pred.bias']:
    if key in d2_sd:
        val = d2_sd[key]
        print(f"{key}:")
        print(f"  shape: {val.shape}")
        print(f"  min: {val.min().item():.6f}, max: {val.max().item():.6f}")
        print(f"  mean: {val.mean().item():.6f}, std: {val.std().item():.6f}")
        if 'bias' in key:
            print(f"  values: {val.tolist()}")
        elif 'cls_score' in key and 'weight' in key:
            print(f"  row 0 (bg) norm: {val[0].norm().item():.4f}")
            print(f"  row 1 (fg) norm: {val[1].norm().item():.4f}")

print("\n=== box_head检查 ===")
for key in ['roi_heads.box_head.fc1.weight', 'roi_heads.box_head.fc1.bias',
            'roi_heads.box_head.fc2.weight', 'roi_heads.box_head.fc2.bias']:
    if key in d2_sd:
        val = d2_sd[key]
        print(f"{key}: shape={val.shape}, min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")

print("\n=== stem检查 ===")
stem_w = d2_sd['backbone.bottom_up.stem.conv1.weight']
print(f"stem.conv1.weight: shape={stem_w.shape}")
print(f"  min={stem_w.min():.4f}, max={stem_w.max():.4f}, mean={stem_w.mean():.4f}")
print(f"  channel-wise means: {stem_w.mean(dim=(2,3)).tolist()[:5]}")

# 检查是否有FrozenBN的标志
print("\n=== BN参数检查 (stem) ===")
for key in d2_sd:
    if 'stem' in key and 'norm' in key:
        print(f"  {key}: {d2_sd[key].shape} min={d2_sd[key].min():.4f} max={d2_sd[key].max():.4f}")

# 检查是否所有BN的running_var都接近1 (FrozenBN特征)
print("\n=== BN running_var检查 ===")
for key in d2_sd:
    if 'running_var' in key and 'stem' in key:
        val = d2_sd[key]
        print(f"  {key}: mean={val.mean():.4f}, min={val.min():.4f}, max={val.max():.4f}")
        # 如果running_var都是1, 说明可能是FrozenBN且未训练
        if val.mean() > 0.99 and val.std() < 0.01:
            print(f"    -> 可能是FrozenBN, running_var未更新")

# 检查trainer中是否有config信息
print("\n=== Trainer检查 ===")
trainer = ckpt.get('trainer', {})
if isinstance(trainer, dict):
    _trainer = trainer.get('_trainer', {})
    if isinstance(_trainer, dict):
        opt = _trainer.get('optimizer', {})
        if isinstance(opt, dict):
            param_groups = opt.get('param_groups', [])
            if param_groups:
                pg = param_groups[0]
                print(f"  optimizer lr: {pg.get('lr', 'N/A')}")
                print(f"  optimizer momentum: {pg.get('momentum', 'N/A')}")
                print(f"  optimizer weight_decay: {pg.get('weight_decay', 'N/A')}")

# 实际加载到torchvision模型并检查
print("\n=== 加载到torchvision后检查 ===")
from maskrcnn_inference import load_detectron2_model
model = load_detectron2_model()

# 检查加载后的cls_score
tv_cls_w = model.roi_heads.box_predictor.cls_score.weight
tv_cls_b = model.roi_heads.box_predictor.cls_score.bias
print(f"torchvision cls_score.weight: shape={tv_cls_w.shape}")
print(f"  row 0 (bg) norm: {tv_cls_w[0].norm().item():.4f}")
print(f"  row 1 (fg) norm: {tv_cls_w[1].norm().item():.4f}")
print(f"torchvision cls_score.bias: {tv_cls_b.tolist()}")

# 比较原始detectron2和加载后的值
print("\n=== 权重值比较 (前5个) ===")
d2_cls_w = d2_sd['roi_heads.box_predictor.cls_score.weight']
d2_cls_b = d2_sd['roi_heads.box_predictor.cls_score.bias']
print(f"d2 cls_score.bias: {d2_cls_b.tolist()}")
print(f"tv cls_score.bias: {tv_cls_b.tolist()}")
print(f"d2 cls_score.weight[0][:5]: {d2_cls_w[0][:5].tolist()}")
print(f"tv cls_score.weight[0][:5]: {tv_cls_w[0][:5].tolist()}")
