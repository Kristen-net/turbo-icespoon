"""
使用MCT(DehazeFormer混合数据集变体)预训练权重
验证去雾效果
"""
import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer\hf_demo")

import os
import torch
import cv2
import numpy as np
from models.dehazeformer import MCT

# ==================== 1. 加载MCT模型 ====================
print("=" * 50)
print("1. 加载MCT预训练模型")
print("=" * 50)

model = MCT()
ckpt_path = r"D:\dehaze_fusion\DehazeFormer\pretrained\saved_models\dehazeformer.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
if missing:
    print(f"    Missing: {missing[:3]}")
if unexpected:
    print(f"    Unexpected: {unexpected[:3]}")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params / 1e6:.2f}M")

# ==================== 2. 测试去雾 ====================
print("\n" + "=" * 50)
print("2. 测试去雾效果")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

def dark_channel(img_bgr, patch=15):
    min_val = np.min(img_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)

out_dir = r"D:\dehaze_fusion\DehazeFormer\test_output"
os.makedirs(out_dir, exist_ok=True)

# 测试1: 合成雾图
print("\n  --- 测试1: 合成雾图 ---")
hazy_dir = r"D:\DATA_ALL\dataset\train\hazy"
hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])

for fname in hazy_files[:3]:
    hazy_path = os.path.join(hazy_dir, fname)
    img = cv2.imread(hazy_path)
    if img is None:
        continue

    # 预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(img_tensor)

    out_img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    dc_before = dark_channel(img)
    dc_after = dark_channel(out_img)

    print(f"  {fname}: 暗通道 {dc_before:.4f} -> {dc_after:.4f} (降低{(1-dc_after/max(dc_before,0.001))*100:.1f}%)")

    if fname == hazy_files[0]:
        cv2.imwrite(os.path.join(out_dir, "synth_input.png"), img)
        cv2.imwrite(os.path.join(out_dir, "synth_output.png"), out_img)

# 测试2: 真实雾图
print("\n  --- 测试2: 真实雾图 ---")
real_dir = r"D:\DATA_ALL\dataset\test\hazy_real"
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.png')])

for fname in real_files[:3]:
    real_path = os.path.join(real_dir, fname)
    img = cv2.imread(real_path)
    if img is None:
        continue

    # resize到合理尺寸
    h, w = img.shape[:2]
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(img_tensor)

    out_img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    dc_before = dark_channel(img)
    dc_after = dark_channel(out_img)

    print(f"  {fname}: 暗通道 {dc_before:.4f} -> {dc_after:.4f} (降低{(1-dc_after/max(dc_before,0.001))*100:.1f}%)")

    if fname == real_files[0]:
        cv2.imwrite(os.path.join(out_dir, "real_input.png"), img)
        cv2.imwrite(os.path.join(out_dir, "real_output.png"), out_img)

# 显存
mem = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"\n  峰值显存: {mem:.0f} MB")
print(f"\n结果保存到: {out_dir}")
print("验证完成!")
