"""
验证DehazeFormer预训练权重加载 + 测试去雾效果
"""
import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")

import os
import torch
import cv2
import numpy as np
from models.dehazeformer import dehazeformer_s

# ==================== 1. 加载权重 ====================
print("=" * 50)
print("1. 加载预训练权重")
print("=" * 50)

ckpt_path = r"D:\dehaze_fusion\DehazeFormer\pretrained\saved_models\dehazeformer.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# 去掉 "basenet." 前缀
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("basenet.", "") if k.startswith("basenet.") else k
    new_state_dict[new_key] = v

print(f"  原始键数: {len(state_dict)}")
print(f"  清洗后键数: {len(new_state_dict)}")

# ==================== 2. 实例化模型并加载 ====================
print("\n" + "=" * 50)
print("2. 加载到DehazeFormer-S")
print("=" * 50)

model = dehazeformer_s()
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"  Missing keys: {len(missing)}")
if missing:
    print(f"    前5个: {missing[:5]}")
print(f"  Unexpected keys: {len(unexpected)}")
if unexpected:
    print(f"    前5个: {unexpected[:5]}")

# ==================== 3. 测试去雾 ====================
print("\n" + "=" * 50)
print("3. 测试去雾效果")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# 读一张合成雾图
hazy_path = None
hazy_dir = r"D:\DATA_ALL\dataset\train\hazy"
for f in sorted(os.listdir(hazy_dir)):
    if f.endswith('.png'):
        hazy_path = os.path.join(hazy_dir, f)
        break

if hazy_path:
    print(f"  测试图: {os.path.basename(hazy_path)}")
    
    img = cv2.imread(hazy_path)
    print(f"  原图尺寸: {img.shape}")
    
    # 预处理: BGR -> RGB, 归一化到[0,1], 转tensor
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 前向
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(img_tensor)
    
    # 后处理
    out_img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    
    # 计算暗通道 (去雾效果指标)
    def dark_channel(img_bgr, patch=15):
        min_val = np.min(img_bgr, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
        dark = cv2.erode(min_val.astype(np.float32), kernel)
        return float(np.mean(dark) / 255.0)
    
    dc_before = dark_channel(img)
    dc_after = dark_channel(out_img)
    
    print(f"  暗通道 - 去雾前: {dc_before:.4f}")
    print(f"  暗通道 - 去雾后: {dc_after:.4f}")
    print(f"  暗通道降低: {(1 - dc_after / max(dc_before, 0.001)) * 100:.1f}%")
    
    # 保存结果
    out_dir = r"D:\dehaze_fusion\DehazeFormer\test_output"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "input_hazy.png"), img)
    cv2.imwrite(os.path.join(out_dir, "output_dehazed.png"), out_img)
    
    # 同时测一张真实雾图
    real_dir = r"D:\DATA_ALL\dataset\test\hazy_real"
    if os.path.exists(real_dir):
        for f in sorted(os.listdir(real_dir))[:1]:
            real_path = os.path.join(real_dir, f)
            real_img = cv2.imread(real_path)
            if real_img is None:
                continue
            
            # resize到512以内
            h, w = real_img.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                real_img = cv2.resize(real_img, (int(w * scale), int(h * scale)))
            
            real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            real_tensor = torch.from_numpy(real_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    real_out = model(real_tensor)
            
            real_out_img = real_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            real_out_img = np.clip(real_out_img * 255, 0, 255).astype(np.uint8)
            real_out_img = cv2.cvtColor(real_out_img, cv2.COLOR_RGB2BGR)
            
            dc_real_before = dark_channel(real_img)
            dc_real_after = dark_channel(real_out_img)
            
            print(f"\n  真实雾图: {f}")
            print(f"  暗通道 - 去雾前: {dc_real_before:.4f}")
            print(f"  暗通道 - 去雾后: {dc_real_after:.4f}")
            print(f"  暗通道降低: {(1 - dc_real_after / max(dc_real_before, 0.001)) * 100:.1f}%")
            
            cv2.imwrite(os.path.join(out_dir, "real_input_hazy.png"), real_img)
            cv2.imwrite(os.path.join(out_dir, "real_output_dehazed.png"), real_out_img)
    
    print(f"\n  结果保存到: {out_dir}")
    
    # 显存使用
    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  显存占用: {mem:.0f} MB")

print("\n验证完成!")
