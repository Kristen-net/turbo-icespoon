"""验证DehazeFormer-S模型能正常实例化"""
import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")

import torch
from models.dehazeformer import dehazeformer_s

# 实例化模型
model = dehazeformer_s()
print(f"Model created successfully")

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# 测试前向传播
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 用256x256随机输入测试
x = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():
    with torch.cuda.amp.autocast():
        out = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
print(f"Device: {device}")
print("\nAll checks passed!")
