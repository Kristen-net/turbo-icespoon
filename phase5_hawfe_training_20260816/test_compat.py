import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
x = torch.rand(2, 3, 192, 192, device='cuda')
y = torch.rand(2, 3, 192, 192, device='cuda')
val = ssim(x, y)
print(f"SSIM test: {val.item():.4f}")
print("torchmetrics OK")

# Test DehazeFormer import
import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
from models.dehazeformer import dehazeformer_s
model = dehazeformer_s().to('cuda')
x = torch.randn(2, 3, 192, 192, device='cuda')
with torch.no_grad():
    y = model(x)
print(f"DehazeFormer-S forward: {x.shape} -> {y.shape}")
print("DehazeFormer OK")

# Test AMP
with torch.amp.autocast('cuda', dtype=torch.float16):
    y = model(x)
print(f"AMP forward: {y.dtype}, shape={y.shape}")
print("AMP OK")

# Test GradScaler
scaler = torch.amp.GradScaler('cuda')
print(f"GradScaler: {scaler.is_enabled()}")
print("All compatibility checks passed!")
