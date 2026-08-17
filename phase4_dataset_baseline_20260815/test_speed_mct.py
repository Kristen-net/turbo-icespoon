"""测试MCT前向+反向传播速度"""
import sys, time
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer\hf_demo")

import torch
import torch.nn.functional as F
from models.dehazeformer import MCT

model = MCT().cuda()
ckpt = torch.load(r"D:\dehaze_fusion\DehazeFormer\pretrained\saved_models\dehazeformer.pth",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"], strict=True)

x = torch.randn(8, 3, 256, 256).cuda()
target = torch.randn(8, 3, 256, 256).cuda()

print("=== 前向传播 (无AMP) ===", flush=True)
t0 = time.time()
out = model(x)
torch.cuda.synchronize()
print(f"  耗时: {time.time()-t0:.3f}s", flush=True)
print(f"  输出: {out.shape}", flush=True)
print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB", flush=True)

print("\n=== 前向+反向 (无AMP) ===", flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model(x)
loss = F.l1_loss(out, target)
loss.backward()
torch.cuda.synchronize()
print(f"  耗时: {time.time()-t0:.3f}s", flush=True)
print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB", flush=True)

print("\n=== 前向+反向 (AMP fp16) ===", flush=True)
torch.cuda.reset_peak_memory_stats()
model.zero_grad()
scaler = torch.amp.GradScaler('cuda')
t0 = time.time()
with torch.amp.autocast('cuda', dtype=torch.float16):
    out = model(x)
    loss = F.l1_loss(out, target)
scaler.scale(loss).backward()
scaler.step(torch.optim.AdamW(model.parameters(), lr=1e-4))
torch.cuda.synchronize()
print(f"  耗时: {time.time()-t0:.3f}s", flush=True)
print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB", flush=True)

print("\n=== 10次前向+反向 (AMP) ===", flush=True)
times = []
for i in range(10):
    model.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model(x)
        loss = F.l1_loss(out, target)
    scaler.scale(loss).backward()
    scaler.step(torch.optim.AdamW(model.parameters(), lr=1e-4))
    torch.cuda.synchronize()
    times.append(time.time()-t0)
print(f"  平均: {sum(times)/len(times):.3f}s/batch", flush=True)
print(f"  每epoch(158batch): {sum(times)/len(times)*158/60:.1f}min", flush=True)
print(f"  100 epochs: {sum(times)/len(times)*158*100/3600:.1f}h", flush=True)
