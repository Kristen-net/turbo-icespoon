"""测试标准DehazeFormer-S速度 (无MCT包装)"""
import sys, time
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")

import torch
import torch.nn.functional as F
from models.dehazeformer import dehazeformer_s

model = dehazeformer_s().cuda()
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

# 测试不同batch/patch组合
configs = [
    (1, 256),
    (4, 256),
    (8, 256),
    (8, 192),
    (16, 256),
    (16, 192),
]

scaler = torch.amp.GradScaler('cuda')

for bs, ps in configs:
    try:
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad()
        
        x = torch.randn(bs, 3, ps, ps).cuda()
        target = torch.randn(bs, 3, ps, ps).cuda()
        
        t0 = time.time()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x)
            loss = F.l1_loss(out, target)
        scaler.scale(loss).backward()
        scaler.unscale_(torch.optim.AdamW(model.parameters(), lr=1e-4))
        scaler.update()
        
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        n_batches = 1266 // bs
        print(f"batch={bs:2d}, patch={ps}: {mem:5.0f} MB, {elapsed:.3f}s/batch, "
              f"epoch={n_batches*elapsed/60:.1f}min, 100ep={n_batches*elapsed*100/3600:.1f}h", flush=True)
        
        del x, target, out, loss
        torch.cuda.empty_cache()
        
    except torch.OutOfMemoryError:
        print(f"batch={bs:2d}, patch={ps}: OOM!", flush=True)
        torch.cuda.empty_cache()
        break
