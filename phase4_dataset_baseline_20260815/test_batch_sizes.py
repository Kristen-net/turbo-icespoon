"""测试不同batch和patch_size的显存占用"""
import sys, time
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer\hf_demo")

import torch
import torch.nn.functional as F
from models.dehazeformer import MCT

model = MCT().cuda()
ckpt = torch.load(r"D:\dehaze_fusion\DehazeFormer\pretrained\saved_models\dehazeformer.pth",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"], strict=True)

scaler = torch.amp.GradScaler('cuda')
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

configs = [
    (1, 256),
    (2, 256),
    (4, 192),
    (4, 128),
    (8, 128),
]

for bs, ps in configs:
    try:
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad()
        opt.zero_grad()
        
        x = torch.randn(bs, 3, ps, ps).cuda()
        target = torch.randn(bs, 3, ps, ps).cuda()
        
        t0 = time.time()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x)
            loss = F.l1_loss(out, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        torch.cuda.synchronize()
        
        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"batch={bs}, patch={ps}: {mem:.0f} MB, {elapsed:.3f}s/batch", flush=True)
        
        # 估算训练时间
        n_batches = 1266 // bs
        epoch_time = n_batches * elapsed
        print(f"  -> 每epoch: {epoch_time/60:.1f}min, 100 epochs: {epoch_time*100/3600:.1f}h", flush=True)
        
        del x, target, out, loss
        torch.cuda.empty_cache()
        
    except torch.OutOfMemoryError:
        print(f"batch={bs}, patch={ps}: OOM!", flush=True)
        torch.cuda.empty_cache()
