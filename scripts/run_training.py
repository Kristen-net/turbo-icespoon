"""W5 正式训练: joint_v2 Stage-A 8ep + Stage-B 40ep

用法 (conda 环境):
    conda run -n dehaze_fusion python -c "
import sys, os; os.chdir(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon');
sys.path.insert(0, r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\src');
exec(open(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\scripts\\run_training.py').read())
    "
"""
import sys
import os
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import torch
from icewave.train.config import load_config
from icewave.train.trainer import Trainer

cfg = load_config(REPO / "configs" / "train" / "joint_v2_train.yaml")
cfg["output_dir"] = str(REPO / "outputs" / "joint_v2_w5")
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
cfg["data"]["root"] = str(REPO / "data" / "synthetic_full")

print(f"Device: {cfg['device']}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    torch.cuda.empty_cache()

print(f"Config: {cfg['model']['version']}")
print(f"Stage-A: {cfg['train']['stages']['warm_up']['epochs']} epochs")
print(f"Stage-B: {cfg['train']['stages']['joint']['epochs']} epochs")
print(f"Batch size: {cfg['train']['batch_size']}")
print()

trainer = Trainer(cfg)
best_psnr = trainer.train_joint_v2()

print(f"\n{'='*60}")
print(f"W5 训练完成: best PSNR = {best_psnr:.2f}")
print(f"输出: {cfg['output_dir']}")
print(f"{'='*60}")
