"""消融实验运行脚本

用法:
    conda run -n dehaze_fusion python -c "
import sys, os; os.chdir(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon');
sys.path.insert(0, r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\src');
exec(open(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\scripts\\run_ablation.py').read())
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

# 消融实验选择 (通过环境变量 ABLATION 选择)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default="ablation_no_boxfeat")
ap.add_argument("--output", type=str, default=None)
args, _ = ap.parse_known_args()

config_name = args.config
output_name = args.output or config_name

cfg = load_config(REPO / "configs" / "train" / f"{config_name}.yaml")
cfg["output_dir"] = str(REPO / "outputs" / output_name)
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
cfg["data"]["root"] = str(REPO / "data" / "synthetic_full")

print(f"=== 消融实验: {config_name} ===")
print(f"Device: {cfg['device']}")
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
print(f"Output: {cfg['output_dir']}")
print()

trainer = Trainer(cfg)
try:
    best_psnr = trainer.train_joint_v2()
    print(f"\n{'='*60}")
    print(f"消融实验 {config_name} 完成: best PSNR = {best_psnr:.2f}")
    print(f"输出: {cfg['output_dir']}")
    print(f"{'='*60}")
except Exception as e:
    print(f"\n消融实验失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
