"""Smoke test: joint_v2 训练 1+1 epoch, 验证无 NaN.

用法 (conda 环境):
    conda run -n dehaze_fusion python -c "
import sys; sys.path.insert(0, r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\src');
exec(open(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\scripts\\smoke_test.py').read())
"
"""
import sys
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
sys.path.insert(0, str(REPO / "src"))

import torch
from icewave.train.config import load_config
from icewave.train.trainer import Trainer

cfg = load_config(REPO / "configs" / "train" / "smoke_test.yaml")
cfg["output_dir"] = str(REPO / "outputs" / "smoke_test")
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
cfg["data"]["root"] = str(REPO / "data" / "synthetic")

print(f"Device: {cfg['device']}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

trainer = Trainer(cfg)
print("Trainer initialized, starting train_joint_v2...")
try:
    best_psnr = trainer.train_joint_v2()
    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED: best PSNR = {best_psnr:.2f}")
    print(f"Output: {cfg['output_dir']}")
    print(f"{'='*60}")
except Exception as e:
    print(f"\nSMOKE TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
