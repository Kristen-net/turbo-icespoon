"""Quick progress checker for M2' training"""
from tensorboard.backend.event_processing import event_accumulator
import os

tb_dir = r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\logs"
ea = event_accumulator.EventAccumulator(tb_dir)
ea.Reload()

psnr = ea.Scalars("val/psnr")
print(f"Epochs: {len(psnr)}/100")
print(f"Latest PSNR: {psnr[-1].value:.2f}")

best_psnr = max(e.value for e in psnr)
best_epoch = [e for e in psnr if e.value == best_psnr][0].step + 1
print(f"Best PSNR: {best_psnr:.2f} (epoch {best_epoch})")

for tag in ["hawfe/alpha_ll", "hawfe/alpha_lh", "hawfe/alpha_hl", "hawfe/alpha_hh", "hawfe/beta"]:
    vals = ea.Scalars(tag)
    print(f"  {tag}: {vals[-1].value:.4f} (init={vals[0].value:.4f})")

# Check if training is done
ckpt_dir = r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\checkpoints"
if os.path.exists(os.path.join(ckpt_dir, "m2p_epoch100.pth")):
    print("\n>>> TRAINING COMPLETE (epoch100.pth found) <<<")
elif len(psnr) >= 100:
    print("\n>>> TRAINING COMPLETE (100 epochs in TB) <<<")
else:
    remaining = 100 - len(psnr)
    print(f"\n>>> {remaining} epochs remaining, ~{remaining * 60}s <<<")

# Check if report is ready
report_path = r"D:\dehaze_fusion\icewave_output\three_way_report.txt"
if os.path.exists(report_path):
    print(">>> REPORT READY <<<")
