"""真实数据下游评估: 用已有 YOLO 权重 + 各去雾模型, 在真实雾图 val 集上评测.

注意: 真实数据无配对清晰图, 只能计算 Δ_gain = mAP_dehazed - mAP_hazy,
无法计算 R (恢复率需 clear 图作 oracle).
"""
import sys
import os
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import json
import cv2
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== 真实数据下游检测评估 === device={device}")

# 路径
yolo_weights = Path(r"D:\dehaze_fusion\yolo_train_output\power_line_yolo\weights\best.pt")
real_hazy_val = Path(r"D:\dehaze_fusion\yolo_dataset\images\val")
real_labels_val = Path(r"D:\dehaze_fusion\yolo_dataset\labels\val")
eval_root = REPO / "outputs" / "real_eval"
eval_root.mkdir(parents=True, exist_ok=True)

class_names = ["insulator", "power_line", "ice", "tower"]

# 实验列表
experiments = {
    "joint_full":      REPO / "outputs" / "joint_v2_w5" / "checkpoints" / "joint_v2_best.pth",
    "no_boxfeat":      REPO / "outputs" / "ablation_no_boxfeat" / "checkpoints" / "joint_v2_best.pth",
    "no_uncertainty":  REPO / "outputs" / "ablation_no_uncertainty" / "checkpoints" / "joint_v2_best.pth",
    "freeze_backbone": REPO / "outputs" / "ablation_freeze_backbone" / "checkpoints" / "joint_v2_best.pth",
}

# ---- 先测 hazy baseline ----
print("\n[Step 1/3] 测试 YOLO 权重 + hazy baseline mAP...")
print(f"  YOLO 权重: {yolo_weights}")
print(f"  Val 图像: {real_hazy_val}")
print(f"  Val 标签: {real_labels_val}")

from icewave.detect.yolo import YOLODetector
from icewave.eval.downstream import evaluate_set

detector = YOLODetector(str(yolo_weights), conf=0.001)

val_imgs = sorted(p for p in real_hazy_val.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
print(f"  图片数量: {len(val_imgs)}")

hazy_result = evaluate_set(detector, val_imgs, real_labels_val, 0.5, class_names)
print(f"  mAP_hazy = {hazy_result['mAP']:.4f}")
for cls, ap in hazy_result["AP_per_class"].items():
    print(f"    {cls}: {ap:.4f}")

# 保存 baseline
baseline_path = eval_root / "hazy_baseline.json"
baseline_path.write_text(json.dumps(hazy_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  → {baseline_path}")


# ---- 生成去雾图 + 检测 ----
def generate_dehazed(ckpt_path: Path, img_paths: list, out_dir: Path, device: str):
    from icewave.models import build_model
    from icewave.train.config import load_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = ckpt_path.parent.parent / "config_snapshot.yaml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    else:
        cfg = {"model": {"version": "joint_v2", "backbone": "s"}}

    version = cfg["model"]["version"]
    backbone = cfg["model"].get("backbone", "s")
    model = build_model(version, backbone).to(device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    from icewave.eval.benchmark import _dehaze

    out_paths = []
    for hp in img_paths:
        img = cv2.imread(str(hp))
        if img is None:
            continue
        with torch.no_grad():
            pred = _dehaze(model, img, device)
        out_path = out_dir / hp.name
        cv2.imwrite(str(out_path), pred)
        out_paths.append(out_path)

    print(f"    生成去雾图: {len(out_paths)} 张 → {out_dir}")
    return out_paths


print(f"\n[Step 2/3] 生成去雾图 + 检测评估...")
all_results = {"hazy_baseline": hazy_result}

for exp_name, ckpt_path in experiments.items():
    if not ckpt_path.exists():
        print(f"\n  [跳过] {exp_name}: checkpoint 不存在")
        continue

    print(f"\n  --- {exp_name} ---")
    dehazed_dir = eval_root / exp_name / "dehazed"
    result_json = eval_root / exp_name / "result.json"

    # 生成去雾图
    print(f"    生成去雾图...")
    dehazed_paths = generate_dehazed(ckpt_path, val_imgs, dehazed_dir, device)

    # 检测评估
    print(f"    运行检测...")
    dehazed_result = evaluate_set(detector, dehazed_paths, real_labels_val, 0.5, class_names)
    all_results[exp_name] = dehazed_result

    delta = dehazed_result["mAP"] - hazy_result["mAP"]
    print(f"    mAP_dehazed = {dehazed_result['mAP']:.4f}  Δ_gain = {delta:+.4f}")
    for cls, ap in dehazed_result["AP_per_class"].items():
        h_ap = hazy_result["AP_per_class"].get(cls, 0)
        print(f"      {cls}: {ap:.4f} ({ap - h_ap:+.4f})")

    # 保存
    result_json.write_text(json.dumps(dehazed_result, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- 汇总表 ----
print(f"\n[Step 3/3] 生成对比表...")
print(f"\n{'='*70}")
print(f"📊 真实数据下游检测对比表 (mAP@0.5, val={len(val_imgs)} 张)")
print(f"{'='*70}")

mAP_hazy = hazy_result["mAP"]

rows = [("Hazy baseline", mAP_hazy, None)]
for exp_name in experiments:
    if exp_name in all_results:
        rows.append((exp_name, all_results[exp_name]["mAP"],
                     all_results[exp_name]["mAP"] - mAP_hazy))

header = f"{'Method':<22} {'mAP@0.5':>10} {'Δ_gain':>10}"
print(header)
print("-" * len(header))
for name, mAP, delta in rows:
    d_str = f"{delta:+.4f}" if delta is not None else "—"
    print(f"{name:<22} {mAP:>10.4f} {d_str:>10}")

# 各类别对比
print(f"\n--- 各类别 mAP 详情 ---")
cls_header = f"{'Method':<22}"
for cls in class_names:
    cls_header += f" {cls:>12}"
print(cls_header)
print("-" * len(cls_header))

# hazy
row = f"{'Hazy baseline':<22}"
for cls in class_names:
    ap = hazy_result["AP_per_class"].get(cls, 0)
    row += f" {ap:>12.4f}"
print(row)

# 各实验
for exp_name in experiments:
    if exp_name not in all_results:
        continue
    r = all_results[exp_name]
    row = f"{exp_name:<22}"
    for cls in class_names:
        ap = r["AP_per_class"].get(cls, 0)
        h_ap = hazy_result["AP_per_class"].get(cls, 0)
        row += f" {ap:>10.4f}({ap-h_ap:+.2f})"
    print(row)

# 保存汇总
summary = {
    "n_val_images": len(val_imgs),
    "yolo_weights": str(yolo_weights),
    "hazy_baseline": hazy_result,
    "experiments": {k: v for k, v in all_results.items() if k != "hazy_baseline"},
}
summary_path = eval_root / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n汇总 → {summary_path}")
print("\n✅ 真实数据评估完成!")
