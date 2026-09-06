"""跨实验下游检测评估: 对所有训练好的 checkpoint 运行下游检测, 计算 mAP + Δ_gain + R.

文档 §6.4 要求的表格:
| Method | mAP_hazy | mAP_dehazed | mAP_clear | Δ_gain | R (%) |

用法:
    conda run -n dehaze_fusion python -c "
import sys, os; os.chdir(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon');
sys.path.insert(0, r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\src');
exec(open(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\scripts\\run_downstream_eval.py').read())
    "
"""
import sys
import os
import json
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import torch
import numpy as np
import cv2

# --- 配置 ---
YOLO_WEIGHTS = str(REPO / "yolov8s.pt")  # YOLOv8s 预训练权重
DATA_ROOT = REPO / "data" / "synthetic_full"
VAL_HAZY = DATA_ROOT / "val" / "hazy"
VAL_CLEAR = DATA_ROOT / "val" / "clear"
VAL_LABELS = DATA_ROOT / "val" / "labels"
OUTPUT_DIR = REPO / "outputs" / "downstream_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 实验列表 (output_name, checkpoint_path)
EXPERIMENTS = [
    ("joint_v2_w5", REPO / "outputs" / "joint_v2_w5" / "checkpoints" / "joint_v2_best.pth"),
    ("ablation_no_boxfeat", REPO / "outputs" / "ablation_no_boxfeat" / "checkpoints" / "joint_v2_best.pth"),
    ("ablation_no_uncertainty", REPO / "outputs" / "ablation_no_uncertainty" / "checkpoints" / "joint_v2_best.pth"),
    ("ablation_freeze_backbone", REPO / "outputs" / "ablation_freeze_backbone" / "checkpoints" / "joint_v2_best.pth"),
    ("ablation_cascade", REPO / "outputs" / "ablation_cascade" / "checkpoints" / "joint_v2_best.pth"),
]

CONF = 0.25
IOU_THR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"=== Downstream Evaluation ===")
print(f"Device: {DEVICE}")
print(f"YOLO weights: {YOLO_WEIGHTS}")
print(f"Data: {DATA_ROOT}")
print()

# --- 加载检测器 ---
from icewave.eval.downstream import (
    downstream_gain, compute_gain_stats, dehaze_directory, evaluate_set,
    DetectorProto, _load_yolo_labels
)

class YOLODetector:
    """YOLOv8 检测器封装, 兼容 downstream.py 的 DetectorProto."""
    def __init__(self, weights_path, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf
        self.class_names = list(self.model.names.values())

    def detect(self, img_bgr, conf=None):
        c = conf or self.conf
        results = self.model(img_bgr, conf=c, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "cls_id": int(box.cls[0]),
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                })
        return dets

# --- 加载去雾模型 ---
from icewave.models import build_model

def load_dehaze_model(ckpt_path, device="cpu"):
    model = build_model("joint_v2", backbone="s").to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def dehaze_single(model, img_bgr, device="cpu"):
    """去雾单张图像, 返回 BGR uint8."""
    from icewave.eval.benchmark import _dehaze
    return _dehaze(model, img_bgr, device)

# --- 主评估循环 ---
if not Path(YOLO_WEIGHTS).exists():
    print(f"[ERROR] YOLO weights not found: {YOLO_WEIGHTS}")
    print("Please download yolov8s.pt first.")
    sys.exit(1)

if not VAL_HAZY.exists():
    print(f"[ERROR] Validation hazy images not found: {VAL_HAZY}")
    sys.exit(1)

print("Loading YOLO detector...")
detector = YOLODetector(YOLO_WEIGHTS, conf=CONF)
print(f"  Classes: {detector.class_names}")
print()

all_results = {}

for exp_name, ckpt_path in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found")
        all_results[exp_name] = {"error": "checkpoint not found"}
        continue

    # 1. 加载去雾模型
    print("  Loading dehazing model...")
    model = load_dehaze_model(ckpt_path, DEVICE)

    # 2. 生成去雾图
    dehazed_dir = OUTPUT_DIR / exp_name / "dehazed"
    dehazed_dir.mkdir(parents=True, exist_ok=True)

    hazy_images = sorted(p for p in VAL_HAZY.iterdir()
                         if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    print(f"  Dehazing {len(hazy_images)} images...")

    for hp in hazy_images:
        img = cv2.imread(str(hp))
        if img is None:
            continue
        pred = dehaze_single(model, img, DEVICE)
        cv2.imwrite(str(dehazed_dir / hp.name), pred)

    # 3. 下游检测评估
    print("  Running downstream evaluation...")
    result = downstream_gain(
        detector, VAL_HAZY, dehazed_dir, VAL_LABELS,
        clear_dir=VAL_CLEAR if VAL_CLEAR.exists() else None,
        iou_thr=IOU_THR,
    )

    # 4. 增益统计
    gain_stats = compute_gain_stats(result)

    # 5. 输出
    mAP_hazy = result["hazy"]["mAP"]
    mAP_dehazed = result["dehazed"]["mAP"]
    delta = result["delta_mAP_dehazed_minus_hazy"]

    line = f"  mAP_hazy={mAP_hazy:.4f}  mAP_dehazed={mAP_dehazed:.4f}  Δ={delta:+.4f}"
    if "clear" in result:
        mAP_clear = result["clear"]["mAP"]
        gap = result["gap_clear_minus_dehazed"]
        R = gain_stats.get("R")
        line += f"  mAP_clear={mAP_clear:.4f}  gap={gap:+.4f}"
        if R is not None:
            line += f"  R={R:.1%}"
    print(line)

    all_results[exp_name] = {
        "mAP_hazy": mAP_hazy,
        "mAP_dehazed": mAP_dehazed,
        "mAP_clear": result.get("clear", {}).get("mAP"),
        "delta_gain": delta,
        "R": gain_stats.get("R"),
        "gain_stats": gain_stats,
    }

    # 释放模型显存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- 汇总表格 ---
print(f"\n\n{'='*80}")
print("=== Summary Table (Document §6.4 format) ===")
print(f"{'='*80}")
header = f"{'Method':<30s} {'mAP_hazy':>10s} {'mAP_dehazed':>12s} {'mAP_clear':>10s} {'Δ_gain':>10s} {'R(%)':>8s}"
print(header)
print("-" * 80)

for name, r in all_results.items():
    if "error" in r:
        print(f"{name:<30s} {'N/A':>10s}")
        continue
    mh = r.get("mAP_hazy", 0)
    md = r.get("mAP_dehazed", 0)
    mc = r.get("mAP_clear", 0) or 0
    dg = r.get("delta_gain", 0)
    R = r.get("R")
    R_str = f"{R*100:.1f}" if R is not None else "N/A"
    print(f"{name:<30s} {mh:>10.4f} {md:>12.4f} {mc:>10.4f} {dg:>+10.4f} {R_str:>8s}")

# 保存 JSON
report_path = OUTPUT_DIR / "downstream_eval_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nReport saved to: {report_path}")
