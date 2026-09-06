"""下游评估 Part 2: 用已微调的 YOLO 权重对各实验做去雾 + 检测评估."""
import sys
import os
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import json
import shutil

import cv2
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== 下游检测评估 (Part 2) === device={device}")

# 路径
yolo_best = REPO / "outputs" / "yolo_finetune" / "yolov8s_synthetic" / "weights" / "best.pt"
eval_root = REPO / "outputs" / "downstream_eval"
val_hazy_dir = eval_root / "val_hazy"
val_clear_dir = eval_root / "val_clear"
val_labels_dir = eval_root / "val_labels"

# 实验列表
experiments = {
    "joint_full":      REPO / "outputs" / "joint_v2_w5" / "checkpoints" / "joint_v2_best.pth",
    "no_boxfeat":      REPO / "outputs" / "ablation_no_boxfeat" / "checkpoints" / "joint_v2_best.pth",
    "no_uncertainty":  REPO / "outputs" / "ablation_no_uncertainty" / "checkpoints" / "joint_v2_best.pth",
    "freeze_backbone": REPO / "outputs" / "ablation_freeze_backbone" / "checkpoints" / "joint_v2_best.pth",
}


def generate_dehazed(ckpt_path: Path, hazy_dir: Path, out_dir: Path, device: str):
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
    print(f"  模型: version={version}, backbone={backbone}")
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

    hazy_paths = sorted(p for p in hazy_dir.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))

    for hp in hazy_paths:
        img = cv2.imread(str(hp))
        if img is None:
            continue
        with torch.no_grad():
            pred = _dehaze(model, img, device)
        cv2.imwrite(str(out_dir / hp.name), pred)

    print(f"  → 生成去雾图: {len(hazy_paths)} 张 → {out_dir}")
    return out_dir


def evaluate_downstream(detector_weights: Path, hazy_dir: Path, dehazed_dir: Path,
                        clear_dir: Path, labels_dir: Path, output_json: Path,
                        conf: float = 0.001):
    from icewave.detect.yolo import YOLODetector
    from icewave.eval.downstream import downstream_gain, compute_gain_stats

    detector = YOLODetector(str(detector_weights), conf=conf)

    result = downstream_gain(
        detector,
        Path(hazy_dir),
        Path(dehazed_dir),
        Path(labels_dir),
        clear_dir=Path(clear_dir),
        iou_thr=0.5,
        class_names=["target"],
    )
    gain_stats = compute_gain_stats(result)
    result["gain_stats"] = gain_stats

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return gain_stats


# ---- 主流程 ----
print(f"YOLO 权重: {yolo_best}")
print(f"Val 集: {len(list(val_hazy_dir.glob('*.png')))} 张")
print()

all_results = {}
for exp_name, ckpt_path in experiments.items():
    if not ckpt_path.exists():
        print(f"[跳过] {exp_name}: checkpoint 不存在")
        continue

    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"  Checkpoint: {ckpt_path}")

    dehazed_dir = eval_root / exp_name / "dehazed"
    result_json = eval_root / exp_name / "downstream_result.json"

    # 生成去雾图
    print(f"  [1/2] 生成去雾图...")
    generate_dehazed(ckpt_path, val_hazy_dir, dehazed_dir, device)

    # 下游评估
    print(f"  [2/2] 下游检测评估...")
    stats = evaluate_downstream(
        yolo_best,
        val_hazy_dir, dehazed_dir, val_clear_dir, val_labels_dir,
        result_json,
        conf=0.001,
    )
    all_results[exp_name] = stats
    print(f"  mAP_hazy={stats['mAP_hazy']:.4f}  "
          f"mAP_dehazed={stats['mAP_dehazed']:.4f}  "
          f"mAP_clear={stats['mAP_clear']:.4f}")
    print(f"  Δ_gain={stats['delta_gain']:+.4f}  "
          f"gap={stats['delta_gap']:+.4f}  "
          f"R={stats['R']:.1%}")

# ---- 汇总表 ----
print(f"\n{'='*60}")
print("📊 下游检测评估对比表 (mAP@0.5)")
print(f"{'='*60}")

# baseline (hazy / clear)
first_key = list(all_results.keys())[0]
mAP_hazy_base = all_results[first_key]["mAP_hazy"]
mAP_clear_base = all_results[first_key]["mAP_clear"]

rows = [
    ("Hazy baseline", mAP_hazy_base, None, None, None),
    ("Oracle (clear)", mAP_clear_base, None, None, None),
]
for exp_name, stats in all_results.items():
    rows.append((
        exp_name,
        stats["mAP_dehazed"],
        stats["delta_gain"],
        stats["delta_gap"],
        stats["R"],
    ))

header = f"{'Method':<22} {'mAP@0.5':>9} {'Δ_gain':>9} {'gap':>9} {'R':>9}"
print(header)
print("-" * len(header))
for name, mAP, delta, gap, R in rows:
    d_str = f"{delta:+.4f}" if delta is not None else "—"
    g_str = f"{gap:+.4f}" if gap is not None else "—"
    r_str = f"{R:.1%}" if R is not None else "—"
    print(f"{name:<22} {mAP:>9.4f} {d_str:>9} {g_str:>9} {r_str:>9}")

# 保存汇总
summary = {
    "mAP_hazy_baseline": mAP_hazy_base,
    "mAP_clear_oracle": mAP_clear_base,
    "experiments": all_results,
}
summary_path = eval_root / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n汇总 → {summary_path}")
print("\n✅ 评估完成!")
