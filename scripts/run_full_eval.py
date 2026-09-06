"""完整下游评估流程: YOLOv8 微调 + 去雾图生成 + mAP/Δ_gain/R 对比.

步骤:
1. 将合成数据划分为 train/val (清晰图用于 YOLO 微调)
2. 微调 YOLOv8s (少量 epoch, 快速验证)
3. 对每个实验 checkpoint: 生成 val 集的去雾图
4. 用微调后的 YOLO 检测: hazy / dehazed / clear 的 mAP@0.5
5. 计算 Δ_gain, gap, R, 生成对比表

用法:
    python scripts/run_full_eval.py
"""
import sys
import os
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import json
import shutil
import random

import cv2
import numpy as np
import torch

random.seed(42)
np.random.seed(42)

# ==============================
# 1. 准备 YOLO 训练数据
# ==============================
def prepare_yolo_data(src_root: Path, dst_root: Path, val_ratio: float = 0.2):
    """划分清晰图为 train/val, 复制到 YOLO 格式目录."""
    dst_root = Path(dst_root)
    (dst_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    clear_dir = src_root / "train" / "clear"
    labels_dir = src_root / "train" / "labels"

    all_imgs = sorted(p for p in clear_dir.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    n_total = len(all_imgs)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    indices = list(range(n_total))
    random.shuffle(indices)
    val_indices = set(indices[:n_val])

    for i, img_path in enumerate(all_imgs):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        split = "val" if i in val_indices else "train"

        # 复制清晰图 (用于 YOLO 训练)
        shutil.copy2(img_path, dst_root / "images" / split / img_path.name)
        # 复制标签
        if label_path.exists():
            shutil.copy2(label_path, dst_root / "labels" / split / f"{stem}.txt")

    print(f"[数据准备] 总计 {n_total} 张: train={n_train}, val={n_val}")
    return n_train, n_val


def create_data_yaml(yolo_root: Path):
    """创建 YOLO data.yaml."""
    yaml_content = f"""path: {yolo_root}
train: images/train
val: images/val
names:
  0: target
"""
    yaml_path = yolo_root / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[数据准备] data.yaml → {yaml_path}")
    return yaml_path


# ==============================
# 2. 微调 YOLOv8s
# ==============================
def finetune_yolo(data_yaml: Path, output_dir: Path, epochs: int = 20,
                  imgsz: int = 256, batch: int = 8):
    """在合成数据上微调 YOLOv8s."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(REPO / "yolov8s.pt"))

    print(f"[YOLO 微调] 开始: epochs={epochs}, imgsz={imgsz}, batch={batch}")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name="yolov8s_synthetic",
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        plots=False,
    )

    best_weights = output_dir / "yolov8s_synthetic" / "weights" / "best.pt"
    print(f"[YOLO 微调] 完成. Best weights → {best_weights}")
    return best_weights


# ==============================
# 3. 生成去雾图
# ==============================
def generate_dehazed(ckpt_path: Path, hazy_dir: Path, out_dir: Path, device: str):
    """用指定 checkpoint 批量生成去雾图."""
    from icewave.models import build_model
    from icewave.train.config import load_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 从 checkpoint 同级目录的 config_snapshot 读取配置
    cfg_path = ckpt_path.parent.parent / "config_snapshot.yaml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    else:
        cfg = {"model": {"version": "joint_v2", "backbone": "s"}}

    model = build_model(cfg["model"]["version"], cfg["model"].get("backbone", "s")).to(device)
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


# ==============================
# 4. 下游检测评估
# ==============================
def evaluate_downstream(detector_weights: Path, hazy_dir: Path, dehazed_dir: Path,
                        clear_dir: Path, labels_dir: Path, output_json: Path,
                        conf: float = 0.001):
    """跑下游检测评估, 返回 mAP/Δ_gain/R."""
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


# ==============================
# 主流程
# ==============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 完整下游评估流程 === device={device}")

    # 路径
    src_data = REPO / "data" / "synthetic_full"
    yolo_root = REPO / "outputs" / "yolo_finetune_data"
    yolo_out = REPO / "outputs" / "yolo_finetune"
    eval_root = REPO / "outputs" / "downstream_eval"

    # 实验列表 (已完成的)
    experiments = {
        "joint_full":      REPO / "outputs" / "joint_v2_w5" / "checkpoints" / "joint_v2_best.pth",
        "no_boxfeat":      REPO / "outputs" / "ablation_no_boxfeat" / "checkpoints" / "joint_v2_best.pth",
        "no_uncertainty":  REPO / "outputs" / "ablation_no_uncertainty" / "checkpoints" / "joint_v2_best.pth",
        "freeze_backbone": REPO / "outputs" / "ablation_freeze_backbone" / "checkpoints" / "joint_v2_best.pth",
    }

    # Step 1: 准备 YOLO 训练数据
    print("\n" + "="*60)
    print("[Step 1/4] 准备 YOLO 训练数据...")
    n_train, n_val = prepare_yolo_data(src_data, yolo_root, val_ratio=0.2)
    data_yaml = create_data_yaml(yolo_root)

    # Step 2: 微调 YOLOv8s
    print("\n" + "="*60)
    print("[Step 2/4] 微调 YOLOv8s...")
    yolo_best = finetune_yolo(
        data_yaml, yolo_out,
        epochs=30, imgsz=256, batch=16,
    )

    # 验证集 hazy / clear / labels 路径
    val_hazy = yolo_root / "images" / "val"  # 不对, 这里只有 clear
    # 我们需要从原始数据的 val 划分找对应的 hazy 图
    # 重新处理: 从原始数据划分 val 的 hazy/clear/labels
    val_hazy_dir = eval_root / "val_hazy"
    val_clear_dir = eval_root / "val_clear"
    val_labels_dir = eval_root / "val_labels"
    val_hazy_dir.mkdir(parents=True, exist_ok=True)
    val_clear_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # 从 yolo_root/images/val 找到 val 的文件名, 再去原目录复制 hazy 版本
    val_clear_imgs = sorted(p for p in (yolo_root / "images" / "val").iterdir()
                            if p.suffix.lower() in (".png", ".jpg"))
    for cp in val_clear_imgs:
        stem = cp.stem
        # 复制清晰图
        shutil.copy2(cp, val_clear_dir / cp.name)
        # 复制对应 hazy 图
        hazy_src = src_data / "train" / "hazy" / cp.name
        if hazy_src.exists():
            shutil.copy2(hazy_src, val_hazy_dir / cp.name)
        # 复制标签
        label_src = src_data / "train" / "labels" / f"{stem}.txt"
        if label_src.exists():
            shutil.copy2(label_src, val_labels_dir / f"{stem}.txt")

    print(f"\n[评估数据] val 集: {len(val_clear_imgs)} 张")
    print(f"  hazy:  {val_hazy_dir}")
    print(f"  clear: {val_clear_dir}")
    print(f"  labels: {val_labels_dir}")

    # Step 3: 生成各实验去雾图 + 评估
    print("\n" + "="*60)
    print("[Step 3/4] 生成去雾图 + 下游检测评估...")

    all_results = {}
    for exp_name, ckpt_path in experiments.items():
        if not ckpt_path.exists():
            print(f"  [跳过] {exp_name}: checkpoint 不存在 {ckpt_path}")
            continue

        print(f"\n--- 实验: {exp_name} ---")
        dehazed_dir = eval_root / exp_name / "dehazed"
        result_json = eval_root / exp_name / "downstream_result.json"

        # 生成去雾图
        print(f"  生成去雾图...")
        generate_dehazed(ckpt_path, val_hazy_dir, dehazed_dir, device)

        # 下游评估
        print(f"  下游检测评估...")
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
        print(f"  Δ_gain={stats['delta_gain']:+.4f}  R={stats['R']:.1%}")

    # Step 4: 生成对比表
    print("\n" + "="*60)
    print("[Step 4/4] 生成对比表...")

    # 计算 hazy baseline (只跑一次)
    from icewave.detect.yolo import YOLODetector
    from icewave.eval.downstream import evaluate_set

    det = YOLODetector(str(yolo_best), conf=0.001)
    hazy_imgs = sorted(p for p in val_hazy_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg"))
    hazy_result = evaluate_set(det, hazy_imgs, val_labels_dir, 0.5, ["target"])
    clear_imgs = sorted(p for p in val_clear_dir.iterdir()
                        if p.suffix.lower() in (".png", ".jpg"))
    clear_result = evaluate_set(det, clear_imgs, val_labels_dir, 0.5, ["target"])
    mAP_hazy_base = hazy_result["mAP"]
    mAP_clear_base = clear_result["mAP"]

    # 表格
    table_rows = []
    table_rows.append(("Cascade baseline (hazy)", mAP_hazy_base, None, None, None))
    table_rows.append(("Oracle (clear)", mAP_clear_base, None, None, None))

    for exp_name, stats in sorted(all_results.items()):
        delta = stats["delta_gain"]
        R = stats["R"]
        gap = stats.get("delta_gap", None)
        table_rows.append((exp_name, stats["mAP_dehazed"], delta, gap, R))

    # 打印表格
    print(f"\n{'Method':<25} {'mAP@0.5':>10} {'Δ_gain':>10} {'gap':>10} {'R':>10}")
    print("-" * 65)
    for name, mAP, delta, gap, R in table_rows:
        d_str = f"{delta:+.4f}" if delta is not None else "—"
        g_str = f"{gap:+.4f}" if gap is not None else "—"
        r_str = f"{R:.1%}" if R is not None else "—"
        print(f"{name:<25} {mAP:>10.4f} {d_str:>10} {g_str:>10} {r_str:>10}")

    # 保存汇总
    summary = {
        "mAP_hazy_baseline": mAP_hazy_base,
        "mAP_clear_oracle": mAP_clear_base,
        "experiments": all_results,
        "table": [
            {"method": n, "mAP": m, "delta_gain": d, "gap": g, "R": r}
            for n, m, d, g, r in table_rows
        ]
    }
    summary_path = eval_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n汇总结果 → {summary_path}")

    print("\n" + "="*60)
    print("✅ 全部完成!")


if __name__ == "__main__":
    main()
