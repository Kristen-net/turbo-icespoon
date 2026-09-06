"""下游任务增益评测 (P1-1 核心实验组件).

回答审稿人最关心的问题: **去雾是否真正提升了下游检测任务的表现**,
而非仅仅提升 PSNR/SSIM 这类图像质量指标。

实验设计 (对应改进方案 "P1-1 下游任务增益指标"):
    同一检测器 (权重冻结) 分别在三种图像上推理:
      1. 雾图     (hazy)   —— 退化输入, 基线
      2. 去雾图   (dehazed) —— 本项目模型输出
      3. 清晰图   (clear)   —— GT 上限 (oracle)
    报告 mAP@0.5:
      ΔmAP = mAP_dehazed - mAP_hazy   (去火增益, >0 说明去雾有效)
      gap  = mAP_clear   - mAP_dehazed (与上限差距, 说明剩余提升空间)

标签格式: YOLO txt (class x_center y_center w h, 归一化坐标),
与项目检测数据集一致; 三套图像共用同一套标签 (逐 stem 对应)。

用法:
    icewave-eval-downstream --detector yolo \
        --detector-weights weights/yolo/power_line_best.pt \
        --hazy-dir data/val/hazy --labels-dir data/val/labels \
        --dehaze-model m4 --clear-dir data/val/clear

mAP 实现为 VOC 风格 all-point 插值 (不依赖 pycocotools),
无 GT 的图像中所有检测均计为 FP —— 这对正确的 mAP 至关重要。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 检测器协议: 任何实现 detect(img_bgr) -> list[dict] 的对象均可
# (YOLODetector / MaskRCNNDetector / 测试用桩)
# ---------------------------------------------------------------------------
class DetectorProto(Protocol):
    def detect(self, img_bgr: np.ndarray, conf: Optional[float] = None) -> list[dict]: ...


def _load_yolo_labels(txt_path: Path, img_w: int, img_h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    """读 YOLO txt 标签 → [(cls_id, (x1, y1, x2, y2)), ...] (绝对像素坐标)."""
    if not txt_path.exists():
        return []
    out = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x1, y1 = (xc - bw / 2) * img_w, (yc - bh / 2) * img_h
        x2, y2 = (xc + bw / 2) * img_w, (yc + bh / 2) * img_h
        out.append((cls_id, (x1, y1, x2, y2)))
    return out


def _iou_xyxy(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    """all-point 插值 AP (PR 曲线下面积, VOC 2010 风格)."""
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_set(detector: DetectorProto, image_paths: list[Path],
                 labels_dir: Path, iou_thr: float = 0.5,
                 class_names: Optional[list[str]] = None) -> dict:
    """在一张图像集合上评测 mAP@0.5 (类别平均, all-point 插值).

    无标签文件的图像参与评测 (其检测全部计为 FP), 保证 mAP 公平性。
    """
    per_img_dets: list[list[tuple[int, tuple, float]]] = []
    per_img_gts: list[list[tuple[int, tuple]]] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = detector.detect(img)
        per_img_dets.append(
            [(int(d["cls_id"]), tuple(float(v) for v in d["bbox"]), float(d["conf"]))
             for d in dets])
        per_img_gts.append(_load_yolo_labels(Path(labels_dir) / f"{img_path.stem}.txt", w, h))

    # 汇总所有出现的类别 (GT 与检测并集)
    cls_ids = {c for gts in per_img_gts for c, _ in gts}
    cls_ids |= {c for dets in per_img_dets for c, _, _ in dets}
    if not cls_ids:
        return {"mAP": 0.0, "AP_per_class": {}, "n_images": len(per_img_gts),
                "n_empty": True}

    ap_per_class = {}
    for cls_id in sorted(cls_ids):
        # 收集该类全部检测 (跨图像), 按置信度降序
        all_dets = []  # (conf, img_idx, box)
        n_gt = 0
        gt_flags: list[list[bool]] = []  # 每图 GT 是否已被匹配
        for i, (dets, gts) in enumerate(zip(per_img_dets, per_img_gts)):
            cls_gts = [b for c, b in gts if c == cls_id]
            n_gt += len(cls_gts)
            gt_flags.append([False] * len(cls_gts))
            for c, box, conf in dets:
                if c == cls_id:
                    all_dets.append((conf, i, box))
        all_dets.sort(key=lambda t: -t[0])

        tp = np.zeros(len(all_dets))
        fp = np.zeros(len(all_dets))
        for k, (conf, i, box) in enumerate(all_dets):
            cls_gts = [b for c, b in per_img_gts[i] if c == cls_id]
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(cls_gts):
                iou = _iou_xyxy(box, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0 and not gt_flags[i][best_j]:
                tp[k] = 1
                gt_flags[i][best_j] = True
            else:
                fp[k] = 1

        if n_gt == 0:
            # 该类无 GT: 有任何检测则 AP=0, 无检测则不计入该类
            if len(all_dets) == 0:
                continue
            ap_per_class[str(cls_id)] = 0.0
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = (tp_cum / n_gt).tolist()
        precisions = (tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)).tolist()
        ap_per_class[str(cls_id)] = _average_precision(recalls, precisions)

    if not ap_per_class:
        return {"mAP": 0.0, "AP_per_class": {}, "n_images": len(per_img_gts),
                "n_empty": True}
    named = {class_names[int(c)] if class_names and int(c) < len(class_names) else c:
             ap for c, ap in ap_per_class.items()}
    return {"mAP": float(np.mean(list(ap_per_class.values()))),
            "AP_per_class": named, "n_images": len(per_img_gts)}


def dehaze_directory(model, hazy_dir: Path, out_dir: Path, device: str = "cpu") -> list[Path]:
    """用去雾模型批量生成去雾图 (供下游检测评测). 复用 benchmark 的 padding 逻辑."""
    from icewave.eval.benchmark import _dehaze

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for hp in sorted(Path(hazy_dir).iterdir()):
        if hp.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        img = cv2.imread(str(hp))
        if img is None:
            continue
        pred = _dehaze(model, img, device)
        out_path = out_dir / hp.name
        cv2.imwrite(str(out_path), pred)
        out_paths.append(out_path)
    return out_paths


def downstream_gain(detector: DetectorProto, hazy_dir: Path, dehazed_dir: Path,
                    labels_dir: Path, clear_dir: Optional[Path] = None,
                    iou_thr: float = 0.5, class_names: Optional[list[str]] = None) -> dict:
    """计算下游任务增益: 同一检测器在 雾图/去雾图/清晰图 上的 mAP 差."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    hazy = sorted(p for p in Path(hazy_dir).iterdir() if p.suffix.lower() in exts)
    dehazed = sorted(p for p in Path(dehazed_dir).iterdir() if p.suffix.lower() in exts)

    result = {
        "iou_thr": iou_thr,
        "hazy": evaluate_set(detector, hazy, labels_dir, iou_thr, class_names),
        "dehazed": evaluate_set(detector, dehazed, labels_dir, iou_thr, class_names),
    }
    result["delta_mAP_dehazed_minus_hazy"] = (
        result["dehazed"]["mAP"] - result["hazy"]["mAP"])

    if clear_dir is not None and Path(clear_dir).is_dir():
        clear = sorted(p for p in Path(clear_dir).iterdir() if p.suffix.lower() in exts)
        result["clear"] = evaluate_set(detector, clear, labels_dir, iou_thr, class_names)
        result["gap_clear_minus_dehazed"] = (
            result["clear"]["mAP"] - result["dehazed"]["mAP"])
    return result


# ---------------------------------------------------------------------------
# §6 增益统计 (Δ_gain, Δ_gap, R + 95% CI)
# ---------------------------------------------------------------------------
def compute_gain_stats(result: dict, n_bootstrap: int = 1000,
                       seed: int = 42) -> dict:
    """从 downstream_gain 结果计算增益统计量.

    计算内容:
        - Δ_gain = mAP_dehazed - mAP_hazy
        - Δ_gap = mAP_clear - mAP_dehazed  (需 clear 字段)
        - R = Δ_gain / (mAP_clear - mAP_hazy)  (归一化恢复率)
        - 95% bootstrap CI (基于 per-image AP 重采样)

    Args:
        result: downstream_gain() 返回的 dict
        n_bootstrap: bootstrap 采样次数
        seed: 随机种子

    Returns:
        dict 含 delta_gain, delta_gap, R, ci_low, ci_high
    """
    rng = np.random.default_rng(seed)
    mAP_hazy = result["hazy"]["mAP"]
    mAP_dehazed = result["dehazed"]["mAP"]
    delta_gain = mAP_dehazed - mAP_hazy

    stats = {
        "delta_gain": delta_gain,
        "mAP_hazy": mAP_hazy,
        "mAP_dehazed": mAP_dehazed,
    }

    if "clear" in result:
        mAP_clear = result["clear"]["mAP"]
        delta_gap = mAP_clear - mAP_dehazed
        denom = mAP_clear - mAP_hazy
        R = delta_gain / denom if abs(denom) > 1e-8 else float("inf")
        stats["delta_gap"] = delta_gap
        stats["R"] = R
        stats["mAP_clear"] = mAP_clear
    else:
        stats["delta_gap"] = None
        stats["R"] = None

    # Bootstrap CI (基于 per-image AP)
    per_img_hazy = result["hazy"].get("per_image", [])
    per_img_dehazed = result["dehazed"].get("per_image", [])
    if per_img_hazy and per_img_dehazed and len(per_img_hazy) == len(per_img_dehazed):
        gains = np.array([d - h for d, h in
                         zip(per_img_dehazed, per_img_hazy)])
        boot_means = []
        n = len(gains)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means.append(gains[idx].mean())
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        stats["delta_gain_ci_low"] = ci_low
        stats["delta_gain_ci_high"] = ci_high
    else:
        stats["delta_gain_ci_low"] = None
        stats["delta_gain_ci_high"] = None

    return stats


def per_haze_level_report(per_image_results: list[dict],
                          haze_levels: list[str]) -> dict:
    """按雾档分组报告 mAP 增益.

    Args:
        per_image_results: 每张图的 {stem, mAP_hazy, mAP_dehazed, ...}
        haze_levels: 与 per_image_results 等长的雾档标签
                     (如 "thin"/"medium"/"dense")

    Returns:
        {level: {n_images, mAP_hazy, mAP_dehazed, delta_gain, R}}
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r, level in zip(per_image_results, haze_levels):
        groups[level].append(r)

    report = {}
    for level, items in sorted(groups.items()):
        hazy_vals = [r.get("mAP_hazy", 0) for r in items]
        dehazed_vals = [r.get("mAP_dehazed", 0) for r in items]
        clear_vals = [r.get("mAP_clear", 0) for r in items]
        mh = np.mean(hazy_vals) if hazy_vals else 0.0
        md = np.mean(dehazed_vals) if dehazed_vals else 0.0
        mc = np.mean(clear_vals) if clear_vals else 0.0
        dg = md - mh
        denom = mc - mh
        R = dg / denom if abs(denom) > 1e-8 else float("inf")
        report[level] = {
            "n_images": len(items),
            "mAP_hazy": float(mh),
            "mAP_dehazed": float(md),
            "mAP_clear": float(mc) if clear_vals else None,
            "delta_gain": float(dg),
            "R": float(R) if mc > 0 else None,
        }
    return report


def paired_t_test(method_a: list[float],
                  method_b: list[float]) -> dict:
    """成对 t 检验: 检验两种方法 mAP 差异是否显著.

    Args:
        method_a: 方法 A 的 per-image mAP 列表
        method_b: 方法 B 的 per-image mAP 列表

    Returns:
        {t_stat, p_value, mean_diff, n, significant}
    """
    from scipy import stats as sp_stats
    a = np.array(method_a, dtype=np.float64)
    b = np.array(method_b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": 0.0,
                "n": n, "significant": False}
    diff = a[:n] - b[:n]
    t_stat, p_value = sp_stats.ttest_rel(a[:n], b[:n])
    if np.isnan(p_value):
        p_value = 1.0
    return {
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "n": n,
        "significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="下游任务增益评测 (mAP@0.5)")
    ap.add_argument("--detector", choices=["yolo", "maskrcnn"], default="yolo")
    ap.add_argument("--detector-weights", required=True)
    ap.add_argument("--hazy-dir", required=True)
    ap.add_argument("--labels-dir", required=True,
                    help="YOLO txt 标签目录 (三套图像共用)")
    ap.add_argument("--clear-dir", default=None, help="清晰图目录 (可选, 计算 oracle 上限)")
    ap.add_argument("--dehazed-dir", default=None,
                    help="已生成的去雾图目录; 省略则用 --dehaze-model 现场生成")
    ap.add_argument("--dehaze-model", default=None, help="去雾模型版本 (m1/m2p/m3/m4)")
    ap.add_argument("--dehaze-weights", default=None, help="去雾模型 checkpoint 路径")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--output", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch as _torch
    device = args.device or ("cuda" if _torch.cuda.is_available() else "cpu")

    # 1. 检测器 (权重冻结)
    if args.detector == "yolo":
        from icewave.detect.yolo import YOLO_CLASSES, YOLODetector
        detector = YOLODetector(args.detector_weights, conf=args.conf)
        class_names = YOLO_CLASSES
    else:
        from icewave.detect.maskrcnn_adapter import MaskRCNNDetector
        detector = MaskRCNNDetector(args.detector_weights, conf=args.conf)
        class_names = None

    # 2. 去雾图: 生成或读取
    dehazed_dir = args.dehazed_dir
    if dehazed_dir is None:
        if not args.dehaze_model:
            ap.error("需要 --dehazed-dir 或 --dehaze-model 之一")
        from icewave.models import build_model, load_checkpoint
        from icewave.utils.paths import OUTPUT_DIR, WEIGHTS_DIR
        model = build_model(args.dehaze_model).to(device)
        ckpt = args.dehaze_weights or (
            WEIGHTS_DIR / "checkpoints" / f"{args.dehaze_model}_best.pth")
        load_checkpoint(model, ckpt, device)
        model.eval()
        dehazed_dir = Path(OUTPUT_DIR) / "downstream" / "dehazed" / args.dehaze_model
        print(f"[生成去雾图] {args.hazy_dir} → {dehazed_dir}")
        dehaze_directory(model, Path(args.hazy_dir), Path(dehazed_dir), device)

    # 3. 增益计算
    print("[评测] 同一检测器在 hazy / dehazed / clear 上的 mAP@0.5 ...")
    result = downstream_gain(detector, Path(args.hazy_dir), Path(dehazed_dir),
                             Path(args.labels_dir),
                             clear_dir=Path(args.clear_dir) if args.clear_dir else None,
                             iou_thr=args.iou_thr, class_names=class_names)
    result["detector"] = args.detector
    result["detector_weights"] = str(args.detector_weights)

    from icewave.utils.paths import OUTPUT_DIR
    out_path = Path(args.output) if args.output else (
        Path(OUTPUT_DIR) / "downstream" / "downstream_gain.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"  mAP@{args.iou_thr}  hazy={result['hazy']['mAP']:.4f}  "
          f"dehazed={result['dehazed']['mAP']:.4f}  "
          f"ΔmAP={result['delta_mAP_dehazed_minus_hazy']:+.4f}")
    if "clear" in result:
        print(f"  clear={result['clear']['mAP']:.4f}  "
              f"gap={result['gap_clear_minus_dehazed']:+.4f}")

    # §6 增益统计
    gain_stats = compute_gain_stats(result)
    result["gain_stats"] = gain_stats
    if gain_stats.get("R") is not None:
        print(f"  Δ_gain={gain_stats['delta_gain']:+.4f}  "
              f"R={gain_stats['R']:.1%}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
