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
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
