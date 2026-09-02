"""推理 CLI (P0-2b): 去雾 + 覆冰检测 + 对比图 + CSV 报告.

旧版问题: phase6/dehaze_inference.py 内嵌"文件哈希变化→自动重训"逻辑,
且权重/数据路径写死 ``D:\\dehaze_fusion`` / ``.trae-cn``。本 CLI:
1. 路径全部参数化 (支持 ${ENV} 与环境变量 ICEWAVE_*);
2. 检测与去雾彻底解耦, **默认关闭自动重训** (--train-if-missing 显式开启);
3. 输出目录规范化: 去雾图 / 冰掩码 / 标注图 / 对比图 / report.csv。

用法:
    icewave-infer --input data/test/hazy --model m4 \
        --detector yolo --detector-weights weights/yolo/power_line_best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch

from icewave.models import build_model, load_checkpoint
from icewave.utils.paths import OUTPUT_DIR, checkpoint_path


def _load_dehaze_model(version: str, weights: str | None, device: str):
    model = build_model(version).to(device)
    ckpt = weights or str(checkpoint_path(version))
    load_checkpoint(model, ckpt, device)
    model.eval()
    return model


def _load_detector(detector: str, weights: str, conf: float):
    if detector == "yolo":
        from icewave.detect.yolo import YOLODetector
        return YOLODetector(weights, conf=conf)
    if detector == "maskrcnn":
        from icewave.detect.maskrcnn_adapter import MaskRCNNDetector
        return MaskRCNNDetector(weights)
    raise ValueError(f"未知检测器: {detector} (可选 yolo/maskrcnn)")


def dehaze_one(model, img_bgr, device: str) -> tuple:
    """去雾单图 (复用 benchmark 的 padding 策略)."""
    from icewave.eval.benchmark import _dehaze
    return _dehaze(model, img_bgr, device)


def main(argv=None):
    ap = argparse.ArgumentParser(description="IceWave 去雾 + 覆冰检测推理")
    ap.add_argument("--input", required=True, help="输入目录 (雾图) 或单张图像")
    ap.add_argument("--output", default=None, help="输出目录 (默认 outputs/infer)")
    ap.add_argument("--model", default="m4", help="去雾模型版本 m1/m2/m2p/m3/m4")
    ap.add_argument("--weights", default=None, help="去雾模型权重路径")
    ap.add_argument("--detector", default=None, choices=["yolo", "maskrcnn"],
                    help="覆冰检测器; 省略则仅去雾")
    ap.add_argument("--detector-weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ice-mask", action="store_true",
                    help="跳过规则式冰掩码生成")
    ap.add_argument("--train-if-missing", action="store_true",
                    help="[危险] 检测器权重缺失时自动重训 (旧版默认行为, 现默认关闭)")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output) if args.output else (Path(OUTPUT_DIR) / "infer")
    (out_dir / "dehazed").mkdir(parents=True, exist_ok=True)
    (out_dir / "ice_mask").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)

    # 1. 去雾模型
    model = _load_dehaze_model(args.model, args.weights, device)
    print(f"[模型] {args.model} @ {device}")

    # 2. 检测器 (可选)
    detector = None
    if args.detector:
        if not args.detector_weights:
            ap.error(f"--detector {args.detector} 需要 --detector-weights")
        from pathlib import Path as _P
        wp = _P(args.detector_weights)
        if not wp.exists() and args.train_if_missing:
            if args.detector == "yolo":
                from icewave.detect.yolo import train_yolo
                train_yolo("ice_detection/configs/data.yaml")
            else:
                ap.error("maskrcnn 不支持自动重训, 请提供权重")
        detector = _load_detector(args.detector, args.detector_weights, args.conf)
        print(f"[检测] {args.detector} 已加载")

    # 3. 收集输入
    inp = Path(args.input)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    if inp.is_dir():
        images = sorted(p for p in inp.iterdir() if p.suffix.lower() in exts)
    else:
        images = [inp] if inp.suffix.lower() in exts else []
    if not images:
        ap.error(f"输入目录无图像: {inp}")

    rows = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[跳过] 无法读取 {img_path}")
            continue
        name = img_path.name

        # 去雾
        dehazed = dehaze_one(model, img, device)
        cv2.imwrite(str(out_dir / "dehazed" / name), dehazed)

        # 检测 (在去雾图上)
        detections = []
        if detector is not None:
            detections = detector.detect(dehazed)

        # 冰掩码 (规则式伪标签, 仅兜底输出)
        ice_mask = None
        if not args.no_ice_mask:
            from icewave.detect.ice_mask import generate_ice_mask
            ice_mask = generate_ice_mask(dehazed, detections or None)
            cv2.imwrite(str(out_dir / "ice_mask" / name), ice_mask)

        # 标注图
        annotated = dehazed.copy()
        if detector is not None:
            annotated = detector.draw(annotated, detections)
        cv2.imwrite(str(out_dir / "annotated" / name), annotated)

        # 对比图 (原图 | 去雾)
        h, w = img.shape[:2]
        compare = cv2.hconcat([img, dehazed])
        cv2.imwrite(str(out_dir / "compare" / name), compare)

        rows.append({
            "image": name,
            "n_detections": len(detections),
            "ice_coverage": float((ice_mask > 0).mean()) if ice_mask is not None else "",
        })

    # 4. CSV 报告
    report_path = out_dir / "report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "n_detections",
                                               "ice_coverage"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 处理 {len(rows)} 张图像 → {out_dir}")
    print(f"  去雾图:   {out_dir / 'dehazed'}")
    print(f"  冰掩码:   {out_dir / 'ice_mask'}")
    print(f"  标注图:   {out_dir / 'annotated'}")
    print(f"  对比图:   {out_dir / 'compare'}")
    print(f"  报告:     {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
