"""公开基准评测 harness (P0-3).

用法:
    icewave-eval-benchmark --config configs/benchmarks.yaml \
        --benchmarks reside_sots_indoor reside_sots_outdoor --models m1 m4

基准定义见 configs/benchmarks.yaml (数据根目录支持 ${ENV:default} 占位符)。
结果写入 outputs/benchmark/<bench>_<model>.json 并汇总 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from icewave.eval.metrics import lpips_score, psnr, ssim
from icewave.models import build_model, load_checkpoint
from icewave.train.config import load_config
from icewave.utils.paths import OUTPUT_DIR, WEIGHTS_DIR


def _dehaze(model, img_bgr: np.ndarray, device: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].float().to(device) / 255.0
    pad = (16 - h % 16) % 16, (16 - w % 16) % 16
    if pad[0] or pad[1]:
        x = torch.nn.functional.pad(x, (0, pad[1], 0, pad[0]), mode="reflect")
    with torch.no_grad():
        pred = model(x)
    pred = pred[:, :, :h, :w].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _iter_pairs(bench_cfg: dict):
    """产出 (hazy_path, gt_path); unpaired 基准 gt 为 None."""
    hazy_dir, gt_dir = Path(bench_cfg["hazy_dir"]), bench_cfg.get("gt_dir")
    gt_dir = Path(gt_dir) if gt_dir else None
    for hp in sorted(hazy_dir.glob("*")):
        if hp.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        gt = None
        if gt_dir is not None:
            gt = gt_dir / hp.name
            if not gt.exists():
                gt = gt_dir / f"{hp.stem}.png"
        yield hp, (gt if (gt and gt.exists()) else None)


def evaluate_model(model, bench_cfg: dict, device: str) -> dict:
    rows = []
    for hazy_path, gt_path in _iter_pairs(bench_cfg):
        img = cv2.imread(str(hazy_path))
        if img is None:
            continue
        pred = _dehaze(model, img, device)
        row = {"name": hazy_path.name}
        if gt_path is not None:
            gt = cv2.imread(str(gt_path))
            if gt is not None and gt.shape[:2] == pred.shape[:2]:
                row["psnr"] = psnr(pred, gt)
                row["ssim"] = ssim(pred, gt)
                lp = lpips_score(pred, gt)
                if lp is not None:
                    row["lpips"] = lp
        rows.append(row)

    paired = [r for r in rows if "psnr" in r]
    summary = {"n_images": len(rows), "n_paired": len(paired)}
    if paired:
        for k in ("psnr", "ssim", "lpips"):
            vals = [r[k] for r in paired if k in r]
            if vals:
                summary[f"mean_{k}"] = float(np.mean(vals))
    summary["per_image"] = rows
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="公开基准评测")
    ap.add_argument("--config", default="configs/benchmarks.yaml")
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="模型版本名或 checkpoint 路径")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bench_cfg = load_config(args.config)["benchmarks"]
    out_dir = Path(OUTPUT_DIR) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_spec in args.models:
        if str(model_spec).endswith(".pth"):
            version = Path(model_spec).stem
            model = build_model("m4").to(device)
            load_checkpoint(model, model_spec, device)
        else:
            version = model_spec
            model = build_model(version).to(device)
            load_checkpoint(model, WEIGHTS_DIR / "checkpoints" / f"{version}_best.pth",
                            device)
        model.eval()

        for bench_name in args.benchmarks:
            if bench_name not in bench_cfg:
                raise KeyError(f"基准 {bench_name} 未在 {args.config} 中定义")
            cfg_b = bench_cfg[bench_name]
            if not Path(cfg_b["hazy_dir"]).is_dir():
                print(f"[跳过] {bench_name}: hazy_dir 不存在 ({cfg_b['hazy_dir']}), "
                      f"请先下载数据集 (见 configs/benchmarks.yaml 注释)")
                continue
            print(f"[评测] {version} @ {bench_name} ...")
            result = evaluate_model(model, cfg_b, device)
            result_path = out_dir / f"{bench_name}_{version}.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → {result_path}  PSNR={result.get('mean_psnr')} "
                  f"SSIM={result.get('mean_ssim')}")
            summary_rows.append({
                "benchmark": bench_name, "model": version,
                "n_paired": result["n_paired"],
                "psnr": result.get("mean_psnr", ""),
                "ssim": result.get("mean_ssim", ""),
                "lpips": result.get("mean_lpips", ""),
            })

    if summary_rows:
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"汇总: {csv_path}")


if __name__ == "__main__":
    main()
