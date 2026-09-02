"""评测指标 (P0-3/P0-4 组件).

关键修复: 冰区指标一律以**人工标注掩码**为参照计算, 废除旧版
"伪标签规则 → 训练 → 同一规则评指标"的循环论证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from icewave.losses.itl import ssim_map


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float64) / 255.0
    gt_f = gt.astype(np.float64) / 255.0
    mse = np.mean((pred_f - gt_f) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """标准 SSIM (RGB 通道平均, 11x11 高斯窗)."""
    x = torch.from_numpy(pred.transpose(2, 0, 1)[None]).float() / 255.0
    y = torch.from_numpy(gt.transpose(2, 0, 1)[None]).float() / 255.0
    return float(ssim_map(x, y).mean().item())


def lpips_score(pred: np.ndarray, gt: np.ndarray, net: str = "alex") -> Optional[float]:
    """LPIPS (可选依赖 lpips 包)."""
    try:
        import lpips as lpips_lib
        import torch
    except ImportError:
        return None
    loss_fn = lpips_lib.LPIPS(net=net)
    x = torch.from_numpy(pred.transpose(2, 0, 1)[None]).float() * 2 - 1
    y = torch.from_numpy(gt.transpose(2, 0, 1)[None]).float() * 2 - 1
    with torch.no_grad():
        return float(loss_fn(x, y).item())


# ---------------------------------------------------------------------------
# 冰区指标 (以人工标注为参照)
# ---------------------------------------------------------------------------
def _otsu_separability(gray: np.ndarray, mask: np.ndarray) -> float:
    """Otsu 可分离性: 冰区/非冰区灰度分布的类间方差 (越大越可分)."""
    g = gray.astype(np.float32)
    m = mask > 127
    if m.sum() == 0 or (~m).sum() == 0:
        return 0.0
    w0, w1 = m.mean(), (~m).mean()
    mu0, mu1 = g[m].mean(), g[~m].mean()
    return float(w0 * w1 * (mu0 - mu1) ** 2)


def _texture_energy(gray: np.ndarray, mask: np.ndarray) -> float:
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    m = mask > 127
    if m.sum() == 0:
        return 0.0
    return float(np.abs(grad[m]).mean())


def _boundary_sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    """冰边界梯度锐度 (边界带内 Sobel 幅值均值)."""
    m = (mask > 127).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    dilated = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    band = cv2.subtract(dilated, m)
    if band.sum() == 0:
        return 0.0
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))
    return float(grad[band > 0].mean())


def _region_iou(pred_mask: np.ndarray, human_mask: np.ndarray) -> float:
    """预测冰掩码与人工标注的 IoU (检测有效性, 越高越好)."""
    p, h = pred_mask > 127, human_mask > 127
    union = (p | h).sum()
    if union == 0:
        return 1.0
    return float((p & h).sum() / union)


def ice_region_metrics(img_bgr: np.ndarray, human_mask: np.ndarray,
                       pred_mask: Optional[np.ndarray] = None) -> dict:
    """冰区指标: 全部以人工标注 human_mask 为参照 (循环论证修复).

    返回 otsu_separability / texture_energy / boundary_sharpness,
    以及提供 pred_mask 时的 region_iou。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = {
        "otsu_separability": _otsu_separability(gray, human_mask),
        "texture_energy": _texture_energy(gray, human_mask),
        "boundary_sharpness": _boundary_sharpness(gray, human_mask),
    }
    if pred_mask is not None:
        out["region_iou_vs_human"] = _region_iou(pred_mask, human_mask)
    return out


def pseudo_label_agreement(pseudo_dir: Path, human_dir: Path) -> dict:
    """伪标签与人工标注的一致性统计 (P0-4: 主动暴露循环论证程度).

    返回 mean_iou / n / coverage_mean, 供论文如实报告。
    """
    ious, coverages = [], []
    for hp in sorted(Path(human_dir).glob("*_ice.png")):
        pp = Path(pseudo_dir) / hp.name
        if not pp.exists():
            continue
        human = cv2.imread(str(hp), cv2.IMREAD_GRAYSCALE)
        pseudo = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if human is None or pseudo is None or human.shape != pseudo.shape:
            continue
        ious.append(_region_iou(pseudo, human))
        coverages.append(float((human > 127).mean()))
    if not ious:
        return {"n": 0}
    return {"n": len(ious), "mean_iou": float(np.mean(ious)),
            "coverage_mean": float(np.mean(coverages))}
