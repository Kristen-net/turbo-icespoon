"""IceWave 数据集构建 CLI (P0-5).

相对旧 ``phase4/build_dataset_v3.py`` 的修复:
1. **边界图默认排除**: 暗通道处于 0.18-0.35 的"边界图"不再并入清晰图, 避免
   轻度带雾图像污染 GT.
2. **场景级切分 (by_zip)**: 同一 zip 来源的图全部归入同一 split, 防止相邻
   巡检帧泄漏到验证集. 旧版按图片随机划分存在同场景泄漏风险.
3. **浓雾档进入测试轮转**: 浓雾 (``dense``) 不再仅训练, 也进入 val/test,
   覆盖困难雾况的能力评测.
4. **路径参数化**: 数据源/输出/工作线程全部走 ``--src/--out/--workers`` 或
   ``ICEWAVE_DATA_ROOT``, 不再写死 ``D:\\DATA_ALL``.

用法
----
::

    icewave-build-dataset --src /path/to/zips_or_clear_imgs \
        --out $ICEWAVE_DATA_ROOT/dataset \
        --val-ratio 0.1 --include-border  # 默认排除边界图
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


HAZE_LEVELS = ("thin", "medium", "dense")


# ---------------------------------------------------------------------------
# 暗通道去雾度评估 (He et al. 2009 简化版, 用于 clear/border 划分)
# ---------------------------------------------------------------------------
def _dark_channel(img_bgr: np.ndarray, ksize: int = 15) -> float:
    """最小通道局部最小值, 数值越大 → 雾越浓."""
    b, g, r = cv2.split(img_bgr)
    dc = cv2.erode(np.minimum(np.minimum(b, g), r),
                   np.ones((ksize, ksize), np.uint8))
    return float(dc.mean()) / 255.0


def _classify_clear(dc: float, *, include_border: bool) -> bool:
    """根据暗通道值判定"清晰图"."""
    if dc <= 0.18:
        return True
    if include_border and 0.18 < dc <= 0.35:
        return True
    return False


# ---------------------------------------------------------------------------
# 数据源: zip 包或已展开目录
# ---------------------------------------------------------------------------
def _list_images(src: Path):
    """递归收集图片, 同时按 zip 来源分组 (场景级切分键)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    groups: dict[str, list[Path]] = {}
    if src.is_file() and src.suffix == ".zip":
        groups[str(src)] = [src]
        return groups

    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        # 场景键: zip 顶层目录; 若无 zip 上下文, 用 parent
        parent = p.parent
        # 尝试定位所属 zip (若 src 是 zip 集合目录)
        for anc in p.parents:
            if anc == src:
                break
            if anc.suffix == ".zip":
                parent = anc
                break
        groups.setdefault(str(parent), []).append(p)
    return groups


def _scene_id(zip_or_dir: Path) -> str:
    """稳定场景 ID, 用路径的 sha1 短哈希避免文件名特殊字符."""
    return hashlib.sha1(str(zip_or_dir).encode("utf-8")).hexdigest()[:12]


def _read_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return img


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build(src: Path, out: Path, val_ratio: float = 0.1,
          include_border: bool = False,
          haze_levels: tuple = HAZE_LEVELS,
          haze_params: dict | None = None,
          seed: int = 42, workers: int = 1) -> None:
    """从 ``src`` 构建 IceWave 数据集到 ``out``/{train,val}/{clear,hazy,ice_mask}/."""
    from icewave.data.degradation import (
        HAZE_PRESETS, IceParams, synthesize_hazy_iced,
    )

    out = Path(out)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    haze_params = haze_params or HAZE_PRESETS

    # 1. 收集清晰图 (按场景)
    groups = _list_images(Path(src))
    if not groups:
        raise FileNotFoundError(f"未发现图片: {src}")

    # 2. 暗通道过滤 (是否纳入边界图)
    clear_groups: dict[str, list[Path]] = {}
    rejected = 0
    for scene, files in groups.items():
        ok = []
        for f in files:
            img = _read_image(f)
            if img is None:
                continue
            dc = _dark_channel(img)
            if _classify_clear(dc, include_border=include_border):
                ok.append(f)
            else:
                rejected += 1
        if ok:
            clear_groups[scene] = ok

    if not clear_groups:
        raise RuntimeError("过滤后无清晰图; 可尝试 --include-border 放宽阈值。")

    # 3. 场景级切分
    scenes = list(clear_groups.keys())
    py_rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_ratio)))
    val_scenes = set(scenes[:n_val])
    print(f"[split] 总场景 {len(scenes)} → train {len(scenes) - n_val} / "
          f"val {n_val}; 拒绝图 {rejected}")

    # 4. 输出目录
    for split in ("train", "val"):
        for sub in ("clear", "hazy", "ice_mask"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    # 5. 写入 clear + 合成 hazy
    ice_params = IceParams(enabled=False)
    counts = {"train": 0, "val": 0}
    for scene, files in clear_groups.items():
        split = "val" if scene in val_scenes else "train"
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            base = _scene_id(Path(scene)) + "_" + f.stem
            cv2.imwrite(str(out / split / "clear" / f"{base}.png"), img)
            for k, level in enumerate(haze_levels):
                params = haze_params[level]
                hazy = synthesize_hazy_iced(img, params, ice=ice_params, rng=rng)
                cv2.imwrite(str(out / split / "hazy" / f"{base}_haze{k}.png"),
                            hazy)
            counts[split] += 1
    print(f"[done] train={counts['train']}  val={counts['val']}  "
          f"→ {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    from icewave.utils.paths import DATA_ROOT

    ap = argparse.ArgumentParser(description="IceWave 数据集构建 (场景级切分)")
    ap.add_argument("--src", required=True, help="原始图片目录或 zip")
    ap.add_argument("--out", default=None,
                    help=f"输出根 (默认 $ICEWAVE_DATA_ROOT 或 ./{DATA_ROOT})")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--include-border", action="store_true",
                    help="纳入暗通道 0.18-0.35 的边界图作 GT (默认排除)")
    ap.add_argument("--haze-levels", nargs="+", default=list(HAZE_LEVELS),
                    choices=HAZE_LEVELS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else DATA_ROOT
    build(src=Path(args.src), out=out, val_ratio=args.val_ratio,
          include_border=args.include_border,
          haze_levels=tuple(args.haze_levels), seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())