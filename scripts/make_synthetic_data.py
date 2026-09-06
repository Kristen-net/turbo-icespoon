#!/usr/bin/env python3
"""生成合成训练数据 (smoke test 用途).

不依赖 torch, 仅用 numpy + PIL 实现简化雾化,
避免 DLL 加载问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def simple_haze(clear: np.ndarray, beta: float = 0.08,
                A: np.ndarray | None = None,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """简化大气散射模型: I = J*t + A*(1-t)."""
    if A is None:
        A = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    h, w, _ = clear.shape
    if rng is None:
        rng = np.random.default_rng(0)
    # 随机透射图 (模拟空间变化的雾)
    t = 1.0 - beta * rng.uniform(0.5, 1.5, (h, w))
    t = np.clip(t, 0.1, 1.0).astype(np.float32)
    t = t[..., None]  # (h, w, 1)
    hazy = clear * t + A * (1.0 - t)
    return np.clip(hazy, 0.0, 1.0).astype(np.float32)


def generate(out_dir: Path, num_images: int = 20, size: int = 256,
             seed: int = 42):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        img_dir = out_dir / split / "clear"
        hazy_dir = out_dir / split / "hazy"
        label_dir = out_dir / split / "labels"
        for d in (img_dir, hazy_dir, label_dir):
            d.mkdir(parents=True, exist_ok=True)

        n = num_images if split == "train" else num_images // 2
        for i in range(n):
            clear = np.zeros((size, size, 3), dtype=np.float32)
            for y in range(size):
                t = y / size
                clear[y, :] = [0.5 + 0.3 * t, 0.6 + 0.2 * t, 0.8 + 0.1 * t]
            bboxes = []
            n_obj = int(rng.integers(1, 4))
            for _ in range(n_obj):
                w = int(rng.integers(30, 80))
                h = int(rng.integers(20, 60))
                x = int(rng.integers(0, size - w))
                y = int(rng.integers(0, size - h))
                color = rng.uniform(0.2, 0.8, 3).astype(np.float32)
                clear[y:y+h, x:x+w] = color
                bboxes.append((x / size, y / size,
                               (x + w) / size, (y + h) / size))

            beta = float(rng.uniform(0.04, 0.12))
            A = rng.uniform(0.7, 0.9, 3).astype(np.float32)
            hazy = simple_haze(clear, beta=beta, A=A, rng=rng)

            name = f"{split}_{i:04d}"
            Image.fromarray((clear * 255).astype(np.uint8)).save(
                img_dir / f"{name}.png")
            Image.fromarray((hazy * 255).clip(0, 255).astype(np.uint8)).save(
                hazy_dir / f"{name}.png")
            lines = []
            for (x1, y1, x2, y2) in bboxes:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (label_dir / f"{name}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    print(f"生成完成: {out_dir}")
    print(f"  train: {num_images} images")
    print(f"  val: {num_images // 2} images")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "synthetic")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.num, args.size, args.seed)
