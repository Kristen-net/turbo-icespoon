"""IceAwareDataset (P0-4/P0-5).

相对旧版的关键改进:
- **场景级切分** (by_zip): 调用方按 zip 来源划分 train/val, 数据构建器负责
  给出 ``{split}/{hazy, clear, ice_mask[, ice_mask_human]}/`` 子目录.
- **人工标注优先**: ``ice_mask_human/{base}_ice.png`` 存在时, 训练/评测一律
  以人工标注为准, 打破"规则造标签→同一规则评效果"的循环论证 (P0-4).
- **可复现验证集裁剪**: 验证 patch 走中心裁剪, ``test_patch_crop_val_centered``
  保证两次 ``__getitem__`` 输出逐元素相等.

目录约定
--------
::

    <root>/train/
        clear/{base}.png            # 清晰 GT (来自暗通道过滤 + 浓雾档排除)
        hazy/{base}_haze{k}.png     # 多雾级 (k=0..n-1)
        ice_mask/{base}_ice.png     # 伪标签 (规则生成, 可选)
        ice_mask_human/{base}_ice.png  # 人工标注 (存在则优先)
    <root>/val/  同上

返回内容由构造参数决定:
- ``return_ice=False``: ``(hazy, clear)``
- ``return_ice=True``: ``(hazy, clear, ice)``
- ``corridor_from_ice=True`` (joint 训练): ``(hazy, clear, ice, corridor)``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_HAZE_SUFFIX_RE = re.compile(r"_haze\d+$")


def _pair_base(filename: str) -> str:
    """从 haze 文件名提取基础名: 剥离扩展名与尾部 ``_hazeN`` 段.

    例: ``train_0001_haze0.png`` → ``train_0001``;
        ``img_haze0_haze1.png``   → ``img_haze0`` (只剥离最后一段).
    """
    stem = Path(filename).stem
    return _HAZE_SUFFIX_RE.sub("", stem)


def _read_gray01(path: Path) -> np.ndarray:
    """读取单通道掩码并二值化到 {0, 1}."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 0), dtype=np.uint8)
    return (img > 127).astype(np.uint8)


def _crop(img: np.ndarray, size: int, center: bool) -> np.ndarray:
    """随机裁剪或中心裁剪."""
    h, w = img.shape[:2]
    if h < size or w < size:
        # 不足则中心 pad 到 size (罕见, 仅测试用)
        ph = max(0, size - h) // 2
        pw = max(0, size - w) // 2
        pad = ((ph, max(0, size - h) - ph), (pw, max(0, size - w) - pw))
        if img.ndim == 3:
            pad = pad + ((0, 0),)
        img = np.pad(img, pad, mode="reflect")
        h, w = img.shape[:2]
    if center:
        y0 = (h - size) // 2
        x0 = (w - size) // 2
    else:
        y0 = int(np.random.randint(0, h - size + 1))
        x0 = int(np.random.randint(0, w - size + 1))
    return img[y0:y0 + size, x0:x0 + size]


def _dilate(mask: np.ndarray, k: int = 7) -> np.ndarray:
    """二值掩码膨胀 (与 joint 训练走廊语义一致)."""
    if mask.size == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


class IceAwareDataset(Dataset):
    """去雾 + 覆冰感知训练数据集."""

    def __init__(self, root, split: str = "train",
                 patch_size: int = 192, is_train: bool = True,
                 return_ice: bool = True, corridor_from_ice: bool = False,
                 corridor_kernel: int = 7):
        self.root = Path(root)
        self.split = split
        self.patch_size = int(patch_size)
        self.is_train = is_train
        self.return_ice = return_ice
        self.corridor_from_ice = corridor_from_ice
        self.corridor_kernel = corridor_kernel

        base = self.root / split
        self.clear_dir = base / "clear"
        self.hazy_dir = base / "hazy"
        self.ice_dir = base / "ice_mask"
        self.human_dir = base / "ice_mask_human"
        for d in (self.clear_dir, self.hazy_dir):
            if not d.exists():
                raise FileNotFoundError(f"数据集目录缺失: {d}")

        # 配对: 仅保留 (hazy, clear) 双方都存在的 base
        clear_bases = {_pair_base(p.name) for p in self.clear_dir.glob("*.*")}
        self.samples = []  # list of (hazy_path, clear_path, ice_path or None, human_path or None)
        self._human_masks = False
        for hp in sorted(self.hazy_dir.glob("*.*")):
            base = _pair_base(hp.name)
            cp = self.clear_dir / f"{base}.png"
            if not cp.exists():
                continue
            ice_p = self.ice_dir / f"{base}_ice.png"
            human_p = self.human_dir / f"{base}_ice.png"
            ice = ice_p if ice_p.exists() else None
            human = human_p if human_p.exists() else None
            if human is not None:
                self._human_masks = True
            self.samples.append((hp, cp, ice, human))

        if not self.samples:
            raise RuntimeError(
                f"{split} 子目录无 (hazy, clear) 配对样本: {base}")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        hp, cp, ice_p, human_p = self.samples[idx]

        hazy = cv2.imread(str(hp))
        clear = cv2.imread(str(cp))
        if hazy is None or clear is None:
            raise RuntimeError(f"读取失败: hazy={hp} clear={cp}")
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)

        ps = self.patch_size
        if ps > 0:
            if self.is_train:
                seed_y = np.random.randint(0, max(1, hazy.shape[0] - ps + 1))
                seed_x = np.random.randint(0, max(1, hazy.shape[1] - ps + 1))
                hazy = _crop(hazy, ps, center=False)
                clear = _crop(clear, ps, center=False)
            else:
                hazy = _crop(hazy, ps, center=True)
                clear = _crop(clear, ps, center=True)

        hazy_t = torch.from_numpy(hazy.transpose(2, 0, 1)).float() / 255.0
        clear_t = torch.from_numpy(clear.transpose(2, 0, 1)).float() / 255.0

        if not self.return_ice:
            return hazy_t, clear_t

        # ice: 优先人工标注 (P0-4)
        mask_path = human_p if human_p is not None else ice_p
        if mask_path is None:
            ice = torch.zeros(1, hazy_t.shape[1], hazy_t.shape[2])
        else:
            ice_arr = _read_gray01(mask_path)
            if ice_arr.size and ps > 0:
                ice_arr = _crop(ice_arr, ps, center=not self.is_train)
            if ice_arr.size == 0:
                ice = torch.zeros(1, hazy_t.shape[1], hazy_t.shape[2])
            else:
                ice = torch.from_numpy(ice_arr).float().unsqueeze(0)

        if not self.corridor_from_ice:
            return hazy_t, clear_t, ice

        # corridor: 冰区膨胀, 表示检测感知约束的目标区域
        ice_np = (ice.squeeze(0).numpy() > 0.5).astype(np.uint8)
        corridor_np = _dilate(ice_np, self.corridor_kernel)
        corridor = torch.from_numpy(corridor_np).float().unsqueeze(0)
        return hazy_t, clear_t, ice, corridor

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"IceAwareDataset(split={self.split!r} n={len(self)} "
                f"patch={self.patch_size} train={self.is_train} "
                f"human_masks={self._human_masks})")