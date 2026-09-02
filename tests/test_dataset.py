"""IceAwareDataset 测试: 配对规则 / 人工标注优先 / 走廊掩码."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from icewave.data.dataset import IceAwareDataset, _pair_base

H, W = 64, 64


def _make_split(root: Path, split: str = "val", n: int = 2, haze_per: int = 2,
                with_ice: bool = True, with_human: bool = False):
    """构造最小数据集目录."""
    (root / split / "hazy").mkdir(parents=True)
    (root / split / "clear").mkdir(parents=True)
    if with_ice:
        (root / split / "ice_mask").mkdir()
    if with_human:
        (root / split / "ice_mask_human").mkdir()
    for i in range(n):
        base = f"img_{i:04d}"
        clear = np.full((H, W, 3), 100 + i * 20, np.uint8)
        cv2.imwrite(str(root / split / "clear" / f"{base}.png"), clear)
        for k in range(haze_per):
            hazy = np.full((H, W, 3), 200, np.uint8)
            cv2.imwrite(str(root / split / "hazy" / f"{base}_haze{k}.png"), hazy)
        if with_ice:
            ice = np.zeros((H, W), np.uint8)
            ice[:, :W // 2] = 255  # 左半冰
            cv2.imwrite(str(root / split / "ice_mask" / f"{base}_ice.png"), ice)
        if with_human:
            human = np.zeros((H, W), np.uint8)
            human[:H // 2, :] = 255  # 上半冰 (与伪标签不同)
            cv2.imwrite(str(root / split / "ice_mask_human" / f"{base}_ice.png"),
                        human)
    return root / split


class TestPairBase:
    @pytest.mark.parametrize("name,base", [
        ("train_0001_haze0.png", "train_0001"),
        ("train_0001_haze12.png", "train_0001"),
        ("train_0001.png", "train_0001"),
        ("img_haze0_haze1.png", "img_haze0"),  # 只剥离最后一个后缀
    ])
    def test_regex(self, name, base):
        assert _pair_base(name) == base


class TestDataset:
    def test_pairing(self, tmp_path):
        split = _make_split(tmp_path / "ds", n=2, haze_per=2)
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0)
        assert len(ds) == 4  # 2 图 × 2 雾级

    def test_missing_clear_excluded(self, tmp_path):
        split = _make_split(tmp_path / "ds", n=2)
        (split / "clear" / "img_0001.png").unlink()
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0)
        assert len(ds) == 2  # 只剩 img_0000 的两个雾级

    def test_getitem_shapes(self, tmp_path):
        _make_split(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0)
        hazy, clear, ice = ds[0]
        assert hazy.shape == (3, H, W) and clear.shape == (3, H, W)
        assert ice.shape == (1, H, W)
        assert set(ice.unique().tolist()) <= {0.0, 1.0}  # 已二值化
        assert float(ice.mean()) == pytest.approx(0.5, abs=0.01)

    def test_patch_crop_val_centered(self, tmp_path):
        """验证集中心裁剪 → 两次取样结果一致 (可复现)."""
        _make_split(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=32)
        a = ds[0]
        b = ds[0]
        torch.testing.assert_close(a[0], b[0])

    def test_no_ice_returns_pair_only(self, tmp_path):
        _make_split(tmp_path / "ds", with_ice=False)
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0,
                             return_ice=False)
        hazy, clear = ds[0]
        assert hazy.shape == (3, H, W)

    def test_human_mask_priority(self, tmp_path):
        """人工标注存在时优先于伪标签 (P0-4 循环论证修复)."""
        _make_split(tmp_path / "ds", with_ice=True, with_human=True)
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0)
        _, _, ice = ds[0]
        # 人工标注: 上半=1, 下半=0 (伪标签是左半=1, 右半=0)
        assert float(ice[:, :H // 2, :].mean()) == pytest.approx(1.0)
        assert float(ice[:, H // 2:, :].mean()) == pytest.approx(0.0)
        # 若错误地取了伪标签: 左半 mean 会是 1.0 (此处应为 0.5)
        assert float(ice[:, :, :W // 2].mean()) == pytest.approx(0.5)

    def test_corridor_from_ice(self, tmp_path):
        _make_split(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0,
                             corridor_from_ice=True)
        hazy, clear, ice, corridor = ds[0]
        assert corridor.shape == (1, H, W)
        # 走廊 = 冰区膨胀 → 覆盖 ⊇ 冰区, 且膨胀出额外区域
        assert bool((corridor >= ice).all())
        assert float((corridor * (1 - ice)).sum()) > 0

    def test_repr(self, tmp_path):
        _make_split(tmp_path / "ds")
        ds = IceAwareDataset(tmp_path / "ds", split="val", patch_size=0)
        r = repr(ds)
        assert "n=4" in r and "human_masks=False" in r
