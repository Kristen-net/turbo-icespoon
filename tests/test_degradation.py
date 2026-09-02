"""复合退化物理模型测试 (P1-1).

关键性质:
- 无冰时严格退化为经典 ASM (与旧合成器兼容);
- 确定性 (同 seed 同输出);
- 元数据 (t_map/A/alpha/thickness) 数值范围正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from icewave.data.degradation import (
    HAZE_PRESETS,
    HazeParams,
    IceParams,
    ice_composite,
    make_ice_thickness,
    make_ice_texture,
    make_transmission_map,
    synthesize_haze,
    synthesize_hazy_iced,
)


def _clear(h=128, w=160, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestDegenerationASM:
    def test_ice_disabled_equals_pure_haze(self):
        """ice=None 或 enabled=False → 与经典 synthesize_haze 完全一致."""
        clear = _clear()
        haze = HazeParams(t_min=0.3, t_max=0.5, airlight=200)
        rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
        out_a = synthesize_hazy_iced(clear, haze, ice=None, rng=rng1)
        out_b, _ = synthesize_haze(clear, haze.t_min, haze.t_max,
                                   haze.airlight, rng2)
        np.testing.assert_array_equal(out_a, out_b)

    def test_ice_disabled_by_flag(self):
        clear = _clear()
        haze = HazeParams()
        a = synthesize_hazy_iced(clear, haze, ice=IceParams(enabled=False),
                                 rng=np.random.default_rng(7))
        b = synthesize_hazy_iced(clear, haze, ice=None,
                                 rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_determinism(self):
        clear = _clear()
        haze, ice = HazeParams(), IceParams(enabled=True)
        a = synthesize_hazy_iced(clear, haze, ice, rng=np.random.default_rng(123))
        b = synthesize_hazy_iced(clear, haze, ice, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(a, b)

    def test_output_range_and_shape(self):
        clear = _clear()
        out = synthesize_hazy_iced(clear, HazeParams(), IceParams(enabled=True),
                                   rng=np.random.default_rng(0))
        assert out.shape == clear.shape
        assert out.dtype == np.uint8


class TestMetadata:
    def test_metadata_ranges(self):
        clear = _clear()
        hazy, meta = synthesize_hazy_iced(
            clear, HazeParams(t_min=0.2, t_max=0.4), IceParams(enabled=True),
            rng=np.random.default_rng(0), with_metadata=True)
        assert 0.05 <= meta["t_map"].min() and meta["t_map"].max() <= 0.95
        assert len(meta["A"]) == 3
        assert 170 <= min(meta["A"]) and max(meta["A"]) <= 245
        assert 0.0 <= meta["ice_alpha"].min() <= meta["ice_alpha"].max() <= 1.0
        assert meta["ice_thickness"].min() >= 0.0

    def test_haze_only_metadata_has_no_ice_keys(self):
        clear = _clear()
        _, meta = synthesize_hazy_iced(clear, HazeParams(), None,
                                       rng=np.random.default_rng(0),
                                       with_metadata=True)
        assert "ice_alpha" not in meta
        assert "t_map" in meta


class TestIceComposite:
    def test_alpha_monotone_in_thickness(self):
        """Beer-Lambert: 冰越厚 → alpha 越大 (更不透明)."""
        rng = np.random.default_rng(0)
        clear = _clear(64, 64)
        ice = IceParams(enabled=True, extinction=2.5, coverage=0.5)
        _, meta_thin = ice_composite(
            clear, ice, rng,
            thickness=np.full((64, 64), 0.1, np.float32))
        _, meta_thick = ice_composite(
            clear, ice, rng,
            thickness=np.full((64, 64), 1.0, np.float32))
        assert meta_thick["alpha"].mean() > meta_thin["alpha"].mean()

    def test_zero_thickness_is_identity(self):
        """d=0 → alpha=0 → 输出应等于原图."""
        clear = _clear(64, 64)
        rng = np.random.default_rng(0)
        out, meta = ice_composite(clear, IceParams(enabled=True), rng,
                                  thickness=np.zeros((64, 64), np.float32))
        assert float(meta["alpha"].max()) == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_array_equal(out, clear)

    def test_coverage_approx(self):
        """目标覆盖比例近似可控 (分位阈值法, 允许较大容差)."""
        clear = _clear(256, 256)
        rng = np.random.default_rng(0)
        d = make_ice_thickness(256, 256, rng, coverage=0.3, max_thickness=1.0)
        actual = float((d > 0).mean())
        assert 0.15 < actual < 0.45, f"覆盖比例偏差过大: {actual:.2f}"


class TestPresets:
    def test_three_levels_ordered(self):
        """thin → medium → dense: 透射率单调下降, 大气光上升."""
        t = [HAZE_PRESETS[k].t_min for k in ("thin", "medium", "dense")]
        a = [HAZE_PRESETS[k].airlight for k in ("thin", "medium", "dense")]
        assert t == sorted(t, reverse=True)
        assert a == sorted(a)

    def test_preset_smoke(self):
        clear = _clear(64, 64)
        for name, params in HAZE_PRESETS.items():
            out = synthesize_hazy_iced(clear, params,
                                       rng=np.random.default_rng(0))
            assert out.shape == clear.shape, name


class TestTransmissionMap:
    def test_range(self):
        t = make_transmission_map(128, 128, np.random.default_rng(0), 0.2, 0.4)
        assert t.shape == (128, 128)
        assert 0.05 <= t.min() and t.max() <= 0.95
        assert t.min() < t.max()  # 空间不均匀
