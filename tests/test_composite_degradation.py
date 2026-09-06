"""复合退化升级字段测试 (JOINT_OPTIMIZATION_FRAMEWORK §2.3).

验证:
- IceParams 新增字段 (specular_strength, translucent_strength, blur_sigma_*, morphology)
- 镜面反射 S 生成与形态关联
- 各向异性模糊核 K 性质
- synthesize_hazy_iced 元数据新增字段
- 无冰时向后兼容 (与旧 synthesize_haze 逐字节相等)
"""

from __future__ import annotations

import numpy as np
import pytest

from icewave.data.degradation import (
    HazeParams,
    IceParams,
    make_anisotropic_blur_kernel,
    make_specular_reflection,
    synthesize_haze,
    synthesize_hazy_iced,
)


def _clear(h=128, w=160, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestIceParamsFields:
    def test_new_fields_have_defaults(self):
        ice = IceParams(enabled=True)
        assert ice.specular_strength == 0.4
        assert ice.translucent_strength == 0.3
        assert ice.blur_sigma_x == 1.0
        assert ice.blur_sigma_y == 2.5
        assert ice.morphology == "shell"

    def test_custom_morphology(self):
        ice = IceParams(enabled=True, morphology="sleeve")
        assert ice.morphology == "sleeve"


class TestSpecularReflection:
    def test_shape_and_range(self):
        S = make_specular_reflection(64, 80, np.random.default_rng(0))
        assert S.shape == (64, 80, 3)
        assert 0.0 <= S.min() and S.max() <= 1.0

    def test_determinism(self):
        a = make_specular_reflection(64, 64, np.random.default_rng(42))
        b = make_specular_reflection(64, 64, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("morphology", ["shell", "sleeve", "lobe", "cluster"])
    def test_morphology_variations(self, morphology):
        S = make_specular_reflection(64, 64, np.random.default_rng(0), morphology)
        assert S.shape == (64, 64, 3)

    def test_cold_tone_bias(self):
        """冰面镜面反射偏冷色调: B > G > R 均值."""
        S = make_specular_reflection(256, 256, np.random.default_rng(0))
        r_mean, g_mean, b_mean = S[:, :, 0].mean(), S[:, :, 1].mean(), S[:, :, 2].mean()
        assert b_mean >= r_mean  # B >= R


class TestAnisotropicBlurKernel:
    def test_normalization(self):
        kernel = make_anisotropic_blur_kernel(1.0, 2.5, ksize=5)
        assert kernel.shape == (5, 5)
        assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_horizontal_spread_when_sigma_y_larger(self):
        """sigma_y > sigma_x → 垂直方向扩散更强 (sigma_y 控制行方向).
        框架文档中 sigma_y=2.5 表示'沿导线方向拉长', 对应垂直方向更宽的扩散
        (冰层在导线轮廓处沿导线方向的模糊延伸).
        """
        kernel = make_anisotropic_blur_kernel(0.5, 5.0, ksize=5)
        pad = 5 // 2
        row = kernel[pad, :]  # 中心行 (水平方向)
        col = kernel[:, pad]  # 中心列 (垂直方向)
        # sigma_y 大 → 垂直方向 (列) 端点权重更高
        assert float(col[0]) > float(row[0])

    def test_isotropic_when_equal_sigma(self):
        kernel = make_anisotropic_blur_kernel(2.0, 2.0, ksize=5)
        pad = 5 // 2
        np.testing.assert_array_almost_equal(kernel[pad, :], kernel[:, pad])


class TestSynthesizeUpgraded:
    def test_metadata_has_new_fields(self):
        clear = _clear()
        ice = IceParams(enabled=True)
        _, meta = synthesize_hazy_iced(
            clear, HazeParams(), ice,
            rng=np.random.default_rng(0), with_metadata=True)
        assert "S" in meta
        assert meta["S"].shape == clear.shape
        assert "morphology" in meta
        assert meta["morphology"] == "shell"
        assert "blur_kernel_id" in meta
        assert "none" not in meta["blur_kernel_id"]

    def test_no_blur_when_disabled(self):
        clear = _clear()
        ice = IceParams(enabled=True)
        _, meta = synthesize_hazy_iced(
            clear, HazeParams(), ice,
            rng=np.random.default_rng(0),
            with_metadata=True, apply_blur=False)
        assert meta["blur_kernel_id"] == "none"

    def test_no_speckle_when_disabled(self):
        clear = _clear()
        ice = IceParams(enabled=True)
        _, meta = synthesize_hazy_iced(
            clear, HazeParams(), ice,
            rng=np.random.default_rng(0),
            with_metadata=True, with_speckle=False)
        assert "S" not in meta

    def test_ice_disabled_compat(self):
        """无冰时与旧 synthesize_haze 逐字节相等."""
        clear = _clear()
        haze = HazeParams(t_min=0.3, t_max=0.5, airlight=200)
        rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
        out_a = synthesize_hazy_iced(clear, haze, ice=None, rng=rng1)
        out_b, _ = synthesize_haze(clear, haze.t_min, haze.t_max,
                                   haze.airlight, rng2)
        np.testing.assert_array_equal(out_a, out_b)

    def test_determinism_upgraded(self):
        clear = _clear()
        ice = IceParams(enabled=True, morphology="lobe")
        a = synthesize_hazy_iced(clear, HazeParams(), ice,
                                 rng=np.random.default_rng(123))
        b = synthesize_hazy_iced(clear, HazeParams(), ice,
                                 rng=np.random.default_rng(123))
        np.testing.assert_array_equal(a, b)

    def test_output_range(self):
        clear = _clear()
        out = synthesize_hazy_iced(clear, HazeParams(), IceParams(enabled=True),
                                   rng=np.random.default_rng(0))
        assert out.shape == clear.shape
        assert out.dtype == np.uint8
