"""复合退化物理模型 (P1-1).

雾天输电线路场景中, 雾和覆冰的成像过程可以分解为两步物理模型:

    1. **冰层复合** (Beer-Lambert): 清晰图 J_ice 与冰层 alpha 线性混合,
       J_ice 与冰厚 d 通过 alpha = 1 - exp(-beta * d) 关联;
    2. **大气散射** (Koschmieder): 雾图 I 与"冰后清晰图"J_ice 通过透射率 t 与
       大气光 A 关联: I = J_ice * t + A * (1 - t).

合成器 ``synthesize_hazy_iced`` 是这两步的组合. 当 ``ice=None`` 或
``IceParams(enabled=False)`` 时, 第一步是恒等映射, 输出与经典 ASM
(``synthesize_haze``) **逐字节相等** —— 该兼容性由 ``test_degradation.py``
的 ``test_ice_disabled_equals_pure_haze`` 保证 (用于旧数据集复现).

关键设计
--------
- **确定性**: 全部随机源由调用方传入 (``np.random.Generator``), 同 seed 同输出.
- **元数据可选**: ``with_metadata=True`` 时返回 ``{t_map, A, ice_alpha,
  ice_thickness}``, 供下游监督 (透射率估计 / 物理一致性损失) 使用.
- **三档预设**: ``HAZE_PRESETS`` (thin / medium / dense), 透射率单调下降,
  大气光单调上升, ``test_presets_three_levels_ordered`` 验证.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 参数数据类
# ---------------------------------------------------------------------------
@dataclass
class HazeParams:
    """大气散射参数 (Koschmieder 模型)."""

    t_min: float = 0.4    # 透射率下限 (越低 → 雾越浓)
    t_max: float = 0.7    # 透射率上限
    airlight: int = 220   # 大气光亮度 (RGB)


@dataclass
class IceParams:
    """覆冰复合参数.

    物理: ``alpha = 1 - exp(-extinction * thickness)``, alpha 越大冰层越不透明.
    """

    enabled: bool = False
    extinction: float = 2.5       # Beer-Lambert 消光系数
    coverage: float = 0.5         # 冰覆盖比例 (0~1)
    max_thickness: float = 1.0    # 归一化厚度上限
    texture_strength: float = 0.15  # 冰面纹理对像素的扰动幅度


HAZE_PRESETS = {
    "thin":   HazeParams(t_min=0.55, t_max=0.85, airlight=200),
    "medium": HazeParams(t_min=0.30, t_max=0.60, airlight=220),
    "dense":  HazeParams(t_min=0.10, t_max=0.30, airlight=235),
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def make_transmission_map(h: int, w: int, rng: np.random.Generator,
                          t_min: float, t_max: float) -> np.ndarray:
    """生成空间不均匀的透射率图.

    由低频径向衰减 + 噪声扰动构成, 取值裁剪到 ``[0.05, 0.95]``. 范围与
    上界约束由 ``test_transmission_map_range`` 验证.
    """
    yy, xx = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32),
                         np.linspace(-1, 1, w, dtype=np.float32), indexing="ij")
    radial = 1.0 - np.clip(np.sqrt(xx ** 2 + yy ** 2), 0, 1) * 0.6  # 中心更透
    noise = rng.normal(0, 0.05, (h, w)).astype(np.float32)
    t = np.clip(radial + noise + 0.5, 0.05, 0.95)
    t = t_min + (t_max - t_min) * t / t.max()
    return np.clip(t, 0.05, 0.95).astype(np.float32)


def make_ice_thickness(h: int, w: int, rng: np.random.Generator,
                       coverage: float, max_thickness: float) -> np.ndarray:
    """生成厚度图 d (h, w), ``(d > 0).mean() ≈ coverage``.

    分位阈值法: 先用低频噪声生成连续场, 取使覆盖比例近似 ``coverage`` 的分位数
    阈值置零. ``test_coverage_approx`` 验证实际覆盖偏差在 ``[coverage*0.5,
    coverage*1.5]`` 之间.
    """
    yy, xx = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32),
                         np.linspace(-1, 1, w, dtype=np.float32), indexing="ij")
    lowfreq = np.exp(-(xx ** 2 + yy ** 2) * 1.5)
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    field_ = lowfreq * 1.2 + noise * 0.5
    # 分位阈值
    threshold = np.quantile(field_, max(0.0, 1.0 - coverage))
    thickness = np.clip(field_ - threshold, 0, None)
    if thickness.max() > 0:
        thickness = thickness / thickness.max() * max_thickness
    return thickness.astype(np.float32)


def make_ice_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """生成冰面高频纹理 (h, w) ∈ [0, 1]."""
    base = rng.normal(0.5, 0.15, (h, w)).astype(np.float32)
    base = np.clip(base, 0.0, 1.0)
    # 局部平滑 (5x5 均值)
    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    pad = 2
    padded = np.pad(base, pad, mode="reflect")
    smoothed = np.zeros_like(base)
    for i in range(5):
        for j in range(5):
            smoothed += padded[i:i + h, j:j + w] * kernel[i, j]
    return smoothed


# ---------------------------------------------------------------------------
# 经典 ASM (无冰), 与旧 build_dataset_v3.py 数值行为一致
# ---------------------------------------------------------------------------
def synthesize_haze(clear: np.ndarray, t_min: float, t_max: float,
                    airlight: int, rng: np.random.Generator) -> tuple:
    """经典大气散射合成.

    返回 ``(hazy_img, [t_map])``. ``t_map`` 仅在调用方需要时取走 (训练/可视化),
    不影响图像数值行为.
    """
    h, w = clear.shape[:2]
    t_map = make_transmission_map(h, w, rng, t_min, t_max)
    # 大气光: 标量或随机色温扰动 (RGB 通道)
    A = np.array([
        int(airlight * rng.uniform(0.95, 1.05)),
        int(airlight * rng.uniform(0.95, 1.05)),
        int(airlight * rng.uniform(0.95, 1.05)),
    ], dtype=np.float32)
    t3 = t_map[:, :, None]
    clear_f = clear.astype(np.float32)
    hazy = clear_f * t3 + A[None, None, :] * (1.0 - t3)
    hazy = np.clip(hazy, 0, 255).astype(np.uint8)
    return hazy, [t_map]


# ---------------------------------------------------------------------------
# 冰层复合 (Beer-Lambert): J_ice = (1 - alpha) * J_clear + alpha * (J_clear + texture)
# ---------------------------------------------------------------------------
def ice_composite(clear: np.ndarray, ice: IceParams, rng: np.random.Generator,
                  thickness: Optional[np.ndarray] = None
                  ) -> tuple:
    """对清晰图施加冰层效果, 返回 ``(out, meta)``.

    ``meta`` 至少含 ``alpha`` (h, w) 掩码, 反映冰层局部不透明度. ``test_zero_thickness_is_identity``
    验证 ``thickness 全 0 → alpha 全 0 → out == clear``.
    """
    h, w = clear.shape[:2]
    if thickness is None:
        thickness = make_ice_thickness(h, w, rng, ice.coverage, ice.max_thickness)
    alpha = (1.0 - np.exp(-ice.extinction * thickness)).astype(np.float32)
    if ice.texture_strength > 0:
        texture = make_ice_texture(h, w, rng)
        clear_f = clear.astype(np.float32)
        ice_color = np.clip(clear_f + 255.0 * ice.texture_strength * texture[:, :, None],
                            0, 255)
        out = (1.0 - alpha[:, :, None]) * clear_f + alpha[:, :, None] * ice_color
    else:
        out = clear.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, {"alpha": alpha, "thickness": thickness}


# ---------------------------------------------------------------------------
# 组合合成 (雾 + 冰), 是训练/评测的主要入口
# ---------------------------------------------------------------------------
def synthesize_hazy_iced(clear: np.ndarray, haze: HazeParams,
                         ice: Optional[IceParams] = None,
                         rng: Optional[np.random.Generator] = None,
                         with_metadata: bool = False):
    """组合合成: 冰层 (若启用) → 大气散射.

    无冰时与 ``synthesize_haze`` **逐字节相等** (供旧管线兼容). 元数据键:
    ``t_map`` (h, w), ``A`` (长度 3 RGB), ``ice_alpha`` (h, w),
    ``ice_thickness`` (h, w). 仅雾时返回 ``t_map`` + ``A`` 无 ice_* 键.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Step 1: 冰层复合 (可选, 无冰即恒等)
    if ice is None or not ice.enabled:
        pre_haze = clear
        ice_meta = {}
    else:
        pre_haze, ice_meta = ice_composite(clear, ice, rng)
        # ice_composite 返回 {"alpha", "thickness"}; 暴露给外部时加 ice_ 前缀
        # 以与 HAZE 域键 (t_map / A) 明确区分, 避免下游误读。
        ice_meta = {"ice_alpha": ice_meta["alpha"],
                    "ice_thickness": ice_meta["thickness"]}

    # Step 2: 大气散射
    hazy, [t_map] = synthesize_haze(pre_haze, haze.t_min, haze.t_max,
                                    haze.airlight, rng)
    # synthesize_haze 返回的 t_map 是 [t_map] 列表以兼容旧多输出

    if not with_metadata:
        return hazy

    # 大气光 (RGB) —— 与 synthesize_haze 中相同规则, 但此处用 rng 一致序列
    A = np.array([
        int(haze.airlight * rng.uniform(0.95, 1.05)),
        int(haze.airlight * rng.uniform(0.95, 1.05)),
        int(haze.airlight * rng.uniform(0.95, 1.05)),
    ], dtype=np.float32).tolist()
    meta = {"t_map": t_map, "A": A}
    meta.update(ice_meta)
    return hazy, meta