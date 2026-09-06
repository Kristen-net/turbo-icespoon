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

    升级字段 (JOINT_OPTIMIZATION_FRAMEWORK §2.3):
    - specular_strength: 镜面反射比例 (R_spec * I_env)
    - translucent_strength: 半透明散射比例 (R_trans * J_trans)
    - blur_sigma_x / blur_sigma_y: 各向异性模糊 σ (沿导线方向拉长)
    - morphology: 冰晶形态, 影响 S 的采样方式
    """

    enabled: bool = False
    extinction: float = 2.5       # Beer-Lambert 消光系数
    coverage: float = 0.5         # 冰覆盖比例 (0~1)
    max_thickness: float = 1.0    # 归一化厚度上限
    texture_strength: float = 0.15  # 冰面纹理对像素的扰动幅度
    # —— 升级字段 (§2.3) ——
    specular_strength: float = 0.4    # 镜面反射比例
    translucent_strength: float = 0.3  # 半透明散射比例
    blur_sigma_x: float = 1.0        # 各向异性模糊 σ_x
    blur_sigma_y: float = 2.5        # σ_y (沿导线方向拉长)
    morphology: str = "shell"       # shell / sleeve / lobe / cluster


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


def make_specular_reflection(h: int, w: int, rng: np.random.Generator,
                             morphology: str = "shell") -> np.ndarray:
    """生成冰面镜面反射图 S (h, w, 3) ∈ [0, 1].

    物理分解 (§2.2): S = R_spec * I_env + R_trans * J_trans
    - R_spec: 强方向性镜面反射, 受冰晶形态影响
    - morphology="shell": 薄壳状, 高频斑点
    - morphology="sleeve": 管状包裹, 沿水平方向拉长
    - morphology="lobe": 瘤状, 低频团块
    - morphology="cluster": 聚簇, 不规则斑块
    """
    if morphology == "lobe":
        base = rng.normal(0.3, 0.1, (h, w)).astype(np.float32)
    elif morphology == "sleeve":
        base = rng.normal(0.4, 0.08, (h, w)).astype(np.float32)
    elif morphology == "cluster":
        base = rng.normal(0.35, 0.15, (h, w)).astype(np.float32)
    else:  # shell (default)
        base = rng.normal(0.5, 0.12, (h, w)).astype(np.float32)

    base = np.clip(base, 0.0, 1.0)
    # 三通道色温微扰 (冰面偏冷色调: B > G > R)
    r = base * rng.uniform(0.85, 0.95)
    g = base * rng.uniform(0.90, 1.00)
    b = base * rng.uniform(0.95, 1.05)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def make_anisotropic_blur_kernel(sigma_x: float, sigma_y: float,
                                  ksize: int = 5) -> np.ndarray:
    """生成各向异性高斯模糊核 (ksize, ksize).

    σ_y > σ_x → 沿水平方向 (导线方向) 扩散更强, 模拟冰层在轮廓处的扩散性模糊。
    """
    pad = ksize // 2
    ax = np.arange(-pad, pad + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    kernel = np.exp(-(xx ** 2) / (2 * sigma_x ** 2)
                    - (yy ** 2) / (2 * sigma_y ** 2))
    kernel = kernel / kernel.sum()
    return kernel


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
                  thickness: Optional[np.ndarray] = None,
                  with_speckle: bool = True
                  ) -> tuple:
    """对清晰图施加冰层效果, 返回 ``(out, meta)``.

    物理模型 (§2.2):
        J_ice = (1 - alpha) * J + alpha * S
    其中 S 包含:
        - 冰面纹理扰动 (既有)
        - 镜面反射 R_spec * I_env (升级, with_speckle=True)
        - 半透明散射 R_trans * J_trans (升级, with_speckle=True)

    ``meta`` 含 ``alpha`` (h, w), ``thickness`` (h, w),
    以及升级字段 ``S`` (h, w, 3), ``morphology`` (str).
    ``test_zero_thickness_is_identity`` 验证 thickness=0 → alpha=0 → out=clear.
    """
    h, w = clear.shape[:2]
    if thickness is None:
        thickness = make_ice_thickness(h, w, rng, ice.coverage, ice.max_thickness)
    alpha = (1.0 - np.exp(-ice.extinction * thickness)).astype(np.float32)
    clear_f = clear.astype(np.float32)

    if ice.texture_strength > 0:
        texture = make_ice_texture(h, w, rng)
        ice_color = np.clip(clear_f + 255.0 * ice.texture_strength * texture[:, :, None],
                            0, 255)
    else:
        ice_color = clear_f.copy()

    S = None
    if with_speckle:
        # 镜面反射 S = R_spec * I_env + R_trans * J_trans
        spec_map = make_specular_reflection(h, w, rng, ice.morphology)
        env_light = np.array([200, 210, 220], dtype=np.float32)  # 环境光偏冷
        S = (ice.specular_strength * spec_map * env_light[None, None, :]
             + ice.translucent_strength * clear_f)  # 半透明散射取清晰图
        S = np.clip(S, 0, 255)
        ice_color = np.clip(ice_color * (1 - 0.3) + S * 0.3, 0, 255)

    out = (1.0 - alpha[:, :, None]) * clear_f + alpha[:, :, None] * ice_color
    out = np.clip(out, 0, 255).astype(np.uint8)

    meta = {"alpha": alpha, "thickness": thickness, "morphology": ice.morphology}
    if S is not None:
        meta["S"] = S
    return out, meta


# ---------------------------------------------------------------------------
# 组合合成 (雾 + 冰), 是训练/评测的主要入口
# ---------------------------------------------------------------------------
def synthesize_hazy_iced(clear: np.ndarray, haze: HazeParams,
                         ice: Optional[IceParams] = None,
                         rng: Optional[np.random.Generator] = None,
                         with_metadata: bool = False,
                         apply_blur: bool = True,
                         with_speckle: bool = True):
    """组合合成: 冰层 (若启用) → 边缘模糊 (若启用) → 大气散射.

    无冰时与 ``synthesize_haze`` **逐字节相等** (供旧管线兼容). 元数据键:
    ``t_map`` (h, w), ``A`` (长度 3 RGB), ``ice_alpha`` (h, w),
    ``ice_thickness`` (h, w), 以及升级字段 ``S`` (h, w, 3),
    ``blur_kernel_id`` (str), ``morphology`` (str).
    仅雾时返回 ``t_map`` + ``A`` 无 ice_* 键.

    升级 (§2.3):
    - ``apply_blur=True``: 冰层边缘各向异性高斯模糊 (K)
    - ``with_speckle=True``: 镜面反射 + 半透明散射 (S)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Step 1: 冰层复合 (可选, 无冰即恒等)
    if ice is None or not ice.enabled:
        pre_blur = clear
        ice_meta = {}
    else:
        pre_blur, full_meta = ice_composite(clear, ice, rng,
                                            with_speckle=with_speckle)
        ice_meta = {"ice_alpha": full_meta["alpha"],
                    "ice_thickness": full_meta["thickness"]}
        if "S" in full_meta:
            ice_meta["S"] = full_meta["S"]
        if "morphology" in full_meta:
            ice_meta["morphology"] = full_meta["morphology"]

    # Step 2: 各向异性边缘模糊 (冰诱导 + 离焦)
    blur_kernel_id = "none"
    if apply_blur and ice is not None and ice.enabled:
        kernel = make_anisotropic_blur_kernel(
            ice.blur_sigma_x, ice.blur_sigma_y, ksize=5)
        alpha_3 = ice_meta["ice_alpha"][:, :, None]
        blurred = _apply_kernel(pre_blur.astype(np.float32), kernel)
        pre_haze = ((1 - alpha_3) * pre_blur.astype(np.float32)
                    + alpha_3 * blurred).clip(0, 255).astype(np.uint8)
        blur_kernel_id = f"sigma({ice.blur_sigma_x},{ice.blur_sigma_y})"
    else:
        pre_haze = pre_blur

    # Step 3: 大气散射
    hazy, [t_map] = synthesize_haze(pre_haze, haze.t_min, haze.t_max,
                                    haze.airlight, rng)

    if not with_metadata:
        return hazy

    A = np.array([
        int(haze.airlight * rng.uniform(0.95, 1.05)),
        int(haze.airlight * rng.uniform(0.95, 1.05)),
        int(haze.airlight * rng.uniform(0.95, 1.05)),
    ], dtype=np.float32).tolist()
    meta = {"t_map": t_map, "A": A}
    meta.update(ice_meta)
    meta["blur_kernel_id"] = blur_kernel_id
    return hazy, meta


def _apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对 (h, w, 3) 图像施加 2D 卷积 (反射边界). 纯 NumPy, 无 scipy 依赖."""
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(k):
        for j in range(k):
            out += padded[i:i + img.shape[0], j:j + img.shape[1]] * kernel[i, j]
    return out