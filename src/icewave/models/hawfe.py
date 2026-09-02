"""HA-WFE 模块与 IceWave 模型 (P0-1a: monkey-patch → 标准 nn.Module 子类).

设计要点
--------
1. ``HAWFE``(v1) 与 ``HAWFEv2`` 的参数结构、初始化、forward 数学与旧实现
   (phase4/ha_wfe.py, phase5/ha_wfe_v2.py) **逐参数一致**, 保证旧 M2/M2p/M3/M4
   检查点可直接加载 (键名兼容性由 tests/test_model_compat.py 验证)。
2. ``IceWaveDehazeFormer`` 以子类方式集成 HA-WFE, 取代旧的运行时
   monkey-patch (``model.forward_features = closure``)。子类化后:
   - forward 可显式接收 ``fog_prompt`` (旧版靠 model.clip_prompt 全局副作用);
   - ONNX/TorchScript 导出不再被闭包阻断;
   - state_dict 键与旧 monkey-patch 版本完全相同 (hawfe.* 挂在顶层模块)。
3. ``build_model(version)`` 统一构造各版本模型:
   m1(基线) / m2(HA-WFE v1) / m2p(HA-WFE v2) / m3、m4、joint(+CLIP雾提示分支)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from icewave.models.dehazeformer import DehazeFormer, dehazeformer_s


# ---------------------------------------------------------------------------
# Haar 小波变换 (无参数, 可微分)
# ---------------------------------------------------------------------------
class HaarDWT2d(nn.Module):
    """单层 Haar 小波二维分解 (与旧实现逐行一致)."""

    def forward(self, x):
        x01 = x[:, :, 0::2, 1::2] / 2.0
        x10 = x[:, :, 1::2, 0::2] / 2.0
        x11 = x[:, :, 1::2, 1::2] / 2.0
        x00 = x[:, :, 0::2, 0::2] / 2.0
        LL = x00 + x01 + x10 + x11
        LH = x00 - x01 + x10 - x11
        HL = x00 + x01 - x10 - x11
        HH = x00 - x01 - x10 + x11
        return LL, LH, HL, HH


class HaarIWT2d(nn.Module):
    """单层 Haar 小波二维逆变换 (与旧实现逐行一致)."""

    def forward(self, LL, LH, HL, HH):
        a = (LL + LH + HL + HH) / 2.0
        b = (LL - LH + HL - HH) / 2.0
        c = (LL + LH - HL - HH) / 2.0
        d = (LL - LH - HL + HH) / 2.0
        B, C, H, W = LL.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d
        return out


class SCA(nn.Module):
    """Simple Channel Attention (DehazeFormer 风格)."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


# ---------------------------------------------------------------------------
# HA-WFE v1 (旧 M2 检查点兼容)
# ---------------------------------------------------------------------------
class HAWFE(nn.Module):
    """HA-WFE v1: 零初始化 + Tanh + 共享 alpha (与 phase4/ha_wfe.py 一致)."""

    def __init__(self, channels, prompt_channels=0):
        super().__init__()
        self.dwt = HaarDWT2d()
        self.iwt = HaarIWT2d()

        self.ll_sca = SCA(channels)
        if prompt_channels > 0:
            self.ll_prompt = nn.Sequential(
                nn.Conv2d(channels + prompt_channels, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
                nn.Tanh(),
            )
        self.has_prompt = prompt_channels > 0

        self.direction_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Tanh(),
        )
        self.hh_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

        self.alpha_ll = nn.Parameter(torch.zeros(1))
        self.alpha_hf = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, F_b, M_h=None):
        orig_dtype = F_b.dtype
        # 与旧实现一致: 模块内部始终以 fp32 计算 (autocast 强制关闭),
        # 保证与旧检查点训练/推理数值行为完全一致。
        with torch.amp.autocast("cuda", enabled=False):
            F_b = F_b.float()

            B, C, H, W = F_b.shape
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                F_b = F.pad(F_b, (0, pad_w, 0, pad_h), "reflect")

            LL, LH, HL, HH = self.dwt(F_b)

            LL_sca = self.ll_sca(LL)
            if self.has_prompt and M_h is not None:
                M_h_ll = F.interpolate(
                    M_h, size=LL.shape[2:], mode="bilinear", align_corners=False
                )
                gamma = self.ll_prompt(torch.cat([LL, M_h_ll], dim=1))
                LL_out = LL + self.alpha_ll * (LL_sca - LL) * (1 + gamma)
            else:
                LL_out = LL + self.alpha_ll * (LL_sca - LL)

            LH_out = LH + self.alpha_hf * (self.direction_gate(LH) * LH)
            HL_out = HL + self.alpha_hf * (self.direction_gate(HL) * HL)
            HH_out = HH + self.alpha_hf * self.hh_enhance(HH)

            F_enhanced = self.iwt(LL_out, LH_out, HL_out, HH_out)

            if pad_h or pad_w:
                # 注意: 旧实现 (phase4/ha_wfe.py) 此处只裁剪 F_enhanced 而保留
                # padding 后的 F_b, 奇数尺寸输入会因形状不匹配崩溃 (从未触发,
                # 因实际特征图恒为偶数)。此处同步裁剪 F_b —— 偶数路径行为不变。
                F_enhanced = F_enhanced[:, :, :H, :W]
                F_b = F_b[:, :, :H, :W]

            result = F_b + self.beta * F_enhanced

        return result.to(orig_dtype) if orig_dtype != torch.float32 else result


# ---------------------------------------------------------------------------
# HA-WFE v2 (M2p/M3/M4 检查点兼容)
# ---------------------------------------------------------------------------
class HAWFEv2(nn.Module):
    """HA-WFE v2: 正值初始化 + Sigmoid 门控 + 子带独立 alpha (与 phase5/ha_wfe_v2.py 一致)."""

    def __init__(self, channels, prompt_channels=0):
        super().__init__()
        self.dwt = HaarDWT2d()
        self.iwt = HaarIWT2d()

        self.ll_sca = SCA(channels)
        if prompt_channels > 0:
            self.ll_prompt = nn.Sequential(
                nn.Conv2d(channels + prompt_channels, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )
        self.has_prompt = prompt_channels > 0

        self.hf_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.hf_correct = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

        self.alpha_ll = nn.Parameter(torch.tensor(0.1))
        self.alpha_lh = nn.Parameter(torch.tensor(0.1))
        self.alpha_hl = nn.Parameter(torch.tensor(0.1))
        self.alpha_hh = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, F_b, M_h=None):
        orig_dtype = F_b.dtype
        # 与旧实现一致: 模块内部始终以 fp32 计算 (autocast 强制关闭)。
        with torch.amp.autocast("cuda", enabled=False):
            F_b = F_b.float()

            B, C, H, W = F_b.shape
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                F_b = F.pad(F_b, (0, pad_w, 0, pad_h), "reflect")

            LL, LH, HL, HH = self.dwt(F_b)

            LL_sca = self.ll_sca(LL)
            if self.has_prompt and M_h is not None:
                M_h_ll = F.interpolate(
                    M_h, size=LL.shape[2:], mode="bilinear", align_corners=False
                )
                gamma = self.ll_prompt(torch.cat([LL, M_h_ll], dim=1))
                LL_out = LL * (1 - self.alpha_ll) + LL_sca * self.alpha_ll * (1 + gamma)
            else:
                LL_out = LL * (1 - self.alpha_ll) + LL_sca * self.alpha_ll

            lh_gate = self.hf_gate(LH)
            lh_corr = self.hf_correct(LH)
            LH_out = LH + self.alpha_lh * lh_gate * lh_corr

            hl_gate = self.hf_gate(HL)
            hl_corr = self.hf_correct(HL)
            HL_out = HL + self.alpha_hl * hl_gate * hl_corr

            hh_gate = self.hf_gate(HH)
            hh_corr = self.hf_correct(HH)
            HH_out = HH + self.alpha_hh * hh_gate * hh_corr

            F_enhanced = self.iwt(LL_out, LH_out, HL_out, HH_out)

            if pad_h or pad_w:
                # 同步裁剪 F_b, 修复旧实现奇数尺寸崩溃 (偶数路径行为不变).
                F_enhanced = F_enhanced[:, :, :H, :W]
                F_b = F_b[:, :, :H, :W]

            result = F_b * (1 - self.beta) + F_enhanced * self.beta

        return result.to(orig_dtype) if orig_dtype != torch.float32 else result


# ---------------------------------------------------------------------------
# IceWave 模型: DehazeFormer + HA-WFE (标准子类, 取代 monkey-patch)
# ---------------------------------------------------------------------------
class IceWaveDehazeFormer(DehazeFormer):
    """DehazeFormer + 瓶颈层 HA-WFE (+ 可选 CLIP 雾提示分支).

    与旧 monkey-patch 集成 (integrate_hawfe_v2_with_prompt) 的区别仅在于
    工程形式: HA-WFE 插入位置、前向数据流、参数命名完全一致, 因此
    旧检查点 ``load_state_dict(strict=True)`` 直接可用。

    参数
    ----
    channels : DehazeFormer-S/B/T 瓶颈通道数, 恒为 96。
    prompt_channels : >0 时启用 LL 分支的 CLIP 雾提示调制 (M3/M4 为 32)。
    hawfe_version : 1 → HAWFE(v1, 对应旧 M2); 2 → HAWFEv2 (M2p/M3/M4)。
    **backbone_kwargs : 透传 DehazeFormer 构造参数 (embed_dims 等)。
    """

    def __init__(self, *, channels: int = 96, prompt_channels: int = 0,
                 hawfe_version: int = 2, **backbone_kwargs):
        super().__init__(**backbone_kwargs)
        if hawfe_version == 1:
            self.hawfe = HAWFE(channels, prompt_channels)
        elif hawfe_version == 2:
            self.hawfe = HAWFEv2(channels, prompt_channels)
        else:
            raise ValueError(f"未知 hawfe_version: {hawfe_version}")
        # 旧代码通过 model.clip_prompt 属性注入运行期提示张量; 保留该通道以兼容
        # 既有调用习惯, 但推荐使用 forward(x, fog_prompt=...) 显式传参。
        self.clip_prompt = None

    def forward_features(self, x, fog_prompt=None):
        prompt = fog_prompt if fog_prompt is not None else self.clip_prompt

        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.hawfe(x, prompt)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x, fog_prompt=None):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x, fog_prompt)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


# ---------------------------------------------------------------------------
# 工厂与检查点工具
# ---------------------------------------------------------------------------
_MODEL_SPECS = {
    # version: (has_hawfe, hawfe_version, prompt_channels)
    "m1": (False, None, 0),
    "m2": (True, 1, 0),
    "m2p": (True, 2, 0),
    "m3": (True, 2, 32),
    "m4": (True, 2, 32),
    "joint": (True, 2, 32),
}


def build_model(version: str = "m4", backbone: str = "s") -> nn.Module:
    """按版本名构造模型 (m1/m2/m2p/m3/m4/joint)."""
    if version not in _MODEL_SPECS:
        raise ValueError(f"未知模型版本 {version!r}, 可选: {sorted(_MODEL_SPECS)}")
    has_hawfe, hawfe_version, prompt_channels = _MODEL_SPECS[version]
    if not has_hawfe:
        if backbone != "s":
            raise ValueError("m1 基线目前仅支持 DehazeFormer-S; 其他骨干请用 m2p/m4 变体")
        return dehazeformer_s()
    return IceWaveDehazeFormer(
        hawfe_version=hawfe_version,
        prompt_channels=prompt_channels,
        **_backbone_kwargs(backbone),
    )


# DehazeFormer 家族工厂参数 (与 vendored dehazeformer.py 各工厂函数逐项一致;
# 瓶颈通道均为 96, HA-WFE 通道数不随骨干变化)。
# 注意: 不可依赖 DehazeFormer 类默认值 —— 其 depths=[16,16,16,8,8] 是 B 档
# 而非 S 档 (tests/test_model_compat.py 曾捕获此差异导致的检查点键错位)。
_BACKBONE_SPECS = {
    "t": {
        "embed_dims": [24, 48, 96, 48, 24],
        "mlp_ratios": [2.0, 4.0, 4.0, 2.0, 2.0],
        "depths": [4, 4, 4, 2, 2],
        "num_heads": [2, 4, 6, 1, 1],
        "attn_ratio": [0, 1 / 2, 1, 0, 0],
    },
    "s": {
        "embed_dims": [24, 48, 96, 48, 24],
        "mlp_ratios": [2.0, 4.0, 4.0, 2.0, 2.0],
        "depths": [8, 8, 8, 4, 4],
        "num_heads": [2, 4, 6, 1, 1],
        "attn_ratio": [1 / 4, 1 / 2, 3 / 4, 0, 0],
    },
    "b": {
        "embed_dims": [24, 48, 96, 48, 24],
        "mlp_ratios": [2.0, 4.0, 4.0, 2.0, 2.0],
        "depths": [16, 16, 16, 8, 8],
        "num_heads": [2, 4, 6, 1, 1],
        "attn_ratio": [1 / 4, 1 / 2, 3 / 4, 0, 0],
    },
}


def _backbone_kwargs(backbone: str) -> dict:
    if backbone not in _BACKBONE_SPECS:
        raise ValueError(f"未知骨干 {backbone!r}, 可选: {sorted(_BACKBONE_SPECS)}")
    kwargs = dict(_BACKBONE_SPECS[backbone])
    kwargs["conv_type"] = ["DWConv"] * 5
    return kwargs


def load_checkpoint(model: nn.Module, path, device: str = "cpu") -> nn.Module:
    """加载检查点, 兼容 {'model': sd} 与裸 state_dict 两种格式."""
    import os

    if not os.path.exists(str(path)):
        raise FileNotFoundError(
            f"检查点不存在: {path}\n"
            f"请运行 scripts/download_weights.py, 或用 ICEWAVE_WEIGHTS_DIR "
            f"环境变量指向权重目录。"
        )
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model
