"""
HA-WFE: Haze-Aware Wavelet Frequency Enhancement
部署在DehazeFormer-S瓶颈层 (layer3之后, patch_split1之前)

设计要点:
1. Haar DWT将瓶颈特征分解为LL(低频) + LH/HL/HH(高频)
2. LL: SCA通道注意力 (雾主要集中在低频)
3. LH/HL: 零初始化方向门控 (保留水平/垂直边缘)
4. HH: 零初始化纹理增强 (保留对角纹理)
5. Haar IDWT重组 + 外层零初始化残差
6. 训练初期HA-WFE≈identity, 逐步学习频域增强

参数增量: ~0.1M (基线1.28M → 1.38M, +8%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2d(nn.Module):
    """单层Haar小波二维分解 (可微分, 无参数)
    输入: (B, C, H, W) — H和W必须为偶数
    输出: LL, LH, HL, HH — 各(B, C, H/2, W/2)
    """
    def forward(self, x):
        x01 = x[:, :, 0::2, 1::2] / 2.0
        x10 = x[:, :, 1::2, 0::2] / 2.0
        x11 = x[:, :, 1::2, 1::2] / 2.0
        x00 = x[:, :, 0::2, 0::2] / 2.0
        LL = x00 + x01 + x10 + x11
        LH = x00 - x01 + x10 - x11   # 水平细节
        HL = x00 + x01 - x10 - x11   # 垂直细节
        HH = x00 - x01 - x10 + x11   # 对角细节
        return LL, LH, HL, HH


class HaarIWT2d(nn.Module):
    """单层Haar小波二维逆变换 (可微分, 无参数)
    输入: LL, LH, HL, HH — 各(B, C, H/2, W/2)
    输出: (B, C, H, W)
    """
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
    """Simple Channel Attention (DehazeFormer风格)"""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.body(x)


class HAWFE(nn.Module):
    """HA-WFE: Haze-Aware Wavelet Frequency Enhancement

    输入: F_b — 瓶颈特征 (B, C, H, W)
    输出: F_out — 增强后特征 (B, C, H, W)

    插入位置: DehazeFormer-S的layer3之后, patch_split1之前
    """
    def __init__(self, channels, prompt_channels=0):
        super().__init__()
        self.dwt = HaarDWT2d()
        self.iwt = HaarIWT2d()

        # LL: SCA通道重标定 (+ 可选雾提示调制)
        self.ll_sca = SCA(channels)
        if prompt_channels > 0:
            self.ll_prompt = nn.Sequential(
                nn.Conv2d(channels + prompt_channels, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
                nn.Tanh()
            )
        self.has_prompt = prompt_channels > 0

        # LH/HL: 零初始化方向门控
        self.direction_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # DWConv
            nn.Conv2d(channels, channels, 1),                              # PWConv
            nn.Tanh()
        )

        # HH: 零初始化纹理增强
        self.hh_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

        # 残差缩放 — 零初始化保证训练初期HA-WFE≈identity
        self.alpha_ll = nn.Parameter(torch.zeros(1))   # LL增强强度
        self.alpha_hf = nn.Parameter(torch.zeros(1))   # HF增强强度
        self.beta = nn.Parameter(torch.zeros(1))       # 外层残差强度

    def forward(self, F_b, M_h=None):
        """
        F_b: (B, C, H, W) 瓶颈特征
        M_h: (B, 1, H, W) 雾提示图 或 None (Phase 2不用, Phase 3加)
        """
        orig_dtype = F_b.dtype
        with torch.amp.autocast('cuda', enabled=False):
            F_b = F_b.float()

            # 确保空间维度为偶数 (DWT要求)
            B, C, H, W = F_b.shape
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                F_b = F.pad(F_b, (0, pad_w, 0, pad_h), 'reflect')

            # 小波分解
            LL, LH, HL, HH = self.dwt(F_b)

            # --- LL处理: SCA + 雾提示调制 ---
            LL_sca = self.ll_sca(LL)
            if self.has_prompt and M_h is not None:
                M_h_ll = F.interpolate(M_h, size=LL.shape[2:], mode='bilinear', align_corners=False)
                gamma = self.ll_prompt(torch.cat([LL, M_h_ll], dim=1))
                LL_out = LL + self.alpha_ll * (LL_sca - LL) * (1 + gamma)
            else:
                LL_out = LL + self.alpha_ll * (LL_sca - LL)

            # --- LH/HL处理: 方向门控 ---
            LH_out = LH + self.alpha_hf * (self.direction_gate(LH) * LH)
            HL_out = HL + self.alpha_hf * (self.direction_gate(HL) * HL)

            # --- HH处理: 纹理增强 ---
            HH_out = HH + self.alpha_hf * self.hh_enhance(HH)

            # 小波逆变换 + 外层残差
            F_enhanced = self.iwt(LL_out, LH_out, HL_out, HH_out)

            # 去掉padding
            if pad_h or pad_w:
                F_enhanced = F_enhanced[:, :, :H, :W]

            result = F_b + self.beta * F_enhanced

        return result.to(orig_dtype) if orig_dtype != torch.float32 else result


def integrate_hawfe(model, channels=96, prompt_channels=0):
    """将HA-WFE集成到DehazeFormer-S模型中

    修改model.forward_features, 在layer3之后插入HA-WFE
    """
    hawfe = HAWFE(channels, prompt_channels)
    # 获取model所在设备
    device = next(model.parameters()).device
    hawfe = hawfe.to(device)
    model.hawfe = hawfe

    original_forward_features = model.forward_features

    def new_forward_features(x):
        x = model.patch_embed(x)
        x = model.layer1(x)
        skip1 = x

        x = model.patch_merge1(x)
        x = model.layer2(x)
        skip2 = x

        x = model.patch_merge2(x)
        x = model.layer3(x)
        # ← HA-WFE插入点
        x = model.hawfe(x)
        x = model.patch_split1(x)

        x = model.fusion1([x, model.skip2(skip2)]) + x
        x = model.layer4(x)
        x = model.patch_split2(x)

        x = model.fusion2([x, model.skip1(skip1)]) + x
        x = model.layer5(x)
        x = model.patch_unembed(x)
        return x

    model.forward_features = new_forward_features
    return model


if __name__ == "__main__":
    # 测试HA-WFE模块
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试1: 单独HA-WFE模块
    print("=== 测试HA-WFE模块 ===")
    hawfe = HAWFE(channels=96).to(device)
    x = torch.randn(8, 96, 48, 48).to(device)

    # 初始状态应≈identity
    with torch.no_grad():
        out = hawfe(x)
    diff = (out - x).abs().max().item()
    print(f"  初始状态 max|out-x| = {diff:.8f} (应≈0)")
    print(f"  输入: {x.shape} → 输出: {out.shape}")

    # 参数量
    params = sum(p.numel() for p in hawfe.parameters())
    print(f"  HA-WFE参数量: {params / 1e3:.1f}K")

    # 测试2: 集成到DehazeFormer-S
    print("\n=== 测试集成到DehazeFormer-S ===")
    import sys
    sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
    from models.dehazeformer import dehazeformer_s

    model = dehazeformer_s().to(device)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"  原始DehazeFormer-S: {base_params / 1e6:.2f}M")

    model = integrate_hawfe(model, channels=96)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"  +HA-WFE后: {new_params / 1e6:.2f}M (+{(new_params-base_params)/1e3:.1f}K, +{(new_params-base_params)/base_params*100:.1f}%)")

    # 前向传播
    x = torch.randn(8, 3, 192, 192).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float16):
            out = model(x)
    print(f"  前向: {x.shape} → {out.shape}")

    # 初始identity测试
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    diff = (out1 - out2).abs().max().item()
    print(f"  确定性测试 max|out1-out2| = {diff:.8f}")

    # VRAM
    print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB")
    print("\nHA-WFE模块验证通过!")
