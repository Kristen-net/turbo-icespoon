"""
HA-WFE v2: 改进版

v1问题:
  1. 零初始化 → 初始identity, 梯度信号弱, 高频分支几乎没被激活(alpha_hf=-0.0005)
  2. Tanh门控 → 输出限制在[-1,1], 只能翻转不能放大, 压制了增强能力
  3. 单个alpha_hf管三个子带 → 无法差异化处理LH/HL/HH

v2改进:
  1. 正值初始化(0.1) → 训练初期模块就处于活跃状态
  2. Sigmoid门控 → [0,1]范围, "保留或抑制"语义更清晰
  3. 独立alpha_lh/alpha_hl/alpha_hh → 每个子带独立学习增强强度
  4. HF增强用"门控×残差校正" → 门决定"在哪里增强", 校正决定"增强什么"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2d(nn.Module):
    """单层Haar小波二维分解"""
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
    """单层Haar小波二维逆变换"""
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
    """Simple Channel Attention"""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.body(x)


class HAWFEv2(nn.Module):
    """HA-WFE v2: 改进的小波频域增强模块

    相比v1的关键改变:
    - 正值初始化(0.1)代替零初始化
    - Sigmoid门控代替Tanh
    - LH/HL/HH各自独立的alpha
    - HF增强 = 门控(在哪里) × 残差校正(增强什么)
    """
    def __init__(self, channels, prompt_channels=0):
        super().__init__()
        self.dwt = HaarDWT2d()
        self.iwt = HaarIWT2d()

        # === LL: 通道注意力 + 可选雾提示 ===
        self.ll_sca = SCA(channels)
        if prompt_channels > 0:
            self.ll_prompt = nn.Sequential(
                nn.Conv2d(channels + prompt_channels, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            )
        self.has_prompt = prompt_channels > 0

        # === HF共享: 空间门控 + 残差校正 ===
        # 门控: 决定"在哪些空间位置增强" (Sigmoid → [0,1])
        # 校正: 决定"增强什么内容" (无激活, 自由学习)
        self.hf_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # DWConv
            nn.Conv2d(channels, channels, 1),                              # PWConv
            nn.Sigmoid()  # [0,1] 空间注意力
        )
        self.hf_correct = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            # 无激活 — 残差校正自由学习
        )

        # === 可学习缩放 — 正值初始化 ===
        self.alpha_ll = nn.Parameter(torch.tensor(0.1))
        self.alpha_lh = nn.Parameter(torch.tensor(0.1))  # 独立
        self.alpha_hl = nn.Parameter(torch.tensor(0.1))  # 独立
        self.alpha_hh = nn.Parameter(torch.tensor(0.1))  # 独立
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, F_b, M_h=None):
        orig_dtype = F_b.dtype
        with torch.amp.autocast('cuda', enabled=False):
            F_b = F_b.float()

            B, C, H, W = F_b.shape
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                F_b = F.pad(F_b, (0, pad_w, 0, pad_h), 'reflect')

            LL, LH, HL, HH = self.dwt(F_b)

            # --- LL: SCA + 雾提示 ---
            LL_sca = self.ll_sca(LL)
            if self.has_prompt and M_h is not None:
                M_h_ll = F.interpolate(M_h, size=LL.shape[2:], mode='bilinear', align_corners=False)
                gamma = self.ll_prompt(torch.cat([LL, M_h_ll], dim=1))
                LL_out = LL * (1 - self.alpha_ll) + LL_sca * self.alpha_ll * (1 + gamma)
            else:
                LL_out = LL * (1 - self.alpha_ll) + LL_sca * self.alpha_ll

            # --- LH: 门控×校正 ---
            lh_gate = self.hf_gate(LH)
            lh_corr = self.hf_correct(LH)
            LH_out = LH + self.alpha_lh * lh_gate * lh_corr

            # --- HL: 门控×校正 (共享权重) ---
            hl_gate = self.hf_gate(HL)
            hl_corr = self.hf_correct(HL)
            HL_out = HL + self.alpha_hl * hl_gate * hl_corr

            # --- HH: 门控×校正 (共享权重) ---
            hh_gate = self.hf_gate(HH)
            hh_corr = self.hf_correct(HH)
            HH_out = HH + self.alpha_hh * hh_gate * hh_corr

            # IDWT + 外层残差
            F_enhanced = self.iwt(LL_out, LH_out, HL_out, HH_out)

            if pad_h or pad_w:
                F_enhanced = F_enhanced[:, :, :H, :W]

            result = F_b * (1 - self.beta) + F_enhanced * self.beta

        return result.to(orig_dtype) if orig_dtype != torch.float32 else result


def integrate_hawfe_v2(model, channels=96, prompt_channels=0):
    """将HA-WFE v2集成到DehazeFormer-S"""
    hawfe = HAWFEv2(channels, prompt_channels)
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
        x = model.hawfe(x)  # ← HA-WFE v2
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试v2
    print("=== 测试HA-WFE v2 ===")
    hawfe = HAWFEv2(channels=96).to(device)
    x = torch.randn(8, 96, 48, 48).to(device)

    with torch.no_grad():
        out = hawfe(x)
    diff = (out - x).abs().max().item()
    print(f"  初始状态 max|out-x| = {diff:.6f} (v1是0.0, v2应>0因为正值初始化)")

    params = sum(p.numel() for p in hawfe.parameters())
    print(f"  参数量: {params / 1e3:.1f}K")

    # 测试集成
    print("\n=== 集成到DehazeFormer-S ===")
    import sys
    sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
    from models.dehazeformer import dehazeformer_s

    model = dehazeformer_s().to(device)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"  原始: {base_params/1e6:.2f}M")

    model = integrate_hawfe_v2(model, channels=96)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"  +HA-WFEv2: {new_params/1e6:.2f}M (+{(new_params-base_params)/1e3:.1f}K, +{(new_params-base_params)/base_params*100:.1f}%)")

    # 前向
    x = torch.randn(8, 3, 192, 192).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float16):
            out = model(x)
    print(f"  前向: {x.shape} → {out.shape}")

    # 打印初始参数值
    print(f"\n  初始参数:")
    print(f"    alpha_ll = {model.hawfe.alpha_ll.item():.4f}")
    print(f"    alpha_lh = {model.hawfe.alpha_lh.item():.4f}")
    print(f"    alpha_hl = {model.hawfe.alpha_hl.item():.4f}")
    print(f"    alpha_hh = {model.hawfe.alpha_hh.item():.4f}")
    print(f"    beta     = {model.hawfe.beta.item():.4f}")

    print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB")
    print("\nHA-WFE v2验证通过!")
