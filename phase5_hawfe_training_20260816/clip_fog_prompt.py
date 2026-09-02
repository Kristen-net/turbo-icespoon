"""
CLIP Fog Prompt Extractor
从雾图中提取CLIP语言引导的空间雾提示特征, 注入HA-WFE模块

核心思想: 用CLIP的语义理解能力指导小波频域处理
- CLIPSurgery返回空间特征 [B, 50, 512] (1 CLS + 49 spatial)
- 取49个空间token → reshape [B, 512, 7, 7]
- 投影到 prompt_channels → [B, C_prompt, 7, 7]
- 插值到HA-WFE的LL空间尺寸
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HazeCLIP_DIR = r"D:\dehaze_fusion\HazeCLIP"
sys.path.insert(0, HazeCLIP_DIR)
sys.path.insert(0, os.path.join(HazeCLIP_DIR, "CLIP"))

from CLIP import clip

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPFogPrompt(nn.Module):
    """CLIP雾提示提取器

    输入: 雾图 [B, 3, H, W] in [0, 1]
    输出: 雾提示特征 M_h [B, prompt_channels, 7, 7]

    流程:
    1. Resize到224x224, CLIP归一化
    2. CLIPSurgery编码 → 空间特征 [B, 49, 512]
    3. 重塑 [B, 512, 7, 7]
    4. 投影 [B, prompt_channels, 7, 7]
    """

    def __init__(self, prompt_channels=32, device="cuda"):
        super().__init__()
        self.prompt_channels = prompt_channels
        self.device = device

        clip_model, _ = clip.load("CS-ViT-B/32", device=device, jit=False)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        self.clip = clip_model

        self.register_buffer("clip_mean", CLIP_MEAN.to(device))
        self.register_buffer("clip_std", CLIP_STD.to(device))

        self.proj = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, prompt_channels, 1),
        ).to(device)

        self._text_features = None
        self._encode_text_prompts()

    def _encode_text_prompts(self):
        prompts = [
            ["clear image", "a clear photo", "a picture of a clear scene"],
            ["hazy image", "a foggy photo", "a picture in the fog"],
            ["clear power line", "clear transmission tower", "clear insulator"],
            ["foggy power line", "foggy transmission tower", "foggy insulator"],
        ]
        text_features = []
        for prompt_group in prompts:
            tokens = clip.tokenize(prompt_group).to(self.device)
            with torch.no_grad():
                feats = self.clip.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feat = feats.mean(dim=0)
                feat = feat / feat.norm()
                text_features.append(feat)
        self._text_features = torch.stack(text_features).to(self.device)

    def get_fog_confidence(self, hazy_img):
        """计算雾置信度 (标量, 用于监控)"""
        with torch.no_grad():
            img_224 = F.interpolate(hazy_img, size=(224, 224), mode='bilinear', align_corners=False)
            img_norm = (img_224 - self.clip_mean) / self.clip_std
            img_norm = img_norm.to(self.clip.visual.conv1.weight.dtype)

            visual_feat = self.clip.encode_image(img_norm)
            visual_feat = visual_feat / visual_feat.norm(dim=-1, keepdim=True)

            cls_feat = visual_feat[:, 0, :]
            sim = (cls_feat @ self._text_features.t())
            probs = sim.softmax(dim=-1)
            fog_prob = probs[:, 1] + probs[:, 3]
            clear_prob = probs[:, 0] + probs[:, 2]
        return fog_prob.mean().item(), clear_prob.mean().item()

    def forward(self, hazy_img):
        """
        输入: hazy_img [B, 3, H, W] in [0, 1] (float32)
        输出: M_h [B, prompt_channels, 7, 7]
        """
        with torch.amp.autocast('cuda', enabled=False):
            hazy_img = hazy_img.float()
            img_224 = F.interpolate(hazy_img, size=(224, 224), mode='bilinear', align_corners=False)
            img_norm = (img_224 - self.clip_mean) / self.clip_std
            img_norm = img_norm.to(self.clip.visual.conv1.weight.dtype)

            with torch.no_grad():
                visual_feat = self.clip.encode_image(img_norm)

            spatial_feat = visual_feat[:, 1:, :]
            B, N, C = spatial_feat.shape
            side = int(N ** 0.5)
            spatial_feat = spatial_feat.transpose(1, 2).reshape(B, C, side, side)
            spatial_feat = spatial_feat.float()

            M_h = self.proj(spatial_feat)
        return M_h

    def get_text_features(self):
        return self._text_features


class HazeCLIPTeacher(nn.Module):
    """HazeCLIP教师模型 (MSBDN), 冻结, 仅推理时使用"""

    def __init__(self, device="cuda"):
        super().__init__()
        from modules.MSBDN import MSBDN

        self.model = MSBDN()
        weight_path = os.path.join(HazeCLIP_DIR, "weights", "model.pth")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.register_buffer("clip_mean", CLIP_MEAN.to(device))
        self.register_buffer("clip_std", CLIP_STD.to(device))

    @torch.no_grad()
    def forward(self, hazy_img):
        """
        输入: hazy_img [B, 3, H, W] in [0, 1]
        输出: y_teacher [B, 3, H, W] in [0, 1]
        """
        with torch.amp.autocast('cuda', enabled=False):
            img_norm = (hazy_img.float() - self.clip_mean) / self.clip_std
            H, W = hazy_img.shape[2:]

            h = H // 16 * 16
            w = W // 16 * 16
            if h != H or w != W:
                img_norm = F.interpolate(img_norm, size=(h, w), mode='bilinear', align_corners=False)

            out = self.model(img_norm)

            out = out * self.clip_std + self.clip_mean
            if h != H or w != W:
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            out = out.clamp(0, 1)
        return out


def integrate_hawfe_v2_with_prompt(model, channels=96, prompt_channels=32):
    """将HA-WFE v2 (带雾提示) 集成到DehazeFormer-S

    与原integrate_hawfe_v2的区别:
    - prompt_channels > 0, 启用ll_prompt
    - forward_features从model.clip_prompt读取雾提示
    """
    from ha_wfe_v2 import HAWFEv2

    hawfe = HAWFEv2(channels, prompt_channels=prompt_channels)
    device = next(model.parameters()).device
    hawfe = hawfe.to(device)
    model.hawfe = hawfe
    model.clip_prompt = None

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
        M_h = model.clip_prompt
        x = model.hawfe(x, M_h)
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
    print(f"Device: {device}")

    print("\n=== 测试 CLIPFogPrompt ===")
    prompt_extractor = CLIPFogPrompt(prompt_channels=32, device=device).to(device)

    dummy = torch.rand(2, 3, 192, 192, device=device)
    M_h = prompt_extractor(dummy)
    print(f"  输入: {dummy.shape} → 雾提示: {M_h.shape}")
    print(f"  M_h range: [{M_h.min().item():.4f}, {M_h.max().item():.4f}]")

    fog_p, clear_p = prompt_extractor.get_fog_confidence(dummy)
    print(f"  雾置信度: {fog_p:.4f}, 清晰置信度: {clear_p:.4f}")

    n_clip = sum(p.numel() for p in prompt_extractor.clip.parameters()) / 1e6
    n_proj = sum(p.numel() for p in prompt_extractor.proj.parameters()) / 1e3
    print(f"  CLIP参数: {n_clip:.1f}M (frozen), 投影层: {n_proj:.1f}K (trainable)")

    print("\n=== 测试 HazeCLIPTeacher ===")
    teacher = HazeCLIPTeacher(device=device).to(device)
    with torch.no_grad():
        y_teacher = teacher(dummy)
    print(f"  输入: {dummy.shape} → 教师输出: {y_teacher.shape}")
    print(f"  y_teacher range: [{y_teacher.min().item():.4f}, {y_teacher.max().item():.4f}]")
    n_teacher = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"  教师参数: {n_teacher:.1f}M (frozen)")

    print("\n=== 测试 integrate_hawfe_v2_with_prompt ===")
    sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
    from models.dehazeformer import dehazeformer_s

    student = dehazeformer_s().to(device)
    base_p = sum(p.numel() for p in student.parameters())
    student = integrate_hawfe_v2_with_prompt(student, channels=96, prompt_channels=32)
    student = student.to(device)
    new_p = sum(p.numel() for p in student.parameters())
    prompt_p = sum(p.numel() for p in student.hawfe.ll_prompt.parameters())
    print(f"  基线: {base_p/1e6:.2f}M → +HA-WFE(prompt): {new_p/1e6:.2f}M "
          f"(+{(new_p-base_p)/1e3:.1f}K, +{(new_p-base_p)/base_p*100:.1f}%)")
    print(f"  其中 ll_prompt: {prompt_p/1e3:.1f}K")

    student.clip_prompt = M_h
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float16):
            pred = student(dummy)
    print(f"  前向: {dummy.shape} → {pred.shape}")

    print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    print("\nCLIP雾提示模块验证通过!")
