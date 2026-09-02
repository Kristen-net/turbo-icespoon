"""CLIP 雾提示提取器与 HazeCLIP 教师模型 (路径全部参数化, P0-1).

与旧实现 (phase5/clip_fog_prompt.py) 的差异:
- HazeCLIP 目录 / 权重路径不再硬编码 ``D:\\dehaze_fusion\\...``, 改为
  参数 > 环境变量 > 仓库默认 (utils.paths) 三级解析;
- CLIP 模块按"完整 HazeCLIP 安装"优先导入。仓库内 source_hazeclip/ 是
  不完整的扁平副本 (缺 build_model.py / simple_tokenizer.py), 仅当其
  可导入时才使用, 否则给出明确的修复指引;
- 教师模型 MSBDN 的导入同时兼容 ``modules.MSBDN``(原仓库布局) 与
  ``MSBDN``(扁平副本布局) 两种布局。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from icewave.utils.paths import CLIP_DIR, HAZECLIP_WEIGHTS

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def import_vendored_clip(clip_dir: Optional[str] = None):
    """导入 CLIP 模块 (HazeCLIP 修改版, CS-ViT-B/32).

    返回 clip 模块对象; 导入失败时抛出带修复指引的 ImportError。
    """
    root = Path(clip_dir) if clip_dir else CLIP_DIR
    candidates = []

    # 布局 1: <root>/CLIP/clip.py (HazeCLIP 原始仓库布局)
    pkg_parent = root
    if (root / "CLIP" / "clip.py").exists():
        candidates.append((pkg_parent, "CLIP.clip"))
    # 布局 2: <root>/clip.py 且附带 build_model.py (完整扁平副本)
    if (root / "clip.py").exists() and (root / "build_model.py").exists():
        candidates.append((root, "clip"))

    for parent, modname in candidates:
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
        try:
            return importlib.import_module(modname)
        except ImportError:
            continue

    raise ImportError(
        f"无法从 {root} 导入 HazeCLIP 的 CLIP 模块。\n"
        f"仓库内 source_hazeclip/ 是不完整的扁平副本 (缺少 build_model.py / "
        f"simple_tokenizer.py)。\n修复方式 (任选其一):\n"
        f"  1. git clone https://github.com/cuaecc/HazeCLIP 并设置环境变量 "
        f"ICEWAVE_CLIP_DIR=<HazeCLIP 目录>;\n"
        f"  2. 将完整的 CLIP/ 子目录 (含 build_model.py, simple_tokenizer.py) "
        f"复制到 source_hazeclip/ 下。"
    )


class CLIPFogPrompt(nn.Module):
    """CLIP 雾提示提取器 (与旧实现 forward 数学一致).

    输入: 雾图 [B, 3, H, W] in [0, 1]; 输出: M_h [B, prompt_channels, 7, 7]。
    """

    def __init__(self, prompt_channels: int = 32, device: str = "cuda",
                 clip_dir: Optional[str] = None):
        super().__init__()
        self.prompt_channels = prompt_channels
        clip = import_vendored_clip(clip_dir)
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

        self._clip_module = clip
        self._text_features = None
        self._encode_text_prompts()

    def _encode_text_prompts(self):
        clip = self._clip_module
        prompts = [
            ["clear image", "a clear photo", "a picture of a clear scene"],
            ["hazy image", "a foggy photo", "a picture in the fog"],
            ["clear power line", "clear transmission tower", "clear insulator"],
            ["foggy power line", "foggy transmission tower", "foggy insulator"],
        ]
        text_features = []
        device = self.clip_mean.device
        for prompt_group in prompts:
            tokens = clip.tokenize(prompt_group).to(device)
            with torch.no_grad():
                feats = self.clip.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feat = feats.mean(dim=0)
                feat = feat / feat.norm()
                text_features.append(feat)
        self._text_features = torch.stack(text_features).to(device)

    def get_fog_confidence(self, hazy_img):
        """雾置信度 (标量, 监控用)."""
        with torch.no_grad():
            img_224 = F.interpolate(
                hazy_img, size=(224, 224), mode="bilinear", align_corners=False
            )
            img_norm = (img_224 - self.clip_mean) / self.clip_std
            img_norm = img_norm.to(self.clip.visual.conv1.weight.dtype)

            visual_feat = self.clip.encode_image(img_norm)
            visual_feat = visual_feat / visual_feat.norm(dim=-1, keepdim=True)

            cls_feat = visual_feat[:, 0, :]
            sim = cls_feat @ self._text_features.t()
            probs = sim.softmax(dim=-1)
            fog_prob = probs[:, 1] + probs[:, 3]
            clear_prob = probs[:, 0] + probs[:, 2]
        return fog_prob.mean().item(), clear_prob.mean().item()

    def forward(self, hazy_img):
        with torch.amp.autocast("cuda", enabled=False):
            hazy_img = hazy_img.float()
            img_224 = F.interpolate(
                hazy_img, size=(224, 224), mode="bilinear", align_corners=False
            )
            img_norm = (img_224 - self.clip_mean) / self.clip_std
            img_norm = img_norm.to(self.clip.visual.conv1.weight.dtype)

            with torch.no_grad():
                visual_feat = self.clip.encode_image(img_norm)

            spatial_feat = visual_feat[:, 1:, :]
            B, N, C = spatial_feat.shape
            side = int(N**0.5)
            spatial_feat = spatial_feat.transpose(1, 2).reshape(B, C, side, side)
            spatial_feat = spatial_feat.float()

            M_h = self.proj(spatial_feat)
        return M_h

    def get_text_features(self):
        return self._text_features


def import_msbdn(hazeclip_dir: Optional[str] = None):
    """导入 MSBDN 网络 (兼容 modules.MSBDN 与扁平 MSBDN 两种布局)."""
    root = Path(hazeclip_dir) if hazeclip_dir else CLIP_DIR
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    for modname, attr in (("modules.MSBDN", "MSBDN"), ("MSBDN", "MSBDN")):
        try:
            mod = importlib.import_module(modname)
            return getattr(mod, attr)
        except ImportError:
            continue
    raise ImportError(
        f"无法导入 MSBDN (尝试目录: {root})。请提供完整 HazeCLIP 仓库 "
        f"(含 modules/MSBDN.py) 并设置 ICEWAVE_CLIP_DIR。"
    )


class HazeCLIPTeacher(nn.Module):
    """HazeCLIP 教师模型 (MSBDN, 冻结, 仅训练期使用)."""

    def __init__(self, device: str = "cuda", weights_path: Optional[str] = None,
                 hazeclip_dir: Optional[str] = None):
        super().__init__()
        MSBDN = import_msbdn(hazeclip_dir)
        self.model = MSBDN()

        weight_path = Path(weights_path) if weights_path else HAZECLIP_WEIGHTS
        if not weight_path.exists():
            raise FileNotFoundError(
                f"HazeCLIP 教师权重不存在: {weight_path}\n"
                f"请运行 scripts/download_weights.py 获取, 或通过 "
                f"ICEWAVE_HAZECLIP_WEIGHTS 环境变量指定位置。"
            )
        state_dict = torch.load(str(weight_path), map_location="cpu",
                                weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.register_buffer("clip_mean", CLIP_MEAN.to(device))
        self.register_buffer("clip_std", CLIP_STD.to(device))

    @torch.no_grad()
    def forward(self, hazy_img):
        with torch.amp.autocast("cuda", enabled=False):
            img_norm = (hazy_img.float() - self.clip_mean) / self.clip_std
            H, W = hazy_img.shape[2:]

            h = H // 16 * 16
            w = W // 16 * 16
            if h != H or w != W:
                img_norm = F.interpolate(
                    img_norm, size=(h, w), mode="bilinear", align_corners=False
                )

            out = self.model(img_norm)

            out = out * self.clip_std + self.clip_mean
            if h != H or w != W:
                out = F.interpolate(
                    out, size=(H, W), mode="bilinear", align_corners=False
                )
            out = out.clamp(0, 1)
        return out
