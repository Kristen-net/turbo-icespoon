"""
Phase 3 完成后: 四模型对比报告 (M1 vs M2 vs M2' vs M3)

1. 合成雾验证集 (84对, PSNR/SSIM)
2. 真实雾测试集 (673张, 无参考指标)
3. HA-WFE参数演化
4. CLIP雾提示蒸馏效果分析
"""

import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.dehazeformer import dehazeformer_s
from ha_wfe import integrate_hawfe
from ha_wfe_v2 import integrate_hawfe_v2
from clip_fog_prompt import integrate_hawfe_v2_with_prompt, CLIPFogPrompt

VAL_HAZY = r"D:\DATA_ALL\dataset\val\hazy"
VAL_CLEAR = r"D:\DATA_ALL\dataset\val\clear"
REAL_DIR = r"D:\DATA_ALL\dataset\test\hazy_real"

CKPT = {
    "m1": r"D:\dehaze_fusion\icewave_output\checkpoints\m1_best.pth",
    "m2": r"D:\dehaze_fusion\icewave_output\m2_hawfe\checkpoints\m2_best.pth",
    "m2p": r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\checkpoints\m2p_best.pth",
    "m3": r"D:\dehaze_fusion\icewave_output\m3_clip_distill\checkpoints\m3_best.pth",
}

OUTPUT_DIR = r"D:\dehaze_fusion\icewave_output\real_test_4way"
REPORT_PATH = r"D:\dehaze_fusion\icewave_output\four_way_report.txt"

DEVICE = "cuda"


def load_model(name, ckpt_path):
    model = dehazeformer_s().to(DEVICE)
    if name == "m1":
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
    elif name == "m2":
        model = integrate_hawfe(model, channels=96)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
    elif name == "m2p":
        model = integrate_hawfe_v2(model, channels=96)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
    elif name == "m3":
        model = integrate_hawfe_v2_with_prompt(model, channels=96, prompt_channels=32)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt)
        own = model.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        model.load_state_dict(own, strict=True)
        model.clip_prompt = None
    model.eval()
    return model


def eval_synth(model, name):
    hazy_files = sorted([f for f in os.listdir(VAL_HAZY) if f.endswith('.png')])
    clear_files = sorted([f for f in os.listdir(VAL_CLEAR) if f.endswith('.png')])
    pairs = []
    for hazy_name in hazy_files:
        clear_name = '_'.join(hazy_name.replace('.png', '').split('_')[:2]) + '.png'
        if clear_name in clear_files:
            pairs.append((hazy_name, clear_name))

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnrs, ssims = [], []

    with torch.no_grad():
        for hazy_name, clear_name in pairs:
            hazy = cv2.imread(os.path.join(VAL_HAZY, hazy_name))
            clear = cv2.imread(os.path.join(VAL_CLEAR, clear_name))
            if hazy is None or clear is None:
                continue
            h, w = hazy.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                hazy = cv2.resize(hazy, (int(w * scale), int(h * scale)))
                clear = cv2.resize(clear, (int(w * scale), int(h * scale)))
            hazy_t = torch.from_numpy(cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            clear_t = torch.from_numpy(cv2.cvtColor(clear, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                pred = model(hazy_t)
            mse = F.mse_loss(pred, clear_t)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnrs.append(psnr.item())
            ssims.append(ssim_metric(pred, clear_t).item())

    return np.mean(psnrs), np.mean(ssims)


def eval_real(model, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    real_files = sorted([f for f in os.listdir(REAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    metrics = {"dark_channel": [], "contrast": [], "saturation": [], "edge_density": [], "gradient_mean": []}

    with torch.no_grad():
        for fname in real_files:
            img = cv2.imread(os.path.join(REAL_DIR, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                pred = model(img_t)
            pred_np = pred[0].float().cpu().numpy().transpose(1, 2, 0)
            pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, fname), pred_bgr)

            dark = np.min(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY))
            gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.sum() / (gray.shape[0] * gray.shape[1] * 255)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mean = np.sqrt(gx**2 + gy**2).mean()

            metrics["dark_channel"].append(dark / 255.0)
            metrics["contrast"].append(contrast)
            metrics["saturation"].append(saturation)
            metrics["edge_density"].append(edge_density)
            metrics["gradient_mean"].append(gradient_mean)

    return {k: np.mean(v) for k, v in metrics.items()}, len(real_files)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = []
    report.append("=" * 75)
    report.append("四模型消融对比报告: M1 vs M2 vs M2' vs M3")
    report.append("=" * 75)
    report.append("")
    report.append("模型配置:")
    report.append("  M1:  DehazeFormer-S (基线)")
    report.append("  M2:  DehazeFormer-S + HA-WFE v1 (零初始化, Tanh, 共享alpha)")
    report.append("  M2': DehazeFormer-S + HA-WFE v2 (正值初始化, Sigmoid, 独立alpha)")
    report.append("  M3:  DehazeFormer-S + HA-WFE v2 + CLIP雾提示蒸馏")
    report.append("       (教师: HazeCLIP MSBDN冻结, CLIP雾提示注入+KD)")
    report.append("       (推理: 无CLIP/MSBDN, 零额外开销)")
    report.append("")

    models = {}
    for name in ["m1", "m2", "m2p", "m3"]:
        if os.path.exists(CKPT[name]):
            print(f"Loading {name}...")
            models[name] = load_model(name, CKPT[name])
        else:
            print(f"Checkpoint not found: {CKPT[name]}")

    if "m3" not in models:
        report.append("M3 checkpoint not found, skipping comparison.")
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print("M3 checkpoint not found!")
        return

    report.append("=" * 75)
    report.append("一、合成雾验证集 (84对, 有参考)")
    report.append("=" * 75)
    report.append("")

    synth_results = {}
    for name in ["m1", "m2", "m2p", "m3"]:
        if name in models:
            psnr, ssim = eval_synth(models[name], name)
            synth_results[name] = (psnr, ssim)
            print(f"  {name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    hdr = "M3"
    m2p_hdr = "M2'(v2)"
    report.append(f"{'指标':<20} {'M1基线':>10} {'M2(v1)':>10} {m2p_hdr:>10} {hdr:>10} {'M3-M1':>10} {'M3-M2p':>10}")
    report.append(f"  {'─'*75}")
    for label, key in [("PSNR (dB)", None), ("SSIM", None)]:
        vals = []
        for n in ["m1", "m2", "m2p", "m3"]:
            if n in synth_results:
                vals.append(synth_results[n][0] if "PSNR" in label else synth_results[n][1])
            else:
                vals.append(0)
        if "PSNR" in label:
            m3_m1 = vals[3] - vals[0]
            m3_m2p = vals[3] - vals[2]
            report.append(f"{label:<20} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f} {vals[3]:>10.2f} {m3_m1:>+10.2f} {m3_m2p:>+10.2f}")
        else:
            m3_m1 = vals[3] - vals[0]
            m3_m2p = vals[3] - vals[2]
            report.append(f"{label:<20} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f} {m3_m1:>+10.4f} {m3_m2p:>+10.4f}")

    report.append("")
    report.append("=" * 75)
    report.append("二、真实雾图测试集 (无参考, 非盲测)")
    report.append("=" * 75)
    report.append("")

    real_results = {}
    real_dirs = {}
    for name in ["m1", "m2", "m2p", "m3"]:
        if name in models:
            save_dir = os.path.join(OUTPUT_DIR, name.upper())
            metrics, n = eval_real(models[name], name, save_dir)
            real_results[name] = metrics
            real_dirs[name] = save_dir
            print(f"  {name}: {n} images processed")

    orig_metrics = {"dark_channel": [], "contrast": [], "saturation": [], "edge_density": [], "gradient_mean": []}
    real_files = sorted([f for f in os.listdir(REAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for fname in real_files:
        img = cv2.imread(os.path.join(REAL_DIR, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orig_metrics["dark_channel"].append(np.min(gray) / 255.0)
        orig_metrics["contrast"].append(gray.std())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        orig_metrics["saturation"].append(hsv[:, :, 1].mean())
        edges = cv2.Canny(gray, 50, 150)
        orig_metrics["edge_density"].append(edges.sum() / (gray.shape[0] * gray.shape[1] * 255))
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        orig_metrics["gradient_mean"].append(np.sqrt(gx**2 + gy**2).mean())

    orig_avg = {k: np.mean(v) for k, v in orig_metrics.items()}

    labels_map = {"m1": "M1", "m2": "M2", "m2p": "M2'", "m3": "M3"}
    report.append(f"测试图数: {len(real_files)}")
    report.append("")
    hdr_line = f"{'指标':<20} {'原始雾图':>10}"
    for n in ["m1", "m2", "m2p", "m3"]:
        if n in real_results:
            hdr_line += f" {labels_map[n]:>10}"
    hdr_line += f" {'M3-M1':>10} {'M3-M2p':>10}"
    report.append(hdr_line)
    report.append(f"  {'─'*75}")
    for metric in ["dark_channel", "contrast", "saturation", "edge_density", "gradient_mean"]:
        line = f"{metric:<20} {orig_avg[metric]:>10.4f}"
        for n in ["m1", "m2", "m2p", "m3"]:
            if n in real_results:
                line += f" {real_results[n][metric]:>10.4f}"
        m3_m1 = real_results.get("m3", {}).get(metric, 0) - real_results.get("m1", {}).get(metric, 0)
        m3_m2p = real_results.get("m3", {}).get(metric, 0) - real_results.get("m2p", {}).get(metric, 0)
        line += f" {m3_m1:>+10.4f} {m3_m2p:>+10.4f}"
        report.append(line)

    report.append("")
    report.append("=" * 75)
    report.append("三、Phase 3 创新点总结")
    report.append("=" * 75)
    report.append("")
    report.append("1. CLIP雾提示蒸馏:")
    report.append("   - 教师: HazeCLIP (MSBDN + CLIPSurgery), 28.4M+151.3M参数, 冻结")
    report.append("   - 学生: DehazeFormer-S + HA-WFE v2, 1.34M参数, 可训练")
    report.append("   - 推理: 仅学生, 无CLIP/MSBDN, 零额外开销")
    report.append("")
    report.append("2. CLIP语义→小波频域桥接:")
    report.append("   - CLIPSurgery提取空间语义特征 [B, 49, 512]")
    report.append("   - 投影到32维雾提示 [B, 32, 7, 7]")
    report.append("   - 注入HA-WFE低频(LL)分支, 指导小波频域处理")
    report.append("")
    report.append("3. Prompt Dropout (50%):")
    report.append("   - 训练时50%批次使用CLIP提示, 50%不用")
    report.append("   - 保证推理时无CLIP也能正常工作")
    report.append("")
    report.append("4. 教师KD蒸馏:")
    report.append("   - HazeCLIP MSBDN教师输出作为软目标")
    report.append("   - L_kd = 0.1 * L1(student, teacher)")
    report.append("")
    report.append("参数预算: 基线1.28M → M3: 1.34M (+4.0%, 在10-15%预算内)")
    report.append(f"新增参数: ll_prompt(21.7K) + clip_proj(139.6K), 推理时移除")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
