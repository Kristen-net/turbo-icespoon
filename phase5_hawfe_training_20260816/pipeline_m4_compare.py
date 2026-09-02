"""
Phase 4 下游评测: M1 vs M4 覆冰检测质量对比

评测维度:
  1. 合成雾PSNR/SSIM (有参考)
  2. 真实雾图像质量 (dark channel, contrast, saturation, edge_density)
  3. 冰区专用指标:
     - 冰区内纹理保持 (edge_density in ice_mask)
     - 冰区边界锐度 (gradient_mean at boundary)
     - 冰/非冰可分离性 (Otsu class variance)
     - 冰区饱和度恢复 (saturation in ice region)

生成五模型对比报告: M1 vs M2 vs M2' vs M3 vs M4
"""

import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from models.dehazeformer import dehazeformer_s
from ha_wfe import integrate_hawfe
from ha_wfe_v2 import integrate_hawfe_v2
from clip_fog_prompt import integrate_hawfe_v2_with_prompt

DEVICE = "cuda"
VAL_HAZY = r"D:\DATA_ALL\dataset\val\hazy"
VAL_CLEAR = r"D:\DATA_ALL\dataset\val\clear"
VAL_ICE = r"D:\DATA_ALL\dataset\val\ice_mask"
REAL_DIR = r"D:\DATA_ALL\dataset\test\hazy_real"
TEST_CLEAR = r"D:\DATA_ALL\dataset\test\clear"
TEST_ICE = r"D:\DATA_ALL\dataset\test\ice_mask"
OUTPUT_DIR = r"D:\dehaze_fusion\icewave_output\m4_eval"
REPORT_PATH = r"D:\dehaze_fusion\icewave_output\five_way_report.txt"

CKPT = {
    "m1": r"D:\dehaze_fusion\icewave_output\checkpoints\m1_best.pth",
    "m2": r"D:\dehaze_fusion\icewave_output\m2_hawfe\checkpoints\m2_best.pth",
    "m2p": r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\checkpoints\m2p_best.pth",
    "m3": r"D:\dehaze_fusion\icewave_output\m3_clip_distill\checkpoints\m3_best.pth",
    "m4": r"D:\dehaze_fusion\icewave_output\m4_itl\checkpoints\m4_best.pth",
}


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
    elif name in ("m3", "m4"):
        model = integrate_hawfe_v2_with_prompt(
            model, channels=96, prompt_channels=32
        )
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.eval()
    return model


def eval_synth(model, name):
    hazy_files = sorted([f for f in os.listdir(VAL_HAZY) if f.endswith('.png')])
    psnrs, ssims = [], []
    with torch.no_grad():
        for hazy_name in hazy_files:
            base = '_'.join(hazy_name.replace('.png', '').split('_')[:2])
            clear_path = os.path.join(VAL_CLEAR, base + '.png')
            if not os.path.exists(clear_path):
                continue
            hazy = cv2.imread(os.path.join(VAL_HAZY, hazy_name))
            clear = cv2.imread(clear_path)
            if hazy is None or clear is None:
                continue

            hazy_t = torch.from_numpy(
                cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            ).float().unsqueeze(0).to(DEVICE) / 255.0
            clear_np = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if name in ("m3", "m4"):
                model.clip_prompt = None

            pred = model(hazy_t).float().clamp(0, 1)
            pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)
            mse = np.mean((pred_np - clear_np) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            psnrs.append(psnr)

            pred_t = torch.from_numpy(pred_np.transpose(2, 0, 1)).unsqueeze(0)
            clear_t = torch.from_numpy(clear_np.transpose(2, 0, 1)).unsqueeze(0)
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
            ssims.append(ssim_fn(pred_t.to(DEVICE), clear_t.to(DEVICE)).item())

    return np.mean(psnrs), np.mean(ssims)


def eval_ice_specific(img_bgr, ice_mask):
    """冰区专用指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask_bool = ice_mask > 127
    mask_inv = ~mask_bool

    n_ice = np.sum(mask_bool)
    n_non = np.sum(mask_inv)
    total = n_ice + n_non

    # 1. 冰区内边缘密度 (纹理保持)
    edges = cv2.Canny(gray[:,:,2], 50, 150)
    if n_ice > 0:
        ice_edge_density = np.sum(edges[mask_bool] > 0) / n_ice
    else:
        ice_edge_density = 0

    # 2. 冰区边界梯度锐度
    # 边界带
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(ice_mask, kernel)
    boundary = (dilated > 0) & (ice_mask == 0)
    n_boundary = np.sum(boundary)
    grad_x = cv2.Sobel(gray[:,:,2], cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray[:,:,2], cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    if n_boundary > 0:
        boundary_grad = np.mean(grad_mag[boundary])
    else:
        boundary_grad = 0

    # 3. 冰/非冰 Otsu可分离性 (饱和度通道)
    if n_ice > 0 and n_non > 0:
        w0 = n_ice / total
        w1 = n_non / total
        mu0 = np.mean(s[mask_bool])
        mu1 = np.mean(s[mask_inv])
        otsu_var = w0 * w1 * (mu0 - mu1) ** 2
    else:
        otsu_var = 0

    # 4. 冰区饱和度
    if n_ice > 0:
        ice_sat = np.mean(s[mask_bool])
        ice_val = np.mean(v[mask_bool])
    else:
        ice_sat = 0
        ice_val = 0

    # 5. 非冰区饱和度 (对比)
    if n_non > 0:
        non_sat = np.mean(s[mask_inv])
    else:
        non_sat = 0

    return {
        "ice_edge_density": float(ice_edge_density),
        "boundary_grad": float(boundary_grad),
        "otsu_var": float(otsu_var),
        "ice_saturation": float(ice_sat),
        "ice_brightness": float(ice_val),
        "non_ice_saturation": float(non_sat),
    }


def eval_real_with_ice(model, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    real_files = sorted([f for f in os.listdir(REAL_DIR) if f.endswith('.png')])

    all_metrics = {
        "dark_channel": [], "contrast": [], "saturation": [],
        "edge_density": [], "gradient_mean": [],
        "ice_edge_density": [], "boundary_grad": [],
        "otsu_var": [], "ice_saturation": [],
        "ice_brightness": [], "non_ice_saturation": [],
    }
    n = 0

    with torch.no_grad():
        for i, fname in enumerate(real_files):
            img = cv2.imread(os.path.join(REAL_DIR, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            img_t = torch.from_numpy(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            ).float().unsqueeze(0).to(DEVICE) / 255.0

            if name in ("m3", "m4"):
                model.clip_prompt = None

            if h % 16 != 0 or w % 16 != 0:
                pad_h = (16 - h % 16) % 16
                pad_w = (16 - w % 16) % 16
                img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                pred = model(img_t).float().clamp(0, 1)

            pred = pred[:, :, :h, :w]
            pred_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(save_dir, fname), pred_bgr)

            gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
            all_metrics["dark_channel"].append(
                float(np.min(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY))))
            all_metrics["contrast"].append(
                float(gray.std()))
            hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV)
            all_metrics["saturation"].append(float(hsv[:, :, 1].mean()))
            edges = cv2.Canny(gray, 50, 150)
            all_metrics["edge_density"].append(
                float(np.sum(edges > 0) / edges.size))
            all_metrics["gradient_mean"].append(
                float(np.mean(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))))

            # 为去雾后图像生成冰掩码
            ice_mask = _quick_ice_mask(pred_bgr)
            ice_metrics = eval_ice_specific(pred_bgr, ice_mask)
            for k, v in ice_metrics.items():
                all_metrics[k].append(v)

            n += 1
            if (i + 1) % 200 == 0:
                print(f"  {name}: {i+1}/{len(real_files)} processed")

    avg = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg, n


def _quick_ice_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    _, low_sat = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v_mean = np.mean(v)
    _, bright = cv2.threshold(v, int(v_mean), 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(low_sat, bright)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def eval_test_set_with_gt_ice(model, name):
    """在测试集上用GT冰掩码评测冰区指标"""
    clear_files = sorted([f for f in os.listdir(TEST_CLEAR) if f.endswith('.png')])
    all_metrics = {
        "ice_edge_density": [], "boundary_grad": [],
        "otsu_var": [], "ice_saturation": [], "ice_brightness": [],
    }

    with torch.no_grad():
        for fname in clear_files:
            base = fname.replace('.png', '')
            ice_path = os.path.join(TEST_ICE, base + '_ice.png')
            if not os.path.exists(ice_path):
                continue
            img = cv2.imread(os.path.join(TEST_CLEAR, fname))
            ice_mask = cv2.imread(ice_path, cv2.IMREAD_GRAYSCALE)
            if img is None or ice_mask is None:
                continue
            metrics = eval_ice_specific(img, ice_mask)
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k].append(v)

    return {k: np.mean(v) for k, v in all_metrics.items() if v}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("五模型对比评测: M1 vs M2 vs M2' vs M3 vs M4")
    print("=" * 70)

    # 加载模型
    models = {}
    for name in ["m1", "m2", "m2p", "m3", "m4"]:
        if not os.path.exists(CKPT[name]):
            print(f"Checkpoint not found: {CKPT[name]}, skipping {name}")
            continue
        print(f"Loading {name}...")
        models[name] = load_model(name, CKPT[name])

    report = []
    report.append("=" * 75)
    report.append("五模型消融对比报告: M1 vs M2 vs M2' vs M3 vs M4")
    report.append("=" * 75)
    report.append("")
    report.append("模型配置:")
    report.append("  M1:  DehazeFormer-S (基线)")
    report.append("  M2:  + HA-WFE v1 (零初始化, Tanh, 共享alpha)")
    report.append("  M2': + HA-WFE v2 (正值初始化, Sigmoid, 独立alpha)")
    report.append("  M3:  + CLIP雾提示蒸馏 (CLIP+MSBDN教师, 推理时移除)")
    report.append("  M4:  + ITL覆冰感知损失 (区域+边界双约束)")
    report.append("")

    # --- 一、合成雾验证集 ---
    report.append("=" * 75)
    report.append("一、合成雾验证集 (84对, 有参考)")
    report.append("=" * 75)
    report.append("")

    hdr = "指标"
    line = f"{hdr:<20} {'M1基线':>10} {'M2(v1)':>10}"
    m2p_h = "M2'(v2)"
    m3_h = "M3"
    m4_h = "M4"
    line += f" {m2p_h:>10} {m3_h:>10} {m4_h:>10} {'M4-M1':>10} {'M4-M3':>10}"
    report.append(line)
    report.append(f"  {'─'*75}")

    synth_results = {}
    print("\n[1] 合成雾验证集评测...")
    for name in ["m1", "m2", "m2p", "m3", "m4"]:
        if name not in models:
            synth_results[name] = (0, 0)
            continue
        psnr, ssim = eval_synth(models[name], name)
        synth_results[name] = (psnr, ssim)
        print(f"  {name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    psnr_line = f"{'PSNR (dB)':<20}"
    ssim_line = f"{'SSIM':<20}"
    for n in ["m1", "m2", "m2p", "m3", "m4"]:
        psnr_line += f" {synth_results[n][0]:>10.2f}"
        ssim_line += f" {synth_results[n][1]:>10.4f}"
    m4_m1 = synth_results["m4"][0] - synth_results["m1"][0]
    m4_m3 = synth_results["m4"][0] - synth_results["m3"][0]
    psnr_line += f" {m4_m1:>+10.2f} {m4_m3:>+10.2f}"
    m4_m1_s = synth_results["m4"][1] - synth_results["m1"][1]
    m4_m3_s = synth_results["m4"][1] - synth_results["m3"][1]
    ssim_line += f" {m4_m1_s:>+10.4f} {m4_m3_s:>+10.4f}"
    report.append(psnr_line)
    report.append(ssim_line)
    report.append("")

    # --- 二、真实雾测试集 ---
    report.append("=" * 75)
    report.append("二、真实雾图测试集 (673张, 无参考)")
    report.append("=" * 75)
    report.append("")

    real_results = {}
    print("\n[2] 真实雾测试集评测...")
    for name in ["m1", "m2", "m2p", "m3", "m4"]:
        if name not in models:
            continue
        save_dir = os.path.join(OUTPUT_DIR, name.upper())
        metrics, n = eval_real_with_ice(models[name], name, save_dir)
        real_results[name] = metrics
        print(f"  {name}: {n} images processed")

    # 原始雾图指标
    real_files = sorted([f for f in os.listdir(REAL_DIR) if f.endswith('.png')])
    orig_metrics = {
        "dark_channel": [], "contrast": [], "saturation": [],
        "edge_density": [], "gradient_mean": [],
    }
    for fname in real_files[:100]:
        img = cv2.imread(os.path.join(REAL_DIR, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orig_metrics["dark_channel"].append(float(np.min(gray)))
        orig_metrics["contrast"].append(float(gray.std()))
        orig_metrics["saturation"].append(
            float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()))
        edges = cv2.Canny(gray, 50, 150)
        orig_metrics["edge_density"].append(float(np.sum(edges > 0) / edges.size))
        orig_metrics["gradient_mean"].append(
            float(np.mean(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))))
    orig_avg = {k: np.mean(v) for k, v in orig_metrics.items()}

    labels_map = {"m1": "M1", "m2": "M2", "m2p": "M2'", "m3": "M3", "m4": "M4"}
    hdr_line = f"{'指标':<20} {'原始雾图':>10}"
    for n in ["m1", "m2", "m2p", "m3", "m4"]:
        if n in real_results:
            hdr_line += f" {labels_map[n]:>10}"
    hdr_line += f" {'M4-M1':>10} {'M4-M3':>10}"
    report.append(hdr_line)
    report.append(f"  {'─'*85}")

    for label, key in [("dark_channel", "dark_channel"), ("contrast", "contrast"),
                       ("saturation", "saturation"), ("edge_density", "edge_density"),
                       ("gradient_mean", "gradient_mean")]:
        line = f"{label:<20} {orig_avg[key]:>10.4f}"
        for n in ["m1", "m2", "m2p", "m3", "m4"]:
            if n in real_results:
                line += f" {real_results[n][key]:>10.4f}"
        m4_m1_v = real_results["m4"][key] - real_results["m1"][key]
        m4_m3_v = real_results["m4"][key] - real_results["m3"][key]
        line += f" {m4_m1_v:>+10.4f} {m4_m3_v:>+10.4f}"
        report.append(line)
    report.append("")

    # --- 三、冰区专用指标 ---
    report.append("=" * 75)
    report.append("三、冰区专用指标 (真实雾去雾后)")
    report.append("=" * 75)
    report.append("")

    ice_hdr = f"{'冰区指标':<20}"
    for n in ["m1", "m2", "m2p", "m3", "m4"]:
        if n in real_results:
            ice_hdr += f" {labels_map[n]:>10}"
    ice_hdr += f" {'M4-M1':>10} {'M4-M3':>10}"
    report.append(ice_hdr)
    report.append(f"  {'─'*85}")

    for label, key in [
        ("冰区纹理保持", "ice_edge_density"),
        ("边界梯度锐度", "boundary_grad"),
        ("Otsu可分离性", "otsu_var"),
        ("冰区饱和度", "ice_saturation"),
    ]:
        line = f"{label:<20}"
        for n in ["m1", "m2", "m2p", "m3", "m4"]:
            if n in real_results:
                line += f" {real_results[n].get(key, 0):>10.4f}"
        m4_m1_v = real_results["m4"].get(key, 0) - real_results["m1"].get(key, 0)
        m4_m3_v = real_results["m4"].get(key, 0) - real_results["m3"].get(key, 0)
        line += f" {m4_m1_v:>+10.4f} {m4_m3_v:>+10.4f}"
        report.append(line)
    report.append("")

    # --- 四、Phase 4 总结 ---
    report.append("=" * 75)
    report.append("四、Phase 4 创新点总结")
    report.append("=" * 75)
    report.append("")
    report.append("1. ITL覆冰感知损失:")
    report.append("   - 区域约束: 冰区内加权L1+SSIM, 防止过度平滑")
    report.append("   - 边界约束: 冰/非冰边界Sobel梯度保持, 维持覆冰边缘锐利")
    report.append("   - L_itl = 0.5*L_region + 0.3*L_boundary")
    report.append("")
    report.append("2. 覆冰伪标签自动生成:")
    report.append("   - HSV饱和度Otsu阈值 → 低饱和掩码")
    report.append("   - 亮度过滤 + 边缘密度 → 冰纹理区")
    report.append("   - 形态学开闭运算 + 连通域过滤 → 清理噪声")
    report.append("   - 训练集629/633含冰, 平均冰区占比35.9%")
    report.append("")
    report.append("3. 训练策略:")
    report.append("   - 从M3最佳检查点初始化 (微调而非从头训练)")
    report.append("   - lr=1e-5, 30 epochs")
    report.append("   - 保持50% prompt dropout + KD蒸馏")
    report.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
