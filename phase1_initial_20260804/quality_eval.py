"""
去雾质量评估脚本
=================
计算 PSNR, SSIM, NIQE 等指标，生成对比可视化

使用方法:
    python quality_eval.py --hazy test_hazy.png --clear test_clear.png --results result1.png result2.png result3.png
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import time


def compute_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if img1.dtype != np.float64:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64:
        img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2):
    """计算 SSIM (Structural Similarity Index)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    if img1.ndim == 3:
        # 多通道，取平均
        ssims = [compute_ssim_single(img1[..., c], img2[..., c]) for c in range(img1.shape[-1])]
        return np.mean(ssims)
    else:
        return compute_ssim_single(img1, img2)


def compute_ssim_single(img1, img2):
    """单通道 SSIM"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_brightness(img):
    """计算平均亮度"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    return float(np.mean(img))


def compute_contrast(img):
    """计算对比度（标准差）"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    return float(np.std(img))


def compute_edge_strength(img):
    """计算边缘强度（Sobel）"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 3:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img = (img * 255).astype(np.uint8)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(sobelx**2 + sobely**2)))


def compute_visibility(img):
    """计算可见度（基于对比度的感知指标）"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = (img * 255).astype(np.float64)
    # 局部对比度
    local_mean = cv2.GaussianBlur(gray, (15, 15), 0)
    local_std = np.sqrt(cv2.GaussianBlur(gray**2, (15, 15), 0) - local_mean**2)
    return float(np.mean(local_std / (local_mean + 1e-8)))


def evaluate(hazy_path, clear_path, result_paths, names=None):
    """评估多个去雾结果"""
    print("=" * 70)
    print("去雾质量评估报告")
    print("=" * 70)
    print()

    # 读取参考图片
    hazy = cv2.imread(hazy_path, cv2.IMREAD_COLOR)
    hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    has_clear = clear_path and os.path.exists(clear_path)
    if has_clear:
        clear = cv2.imread(clear_path, cv2.IMREAD_COLOR)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        print(f"参考清晰图: {clear_path}")
        print(f"雾天输入图: {hazy_path}")
    else:
        print(f"雾天输入图: {hazy_path}")
        print(f"（无参考清晰图，仅计算无参考指标）")

    print(f"图像尺寸: {hazy.shape}")
    print()

    if names is None:
        names = [f"Result_{i+1}" for i in range(len(result_paths))]

    # 评估每个结果
    print("-" * 70)
    print(f"{'方法':<20} {'PSNR(dB)':<12} {'SSIM':<10} {'亮度':<10} {'对比度':<10} {'边缘强度':<12} {'可见度':<10}")
    print("-" * 70)

    all_metrics = []

    # 雾天原图指标
    hazy_brightness = compute_brightness(hazy)
    hazy_contrast = compute_contrast(hazy)
    hazy_edge = compute_edge_strength(hazy)
    hazy_vis = compute_visibility(hazy)

    if has_clear:
        hazy_psnr = compute_psnr(hazy, clear)
        hazy_ssim = compute_ssim(hazy, clear)
        print(f"{'[雾天原图]':<20} {hazy_psnr:<12.2f} {hazy_ssim:<10.4f} {hazy_brightness:<10.4f} {hazy_contrast:<10.4f} {hazy_edge:<12.2f} {hazy_vis:<10.4f}")
    else:
        print(f"{'[雾天原图]':<20} {'N/A':<12} {'N/A':<10} {hazy_brightness:<10.4f} {hazy_contrast:<10.4f} {hazy_edge:<12.2f} {hazy_vis:<10.4f}")

    # 清晰图指标
    if has_clear:
        clear_brightness = compute_brightness(clear)
        clear_contrast = compute_contrast(clear)
        clear_edge = compute_edge_strength(clear)
        clear_vis = compute_visibility(clear)
        print(f"{'[清晰参考]':<20} {'∞':<12} {'1.0000':<10} {clear_brightness:<10.4f} {clear_contrast:<10.4f} {clear_edge:<12.2f} {clear_vis:<10.4f}")
        print("-" * 70)

    # 各去雾结果
    for name, path in zip(names, result_paths):
        if not os.path.exists(path):
            print(f"{name:<20} 文件不存在: {path}")
            continue

        result = cv2.imread(path, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        brightness = compute_brightness(result)
        contrast = compute_contrast(result)
        edge = compute_edge_strength(result)
        vis = compute_visibility(result)

        if has_clear:
            psnr = compute_psnr(result, clear)
            ssim = compute_ssim(result, clear)
            print(f"{name:<20} {psnr:<12.2f} {ssim:<10.4f} {brightness:<10.4f} {contrast:<10.4f} {edge:<12.2f} {vis:<10.4f}")
            all_metrics.append({
                'name': name, 'psnr': psnr, 'ssim': ssim,
                'brightness': brightness, 'contrast': contrast,
                'edge': edge, 'visibility': vis
            })
        else:
            print(f"{name:<20} {'N/A':<12} {'N/A':<10} {brightness:<10.4f} {contrast:<10.4f} {edge:<12.2f} {vis:<10.4f}")
            all_metrics.append({
                'name': name, 'psnr': None, 'ssim': None,
                'brightness': brightness, 'contrast': contrast,
                'edge': edge, 'visibility': vis
            })

    print("-" * 70)
    print()

    # 生成对比可视化
    if len(result_paths) > 0:
        _generate_comparison(hazy_path, clear_path, result_paths, names, all_metrics)

    return all_metrics


def _generate_comparison(hazy_path, clear_path, result_paths, names, metrics):
    """生成对比可视化图片"""
    n = len(result_paths) + (1 if clear_path else 0) + 1  # 雾天 + 结果 + 清晰
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig_w = 5 * cols
    fig_h = 5 * rows + 2  # 额外空间放标题

    # 使用 OpenCV 创建拼接图
    target_h, target_w = 480, 640
    canvas = np.ones((target_h * rows + 40 * rows, target_w * cols, 3), dtype=np.uint8) * 30

    images = [(hazy_path, "雾天原图")]
    if clear_path and os.path.exists(clear_path):
        images.append((clear_path, "清晰参考"))
    for name, path in zip(names, result_paths):
        if os.path.exists(path):
            images.append((path, name))

    for idx, (path, title) in enumerate(images):
        row = idx // cols
        col = idx % cols
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (target_w, target_h))
            # 添加标题条
            title_bar = np.ones((40, target_w, 3), dtype=np.uint8) * 50
            cv2.putText(title_bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_start = row * (target_h + 40)
            canvas[y_start:y_start+40, col*target_w:(col+1)*target_w] = title_bar
            canvas[y_start+40:y_start+40+target_h, col*target_w:(col+1)*target_w] = img

    output_path = os.path.join(os.path.dirname(result_paths[0]), "comparison_grid.png")
    cv2.imwrite(output_path, canvas)
    print(f"对比可视化已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="去雾质量评估")
    parser.add_argument('--hazy', type=str, required=True, help='雾天原图路径')
    parser.add_argument('--clear', type=str, default=None, help='清晰参考图路径（可选）')
    parser.add_argument('--results', type=str, nargs='+', required=True, help='去雾结果图片路径')
    parser.add_argument('--names', type=str, nargs='+', default=None, help='结果名称')

    args = parser.parse_args()

    if args.names and len(args.names) != len(args.results):
        print("错误: --names 数量必须与 --results 一致")
        sys.exit(1)

    evaluate(args.hazy, args.clear, args.results, args.names)


if __name__ == '__main__':
    main()
