"""
合成雾天图像生成器
===================
基于大气散射模型生成雾天输电线路图像
I(x) = J(x)t(x) + A(1-t(x))
  I: 雾天图像, J: 清晰图像, t: 透射率, A: 大气光

使用方法:
    python generate_haze.py --input clear.png --output hazy.png --density 0.6
    python generate_haze.py --batch --input-dir clear_images/ --output-dir hazy_images/
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2


def atmospheric_light(img, top_ratio=0.001):
    """估计大气光 A"""
    # 取最亮像素的均值作为大气光
    flat = img.reshape(-1, 3)
    # 按亮度排序
    brightness = flat.sum(axis=1)
    top_n = max(1, int(len(flat) * top_ratio))
    top_indices = np.argsort(brightness)[-top_n:]
    A = flat[top_indices].mean(axis=0)
    return A


def estimate_transmission(img, A, beta=1.0, omega=0.95):
    """估计透射率 t（基于暗通道先验）"""
    # 暗通道
    dark = dark_channel(img)
    # 透射率
    t = 1 - omega * dark / np.max(dark)
    # 介质衰减系数
    t = np.maximum(t, 0.1)
    return t


def dark_channel(img, patch_size=15):
    """计算暗通道"""
    min_channels = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channels, kernel)
    return dark


def generate_haze(img, density=0.5, beta=1.0, light_strength=0.8, seed=None):
    """
    基于大气散射模型生成雾天图像
    I(x) = J(x) * t(x) + A * (1 - t(x))

    Args:
        img: 清晰图像 (H, W, 3), 0-255
        density: 雾浓度 [0, 1], 0=无雾, 1=浓雾
        beta: 介质衰减系数
        light_strength: 大气光强度
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = img.astype(np.float64) / 255.0
    H, W, _ = img.shape

    # 大气光 A
    A = np.array([light_strength, light_strength, light_strength])
    A = A + np.random.uniform(-0.05, 0.05, size=3)
    A = np.clip(A, 0.5, 1.0)

    # 透射率 t：基于深度图模拟
    # 简化：使用随机梯度场模拟深度
    depth = _generate_depth_map(H, W, density, seed)

    # 透射率 = exp(-beta * depth)
    t = np.exp(-beta * density * 3.0 * depth)
    t = np.clip(t, 0.1, 1.0)
    t = np.stack([t, t, t], axis=2)

    # 雾天图像
    A_expanded = A.reshape(1, 1, 3)
    hazy = img * t + A_expanded * (1 - t)

    # 添加噪声
    noise = np.random.normal(0, 0.005, img.shape)
    hazy = hazy + noise

    hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)
    return hazy


def _generate_depth_map(H, W, density, seed=None):
    """生成深度图（模拟输电线路场景的远近关系）"""
    if seed is not None:
        np.random.seed(seed)

    # 基础梯度：从上到下深度递增（天空->地面）
    y_gradient = np.linspace(0.3, 1.0, H).reshape(H, 1)
    depth = np.tile(y_gradient, (1, W))

    # 添加随机起伏
    noise = np.random.uniform(-0.1, 0.1, (H, W))
    depth = depth + noise

    # 平滑
    depth = cv2.GaussianBlur(depth, (31, 31), 0)

    # 归一化到 [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # 根据雾密度调整
    depth = depth * (0.5 + density * 0.5)

    return depth


def generate_power_line_haze(img, density=0.5, seed=None):
    """专门针对输电线路场景的雾生成"""
    if seed is not None:
        np.random.seed(seed)

    img = img.astype(np.float64) / 255.0
    H, W, _ = img.shape

    # 输电线路场景的深度特征：
    # - 上方天空（远，雾浓）
    # - 中间线路（中等距离）
    # - 下方地面/塔基（近，雾淡）

    depth = np.zeros((H, W))

    # 天空区域（上方 40%）
    sky_h = int(H * 0.4)
    depth[:sky_h, :] = np.linspace(0.8, 0.6, sky_h).reshape(-1, 1)

    # 线路区域（中间 30%）
    line_h = int(H * 0.3)
    depth[sky_h:sky_h + line_h, :] = np.linspace(0.6, 0.4, line_h).reshape(-1, 1)

    # 地面区域（下方 30%）
    ground_h = H - sky_h - line_h
    depth[sky_h + line_h:, :] = np.linspace(0.4, 0.2, ground_h).reshape(-1, 1)

    # 添加随机起伏
    noise = np.random.uniform(-0.1, 0.1, (H, W))
    depth = depth + noise
    depth = cv2.GaussianBlur(depth, (41, 41), 0)
    depth = np.clip(depth, 0.1, 1.0)

    # 透射率
    t = np.exp(-3.0 * density * depth)
    t = np.clip(t, 0.1, 1.0)
    t = np.stack([t, t, t], axis=2)

    # 大气光（偏白偏蓝，模拟雾天）
    A = np.array([0.85, 0.88, 0.92])  # 偏蓝白色
    A = A + np.random.uniform(-0.03, 0.03, 3)
    A = np.clip(A, 0.6, 1.0)
    A_expanded = A.reshape(1, 1, 3)

    # 生成雾天图像
    hazy = img * t + A_expanded * (1 - t)

    # 添加大气散射噪声
    noise = np.random.normal(0, 0.003, img.shape)
    hazy = hazy + noise

    return np.clip(hazy * 255, 0, 255).astype(np.uint8)


def batch_generate(input_dir, output_dir, density_range=(0.3, 0.8), pattern='power_line'):
    """批量生成雾天图像"""
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(images)} 张图片")

    for i, filename in enumerate(images):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  跳过: {filename}")
            continue

        # 随机雾浓度
        density = random.uniform(*density_range)

        if pattern == 'power_line':
            hazy = generate_power_line_haze(img, density=density, seed=i)
        else:
            hazy = generate_haze(img, density=density, seed=i)

        # 保存
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_dir, f"{name}_hazy{ext}")
        cv2.imwrite(out_path, hazy)

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(images)}")

    print(f"完成! 共生成 {len(images)} 张雾天图像")


def main():
    parser = argparse.ArgumentParser(description="合成雾天图像生成器")
    parser.add_argument('--input', type=str, help='输入清晰图片路径')
    parser.add_argument('--output', type=str, default='hazy.png', help='输出雾天图片路径')
    parser.add_argument('--density', type=float, default=0.5, help='雾浓度 (0-1)')
    parser.add_argument('--pattern', type=str, default='power_line', choices=['power_line', 'general'],
                        help='雾模式: power_line=输电线路场景, general=通用场景')
    parser.add_argument('--batch', action='store_true', help='批量模式')
    parser.add_argument('--input-dir', type=str, help='批量输入目录')
    parser.add_argument('--output-dir', type=str, help='批量输出目录')
    parser.add_argument('--density-min', type=float, default=0.3, help='批量模式最小雾浓度')
    parser.add_argument('--density-max', type=float, default=0.8, help='批量模式最大雾浓度')

    args = parser.parse_args()

    if args.batch:
        if not args.input_dir or not args.output_dir:
            print("批量模式需要 --input-dir 和 --output-dir")
            sys.exit(1)
        batch_generate(args.input_dir, args.output_dir,
                       density_range=(args.density_min, args.density_max),
                       pattern=args.pattern)
    else:
        if not args.input:
            print("需要 --input 参数")
            sys.exit(1)

        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图片: {args.input}")
            sys.exit(1)

        print(f"输入: {args.input}")
        print(f"雾浓度: {args.density}")
        print(f"雾模式: {args.pattern}")

        if args.pattern == 'power_line':
            hazy = generate_power_line_haze(img, density=args.density, seed=42)
        else:
            hazy = generate_haze(img, density=args.density, seed=42)

        cv2.imwrite(args.output, hazy)
        print(f"输出: {args.output}")

        # 输出统计
        print(f"\n原始图片: 均值={img.mean():.1f}, 对比度={img.std():.1f}")
        print(f"雾天图片: 均值={hazy.mean():.1f}, 对比度={hazy.std():.1f}")


if __name__ == '__main__':
    main()
