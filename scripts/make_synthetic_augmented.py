#!/usr/bin/env python3
"""增强版合成数据生成器.

相比原版增强点:
- 1000+ 训练样本, 200+ 验证样本
- 三级雾浓度 (薄/中/浓), 浓度范围更广
- 更小的目标尺寸 (模拟真实场景中的远距离目标)
- 4 个类别 (insulator/power_line/ice/tower), 形状各异
- 更复杂的背景 (纹理, 噪声, 渐变)
- 空间变化的透射率 (深度相关雾)
- 高斯噪声模拟传感器噪声
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# 类别定义: 0=insulator(绝缘子), 1=power_line(输电线), 2=ice(覆冰), 3=tower(塔架)
CLASS_NAMES = ["insulator", "power_line", "ice", "tower"]
CLASS_COLORS = {
    0: (0.85, 0.75, 0.55),  # 绝缘子: 棕黄色
    1: (0.3, 0.3, 0.35),     # 输电线: 深灰
    2: (0.85, 0.92, 0.98),   # 覆冰: 淡蓝白
    3: (0.4, 0.42, 0.45),    # 塔架: 钢灰色
}

# 雾浓度等级
HAZE_LEVELS = {
    "thin":   (0.03, 0.08),   # 薄雾
    "medium": (0.08, 0.18),   # 中雾
    "dense":  (0.18, 0.35),   # 浓雾
}


def add_texture_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """生成带纹理的天空背景."""
    bg = np.zeros((size, size, 3), dtype=np.float32)
    # 垂直渐变 (天空)
    for y in range(size):
        t = y / size
        # 从淡蓝到灰白
        r = 0.55 + 0.25 * (1 - t) + 0.1 * rng.normal(0, 0.02)
        g = 0.65 + 0.20 * (1 - t) + 0.1 * rng.normal(0, 0.02)
        b = 0.80 + 0.15 * (1 - t) + 0.1 * rng.normal(0, 0.02)
        bg[y, :] = [r, g, b]
    # 添加随机噪声纹理
    noise = rng.normal(0, 0.03, (size, size, 3)).astype(np.float32)
    bg = np.clip(bg + noise, 0.0, 1.0)
    return bg


def draw_insulator(img: np.ndarray, x: int, y: int, w: int, h: int,
                   rng: np.random.Generator) -> tuple[int, int, int, int]:
    """绘制绝缘子 (椭圆形/圆柱形堆叠)."""
    color = np.array(CLASS_COLORS[0], dtype=np.float32)
    # 主体: 多个圆盘堆叠
    n_discs = max(2, int(h / 8))
    disc_h = max(3, h // n_discs)
    for i in range(n_discs):
        cy = y + i * disc_h + disc_h // 2
        # 每个圆盘宽度略有变化
        dw = int(w * (0.8 + 0.2 * np.sin(i * 1.2)))
        dx = x + (w - dw) // 2
        # 绘制椭圆盘
        for py in range(max(0, cy - disc_h // 2), min(img.shape[0], cy + disc_h // 2)):
            for px in range(max(0, dx), min(img.shape[1], dx + dw)):
                dxn = (px - (dx + dw / 2)) / (dw / 2)
                dyn = (py - cy) / (disc_h / 2)
                if dxn * dxn + dyn * dyn <= 1.0:
                    shade = 0.85 + 0.15 * (1 - abs(dxn))
                    img[py, px] = color * shade
    return (x, y, x + w, y + h)


def draw_power_line(img: np.ndarray, x: int, y: int, w: int, h: int,
                    rng: np.random.Generator) -> tuple[int, int, int, int]:
    """绘制输电线 (细长线条, 略带下垂)."""
    color = np.array(CLASS_COLORS[1], dtype=np.float32)
    thickness = max(1, h // 10)
    # 绘制下垂曲线 (悬链线近似)
    for px in range(x, min(x + w, img.shape[1])):
        t = (px - x) / max(1, w)
        sag = int(8 * t * (1 - t) * h)  # 下垂量
        py = y + sag
        for dy in range(-thickness // 2, thickness // 2 + 1):
            if 0 <= py + dy < img.shape[0]:
                shade = 0.9 + 0.1 * rng.random()
                img[py + dy, px] = color * shade
    return (x, y, x + w, y + h + 4)


def draw_ice(img: np.ndarray, x: int, y: int, w: int, h: int,
             rng: np.random.Generator) -> tuple[int, int, int, int]:
    """绘制覆冰物体 (不规则形状, 半透明感)."""
    color = np.array(CLASS_COLORS[2], dtype=np.float32)
    # 主体: 圆角矩形
    for py in range(y, min(y + h, img.shape[0])):
        for px in range(x, min(x + w, img.shape[1])):
            # 到中心的距离
            dxn = abs(px - (x + w / 2)) / (w / 2)
            dyn = abs(py - (y + h / 2)) / (h / 2)
            if dxn < 1.0 and dyn < 1.0:
                # 圆角
                corner_dist = max(0, dxn - 0.7) ** 2 + max(0, dyn - 0.7) ** 2
                if corner_dist < 0.09:
                    shade = 0.8 + 0.2 * (1 - dxn) + 0.05 * rng.normal(0, 0.1)
                    img[py, px] = color * shade
    # 添加高光
    highlight_x = x + int(w * 0.3)
    highlight_w = max(2, w // 6)
    for py in range(y + 2, min(y + h - 2, img.shape[0])):
        for px in range(highlight_x, min(highlight_x + highlight_w, img.shape[1])):
            if img[py, px, 0] > 0.5:  # 只在冰面上
                img[py, px] = np.clip(img[py, px] * 1.15, 0, 1)
    return (x, y, x + w, y + h)


def draw_tower(img: np.ndarray, x: int, y: int, w: int, h: int,
               rng: np.random.Generator) -> tuple[int, int, int, int]:
    """绘制塔架 (桁架结构)."""
    color = np.array(CLASS_COLORS[3], dtype=np.float32)
    thickness = max(2, w // 12)
    # 左右两根立柱
    left_x = x + thickness
    right_x = x + w - thickness
    for py in range(y, min(y + h, img.shape[0])):
        for dx in range(thickness):
            if 0 <= left_x + dx < img.shape[1]:
                shade = 0.85 + 0.1 * rng.random()
                img[py, left_x + dx] = color * shade
            if 0 <= right_x + dx < img.shape[1]:
                shade = 0.85 + 0.1 * rng.random()
                img[py, right_x + dx] = color * shade
    # 横梁 (每隔一段)
    n_beams = max(3, h // 20)
    for i in range(n_beams):
        by = y + int(i * h / n_beams) + h // (2 * n_beams)
        if by >= img.shape[0]:
            break
        for px in range(left_x, right_x + thickness):
            if 0 <= px < img.shape[1]:
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    if 0 <= by + dy < img.shape[0]:
                        img[by + dy, px] = color * (0.9 + 0.1 * rng.random())
    # 顶部尖顶
    top_y = y
    mid_x = x + w // 2
    for py in range(max(0, y - h // 4), y):
        t = (y - py) / max(1, h // 4)
        span = int(w * 0.4 * (1 - t))
        for dx in range(-span, span + 1):
            px = mid_x + dx
            if 0 <= px < img.shape[1]:
                for ddy in range(-thickness // 2, thickness // 2 + 1):
                    if 0 <= py + ddy < img.shape[0]:
                        img[py + ddy, px] = color * (0.85 + 0.15 * rng.random())
    return (x, max(0, y - h // 4), x + w, y + h)


def draw_object(img: np.ndarray, cls_id: int, x: int, y: int, w: int, h: int,
                rng: np.random.Generator) -> tuple[int, int, int, int]:
    """根据类别绘制目标."""
    if cls_id == 0:
        return draw_insulator(img, x, y, w, h, rng)
    elif cls_id == 1:
        return draw_power_line(img, x, y, w, h, rng)
    elif cls_id == 2:
        return draw_ice(img, x, y, w, h, rng)
    else:
        return draw_tower(img, x, y, w, h, rng)


def get_object_size_range(cls_id: int, size: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """根据类别获取目标尺寸范围 (宽, 高)."""
    if cls_id == 0:  # 绝缘子: 小到中等
        w = int(rng.integers(size * 0.04, size * 0.12))
        h = int(rng.integers(size * 0.08, size * 0.20))
    elif cls_id == 1:  # 输电线: 细长
        w = int(rng.integers(size * 0.3, size * 0.8))
        h = int(rng.integers(size * 0.02, size * 0.06))
    elif cls_id == 2:  # 覆冰: 小
        w = int(rng.integers(size * 0.03, size * 0.10))
        h = int(rng.integers(size * 0.05, size * 0.15))
    else:  # 塔架: 大
        w = int(rng.integers(size * 0.10, size * 0.25))
        h = int(rng.integers(size * 0.15, size * 0.40))
    return w, h


def depth_guided_transmission(size: int, beta: float,
                              rng: np.random.Generator) -> np.ndarray:
    """生成深度引导的透射率图 (远处雾更浓)."""
    # 垂直深度梯度 (上方更远)
    depth = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        depth[y, :] = 0.3 + 0.7 * (y / size)  # 上方=0.3(远), 下方=1.0(近)
    # 添加空间噪声
    noise = rng.uniform(0.8, 1.2, (size, size)).astype(np.float32)
    t = 1.0 - beta * depth * noise
    t = np.clip(t, 0.05, 1.0)
    return t[..., None]  # (h, w, 1)


def add_gaussian_noise(img: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """添加高斯噪声模拟传感器噪声."""
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def apply_haze(clear: np.ndarray, beta: float, A: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """应用大气散射模型: I = J*t + A*(1-t)."""
    t = depth_guided_transmission(clear.shape[0], beta, rng)
    hazy = clear * t + A * (1.0 - t)
    return np.clip(hazy, 0.0, 1.0).astype(np.float32)


def generate(out_dir: Path, num_train: int = 1000, num_val: int = 200,
             size: int = 512, seed: int = 42):
    """生成增强合成数据集."""
    rng = np.random.default_rng(seed)

    for split, n_images in [("train", num_train), ("val", num_val)]:
        img_dir = out_dir / split / "clear"
        hazy_dir = out_dir / split / "hazy"
        label_dir = out_dir / split / "labels"
        for d in (img_dir, hazy_dir, label_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 雾浓度等级分布: 薄30%, 中40%, 浓30%
        haze_dist = ["thin"] * int(n_images * 0.3) + \
                    ["medium"] * int(n_images * 0.4) + \
                    ["dense"] * (n_images - int(n_images * 0.3) - int(n_images * 0.4))
        rng.shuffle(haze_dist)

        for i in range(n_images):
            # 背景
            clear = add_texture_background(size, rng)

            # 随机目标数量 (1-6个)
            n_obj = int(rng.integers(1, 7))
            bboxes = []  # (cls_id, x1, y1, x2, y2)

            for _ in range(n_obj):
                # 随机类别
                cls_id = int(rng.integers(0, 4))
                w, h = get_object_size_range(cls_id, size, rng)
                # 随机位置 (确保在图像内)
                x = int(rng.integers(0, max(1, size - w)))
                y = int(rng.integers(size * 0.1, max(size * 0.1 + 1, size - h)))

                # 绘制目标
                x1, y1, x2, y2 = draw_object(clear, cls_id, x, y, w, h, rng)

                # 裁剪到图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(size, x2)
                y2 = min(size, y2)
                if x2 > x1 and y2 > y1:
                    bboxes.append((cls_id, x1, y1, x2, y2))

            # 雾浓度
            level = haze_dist[i]
            beta_range = HAZE_LEVELS[level]
            beta = float(rng.uniform(*beta_range))

            # 大气光 (随雾浓度变化, 浓雾更白)
            A_brightness = rng.uniform(0.75, 0.95)
            A = np.array([A_brightness, A_brightness, A_brightness + 0.02],
                         dtype=np.float32)
            A = np.clip(A, 0, 1)

            # 加雾
            hazy = apply_haze(clear, beta=beta, A=A, rng=rng)

            # 添加传感器噪声 (雾天噪声更大)
            noise_sigma = 0.005 + beta * 0.05
            hazy = add_gaussian_noise(hazy, noise_sigma, rng)

            # 保存图像
            name = f"{split}_{i:05d}"
            Image.fromarray((clear * 255).astype(np.uint8)).save(
                img_dir / f"{name}.png")
            Image.fromarray((hazy * 255).clip(0, 255).astype(np.uint8)).save(
                hazy_dir / f"{name}.png")

            # 保存 YOLO 格式标注
            lines = []
            for (cls_id, x1, y1, x2, y2) in bboxes:
                cx = (x1 + x2) / 2 / size
                cy = (y1 + y2) / 2 / size
                bw = (x2 - x1) / size
                bh = (y2 - y1) / size
                if bw > 0.005 and bh > 0.005:  # 过滤过小目标
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (label_dir / f"{name}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

        print(f"  {split}: {n_images} images "
              f"(thin={haze_dist.count('thin')}, "
              f"medium={haze_dist.count('medium')}, "
              f"dense={haze_dist.count('dense')})")

    print(f"\n生成完成: {out_dir}")
    print(f"  train: {num_train} images")
    print(f"  val: {num_val} images")
    print(f"  size: {size}x{size}")
    print(f"  classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"  haze levels: thin/medium/dense")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "synthetic_augmented")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.train, args.val, args.size, args.seed)
