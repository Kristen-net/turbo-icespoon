"""
⚠️ 此文件已迁移到 `src/icewave/detect/ice_mask.py` (审计编号 H7)。

历史开发过程文件, 仅供追溯。新代码 / 推理 / 训练请使用新包:

    from icewave.detect.ice_mask import generate_ice_mask, pseudo_ice_mask_simple

新版接口在原基础上做了模块化拆分 (generate_ice_mask / pseudo_ice_mask_simple),
并支持走廊约束 + YOLO 检测结果双重来源。

详见 docs/AUDIT_REPORT.md §H7。

覆冰掩码自动标注器

生成覆冰区域伪标签, 用于ITL损失训练

冰区特征 (HSV):
  - 低饱和度 (S < 50): 冰/雾/白色物体
  - 高亮度 (V > 100): 冰区偏亮
  - 边缘丰富: 冰表面有纹理

策略:
  1. HSV饱和度通道 Otsu阈值 → 低饱和掩码
  2. 亮度阈值过滤 (V > mean_V) → 去除暗色低饱和区(阴影)
  3. Canny边缘密度 → 冰纹理区
  4. 形态学开闭运算清理噪声
  5. 连通域过滤 (去除小区域)

输出: 二值掩码 [0=非冰, 1=冰], 保存为8位PNG (0/255)
"""

import os
import numpy as np
import cv2
from tqdm import tqdm


def generate_ice_mask(img_bgr):
    """为单张图像生成覆冰掩码

    输入: BGR图像 (H, W, 3)
    输出: 二值掩码 (H, W), uint8, 0或255
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 1. 饱和度 Otsu → 低饱和掩码
    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    _, low_sat_mask = cv2.threshold(
        s_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. 亮度过滤: 只保留高亮的低饱和区域 (冰/雾偏亮)
    v_mean = np.mean(v)
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    _, bright_mask = cv2.threshold(
        v_blur, int(v_mean), 255, cv2.THRESH_BINARY
    )

    # 3. 边缘密度: 冰表面有丰富纹理
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.boxFilter(
        (edges > 0).astype(np.float32), ddepth=-1, ksize=(15, 15)
    )
    _, edge_mask = cv2.threshold(
        (edge_density * 255).astype(np.uint8), 10, 255, cv2.THRESH_BINARY
    )

    # 4. 组合: 低饱和 + 高亮 + (边缘纹理 OR 大面积低饱和)
    ice_mask = cv2.bitwise_and(low_sat_mask, bright_mask)

    # 大面积低饱和区直接判为冰/雾区
    # 用大核膨胀后再腐蚀, 填充孔洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel_open)

    # 5. 连通域过滤: 去除小区域 (< 200像素)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ice_mask, connectivity=8
    )
    cleaned = np.zeros_like(ice_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 200:
            cleaned[labels == i] = 255

    return cleaned


def process_directory(img_dir, output_dir, prefix="train"):
    """批量生成覆冰掩码"""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    stats = {"total": len(files), "with_ice": 0, "ice_ratios": []}

    for fname in tqdm(files, desc="Generating ice masks"):
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue

        mask = generate_ice_mask(img)
        ice_ratio = np.sum(mask > 0) / mask.size
        if ice_ratio > 0.01:
            stats["with_ice"] += 1
        stats["ice_ratios"].append(float(ice_ratio))

        out_name = fname.replace('.png', '_ice.png')
        cv2.imwrite(os.path.join(output_dir, out_name), mask)

    stats["mean_ice_ratio"] = float(np.mean(stats["ice_ratios"]))
    stats["median_ice_ratio"] = float(np.median(stats["ice_ratios"]))
    return stats


if __name__ == "__main__":
    # 训练集覆冰掩码
    train_clear = r"D:\DATA_ALL\dataset\train\clear"
    train_ice_dir = r"D:\DATA_ALL\dataset\train\ice_mask"
    print("=== 生成训练集覆冰掩码 ===")
    train_stats = process_directory(train_clear, train_ice_dir, prefix="train")
    print(f"  总数: {train_stats['total']}, 含冰: {train_stats['with_ice']}")
    print(f"  平均冰区占比: {train_stats['mean_ice_ratio']:.4f}")
    print(f"  中位数冰区占比: {train_stats['median_ice_ratio']:.4f}")

    # 验证集覆冰掩码
    val_clear = r"D:\DATA_ALL\dataset\val\clear"
    val_ice_dir = r"D:\DATA_ALL\dataset\val\ice_mask"
    print("\n=== 生成验证集覆冰掩码 ===")
    val_stats = process_directory(val_clear, val_ice_dir, prefix="val")
    print(f"  总数: {val_stats['total']}, 含冰: {val_stats['with_ice']}")
    print(f"  平均冰区占比: {val_stats['mean_ice_ratio']:.4f}")

    # 测试集覆冰掩码
    test_clear = r"D:\DATA_ALL\dataset\test\clear"
    test_ice_dir = r"D:\DATA_ALL\dataset\test\ice_mask"
    print("\n=== 生成测试集覆冰掩码 ===")
    test_stats = process_directory(test_clear, test_ice_dir, prefix="test")
    print(f"  总数: {test_stats['total']}, 含冰: {test_stats['with_ice']}")
    print(f"  平均冰区占比: {test_stats['mean_ice_ratio']:.4f}")
    print("\n覆冰掩码生成完成!")
