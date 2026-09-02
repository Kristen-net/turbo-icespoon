"""
四层融合去雾推理脚本
融合四个赛道: HazeCLIP + DehazeSB + WDMamba + DiffDehaze(DCP)

融合策略:
  Level 1: 像素级加权平均融合
  Level 2: 特征级自适应权重融合 (对比度+饱和度+边缘)
  Level 3: 决策级质量选择融合 (暗通道+梯度)
  Level 4: 级联融合 (DCP→HazeCLIP→WDMamba 逐级精炼)
"""
import os
import sys
import time
import cv2
import torch
import numpy as np

# 各赛道输出目录
HAZECLIP_DIR = r"D:\dehaze_fusion\HazeCLIP\outputs"
DEHAZESB_DIR = r"D:\dehaze_fusion\DehazeSB\output\dehazesb_results"
WDMAMBA_DIR = r"D:\dehaze_fusion\WDMamba\output\wdmamba_results"
DIFFDEHAZE_DIR = r"D:\dehaze_fusion\DiffDehaze-GAN\output\diffdehaze_results"

# 融合输出目录
FUSION_OUTPUT_DIR = r"D:\dehaze_fusion\fusion_output"
os.makedirs(FUSION_OUTPUT_DIR, exist_ok=True)

# 统一尺寸
TARGET_SIZE = 256


def load_image(path, target_size=TARGET_SIZE):
    """加载图像并缩放到统一尺寸"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def compute_contrast(img):
    """计算图像对比度 (标准差)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def compute_saturation(img):
    """计算图像饱和度 (S通道均值)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def compute_edge_density(img):
    """计算边缘密度 (Canny边缘像素比例)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))


def compute_dark_channel(img, patch_size=7):
    """计算暗通道值 (越低越好, 表示去雾越彻底)"""
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val, kernel)
    return float(np.mean(dark) / 255.0)


def compute_gradient_mean(img):
    """计算梯度均值 (清晰度指标, 越高越清晰)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def compute_quality_scores(img):
    """计算综合质量分数"""
    contrast = compute_contrast(img)
    saturation = compute_saturation(img)
    edge = compute_edge_density(img)
    dark = compute_dark_channel(img)
    gradient = compute_gradient_mean(img)

    # 归一化分数 (0-1)
    scores = {
        'contrast': contrast / 80.0,        # 典型范围 0-80
        'saturation': saturation / 255.0,     # 0-255
        'edge': edge / 0.3,                    # 典型范围 0-0.3
        'dark_channel': 1.0 - min(dark, 1.0),  # 反转: 暗通道越低越好
        'gradient': gradient / 30.0,            # 典型范围 0-30
    }

    # 综合分数
    total = (
        scores['contrast'] * 0.2 +
        scores['saturation'] * 0.15 +
        scores['edge'] * 0.25 +
        scores['dark_channel'] * 0.2 +
        scores['gradient'] * 0.2
    )
    return total, scores


def level1_pixel_fusion(images, weights=None):
    """Level 1: 像素级加权平均融合"""
    if weights is None:
        weights = [1.0 / len(images)] * len(images)

    # 归一化权重
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    result = np.zeros_like(images[0], dtype=np.float32)
    for img, w in zip(images, weights):
        result += img.astype(np.float32) * w

    return np.clip(result, 0, 255).astype(np.uint8)


def level2_feature_fusion(images):
    """Level 2: 特征级自适应权重融合"""
    # 计算每个图像的特征质量
    contrasts = [compute_contrast(img) for img in images]
    edges = [compute_edge_density(img) for img in images]
    gradients = [compute_gradient_mean(img) for img in images]

    # 归一化特征
    c_sum = sum(contrasts) + 1e-8
    e_sum = sum(edges) + 1e-8
    g_sum = sum(gradients) + 1e-8

    # 特征权重
    weights = []
    for i in range(len(images)):
        w = (contrasts[i] / c_sum + edges[i] / e_sum + gradients[i] / g_sum) / 3.0
        weights.append(w)

    print(f"  Level 2 特征权重: {[f'{w:.3f}' for w in weights]}")
    return level1_pixel_fusion(images, weights)


def level3_decision_fusion(images, names):
    """Level 3: 决策级质量选择融合"""
    scores = []
    for i, img in enumerate(images):
        score, metrics = compute_quality_scores(img)
        scores.append(score)
        print(f"  {names[i]}: 分数={score:.4f} (对比度={metrics['contrast']:.3f}, "
              f"边缘={metrics['edge']:.3f}, 暗通道={metrics['dark_channel']:.3f}, "
              f"梯度={metrics['gradient']:.3f})")

    # 软max权重
    scores_arr = np.array(scores)
    exp_scores = np.exp(scores_arr * 3.0)  # 放大差异
    weights = exp_scores / np.sum(exp_scores)

    print(f"  Level 3 决策权重: {[f'{w:.3f}' for w in weights]}")
    return level1_pixel_fusion(images, weights.tolist())


def level4_cascade_fusion(images, names):
    """Level 4: 级联融合 (基于物理→学习模型的逐级精炼)"""
    # 级联顺序: DCP(物理) → HazeCLIP(轻量) → WDMamba(深度)
    # 逐级取加权平均, 后一级权重更大
    cascade_order = []
    for name in ['DCP', 'HazeCLIP', 'WDMamba', 'DehazeSB']:
        for i, n in enumerate(names):
            if name.lower() in n.lower():
                cascade_order.append(i)
                break

    if len(cascade_order) < 2:
        return level1_pixel_fusion(images)

    # 逐级融合, 后一级权重递增
    result = images[cascade_order[0]].astype(np.float32)
    for level, idx in enumerate(cascade_order[1:], 1):
        w = level / (level + 1)  # 后一级权重
        result = result * (1 - w) + images[idx].astype(np.float32) * w

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    print("=" * 60)
    print("四层融合去雾推理脚本")
    print("融合: HazeCLIP + DehazeSB + WDMamba + DiffDehaze(DCP)")
    print("=" * 60)

    # 测试图像
    test_images = ['1.png', '2.png']

    for img_name in test_images:
        base_name = os.path.splitext(img_name)[0]
        print(f"\n{'='*60}")
        print(f"处理: {img_name}")
        print(f"{'='*60}")

        # 加载各赛道输出
        track_images = []
        track_names = []

        # HazeCLIP
        hazeclip_path = os.path.join(HAZECLIP_DIR, img_name)
        if os.path.exists(hazeclip_path):
            img = load_image(hazeclip_path)
            if img is not None:
                track_images.append(img)
                track_names.append('HazeCLIP')
                print(f"  HazeCLIP: 已加载 ({hazeclip_path})")

        # DehazeSB
        dehazesb_path = os.path.join(DEHAZESB_DIR, f"{base_name}_dehazed.png")
        if os.path.exists(dehazesb_path):
            img = load_image(dehazesb_path)
            if img is not None:
                track_images.append(img)
                track_names.append('DehazeSB')
                print(f"  DehazeSB: 已加载 ({dehazesb_path})")

        # WDMamba
        wdmamba_path = os.path.join(WDMAMBA_DIR, f"{base_name}_wdmamba_dehazed.png")
        if os.path.exists(wdmamba_path):
            img = load_image(wdmamba_path)
            if img is not None:
                track_images.append(img)
                track_names.append('WDMamba')
                print(f"  WDMamba: 已加载 ({wdmamba_path})")

        # DiffDehaze (DCP)
        diffdehaze_path = os.path.join(DIFFDEHAZE_DIR, f"{base_name}_dcp_dehazed.png")
        if os.path.exists(diffdehaze_path):
            img = load_image(diffdehaze_path)
            if img is not None:
                track_images.append(img)
                track_names.append('DCP')
                print(f"  DCP: 已加载 ({diffdehaze_path})")

        if len(track_images) < 2:
            print(f"  跳过: 不足2个赛道输出 (只有 {len(track_images)} 个)")
            continue

        print(f"\n  参与融合的赛道: {track_names}")

        # Level 1: 像素级等权融合
        print(f"\n  --- Level 1: 像素级等权融合 ---")
        l1_result = level1_pixel_fusion(track_images)
        l1_path = os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L1_pixel.png")
        cv2.imwrite(l1_path, l1_result)
        score1, metrics1 = compute_quality_scores(l1_result)
        print(f"  质量分数: {score1:.4f}")
        print(f"  已保存: {base_name}_L1_pixel.png")

        # Level 2: 特征级自适应融合
        print(f"\n  --- Level 2: 特征级自适应融合 ---")
        l2_result = level2_feature_fusion(track_images)
        l2_path = os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L2_feature.png")
        cv2.imwrite(l2_path, l2_result)
        score2, metrics2 = compute_quality_scores(l2_result)
        print(f"  质量分数: {score2:.4f}")
        print(f"  已保存: {base_name}_L2_feature.png")

        # Level 3: 决策级质量选择融合
        print(f"\n  --- Level 3: 决策级质量选择融合 ---")
        l3_result = level3_decision_fusion(track_images, track_names)
        l3_path = os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L3_decision.png")
        cv2.imwrite(l3_path, l3_result)
        score3, metrics3 = compute_quality_scores(l3_result)
        print(f"  质量分数: {score3:.4f}")
        print(f"  已保存: {base_name}_L3_decision.png")

        # Level 4: 级联融合
        print(f"\n  --- Level 4: 级联融合 ---")
        l4_result = level4_cascade_fusion(track_images, track_names)
        l4_path = os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L4_cascade.png")
        cv2.imwrite(l4_path, l4_result)
        score4, metrics4 = compute_quality_scores(l4_result)
        print(f"  质量分数: {score4:.4f}")
        print(f"  已保存: {base_name}_L4_cascade.png")

        # 最终推荐: 选择质量分数最高的融合结果
        scores = {
            'L1_pixel': score1,
            'L2_feature': score2,
            'L3_decision': score3,
            'L4_cascade': score4,
        }
        best_level = max(scores, key=scores.get)
        best_result = {
            'L1_pixel': l1_result,
            'L2_feature': l2_result,
            'L3_decision': l3_result,
            'L4_cascade': l4_result,
        }[best_level]

        final_path = os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_fusion_final.png")
        cv2.imwrite(final_path, best_result)
        print(f"\n  >>> 最佳融合策略: {best_level} (分数={scores[best_level]:.4f})")
        print(f"  >>> 最终结果: {base_name}_fusion_final.png")

        # 汇总
        print(f"\n  === 融合结果汇总 ===")
        for level, score in scores.items():
            print(f"    {level}: {score:.4f}")

    print(f"\n{'='*60}")
    print(f"融合推理完成!")
    print(f"输出目录: {FUSION_OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
