"""
三赛道融合去雾推理脚本
融合三个有效赛道: HazeCLIP + DehazeSB + DiffDehaze(DCP)
(WDMamba 因 Windows 纯 PyTorch 实现精度问题被排除)

融合策略:
  Level 1: 像素级加权平均融合
  Level 2: 特征级自适应权重融合 (对比度+饱和度+边缘)
  Level 3: 决策级质量选择融合 (暗通道+梯度)
  Level 4: 级联融合 (DCP→HazeCLIP→DehazeSB 逐级精炼)
"""
import os
import cv2
import numpy as np

# 各赛道输出目录
HAZECLIP_DIR = r"D:\dehaze_fusion\HazeCLIP\outputs"
DEHAZESB_DIR = r"D:\dehaze_fusion\DehazeSB\output\dehazesb_results"
DIFFDEHAZE_DIR = r"D:\dehaze_fusion\DiffDehaze-GAN\output\diffdehaze_results"

# 融合输出目录
FUSION_OUTPUT_DIR = r"D:\dehaze_fusion\fusion_3tracks_output"
os.makedirs(FUSION_OUTPUT_DIR, exist_ok=True)

# 统一尺寸
TARGET_SIZE = 512


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
        'contrast': contrast / 80.0,
        'saturation': saturation / 255.0,
        'edge': edge / 0.3,
        'dark_channel': 1.0 - min(dark, 1.0),
        'gradient': gradient / 30.0,
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

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    result = np.zeros_like(images[0], dtype=np.float32)
    for img, w in zip(images, weights):
        result += img.astype(np.float32) * w

    return np.clip(result, 0, 255).astype(np.uint8)


def level2_feature_fusion(images, names):
    """Level 2: 特征级自适应权重融合"""
    contrasts = [compute_contrast(img) for img in images]
    edges = [compute_edge_density(img) for img in images]
    gradients = [compute_gradient_mean(img) for img in images]

    c_sum = sum(contrasts) + 1e-8
    e_sum = sum(edges) + 1e-8
    g_sum = sum(gradients) + 1e-8

    weights = []
    for i in range(len(images)):
        w = (contrasts[i] / c_sum + edges[i] / e_sum + gradients[i] / g_sum) / 3.0
        weights.append(w)

    print(f"  各赛道特征权重:")
    for n, w in zip(names, weights):
        print(f"    {n}: {w:.3f}")
    return level1_pixel_fusion(images, weights)


def level3_decision_fusion(images, names):
    """Level 3: 决策级质量选择融合"""
    scores = []
    for i, img in enumerate(images):
        score, metrics = compute_quality_scores(img)
        scores.append(score)
        print(f"    {names[i]}: 分数={score:.4f} (对比度={metrics['contrast']:.3f}, "
              f"边缘={metrics['edge']:.3f}, 暗通道={metrics['dark_channel']:.3f}, "
              f"梯度={metrics['gradient']:.3f})")

    # 软max权重
    scores_arr = np.array(scores)
    exp_scores = np.exp(scores_arr * 3.0)
    weights = exp_scores / np.sum(exp_scores)

    print(f"  决策权重: {[f'{w:.3f}' for w in weights]}")
    return level1_pixel_fusion(images, weights.tolist())


def level4_cascade_fusion(images, names):
    """Level 4: 级联融合 (物理→轻量→深度 逐级精炼)"""
    # 级联顺序: DCP(物理) → HazeCLIP(轻量) → DehazeSB(扩散深度)
    cascade_order = []
    for name in ['DCP', 'HazeCLIP', 'DehazeSB']:
        for i, n in enumerate(names):
            if name.lower() in n.lower():
                cascade_order.append(i)
                break

    if len(cascade_order) < 2:
        return level1_pixel_fusion(images)

    result = images[cascade_order[0]].astype(np.float32)
    for level, idx in enumerate(cascade_order[1:], 1):
        w = level / (level + 1)
        result = result * (1 - w) + images[idx].astype(np.float32) * w

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    print("=" * 60)
    print("三赛道融合去雾推理")
    print("融合: HazeCLIP + DehazeSB + DCP")
    print("(WDMamba 因纯PyTorch精度问题排除)")
    print("=" * 60)

    # 测试图像列表: (文件名, 基础名)
    test_images = [
        ('1.png', '1'),
        ('2.png', '2'),
        ('ice1189.jpg', 'ice1189'),
    ]

    all_results = {}

    for img_file, base_name in test_images:
        print(f"\n{'='*60}")
        print(f"处理: {img_file}")
        print(f"{'='*60}")

        # 加载各赛道输出
        track_images = []
        track_names = []

        # HazeCLIP
        hazeclip_path = os.path.join(HAZECLIP_DIR, img_file)
        if os.path.exists(hazeclip_path):
            img = load_image(hazeclip_path)
            if img is not None:
                track_images.append(img)
                track_names.append('HazeCLIP')
                print(f"  [OK] HazeCLIP")
        else:
            print(f"  [--] HazeCLIP: 未找到")

        # DehazeSB
        dehazesb_path = os.path.join(DEHAZESB_DIR, f"{base_name}_dehazed.png")
        if os.path.exists(dehazesb_path):
            img = load_image(dehazesb_path)
            if img is not None:
                track_images.append(img)
                track_names.append('DehazeSB')
                print(f"  [OK] DehazeSB")
        else:
            print(f"  [--] DehazeSB: 未找到")

        # DCP
        diffdehaze_path = os.path.join(DIFFDEHAZE_DIR, f"{base_name}_dcp_dehazed.png")
        if os.path.exists(diffdehaze_path):
            img = load_image(diffdehaze_path)
            if img is not None:
                track_images.append(img)
                track_names.append('DCP')
                print(f"  [OK] DCP")
        else:
            print(f"  [--] DCP: 未找到")

        if len(track_images) < 2:
            print(f"  跳过: 不足2个赛道输出")
            continue

        print(f"\n  参与融合: {track_names}")

        # Level 1: 像素级等权融合
        print(f"\n  --- Level 1: 像素级等权融合 ---")
        l1 = level1_pixel_fusion(track_images)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L1_pixel.png"), l1)
        s1, _ = compute_quality_scores(l1)
        print(f"  分数: {s1:.4f}")

        # Level 2: 特征级自适应融合
        print(f"\n  --- Level 2: 特征级自适应融合 ---")
        l2 = level2_feature_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L2_feature.png"), l2)
        s2, _ = compute_quality_scores(l2)
        print(f"  分数: {s2:.4f}")

        # Level 3: 决策级质量选择融合
        print(f"\n  --- Level 3: 决策级质量选择融合 ---")
        l3 = level3_decision_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L3_decision.png"), l3)
        s3, _ = compute_quality_scores(l3)
        print(f"  分数: {s3:.4f}")

        # Level 4: 级联融合
        print(f"\n  --- Level 4: 级联融合 ---")
        l4 = level4_cascade_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L4_cascade.png"), l4)
        s4, _ = compute_quality_scores(l4)
        print(f"  分数: {s4:.4f}")

        # 最佳结果
        scores = {'L1_pixel': s1, 'L2_feature': s2, 'L3_decision': s3, 'L4_cascade': s4}
        results = {'L1_pixel': l1, 'L2_feature': l2, 'L3_decision': l3, 'L4_cascade': l4}
        best_level = max(scores, key=scores.get)

        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_fusion_final.png"), results[best_level])
        print(f"\n  >>> 最佳策略: {best_level} (分数={scores[best_level]:.4f})")

        print(f"\n  === 汇总 ===")
        for level, score in scores.items():
            print(f"    {level}: {score:.4f}")

        all_results[base_name] = {
            'scores': scores,
            'best': best_level,
            'best_score': scores[best_level],
        }

    print(f"\n{'='*60}")
    print(f"全部完成! 输出目录: {FUSION_OUTPUT_DIR}")
    print(f"{'='*60}")

    # 最终总览
    print("\n=== 总览 ===")
    for name, info in all_results.items():
        print(f"  {name}: 最佳={info['best']} ({info['best_score']:.4f})")


if __name__ == "__main__":
    main()
