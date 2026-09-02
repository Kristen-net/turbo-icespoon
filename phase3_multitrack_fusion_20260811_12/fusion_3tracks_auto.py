"""
三赛道融合去雾 v3 - 自适应权重融合
修复: 
  1. 以最高分辨率赛道为基准 (不再降级匹配)
  2. 自适应权重: 最优赛道获70%+权重, 不再被差结果稀释
  3. 新增 L5 自适应融合: 质量驱动的动态权重分配
"""
import os
import cv2
import numpy as np

HAZECLIP_DIR = r"D:\dehaze_fusion\HazeCLIP\outputs"
DEHAZESB_DIR = r"D:\dehaze_fusion\DehazeSB\output\dehazesb_results"
DIFFDEHAZE_DIR = r"D:\dehaze_fusion\DiffDehaze-GAN\output\diffdehaze_results"
FUSION_OUTPUT_DIR = r"D:\dehaze_fusion\fusion_3tracks_output"
os.makedirs(FUSION_OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')


def get_image_list():
    images = []
    for ext in IMAGE_EXTS:
        for f in os.listdir(HAZECLIP_DIR):
            if f.lower().endswith(ext):
                images.append(f)
    return sorted(images)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def align_to_highest(images):
    """以最高分辨率图像为基准, 将其他图像放大对齐"""
    # 找到最大分辨率
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    aligned = []
    for img in images:
        if img.shape[:2] != (max_h, max_w):
            img = cv2.resize(img, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        aligned.append(img)
    return aligned


def compute_contrast(img):
    return float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

def compute_saturation(img):
    return float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]))

def compute_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

def compute_dark_channel(img, patch_size=15):
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)

def compute_gradient_mean(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def compute_quality_scores(img):
    contrast = compute_contrast(img)
    saturation = compute_saturation(img)
    edge = compute_edge_density(img)
    dark = compute_dark_channel(img)
    gradient = compute_gradient_mean(img)
    scores = {
        'contrast': contrast / 80.0,
        'saturation': saturation / 255.0,
        'edge': edge / 0.3,
        'dark_channel': 1.0 - min(dark, 1.0),
        'gradient': gradient / 30.0,
    }
    # 暗通道权重提高到0.35 (去雾核心指标), 边缘0.25, 其余各0.1-0.15
    total = (scores['dark_channel'] * 0.35 +
             scores['edge'] * 0.25 +
             scores['contrast'] * 0.15 +
             scores['gradient'] * 0.15 +
             scores['saturation'] * 0.10)
    return total, scores


def weighted_fusion(images, weights):
    """加权像素融合"""
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    result = np.zeros_like(images[0], dtype=np.float32)
    for img, w in zip(images, weights):
        result += img.astype(np.float32) * w
    return np.clip(result, 0, 255).astype(np.uint8)


def level1_equal_fusion(images):
    """L1: 等权平均"""
    return weighted_fusion(images, [1.0/len(images)] * len(images))


def level2_feature_fusion(images, names):
    """L2: 特征自适应权重"""
    contrasts = [compute_contrast(img) for img in images]
    edges = [compute_edge_density(img) for img in images]
    gradients = [compute_gradient_mean(img) for img in images]
    c_sum = sum(contrasts) + 1e-8
    e_sum = sum(edges) + 1e-8
    g_sum = sum(gradients) + 1e-8
    weights = [(contrasts[i]/c_sum + edges[i]/e_sum + gradients[i]/g_sum) / 3.0
               for i in range(len(images))]
    return weighted_fusion(images, weights)


def level3_decision_fusion(images, names):
    """L3: 决策级融合 (高温softmax, 最优赛道获大权重)"""
    scores = []
    for img in images:
        score, _ = compute_quality_scores(img)
        scores.append(score)
    scores_arr = np.array(scores)
    # 温度=8.0 (原来3.0), 让差异更极端
    exp_scores = np.exp(scores_arr * 8.0)
    weights = exp_scores / np.sum(exp_scores)
    return weighted_fusion(images, weights.tolist())


def level4_cascade_fusion(images, names):
    """L4: 级联融合 (DCP→HazeCLIP→DehazeSB 逐级精炼)"""
    cascade_order = []
    for name in ['DCP', 'HazeCLIP', 'DehazeSB']:
        for i, n in enumerate(names):
            if name.lower() in n.lower():
                cascade_order.append(i)
                break
    if len(cascade_order) < 2:
        return weighted_fusion(images, [1.0/len(images)] * len(images))
    result = images[cascade_order[0]].astype(np.float32)
    for level, idx in enumerate(cascade_order[1:], 1):
        w = level / (level + 1)
        result = result * (1 - w) + images[idx].astype(np.float32) * w
    return np.clip(result, 0, 255).astype(np.uint8)


def level5_adaptive_fusion(images, names):
    """L5: 自适应融合 - 最优赛道为主导(70%+), 其余补充"""
    scores = []
    for img in images:
        score, metrics = compute_quality_scores(img)
        scores.append((score, metrics))
    
    # 按质量分数排序
    sorted_idx = sorted(range(len(images)), key=lambda i: scores[i][0], reverse=True)
    
    best_idx = sorted_idx[0]
    best_score = scores[best_idx][0]
    
    # 计算权重: 最优赛道获基础权重 0.7, 其余按质量分配 0.3
    base_weight = 0.7
    remaining = 0.3
    
    weights = [0.0] * len(images)
    weights[best_idx] = base_weight
    
    if len(sorted_idx) > 1:
        # 其余赛道按质量分数分配剩余权重
        other_scores = [scores[i][0] for i in sorted_idx[1:]]
        other_sum = sum(other_scores) + 1e-8
        for i, idx in enumerate(sorted_idx[1:]):
            weights[idx] = remaining * (other_scores[i] / other_sum)
    
    # 如果最优赛道远超其他 (差距>50%), 给予更高权重
    if len(sorted_idx) > 1:
        second_score = scores[sorted_idx[1]][0]
        if best_score > 0 and (best_score - second_score) / max(best_score, 0.01) > 0.3:
            # 差距大, 最优赛道权重提升到 85%
            extra = 0.15
            weights[best_idx] += extra
            for idx in sorted_idx[1:]:
                weights[idx] *= (1 - extra / remaining) if remaining > 0 else 1
    
    result = weighted_fusion(images, weights)
    return result, weights


def main():
    print("=" * 60)
    print("三赛道融合去雾 v3 (自适应权重)")
    print("=" * 60)

    image_files = get_image_list()
    if not image_files:
        print("错误: HazeCLIP 输出目录中没有图像")
        return

    print(f"检测到 {len(image_files)} 张图像")

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]

        print(f"\n{'='*60}")
        print(f"处理: {img_file}")

        track_images = []
        track_names = []

        # HazeCLIP
        hazeclip_path = os.path.join(HAZECLIP_DIR, img_file)
        if os.path.exists(hazeclip_path):
            img = load_image(hazeclip_path)
            if img is not None:
                track_images.append(img)
                track_names.append('HazeCLIP')
                dc = compute_dark_channel(img)
                print(f"  [OK] HazeCLIP  DC={dc:.3f} {img.shape[1]}x{img.shape[0]}")

        # DehazeSB
        for suffix in ['_dehazed.png', '_dehazed.jpg']:
            dehazesb_path = os.path.join(DEHAZESB_DIR, f"{base_name}{suffix}")
            if os.path.exists(dehazesb_path):
                img = load_image(dehazesb_path)
                if img is not None:
                    track_images.append(img)
                    track_names.append('DehazeSB')
                    dc = compute_dark_channel(img)
                    print(f"  [OK] DehazeSB   DC={dc:.3f} {img.shape[1]}x{img.shape[0]}")
                    break

        # DCP
        for suffix in ['_dcp_dehazed.png', '_dcp_dehazed.jpg']:
            dcp_path = os.path.join(DIFFDEHAZE_DIR, f"{base_name}{suffix}")
            if os.path.exists(dcp_path):
                img = load_image(dcp_path)
                if img is not None:
                    track_images.append(img)
                    track_names.append('DCP')
                    dc = compute_dark_channel(img)
                    print(f"  [OK] DCP        DC={dc:.3f} {img.shape[1]}x{img.shape[0]}")
                    break

        if len(track_images) < 2:
            if len(track_images) == 1:
                cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_fusion_final.png"), track_images[0])
                print(f"  只有1个赛道, 直接输出")
            continue

        # 以最高分辨率为基准对齐
        track_images = align_to_highest(track_images)
        print(f"  对齐尺寸: {track_images[0].shape[1]}x{track_images[0].shape[0]}")

        # 各赛道质量评分
        print(f"  质量评分:")
        for i, (img, name) in enumerate(zip(track_images, track_names)):
            score, metrics = compute_quality_scores(img)
            print(f"    {name:12s}: score={score:.4f}  DC={metrics['dark_channel']:.3f}  edge={metrics['edge']:.3f}  grad={metrics['gradient']:.3f}")

        # L1-L4
        l1 = level1_equal_fusion(track_images)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L1_pixel.png"), l1)
        s1, _ = compute_quality_scores(l1)

        l2 = level2_feature_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L2_feature.png"), l2)
        s2, _ = compute_quality_scores(l2)

        l3 = level3_decision_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L3_decision.png"), l3)
        s3, _ = compute_quality_scores(l3)

        l4 = level4_cascade_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L4_cascade.png"), l4)
        s4, _ = compute_quality_scores(l4)

        # L5 自适应融合
        l5, l5_weights = level5_adaptive_fusion(track_images, track_names)
        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_L5_adaptive.png"), l5)
        s5, _ = compute_quality_scores(l5)

        # 选择最佳
        scores = {'L1': s1, 'L2': s2, 'L3': s3, 'L4': s4, 'L5': s5}
        results = {'L1': l1, 'L2': l2, 'L3': l3, 'L4': l4, 'L5': l5}
        best = max(scores, key=scores.get)

        cv2.imwrite(os.path.join(FUSION_OUTPUT_DIR, f"{base_name}_fusion_final.png"), results[best])
        
        print(f"  L5 权重分配: {dict(zip(track_names, [f'{w:.2%}' for w in l5_weights]))}")
        print(f"  最佳: {best} (score={scores[best]:.4f})")
        for level, score in scores.items():
            marker = " <<<" if level == best else ""
            print(f"    {level}: {score:.4f}{marker}")

    print(f"\n完成! 输出: {FUSION_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
