"""
端到端去雾-覆冰检测管线 (三赛道融合版)
输入: 有雾图像
输出: YOLOv8 检测 + 覆冰覆盖率估计
"""
import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 路径配置
FUSION_DIR = r"D:\dehaze_fusion\fusion_3tracks_output"
HAZECLIP_DIR = r"D:\dehaze_fusion\HazeCLIP"
OUTPUT_DIR = r"D:\dehaze_fusion\end2end_3tracks_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLOv8 模型
model = YOLO('yolov8n.pt')

# 中文字体
font_path = r"C:\Windows\Fonts\msyh.ttc"
zh_font = FontProperties(fname=font_path, size=10)
zh_title = FontProperties(fname=font_path, size=12, weight='bold')


def compute_dark_channel(img, patch_size=15):
    """计算暗通道 (归一化 0-1)"""
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)


def compute_saturation(img):
    """计算饱和度均值 (0-255)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def estimate_ice_coverage(img):
    """
    估计覆冰覆盖率
    基于白色/透明区域比例 + 边缘结构分析
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 白色/冰区域: 低饱和度 + 高明度
    low_sat = hsv[:, :, 1] < 50       # 低饱和
    high_val = hsv[:, :, 2] > 180     # 高明度
    white_mask = low_sat & high_val
    
    # 冰的纹理: 用边缘密度过滤 (冰有纹理, 天空无纹理)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = cv2.GaussianBlur((edges > 0).astype(np.float32), (21, 21), 0)
    has_texture = edge_density > 0.02  # 有一定边缘的区域
    
    # 冰区域 = 白色且有纹理
    ice_mask = white_mask & has_texture
    
    # 只考虑上半部分 (输电线和塔通常在画面上方)
    h, w = img.shape[:2]
    upper_mask = np.zeros((h, w), dtype=bool)
    upper_mask[:int(h*0.85), :] = True
    
    ice_mask = ice_mask & upper_mask
    coverage = float(np.sum(ice_mask) / np.sum(upper_mask))
    
    return coverage, ice_mask


def run_detection(img_path, label):
    """运行 YOLOv8 检测"""
    results = model(img_path, conf=0.25, verbose=False)
    img = cv2.imread(img_path)
    
    annotated = results[0].plot()
    
    num_objects = len(results[0].boxes)
    classes = results[0].boxes.cls.cpu().numpy() if num_objects > 0 else []
    
    # 类别名称
    cls_names = [results[0].names[int(c)] for c in classes]
    
    return annotated, num_objects, cls_names


def process_image(img_name, base_name):
    """处理单组图像"""
    print(f"\n{'='*60}")
    print(f"处理: {img_name}")
    print(f"{'='*60}")
    
    # 有雾原图 (从 HazeCLIP images 取)
    haze_path = os.path.join(HAZECLIP_DIR, "images", img_name)
    if not os.path.exists(haze_path):
        # 尝试 jpg 扩展名
        haze_path = os.path.join(HAZECLIP_DIR, "images", f"{base_name}.jpg")
    haze_img = cv2.imread(haze_path)
    
    # 融合去雾结果
    fusion_path = os.path.join(FUSION_DIR, f"{base_name}_fusion_final.png")
    dehaze_img = cv2.imread(fusion_path)
    
    if haze_img is None or dehaze_img is None:
        print(f"  跳过: 图像读取失败")
        return None
    
    # 统一尺寸 (用融合图像尺寸)
    dh, dw = dehaze_img.shape[:2]
    haze_resized = cv2.resize(haze_img, (dw, dh))
    
    # YOLO 检测
    print("  有雾图 YOLO 检测...")
    haze_det, haze_count, haze_classes = run_detection(haze_path, "hazy")
    print(f"    检测到 {haze_count} 个目标: {haze_classes[:5]}")
    
    # 保存融合图为临时文件用于检测
    tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{base_name}_fusion.png")
    cv2.imwrite(tmp_path, dehaze_img)
    
    print("  去雾图 YOLO 检测...")
    dehaze_det, dehaze_count, dehaze_classes = run_detection(tmp_path, "dehazed")
    print(f"    检测到 {dehaze_count} 个目标: {dehaze_classes[:5]}")
    
    # 覆冰覆盖率估计
    haze_cov, haze_mask = estimate_ice_coverage(haze_resized)
    dehaze_cov, dehaze_mask = estimate_ice_coverage(dehaze_img)
    print(f"  有雾覆冰覆盖率: {haze_cov:.4f} ({haze_cov*100:.2f}%)")
    print(f"  去雾覆冰覆盖率: {dehaze_cov:.4f} ({dehaze_cov*100:.2f}%)")
    
    # 质量指标
    haze_dark = compute_dark_channel(haze_resized)
    dehaze_dark = compute_dark_channel(dehaze_img)
    haze_sat = compute_saturation(haze_resized)
    dehaze_sat = compute_saturation(dehaze_img)
    
    # 保存检测结果图
    haze_det_path = os.path.join(OUTPUT_DIR, f"{base_name}_haze_det.png")
    dehaze_det_path = os.path.join(OUTPUT_DIR, f"{base_name}_dehaze_det.png")
    cv2.imwrite(haze_det_path, haze_det)
    cv2.imwrite(dehaze_det_path, dehaze_det)
    
    # 生成对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行: 有雾
    axes[0, 0].imshow(cv2.cvtColor(haze_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'有雾原图\n暗通道={haze_dark:.3f}, 饱和度={haze_sat:.1f}', fontproperties=zh_title)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(haze_det, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'有雾 YOLOv8 检测\n{haze_count} 个目标', fontproperties=zh_title)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(haze_mask, cmap='Blues')
    axes[0, 2].set_title(f'有雾覆冰估计\n覆盖率={haze_cov*100:.1f}%', fontproperties=zh_title)
    axes[0, 2].axis('off')
    
    # 第二行: 去雾
    axes[1, 0].imshow(cv2.cvtColor(dehaze_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'三赛道融合去雾\n暗通道={dehaze_dark:.3f}, 饱和度={dehaze_sat:.1f}', fontproperties=zh_title)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(dehaze_det, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'去雾 YOLOv8 检测\n{dehaze_count} 个目标', fontproperties=zh_title)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(dehaze_mask, cmap='Blues')
    axes[1, 2].set_title(f'去雾覆冰估计\n覆盖率={dehaze_cov*100:.1f}%', fontproperties=zh_title)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
    plt.savefig(comp_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  对比图已保存: {comp_path}")
    
    # 清理临时文件
    os.remove(tmp_path)
    
    return {
        'name': img_name,
        'haze_dark': haze_dark,
        'dehaze_dark': dehaze_dark,
        'dark_improvement': (haze_dark - dehaze_dark) / haze_dark * 100,
        'haze_saturation': haze_sat,
        'dehaze_saturation': dehaze_sat,
        'sat_improvement': (dehaze_sat - haze_sat) / haze_sat * 100,
        'haze_detection_count': haze_count,
        'dehaze_detection_count': dehaze_count,
        'detection_improvement': dehaze_count - haze_count,
        'haze_classes': haze_classes,
        'dehaze_classes': dehaze_classes,
        'haze_ice_coverage': haze_cov,
        'dehaze_ice_coverage': dehaze_cov,
        'coverage_change_pct': (dehaze_cov - haze_cov) / haze_cov * 100,
    }


def main():
    print("=" * 60)
    print("端到端去雾-覆冰检测管线 (三赛道融合版)")
    print("=" * 60)
    
    test_images = [
        ('1.png', '1'),
        ('2.png', '2'),
        ('ice1189.jpg', 'ice1189'),
    ]
    
    results = []
    for img_file, base_name in test_images:
        r = process_image(img_file, base_name)
        if r:
            results.append(r)
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "end2end_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 文本报告
    txt_path = os.path.join(OUTPUT_DIR, "end2end_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("端到端去雾-覆冰检测报告 (三赛道融合版)\n")
        f.write("融合赛道: HazeCLIP + DehazeSB + DCP\n")
        f.write("=" * 60 + "\n\n")
        
        for r in results:
            f.write(f"图像: {r['name']}\n")
            f.write(f"  暗通道: {r['haze_dark']:.4f} → {r['dehaze_dark']:.4f} ({r['dark_improvement']:+.1f}%)\n")
            f.write(f"  饱和度: {r['haze_saturation']:.1f} → {r['dehaze_saturation']:.1f} ({r['sat_improvement']:+.1f}%)\n")
            f.write(f"  YOLO检测: {r['haze_detection_count']} → {r['dehaze_detection_count']} 个目标 ({r['detection_improvement']:+d})\n")
            f.write(f"  覆冰覆盖率: {r['haze_ice_coverage']*100:.2f}% → {r['dehaze_ice_coverage']*100:.2f}% ({r['coverage_change_pct']:+.1f}%)\n")
            f.write("\n")
    
    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    print("\n=== 结果总览 ===")
    for r in results:
        print(f"  {r['name']}:")
        print(f"    YOLO: {r['haze_detection_count']}→{r['dehaze_detection_count']} ({r['detection_improvement']:+d})")
        print(f"    覆冰率: {r['haze_ice_coverage']*100:.1f}%→{r['dehaze_ice_coverage']*100:.1f}%")
        print(f"    暗通道↓: {r['dark_improvement']:.1f}%, 饱和度↑: {r['sat_improvement']:.1f}%")


if __name__ == "__main__":
    main()
