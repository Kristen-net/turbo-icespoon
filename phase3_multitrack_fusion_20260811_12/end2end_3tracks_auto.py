"""
端到端去雾-覆冰检测 - 自动检测所有融合结果
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

FUSION_DIR = r"D:\dehaze_fusion\fusion_3tracks_output"
HAZECLIP_IMAGES = r"D:\dehaze_fusion\HazeCLIP\images"
OUTPUT_DIR = r"D:\dehaze_fusion\end2end_3tracks_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(r'C:\Users\2457025871\yolov8n.pt')  # 使用本地权重，避免GitHub下载

font_path = r"C:\Windows\Fonts\msyh.ttc"
zh_font = FontProperties(fname=font_path, size=10)
zh_title = FontProperties(fname=font_path, size=12, weight='bold')


def get_fusion_images():
    """自动检测所有 fusion_final 图像"""
    images = []
    for f in os.listdir(FUSION_DIR):
        if f.endswith('_fusion_final.png'):
            images.append(f)
    return sorted(images)


def compute_dark_channel(img, patch_size=15):
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)

def compute_saturation(img):
    return float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]))


def estimate_ice_coverage(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_sat = hsv[:, :, 1] < 50
    high_val = hsv[:, :, 2] > 180
    white_mask = low_sat & high_val
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = cv2.GaussianBlur((edges > 0).astype(np.float32), (21, 21), 0)
    has_texture = edge_density > 0.02
    ice_mask = white_mask & has_texture
    h, w = img.shape[:2]
    upper_mask = np.zeros((h, w), dtype=bool)
    upper_mask[:int(h*0.85), :] = True
    ice_mask = ice_mask & upper_mask
    coverage = float(np.sum(ice_mask) / np.sum(upper_mask))
    return coverage, ice_mask


def run_detection(img_path):
    results = model(img_path, conf=0.10, verbose=False)  # 降低置信度到0.10提高召回率
    annotated = results[0].plot()
    num_objects = len(results[0].boxes)
    classes = results[0].boxes.cls.cpu().numpy() if num_objects > 0 else []
    cls_names = [results[0].names[int(c)] for c in classes]
    return annotated, num_objects, cls_names


def process_image(fusion_file):
    base_name = fusion_file.replace('_fusion_final.png', '')
    
    # 找原始有雾图
    haze_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        p = os.path.join(HAZECLIP_IMAGES, f"{base_name}{ext}")
        if os.path.exists(p):
            haze_path = p
            break
    # 也尝试 .rf. 格式
    if haze_path is None:
        for f in os.listdir(HAZECLIP_IMAGES):
            if f.startswith(base_name) and not f.endswith('_dehazed') and not '_wdmamba' in f:
                haze_path = os.path.join(HAZECLIP_IMAGES, f)
                break
    
    if haze_path is None:
        print(f"  跳过: 找不到原图 {base_name}")
        return None

    haze_img = cv2.imread(haze_path)
    fusion_path = os.path.join(FUSION_DIR, fusion_file)
    dehaze_img = cv2.imread(fusion_path)

    if haze_img is None or dehaze_img is None:
        print(f"  跳过: 图像读取失败")
        return None

    dh, dw = dehaze_img.shape[:2]
    haze_resized = cv2.resize(haze_img, (dw, dh))

    # YOLO 检测
    print(f"  有雾 YOLO...")
    haze_det, haze_count, haze_classes = run_detection(haze_path)
    print(f"    {haze_count} 个目标: {haze_classes[:5]}")

    tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{base_name}.png")
    cv2.imwrite(tmp_path, dehaze_img)
    print(f"  去雾 YOLO...")
    dehaze_det, dehaze_count, dehaze_classes = run_detection(tmp_path)
    print(f"    {dehaze_count} 个目标: {dehaze_classes[:5]}")
    os.remove(tmp_path)

    # 覆冰估计
    haze_cov, haze_mask = estimate_ice_coverage(haze_resized)
    dehaze_cov, dehaze_mask = estimate_ice_coverage(dehaze_img)
    print(f"  覆冰率: {haze_cov*100:.1f}% → {dehaze_cov*100:.1f}%")

    # 质量指标
    haze_dark = compute_dark_channel(haze_resized)
    dehaze_dark = compute_dark_channel(dehaze_img)
    haze_sat = compute_saturation(haze_resized)
    dehaze_sat = compute_saturation(dehaze_img)

    # 保存检测图
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_haze_det.png"), haze_det)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_dehaze_det.png"), dehaze_det)

    # 对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(haze_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'有雾原图\n暗通道={haze_dark:.3f}, 饱和度={haze_sat:.1f}', fontproperties=zh_title)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(haze_det, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'有雾 YOLOv8\n{haze_count} 个目标', fontproperties=zh_title)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(haze_mask, cmap='Blues')
    axes[0, 2].set_title(f'有雾覆冰\n{haze_cov*100:.1f}%', fontproperties=zh_title)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(dehaze_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'三赛道融合去雾\n暗通道={dehaze_dark:.3f}, 饱和度={dehaze_sat:.1f}', fontproperties=zh_title)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(dehaze_det, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'去雾 YOLOv8\n{dehaze_count} 个目标', fontproperties=zh_title)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(dehaze_mask, cmap='Blues')
    axes[1, 2].set_title(f'去雾覆冰\n{dehaze_cov*100:.1f}%', fontproperties=zh_title)
    axes[1, 2].axis('off')

    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
    plt.savefig(comp_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'name': base_name,
        'haze_dark': haze_dark, 'dehaze_dark': dehaze_dark,
        'dark_improvement': (haze_dark - dehaze_dark) / haze_dark * 100 if haze_dark > 0 else 0,
        'haze_saturation': haze_sat, 'dehaze_saturation': dehaze_sat,
        'sat_improvement': (dehaze_sat - haze_sat) / haze_sat * 100 if haze_sat > 0 else 0,
        'haze_detection': haze_count, 'dehaze_detection': dehaze_count,
        'detection_delta': dehaze_count - haze_count,
        'haze_ice_coverage': haze_cov, 'dehaze_ice_coverage': dehaze_cov,
        'coverage_change': (dehaze_cov - haze_cov) / haze_cov * 100 if haze_cov > 0 else 0,
    }


def main():
    print("=" * 60)
    print("端到端去雾-覆冰检测 (自动检测)")
    print("=" * 60)

    fusion_files = get_fusion_images()
    if not fusion_files:
        print("错误: 融合目录中没有 _fusion_final.png 图像")
        return

    print(f"检测到 {len(fusion_files)} 张融合图\n")

    results = []
    for f in fusion_files:
        print(f"\n{'='*60}")
        print(f"处理: {f}")
        r = process_image(f)
        if r:
            results.append(r)

    # 保存报告
    if results:
        with open(os.path.join(OUTPUT_DIR, "end2end_report.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("结果总览:")
        for r in results:
            print(f"  {r['name']}:")
            print(f"    YOLO: {r['haze_detection']}→{r['dehaze_detection']} ({r['detection_delta']:+d})")
            print(f"    覆冰率: {r['haze_ice_coverage']*100:.1f}%→{r['dehaze_ice_coverage']*100:.1f}%")
            print(f"    暗通道↓: {r['dark_improvement']:.1f}%  饱和度↑: {r['sat_improvement']:.1f}%")

    print(f"\n完成! 输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
