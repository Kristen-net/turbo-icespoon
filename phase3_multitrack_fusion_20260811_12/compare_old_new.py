"""
新旧融合结果质量对比
旧: fusion_output (四赛道: HazeCLIP + DehazeSB + WDMamba + DCP)
新: fusion_3tracks_output (三赛道: HazeCLIP + DehazeSB + DCP)
"""
import os
import cv2
import numpy as np
import json

OLD_DIR = r"D:\dehaze_fusion\fusion_output"
NEW_DIR = r"D:\dehaze_fusion\fusion_3tracks_output"
HAZECLIP_DIR = r"D:\dehaze_fusion\HazeCLIP\images"

TARGET_SIZE = 512

def load_image(path, target_size=TARGET_SIZE):
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

def compute_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))

def compute_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

def compute_dark_channel(img, patch_size=7):
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val, kernel)
    return float(np.mean(dark) / 255.0)

def compute_gradient_mean(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

def compute_psnr(img1, img2):
    """计算 PSNR (峰值信噪比)"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0**2 / mse))

def compute_all_metrics(img):
    return {
        'contrast': compute_contrast(img),
        'saturation': compute_saturation(img),
        'edge': compute_edge_density(img),
        'dark_channel': compute_dark_channel(img),
        'gradient': compute_gradient_mean(img),
    }

def compute_quality_score(metrics):
    """计算综合质量分数"""
    scores = {
        'contrast': metrics['contrast'] / 80.0,
        'saturation': metrics['saturation'] / 255.0,
        'edge': metrics['edge'] / 0.3,
        'dark_channel': 1.0 - min(metrics['dark_channel'], 1.0),
        'gradient': metrics['gradient'] / 30.0,
    }
    total = (
        scores['contrast'] * 0.2 +
        scores['saturation'] * 0.15 +
        scores['edge'] * 0.25 +
        scores['dark_channel'] * 0.2 +
        scores['gradient'] * 0.2
    )
    return total

# 测试图像
test_images = [
    ('1.png', '1'),
    ('2.png', '2'),
]

# 融合层级
levels = ['L1_pixel', 'L2_feature', 'L3_decision', 'L4_cascade', 'fusion_final']

print("=" * 80)
print("新旧融合结果质量对比")
print("旧: 四赛道 (HazeCLIP + DehazeSB + WDMamba + DCP)")
print("新: 三赛道 (HazeCLIP + DehazeSB + DCP, 排除 WDMamba)")
print("=" * 80)

all_results = []

for img_file, base_name in test_images:
    print(f"\n{'='*80}")
    print(f"图像: {img_file}")
    print(f"{'='*80}")
    
    # 加载原图
    haze_path = os.path.join(HAZECLIP_DIR, img_file)
    haze_img = load_image(haze_path)
    haze_metrics = compute_all_metrics(haze_img)
    haze_score = compute_quality_score(haze_metrics)
    
    print(f"\n  原图 (有雾): 综合分数={haze_score:.4f}")
    print(f"    暗通道={haze_metrics['dark_channel']:.4f}, 饱和度={haze_metrics['saturation']:.1f}, "
          f"对比度={haze_metrics['contrast']:.2f}, 边缘={haze_metrics['edge']:.4f}, 梯度={haze_metrics['gradient']:.2f}")
    
    for level in levels:
        old_path = os.path.join(OLD_DIR, f"{base_name}_{level}.png")
        new_path = os.path.join(NEW_DIR, f"{base_name}_{level}.png")
        
        old_img = load_image(old_path) if os.path.exists(old_path) else None
        new_img = load_image(new_path) if os.path.exists(new_path) else None
        
        if old_img is None and new_img is None:
            continue
        
        print(f"\n  --- {level} ---")
        
        if old_img is not None:
            old_m = compute_all_metrics(old_img)
            old_s = compute_quality_score(old_m)
            print(f"  旧(4赛道): 分数={old_s:.4f}  暗通道={old_m['dark_channel']:.4f}  饱和度={old_m['saturation']:.1f}  "
                  f"对比度={old_m['contrast']:.2f}  边缘={old_m['edge']:.4f}  梯度={old_m['gradient']:.2f}")
        else:
            old_s = None
            old_m = None
            print(f"  旧(4赛道): 不存在")
        
        if new_img is not None:
            new_m = compute_all_metrics(new_img)
            new_s = compute_quality_score(new_m)
            print(f"  新(3赛道): 分数={new_s:.4f}  暗通道={new_m['dark_channel']:.4f}  饱和度={new_m['saturation']:.1f}  "
                  f"对比度={new_m['contrast']:.2f}  边缘={new_m['edge']:.4f}  梯度={new_m['gradient']:.2f}")
        else:
            new_s = None
            new_m = None
            print(f"  新(3赛道): 不存在")
        
        if old_s is not None and new_s is not None:
            delta = new_s - old_s
            delta_pct = (delta / old_s) * 100 if old_s > 0 else 0
            psnr = compute_psnr(old_img, new_img)
            
            # 各指标变化
            d_dark = (old_m['dark_channel'] - new_m['dark_channel']) / old_m['dark_channel'] * 100 if old_m['dark_channel'] > 0 else 0
            d_sat = (new_m['saturation'] - old_m['saturation']) / old_m['saturation'] * 100 if old_m['saturation'] > 0 else 0
            d_con = (new_m['contrast'] - old_m['contrast']) / old_m['contrast'] * 100 if old_m['contrast'] > 0 else 0
            d_edge = (new_m['edge'] - old_m['edge']) / old_m['edge'] * 100 if old_m['edge'] > 0 else 0
            d_grad = (new_m['gradient'] - old_m['gradient']) / old_m['gradient'] * 100 if old_m['gradient'] > 0 else 0
            
            print(f"  变化: 分数 {delta:+.4f} ({delta_pct:+.1f}%)  PSNR={psnr:.2f}dB")
            print(f"        暗通道↓{d_dark:+.1f}%  饱和度↑{d_sat:+.1f}%  对比度↑{d_con:+.1f}%  边缘↑{d_edge:+.1f}%  梯度↑{d_grad:+.1f}%")
            
            all_results.append({
                'image': img_file,
                'level': level,
                'old_score': old_s,
                'new_score': new_s,
                'delta': delta,
                'delta_pct': delta_pct,
                'psnr': psnr,
                'old_dark': old_m['dark_channel'],
                'new_dark': new_m['dark_channel'],
                'old_saturation': old_m['saturation'],
                'new_saturation': new_m['saturation'],
                'old_contrast': old_m['contrast'],
                'new_contrast': new_m['contrast'],
                'old_edge': old_m['edge'],
                'new_edge': new_m['edge'],
                'old_gradient': old_m['gradient'],
                'new_gradient': new_m['gradient'],
            })

# 最终结果对比 (fusion_final)
print(f"\n{'='*80}")
print("最终融合结果 (fusion_final) 对比汇总")
print(f"{'='*80}")
print(f"{'图像':<10} {'旧(4赛道)':>10} {'新(3赛道)':>10} {'变化':>10} {'变化%':>10} {'PSNR':>10}")
print("-" * 60)

for r in all_results:
    if r['level'] == 'fusion_final':
        print(f"{r['image']:<10} {r['old_score']:>10.4f} {r['new_score']:>10.4f} {r['delta']:>+10.4f} {r['delta_pct']:>+9.1f}% {r['psnr']:>9.2f}dB")

# 各层级平均对比
print(f"\n{'='*80}")
print("各层级平均质量分数对比")
print(f"{'='*80}")
print(f"{'层级':<15} {'旧(4赛道)平均':>15} {'新(3赛道)平均':>15} {'变化':>10} {'变化%':>10}")
print("-" * 65)

for level in levels:
    level_results = [r for r in all_results if r['level'] == level]
    if len(level_results) < 2:
        continue
    old_avg = np.mean([r['old_score'] for r in level_results])
    new_avg = np.mean([r['new_score'] for r in level_results])
    delta = new_avg - old_avg
    delta_pct = (delta / old_avg) * 100 if old_avg > 0 else 0
    print(f"{level:<15} {old_avg:>15.4f} {new_avg:>15.4f} {delta:>+10.4f} {delta_pct:>+9.1f}%")

# 指标维度平均对比
print(f"\n{'='*80}")
print("最终结果各指标维度平均变化 (%)")
print(f"{'='*80}")

final_results = [r for r in all_results if r['level'] == 'fusion_final']
if final_results:
    avg_dark_old = np.mean([r['old_dark'] for r in final_results])
    avg_dark_new = np.mean([r['new_dark'] for r in final_results])
    avg_sat_old = np.mean([r['old_saturation'] for r in final_results])
    avg_sat_new = np.mean([r['new_saturation'] for r in final_results])
    avg_con_old = np.mean([r['old_contrast'] for r in final_results])
    avg_con_new = np.mean([r['new_contrast'] for r in final_results])
    avg_edge_old = np.mean([r['old_edge'] for r in final_results])
    avg_edge_new = np.mean([r['new_edge'] for r in final_results])
    avg_grad_old = np.mean([r['old_gradient'] for r in final_results])
    avg_grad_new = np.mean([r['new_gradient'] for r in final_results])
    
    print(f"  暗通道: {avg_dark_old:.4f} → {avg_dark_new:.4f} (下降 {(avg_dark_old-avg_dark_new)/avg_dark_old*100:.1f}%)")
    print(f"  饱和度: {avg_sat_old:.1f} → {avg_sat_new:.1f} (提升 {(avg_sat_new-avg_sat_old)/avg_sat_old*100:.1f}%)")
    print(f"  对比度: {avg_con_old:.2f} → {avg_con_new:.2f} (提升 {(avg_con_new-avg_con_old)/avg_con_old*100:.1f}%)")
    print(f"  边缘密度: {avg_edge_old:.4f} → {avg_edge_new:.4f} (提升 {(avg_edge_new-avg_edge_old)/avg_edge_old*100:.1f}%)")
    print(f"  梯度: {avg_grad_old:.2f} → {avg_grad_new:.2f} (提升 {(avg_grad_new-avg_grad_old)/avg_grad_old*100:.1f}%)")

# 保存 JSON
report_path = os.path.join(NEW_DIR, "comparison_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n详细报告已保存: {report_path}")
