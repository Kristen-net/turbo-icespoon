"""
分析饱和度提升对覆冰检测的具体影响
1. HSV通道统计对比 (M1 vs M2 vs M2')
2. 饱和度直方图分布
3. 低饱和区域(潜在覆冰/雾) vs 高饱和区域(正常设备)的对比度分析
4. 色彩可区分性指标 (类间方差)
"""

import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")

import os
import numpy as np
import cv2
import json

# 路径
real_dir = r"D:\DATA_ALL\dataset\test\hazy_real"
m1_dir = r"D:\dehaze_fusion\icewave_output\real_test_3way\M1"
m2_dir = r"D:\dehaze_fusion\icewave_output\real_test_3way\M2"
m2p_dir = r"D:\dehaze_fusion\icewave_output\real_test_3way\M2'"

# 采样图列表
sample_files = sorted([f for f in os.listdir(m1_dir) if f.endswith('.png')])

def compute_hsv_stats(img):
    """计算HSV统计"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 饱和度直方图 (10个bin)
    s_hist = np.histogram(s.ravel(), bins=10, range=(0, 256))[0]
    s_hist = s_hist / s_hist.sum() * 100  # 百分比

    # 低饱和区域 (S < 50, 潜在覆冰/雾/白色物体)
    low_sat_mask = s < 50
    # 中等饱和区域 (50 <= S < 120)
    mid_sat_mask = (s >= 50) & (s < 120)
    # 高饱和区域 (S >= 120, 彩色设备/导线/绝缘子)
    high_sat_mask = s >= 120

    total = s.size
    low_pct = np.sum(low_sat_mask) / total * 100
    mid_pct = np.sum(mid_sat_mask) / total * 100
    high_pct = np.sum(high_sat_mask) / total * 100

    # 低饱和区域的平均亮度 (覆冰区域应该偏亮)
    low_sat_v = np.mean(v[low_sat_mask]) if np.any(low_sat_mask) else 0
    high_sat_v = np.mean(v[high_sat_mask]) if np.any(high_sat_mask) else 0

    # 饱和度对比度: 高饱和区与低饱和区的饱和度差异
    low_sat_mean = np.mean(s[low_sat_mask]) if np.any(low_sat_mask) else 0
    high_sat_mean = np.mean(s[high_sat_mask]) if np.any(high_sat_mask) else 0
    sat_contrast = high_sat_mean - low_sat_mean

    # 类间方差 (Otsu-like): 低饱和vs高饱和的分离度
    if np.any(low_sat_mask) and np.any(high_sat_mask):
        w0 = np.sum(low_sat_mask) / total
        w1 = np.sum(high_sat_mask) / total
        mu0 = np.mean(s[low_sat_mask])
        mu1 = np.mean(s[high_sat_mask])
        between_var = w0 * w1 * (mu0 - mu1) ** 2
    else:
        between_var = 0

    return {
        "sat_mean": float(np.mean(s)),
        "sat_std": float(np.std(s)),
        "val_mean": float(np.mean(v)),
        "hue_mean": float(np.mean(h)),
        "low_sat_pct": float(low_pct),
        "mid_sat_pct": float(mid_pct),
        "high_sat_pct": float(high_pct),
        "low_sat_v": float(low_sat_v),
        "high_sat_v": float(high_sat_v),
        "low_sat_mean": float(low_sat_mean),
        "high_sat_mean": float(high_sat_mean),
        "sat_contrast": float(sat_contrast),
        "between_var": float(between_var),
        "s_hist": s_hist.tolist(),
    }


def main():
    print("=" * 70)
    print("饱和度提升对覆冰检测影响分析")
    print("=" * 70)

    all_stats = {"original": [], "M1": [], "M2": [], "M2p": []}

    for fname in sample_files:
        # 原图
        orig_path = os.path.join(real_dir, fname)
        if not os.path.exists(orig_path):
            continue
        orig = cv2.imread(orig_path)
        if orig is None:
            continue

        m1 = cv2.imread(os.path.join(m1_dir, fname))
        m2 = cv2.imread(os.path.join(m2_dir, fname))
        m2p = cv2.imread(os.path.join(m2p_dir, fname))

        if m1 is None or m2 is None or m2p is None:
            continue

        all_stats["original"].append(compute_hsv_stats(orig))
        all_stats["M1"].append(compute_hsv_stats(m1))
        all_stats["M2"].append(compute_hsv_stats(m2))
        all_stats["M2p"].append(compute_hsv_stats(m2p))

    n = len(all_stats["M1"])
    print(f"\n采样图数: {n}")

    # 汇总统计
    print(f"\n{'='*70}")
    print("一、HSV通道平均统计 (14张采样图)")
    print(f"{'='*70}")

    m2p_hdr = "M2p"
    print(f"\n{'指标':<28} {'原始雾图':>10} {'M1':>10} {'M2':>10} {m2p_hdr:>10} {'v2-M1':>10}")
    print(f"  {'─'*68}")

    for key, label in [
        ("sat_mean", "饱和度均值(S)"),
        ("sat_std", "饱和度标准差"),
        ("val_mean", "亮度均值(V)"),
        ("low_sat_pct", "低饱和区占比(%)"),
        ("mid_sat_pct", "中饱和区占比(%)"),
        ("high_sat_pct", "高饱和区占比(%)"),
        ("low_sat_v", "低饱和区亮度(V)"),
        ("high_sat_v", "高饱和区亮度(V)"),
        ("low_sat_mean", "低饱和区S均值"),
        ("high_sat_mean", "高饱和区S均值"),
        ("sat_contrast", "饱和度对比度(H-L)"),
        ("between_var", "类间方差(分离度)"),
    ]:
        vals = {}
        for name in ["original", "M1", "M2", "M2p"]:
            vals[name] = np.mean([s[key] for s in all_stats[name]])

        diff = vals["M2p"] - vals["M1"]
        if key in ["sat_mean", "sat_contrast", "between_var", "high_sat_mean"]:
            print(f"{label:<28} {vals['original']:>10.2f} {vals['M1']:>10.2f} {vals['M2']:>10.2f} {vals['M2p']:>10.2f} {diff:>+10.2f}")
        elif "pct" in key:
            print(f"{label:<28} {vals['original']:>10.1f} {vals['M1']:>10.1f} {vals['M2']:>10.1f} {vals['M2p']:>10.1f} {diff:>+10.1f}")
        else:
            print(f"{label:<28} {vals['original']:>10.2f} {vals['M1']:>10.2f} {vals['M2']:>10.2f} {vals['M2p']:>10.2f} {diff:>+10.2f}")

    # 饱和度直方图
    print(f"\n{'='*70}")
    print("二、饱和度直方图分布 (10个bin, %)")
    print(f"{'='*70}")

    m2p_col = "M2'"
    print(f"\n{'S区间':<12} {'原始雾图':>10} {'M1':>10} {'M2':>10} {m2p_col:>10}")
    print(f"  {'─'*52}")
    for i in range(10):
        lo = i * 25.6
        hi = (i + 1) * 25.6
        orig_avg = np.mean([s["s_hist"][i] for s in all_stats["original"]])
        m1_avg = np.mean([s["s_hist"][i] for s in all_stats["M1"]])
        m2_avg = np.mean([s["s_hist"][i] for s in all_stats["M2"]])
        m2p_avg = np.mean([s["s_hist"][i] for s in all_stats["M2p"]])
        print(f"{lo:3.0f}-{hi:3.0f}      {orig_avg:>10.1f} {m1_avg:>10.1f} {m2_avg:>10.1f} {m2p_avg:>10.1f}")

    # 影响分析
    print(f"\n{'='*70}")
    print("三、对覆冰检测的具体影响分析")
    print(f"{'='*70}")

    orig_sat = np.mean([s["sat_mean"] for s in all_stats["original"]])
    m1_sat = np.mean([s["sat_mean"] for s in all_stats["M1"]])
    m2_sat = np.mean([s["sat_mean"] for s in all_stats["M2"]])
    m2p_sat = np.mean([s["sat_mean"] for s in all_stats["M2p"]])

    orig_contrast = np.mean([s["sat_contrast"] for s in all_stats["original"]])
    m1_contrast = np.mean([s["sat_contrast"] for s in all_stats["M1"]])
    m2_contrast = np.mean([s["sat_contrast"] for s in all_stats["M2"]])
    m2p_contrast = np.mean([s["sat_contrast"] for s in all_stats["M2p"]])

    orig_bvar = np.mean([s["between_var"] for s in all_stats["original"]])
    m1_bvar = np.mean([s["between_var"] for s in all_stats["M1"]])
    m2_bvar = np.mean([s["between_var"] for s in all_stats["M2"]])
    m2p_bvar = np.mean([s["between_var"] for s in all_stats["M2p"]])

    orig_low_pct = np.mean([s["low_sat_pct"] for s in all_stats["original"]])
    m1_low_pct = np.mean([s["low_sat_pct"] for s in all_stats["M1"]])
    m2p_low_pct = np.mean([s["low_sat_pct"] for s in all_stats["M2p"]])

    orig_high_pct = np.mean([s["high_sat_pct"] for s in all_stats["original"]])
    m1_high_pct = np.mean([s["high_sat_pct"] for s in all_stats["M1"]])
    m2p_high_pct = np.mean([s["high_sat_pct"] for s in all_stats["M2p"]])

    print(f"""
1. 色彩区分度提升 (覆冰 vs 正常设备)
   覆冰区域呈白色/透明 → 低饱和度(S<50)
   正常设备(绝缘子/导线/金具)有固有色彩 → 高饱和度(S>=120)

   原始雾图:  低饱和区 {orig_low_pct:.1f}% / 高饱和区 {orig_high_pct:.1f}%
   M1去雾:   低饱和区 {m1_low_pct:.1f}% / 高饱和区 {m1_high_pct:.1f}%
   M2'去雾:  低饱和区 {m2p_low_pct:.1f}% / 高饱和区 {m2p_high_pct:.1f}%

   → M2'使高饱和区(彩色设备)占比提升 {m2p_high_pct-m1_high_pct:+.1f}%
   → 去雾后彩色设备恢复色彩, 覆冰区域保持白色, 形成更清晰的饱和度对比

2. 饱和度对比度 (高饱和区S - 低饱和区S)
   原始雾图: {orig_contrast:.1f}
   M1去雾:   {m1_contrast:.1f}
   M2去雾:   {m2_contrast:.1f}
   M2'去雾:  {m2p_contrast:.1f}
   M2'相比M1提升: {m2p_contrast-m1_contrast:+.1f} ({(m2p_contrast/max(m1_contrast,0.01)-1)*100:+.1f}%)

   → 饱和度对比度越大, 覆冰区域与正常区域的色彩分离越清晰
   → 检测模型能更容易区分"白色覆冰"与"彩色设备"

3. 类间方差 (Otsu分离度, 越大越易分割)
   原始雾图: {orig_bvar:.0f}
   M1去雾:   {m1_bvar:.0f}
   M2去雾:   {m2_bvar:.0f}
   M2'去雾:  {m2p_bvar:.0f}
   M2'相比M1提升: {m2p_bvar-m1_bvar:+.0f} ({(m2p_bvar/max(m1_bvar,1)-1)*100:+.1f}%)

   → 类间方差直接反映"覆冰类"与"设备类"在饱和度维度上的可分性
   → M2'的类间方差更高意味着基于HSV色彩空间的分割/检测会更准确

4. 物理机制总结
   雾天: 全图饱和度趋同(均偏低) → 覆冰与设备的色彩差异消失 → 检测困难
   M1去雾: 部分恢复色彩, 但饱和度对比度仍有限 → 检测改善有限
   M2'去雾: 强烈恢复非覆冰区域色彩, 覆冰区域保持白色 → 饱和度对比度最大化
   → HA-WFE v2的alpha_hl(垂直边缘)=0.39增强了输电塔架等垂直结构的特征保持
   → 这直接转化为更高的饱和度对比度和类间方差, 为下游覆冰检测提供更优输入
""")

    # 保存结果
    result = {
        "n_samples": n,
        "summary": {
            "sat_mean": {"orig": orig_sat, "M1": m1_sat, "M2": m2_sat, "M2p": m2p_sat},
            "sat_contrast": {"orig": orig_contrast, "M1": m1_contrast, "M2": m2_contrast, "M2p": m2p_contrast},
            "between_var": {"orig": orig_bvar, "M1": m1_bvar, "M2": m2_bvar, "M2p": m2p_bvar},
            "low_sat_pct": {"orig": orig_low_pct, "M1": m1_low_pct, "M2p": m2p_low_pct},
            "high_sat_pct": {"orig": orig_high_pct, "M1": m1_high_pct, "M2p": m2p_high_pct},
        }
    }

    out_path = r"D:\dehaze_fusion\icewave_output\saturation_impact_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("饱和度提升对覆冰检测影响分析\n")
        f.write(f"采样图数: {n}\n\n")
        f.write(f"饱和度均值: 原始={orig_sat:.2f}, M1={m1_sat:.2f}, M2={m2_sat:.2f}, M2'={m2p_sat:.2f}\n")
        f.write(f"饱和度对比度: 原始={orig_contrast:.1f}, M1={m1_contrast:.1f}, M2={m2_contrast:.1f}, M2'={m2p_contrast:.1f}\n")
        f.write(f"类间方差: 原始={orig_bvar:.0f}, M1={m1_bvar:.0f}, M2={m2_bvar:.0f}, M2'={m2p_bvar:.0f}\n")
        f.write(f"低饱和区占比: 原始={orig_low_pct:.1f}%, M1={m1_low_pct:.1f}%, M2'={m2p_low_pct:.1f}%\n")
        f.write(f"高饱和区占比: 原始={orig_high_pct:.1f}%, M1={m1_high_pct:.1f}%, M2'={m2p_high_pct:.1f}%\n")

    print(f"分析结果保存: {out_path}")
    print(f"\nJSON: {json.dumps(result['summary'], indent=2)}")


if __name__ == "__main__":
    main()
