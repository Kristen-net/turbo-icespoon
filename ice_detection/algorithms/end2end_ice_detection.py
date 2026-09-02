"""
端到端冰检测管线
==================
功能:
  1. 对有雾原图和融合去雾图分别进行 YOLOv8 目标检测
  2. 自定义覆冰检测模块 (霍夫变换 + 纹理分析 + 颜色分析)
  3. 对比去雾前后的检测效果 (检测置信度、边缘密度、覆冰评分)
  4. 生成可视化对比结果和定量指标

适用于 RTX 5060 8GB 显存
"""
import os
import sys
import time
import json
import cv2
import torch
import numpy as np
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
HAZY_DIR = r"D:\dehaze_fusion\HazeCLIP\images"          # 有雾原图
DEHAZED_DIR = r"D:\dehaze_fusion\fusion_output"           # 融合去雾结果
OUTPUT_DIR = r"D:\dehaze_fusion\end2end_output"           # 端到端输出
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试图像对 (有雾原图名 → 去雾结果名)
IMAGE_PAIRS = [
    ("1.png", "1_fusion_final.png"),
    ("2.png", "2_fusion_final.png"),
]

# 额外冰检测测试图
ICE_TEST_IMAGES = [
    (r"D:\dehaze_fusion\DehazeSB\test_data\ice1189.jpg",
     r"D:\dehaze_fusion\HazeCLIP\outputs\ice1189_jpg.rf.fd764f5157fdc38ee4912f150e03300a.jpg"),
]

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# 模块 1: YOLOv8 目标检测
# ============================================================
class YOLOv8Detector:
    """YOLOv8 通用目标检测器"""

    def __init__(self, model_name="yolov8n.pt"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 YOLOv8 模型 (首次使用自动下载)"""
        from ultralytics import YOLO
        print(f"\n[YOLOv8] 加载模型: {self.model_name}")

        # 检查本地是否已有模型文件
        local_path = os.path.join(os.path.expanduser("~"), ".cache", "torch", "ultralytics", self.model_name)
        if os.path.exists(local_path):
            print(f"  本地缓存: {local_path}")

        try:
            self.model = YOLO(self.model_name)
            print(f"  模型加载成功")
        except Exception as e:
            print(f"  模型加载失败: {e}")
            print(f"  尝试手动下载...")
            raise

    def detect(self, img, conf_threshold=0.25):
        """
        对图像进行目标检测
        返回: detections list [{class, confidence, bbox}]
        """
        results = self.model(img, conf=conf_threshold, verbose=False, device=device)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

        return detections, results

    def draw_detections(self, img, detections):
        """在图像上绘制检测框"""
        annotated = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class']

            # 根据置信度选择颜色
            if conf > 0.7:
                color = (0, 255, 0)    # 绿色 - 高置信度
            elif conf > 0.4:
                color = (0, 255, 255)  # 黄色 - 中置信度
            else:
                color = (0, 128, 255)  # 橙色 - 低置信度

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated


# ============================================================
# 模块 2: 自定义覆冰检测
# ============================================================
class IceDetector:
    """
    输电线路覆冰检测模块
    基于: 霍夫直线检测 + LBP 纹理分析 + 颜色/边缘特征
    """

    def __init__(self):
        self.line_params = {
            'rho': 1,
            'theta': np.pi / 180,
            'threshold': 50,
            'minLineLength': 50,
            'maxLineGap': 20
        }

    def detect_power_lines(self, img):
        """使用霍夫变换检测电力线"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, **self.line_params)

        line_mask = np.zeros(gray.shape, dtype=np.uint8)
        line_count = 0

        if lines is not None:
            lines = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in lines:
                cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
                line_count += 1

        return line_mask, edges, line_count

    def compute_lbp_texture(self, gray, radius=3, n_points=24):
        """计算 LBP (Local Binary Pattern) 纹理特征"""
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        # 计算纹理能量 (方差)
        texture_energy = np.std(lbp.astype(np.float64))
        return lbp, texture_energy

    def detect_ice_regions(self, img, line_mask):
        """
        检测电力线上的覆冰区域
        覆冰特征: 高亮度 + 低饱和度 + 高纹理粗糙度
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, s, v = cv2.split(hsv)

        # 覆冰颜色特征: 高亮度 (V > 120), 低饱和度 (S < 80)
        ice_color_mask = (v > 120) & (s < 80)

        # 边缘密度 (覆冰表面粗糙, 边缘密集)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0) / 255.0
        ice_edge_mask = edge_density > 0.15

        # 纹理分析: 局部标准差
        local_std = cv2.GaussianBlur(gray.astype(np.float32), (21, 21), 0)
        local_std = np.abs(gray.astype(np.float32) - local_std)
        ice_texture_mask = local_std > 15

        # 综合覆冰掩码: 颜色 + 边缘 + 纹理, 且在电力线附近
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        line_dilated = cv2.dilate(line_mask, kernel, iterations=2)

        ice_mask = (ice_color_mask & ice_edge_mask & ice_texture_mask).astype(np.uint8) * 255
        ice_mask = cv2.bitwise_and(ice_mask, line_dilated)

        # 形态学后处理
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        return ice_mask

    def compute_ice_metrics(self, img, ice_mask, line_mask):
        """计算覆冰指标"""
        # 覆冰覆盖率 (覆冰面积 / 电力线面积)
        line_pixels = max(np.count_nonzero(line_mask), 1)
        ice_pixels = np.count_nonzero(ice_mask)
        ice_coverage = ice_pixels / line_pixels

        # 覆冰严重度评分 (0-100)
        if ice_coverage < 0.05:
            severity = "轻微"
            score = ice_coverage * 100
        elif ice_coverage < 0.15:
            severity = "轻度"
            score = ice_coverage * 300
        elif ice_coverage < 0.30:
            severity = "中度"
            score = ice_coverage * 200 + 15
        elif ice_coverage < 0.50:
            severity = "严重"
            score = ice_coverage * 150 + 25
        else:
            severity = "极严重"
            score = min(ice_coverage * 100 + 50, 100)

        # 平均亮度 (覆冰区域)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if ice_pixels > 0:
            mean_brightness = float(gray[ice_mask > 0].mean())
        else:
            mean_brightness = 0.0

        return {
            'ice_coverage': float(ice_coverage),
            'ice_pixels': int(ice_pixels),
            'line_pixels': int(line_pixels),
            'severity': severity,
            'severity_score': float(min(score, 100)),
            'mean_brightness': mean_brightness
        }

    def detect(self, img):
        """
        完整覆冰检测流程
        返回: {line_mask, ice_mask, edges, metrics}
        """
        # 1. 电力线检测
        line_mask, edges, line_count = self.detect_power_lines(img)

        # 2. 覆冰区域检测
        ice_mask = self.detect_ice_regions(img, line_mask)

        # 3. 覆冰指标计算
        metrics = self.compute_ice_metrics(img, ice_mask, line_mask)
        metrics['line_count'] = line_count

        # 4. LBP 纹理分析
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            _, texture_energy = self.compute_lbp_texture(gray)
            metrics['texture_energy'] = float(texture_energy)
        except ImportError:
            metrics['texture_energy'] = -1.0

        return {
            'line_mask': line_mask,
            'ice_mask': ice_mask,
            'edges': edges,
            'metrics': metrics
        }

    def visualize(self, img, result, alpha=0.4):
        """生成覆冰检测可视化"""
        vis = img.copy()
        ice_mask = result['ice_mask']
        line_mask = result['line_mask']
        metrics = result['metrics']

        # 覆冰区域: 红色半透明叠加
        ice_overlay = vis.copy()
        ice_overlay[ice_mask > 0] = [0, 0, 255]  # 红色
        vis = cv2.addWeighted(ice_overlay, alpha, vis, 1 - alpha, 0)

        # 电力线: 蓝色叠加
        line_overlay = vis.copy()
        line_overlay[line_mask > 0] = [255, 100, 0]  # 橙蓝色
        vis = cv2.addWeighted(line_overlay, alpha * 0.7, vis, 1 - alpha * 0.7, 0)

        # 绘制覆冰区域轮廓
        contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        # 信息面板
        info_lines = [
            f"Line Count: {metrics.get('line_count', 0)}",
            f"Ice Coverage: {metrics['ice_coverage']:.4f}",
            f"Severity: {metrics['severity']} ({metrics['severity_score']:.1f}/100)",
            f"Texture Energy: {metrics.get('texture_energy', -1):.2f}",
        ]
        panel_h = len(info_lines) * 25 + 15
        cv2.rectangle(vis, (5, 5), (320, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (320, panel_h), (255, 255, 255), 1)
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return vis


# ============================================================
# 模块 3: 图像质量评估
# ============================================================
def compute_image_quality(img):
    """计算图像质量指标"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 对比度 (标准差)
    contrast = float(gray.std())

    # 平均亮度
    brightness = float(gray.mean())

    # 饱和度
    saturation = float(hsv[:, :, 1].mean())

    # 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (edges.shape[0] * edges.shape[1])

    # 暗通道
    min_val = cv2.erode(img.min(axis=2).astype(np.uint8), np.ones((7, 7), np.uint8))
    dark_channel = float(min_val.mean()) / 255.0

    # 平均梯度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = float(np.mean(np.sqrt(gx**2 + gy**2)))

    return {
        'contrast': contrast,
        'brightness': brightness,
        'saturation': saturation,
        'edge_density': edge_density,
        'dark_channel': dark_channel,
        'gradient': gradient
    }


# ============================================================
# 模块 4: 端到端管线
# ============================================================
def run_pipeline():
    """运行端到端冰检测管线"""

    print("=" * 70)
    print("端到端冰检测管线")
    print("去雾前 → 去雾后 对比分析")
    print("=" * 70)

    # 初始化检测器
    print("\n[1/5] 初始化检测器...")
    # 使用本地下载的 yolov8n.pt
    yolo_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n.pt"  # 回退到自动下载
    yolo_detector = YOLOv8Detector(model_name=yolo_model_path)
    ice_detector = IceDetector()
    print("  检测器初始化完成")

    all_results = []

    # 处理标准图像对
    test_pairs = []
    for hazy_name, dehazed_name in IMAGE_PAIRS:
        hazy_path = os.path.join(HAZY_DIR, hazy_name)
        dehazed_path = os.path.join(DEHAZED_DIR, dehazed_name)
        if os.path.exists(hazy_path) and os.path.exists(dehazed_path):
            test_pairs.append((hazy_path, dehazed_path, hazy_name))
        else:
            print(f"  跳过: {hazy_name} (文件不存在)")

    # 处理额外冰检测图
    for hazy_path, dehazed_path in ICE_TEST_IMAGES:
        if os.path.exists(hazy_path) and os.path.exists(dehazed_path):
            test_pairs.append((hazy_path, dehazed_path, os.path.basename(hazy_path)))
        else:
            print(f"  跳过: {os.path.basename(hazy_path)} (文件不存在)")

    print(f"\n[2/5] 共 {len(test_pairs)} 组图像对")

    for idx, (hazy_path, dehazed_path, img_name) in enumerate(test_pairs):
        print(f"\n{'='*60}")
        print(f"[3/5] 处理 [{idx+1}/{len(test_pairs)}]: {img_name}")
        print(f"  有雾: {hazy_path}")
        print(f"  去雾: {dehazed_path}")

        # 读取图像
        hazy_img = cv2.imread(hazy_path)
        dehazed_img = cv2.imread(dehazed_path)

        if hazy_img is None or dehazed_img is None:
            print(f"  错误: 无法读取图像")
            continue

        # 确保尺寸一致
        h, w = hazy_img.shape[:2]
        if dehazed_img.shape[:2] != (h, w):
            dehazed_img = cv2.resize(dehazed_img, (w, h))

        print(f"  图像尺寸: {w}x{h}")

        # ---- 2.1 YOLOv8 目标检测 ----
        print(f"\n  --- YOLOv8 目标检测 ---")
        t0 = time.time()
        hazy_dets, hazy_raw = yolo_detector.detect(hazy_img, conf_threshold=0.15)
        t1 = time.time()
        dehazed_dets, dehazed_raw = yolo_detector.detect(dehazed_img, conf_threshold=0.15)
        t2 = time.time()

        print(f"  有雾图: {len(hazy_dets)} 个目标, 耗时 {t1-t0:.3f}s")
        print(f"  去雾图: {len(dehazed_dets)} 个目标, 耗时 {t2-t1:.3f}s")

        # 打印检测结果
        if hazy_dets:
            print(f"  有雾检测:")
            for d in hazy_dets[:5]:
                print(f"    {d['class']}: {d['confidence']:.3f}")
        if dehazed_dets:
            print(f"  去雾检测:")
            for d in dehazed_dets[:5]:
                print(f"    {d['class']}: {d['confidence']:.3f}")

        # ---- 2.2 覆冰检测 ----
        print(f"\n  --- 覆冰检测 ---")
        t0 = time.time()
        hazy_ice = ice_detector.detect(hazy_img)
        t1 = time.time()
        dehazed_ice = ice_detector.detect(dehazed_img)
        t2 = time.time()

        print(f"  有雾覆冰: {hazy_ice['metrics']['ice_coverage']:.4f}, "
              f"严重度: {hazy_ice['metrics']['severity']}, 耗时 {t1-t0:.3f}s")
        print(f"  去雾覆冰: {dehazed_ice['metrics']['ice_coverage']:.4f}, "
              f"严重度: {dehazed_ice['metrics']['severity']}, 耗时 {t2-t1:.3f}s")

        # ---- 2.3 图像质量评估 ----
        hazy_quality = compute_image_quality(hazy_img)
        dehazed_quality = compute_image_quality(dehazed_img)

        print(f"\n  --- 质量指标对比 ---")
        print(f"  {'指标':<15} {'有雾':>10} {'去雾':>10} {'提升':>10}")
        for key in ['contrast', 'brightness', 'saturation', 'edge_density', 'gradient']:
            h_val = hazy_quality[key]
            d_val = dehazed_quality[key]
            change = ((d_val - h_val) / max(abs(h_val), 1e-6)) * 100
            print(f"  {key:<15} {h_val:>10.2f} {d_val:>10.2f} {change:>+9.1f}%")
        print(f"  {'dark_channel':<15} {hazy_quality['dark_channel']:>10.4f} "
              f"{dehazed_quality['dark_channel']:>10.4f}")

        # ---- 2.4 生成可视化 ----
        print(f"\n  --- 生成可视化 ---")
        base_name = os.path.splitext(img_name)[0]

        # YOLOv8 检测可视化
        hazy_yolo_vis = yolo_detector.draw_detections(hazy_img, hazy_dets)
        dehazed_yolo_vis = yolo_detector.draw_detections(dehazed_img, dehazed_dets)

        # 覆冰检测可视化
        hazy_ice_vis = ice_detector.visualize(hazy_img, hazy_ice)
        dehazed_ice_vis = ice_detector.visualize(dehazed_img, dehazed_ice)

        # 拼接对比图 (2x3 网格)
        # 第一行: 有雾原图 | 有雾YOLO检测 | 有雾覆冰检测
        # 第二行: 去雾图 | 去雾YOLO检测 | 去雾覆冰检测
        target_h = 400
        scale = target_h / h
        target_w = int(w * scale)

        hazy_r = cv2.resize(hazy_img, (target_w, target_h))
        dehazed_r = cv2.resize(dehazed_img, (target_w, target_h))
        hazy_yolo_r = cv2.resize(hazy_yolo_vis, (target_w, target_h))
        dehazed_yolo_r = cv2.resize(dehazed_yolo_vis, (target_w, target_h))
        hazy_ice_r = cv2.resize(hazy_ice_vis, (target_w, target_h))
        dehazed_ice_r = cv2.resize(dehazed_ice_vis, (target_w, target_h))

        # 添加标题
        def add_title(img, title):
            titled = np.zeros((30 + img.shape[0], img.shape[1], 3), dtype=np.uint8)
            titled[30:] = img
            cv2.putText(titled, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            return titled

        hazy_r = add_title(hazy_r, "Hazy (Original)")
        dehazed_r = add_title(dehazed_r, "Dehazed (Fusion)")
        hazy_yolo_r = add_title(hazy_yolo_r, f"Hazy YOLO ({len(hazy_dets)} dets)")
        dehazed_yolo_r = add_title(dehazed_yolo_r, f"Dehazed YOLO ({len(dehazed_dets)} dets)")
        hazy_ice_r = add_title(hazy_ice_r, f"Hazy Ice ({hazy_ice['metrics']['severity']})")
        dehazed_ice_r = add_title(dehazed_ice_r, f"Dehazed Ice ({dehazed_ice['metrics']['severity']})")

        # 水平拼接
        row1 = np.hstack([hazy_r, hazy_yolo_r, hazy_ice_r])
        row2 = np.hstack([dehazed_r, dehazed_yolo_r, dehazed_ice_r])
        comparison = np.vstack([row1, row2])

        # 添加分隔线
        cv2.line(comparison, (0, row1.shape[0]), (comparison.shape[1], row1.shape[0]),
                (0, 255, 0), 2)

        comparison_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"  对比图已保存: {base_name}_comparison.png ({comparison.shape[1]}x{comparison.shape[0]})")

        # 保存单独结果
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_hazy_yolo.png"), hazy_yolo_vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_dehazed_yolo.png"), dehazed_yolo_vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_hazy_ice.png"), hazy_ice_vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_dehazed_ice.png"), dehazed_ice_vis)

        # ---- 2.5 汇总结果 ----
        result = {
            'image': img_name,
            'size': f"{w}x{h}",
            'yolo': {
                'hazy': {
                    'num_detections': len(hazy_dets),
                    'avg_confidence': float(np.mean([d['confidence'] for d in hazy_dets])) if hazy_dets else 0,
                    'max_confidence': float(max([d['confidence'] for d in hazy_dets])) if hazy_dets else 0,
                    'classes': list(set(d['class'] for d in hazy_dets)),
                    'time': float(t1 - t0)
                },
                'dehazed': {
                    'num_detections': len(dehazed_dets),
                    'avg_confidence': float(np.mean([d['confidence'] for d in dehazed_dets])) if dehazed_dets else 0,
                    'max_confidence': float(max([d['confidence'] for d in dehazed_dets])) if dehazed_dets else 0,
                    'classes': list(set(d['class'] for d in dehazed_dets)),
                    'time': float(t2 - t1)
                }
            },
            'ice_detection': {
                'hazy': hazy_ice['metrics'],
                'dehazed': dehazed_ice['metrics']
            },
            'image_quality': {
                'hazy': hazy_quality,
                'dehazed': dehazed_quality
            }
        }
        all_results.append(result)

        # 打印检测提升摘要
        hazy_ndet = len(hazy_dets)
        dehazed_ndet = len(dehazed_dets)
        det_change = dehazed_ndet - hazy_ndet

        hazy_conf = result['yolo']['hazy']['avg_confidence']
        dehazed_conf = result['yolo']['dehazed']['avg_confidence']
        conf_change = ((dehazed_conf - hazy_conf) / max(hazy_conf, 1e-6)) * 100

        print(f"\n  --- 检测提升摘要 ---")
        print(f"  目标数: {hazy_ndet} → {dehazed_ndet} ({'+' if det_change>=0 else ''}{det_change})")
        print(f"  平均置信度: {hazy_conf:.4f} → {dehazed_conf:.4f} ({'+' if conf_change>=0 else ''}{conf_change:.1f}%)")
        print(f"  覆冰覆盖率: {hazy_ice['metrics']['ice_coverage']:.4f} → {dehazed_ice['metrics']['ice_coverage']:.4f}")
        print(f"  覆冰严重度: {hazy_ice['metrics']['severity']} → {dehazed_ice['metrics']['severity']}")
        print(f"  对比度: {hazy_quality['contrast']:.2f} → {dehazed_quality['contrast']:.2f}")
        print(f"  边缘密度: {hazy_quality['edge_density']:.6f} → {dehazed_quality['edge_density']:.6f}")

    # ---- 3. 汇总报告 ----
    print(f"\n{'='*70}")
    print(f"[4/5] 生成汇总报告...")

    report_path = os.path.join(OUTPUT_DIR, "end2end_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  JSON 报告: {report_path}")

    # 文本报告
    txt_path = os.path.join(OUTPUT_DIR, "end2end_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("端到端冰检测管线 - 实验报告\n")
        f.write(f"去雾方法: 四层融合 (像素级 + 特征级 + 决策级 + 级联)\n")
        f.write(f"检测方法: YOLOv8n + 自定义覆冰检测\n")
        f.write(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("=" * 70 + "\n\n")

        for r in all_results:
            f.write(f"图像: {r['image']} ({r['size']})\n")
            f.write("-" * 50 + "\n")

            f.write(f"  YOLOv8 目标检测:\n")
            f.write(f"    有雾: {r['yolo']['hazy']['num_detections']} 目标, "
                   f"平均置信度 {r['yolo']['hazy']['avg_confidence']:.4f}, "
                   f"耗时 {r['yolo']['hazy']['time']:.3f}s\n")
            f.write(f"    去雾: {r['yolo']['dehazed']['num_detections']} 目标, "
                   f"平均置信度 {r['yolo']['dehazed']['avg_confidence']:.4f}, "
                   f"耗时 {r['yolo']['dehazed']['time']:.3f}s\n")

            det_imp = r['yolo']['dehazed']['num_detections'] - r['yolo']['hazy']['num_detections']
            conf_h = r['yolo']['hazy']['avg_confidence']
            conf_d = r['yolo']['dehazed']['avg_confidence']
            conf_imp = ((conf_d - conf_h) / max(conf_h, 1e-6)) * 100
            f.write(f"    提升: 目标数 {'+' if det_imp>=0 else ''}{det_imp}, "
                   f"置信度 {'+' if conf_imp>=0 else ''}{conf_imp:.1f}%\n\n")

            f.write(f"  覆冰检测:\n")
            f.write(f"    有雾: 覆盖率 {r['ice_detection']['hazy']['ice_coverage']:.4f}, "
                   f"严重度 {r['ice_detection']['hazy']['severity']} "
                   f"({r['ice_detection']['hazy']['severity_score']:.1f}/100)\n")
            f.write(f"    去雾: 覆盖率 {r['ice_detection']['dehazed']['ice_coverage']:.4f}, "
                   f"严重度 {r['ice_detection']['dehazed']['severity']} "
                   f"({r['ice_detection']['dehazed']['severity_score']:.1f}/100)\n\n")

            f.write(f"  图像质量:\n")
            for key in ['contrast', 'brightness', 'saturation', 'edge_density', 'gradient', 'dark_channel']:
                h_val = r['image_quality']['hazy'][key]
                d_val = r['image_quality']['dehazed'][key]
                f.write(f"    {key}: {h_val:.4f} → {d_val:.4f}\n")
            f.write("\n" + "=" * 70 + "\n\n")

    print(f"  文本报告: {txt_path}")

    # ---- 4. 总结 ----
    print(f"\n[5/5] 端到端管线完成!")
    print(f"  处理图像对: {len(all_results)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  对比图: {len(all_results)} 张")
    print(f"  报告: end2end_report.json, end2end_report.txt")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_pipeline()
