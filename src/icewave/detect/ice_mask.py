"""覆冰掩码生成 (规则式, 从 phase6/dehaze_inference.py 抽取为可复用模块).

注意: 本模块是**伪标签生成器**, 仅用于 (a) 训练数据预标注, (b) 无人工
标注时的兜底推理输出。论文实验指标必须基于 ice_mask_human/ 人工标注计算
(见 icewave.eval.metrics.ice_region_metrics), 以打破"规则造标签→规则评效果"
的循环论证。
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


def generate_ice_mask(img_bgr: np.ndarray,
                      yolo_detections: Optional[Sequence[dict]] = None) -> np.ndarray:
    """输电线路覆冰掩码 (白色×纹理×走廊 三重 AND 约束).

    参数
    ----
    img_bgr : BGR 图像。
    yolo_detections : 可选, 检测器输出 [{'class': str, 'bbox': (x1,y1,x2,y2)}, ...];
        提供 power_line/insulator 框时用作走廊约束, 否则退化为 Hough 直线走廊。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # Step 1: 颜色筛选 — 低饱和 + 较高亮度 (覆冰白色特征)
    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    _, low_sat = cv2.threshold(s_blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v_mean = np.mean(v)
    _, bright = cv2.threshold(v, int(v_mean * 0.7), 255, cv2.THRESH_BINARY)
    ice_color = cv2.bitwise_and(low_sat, bright)

    # Step 2: 走廊 (检测框 或 Hough 直线)
    corridor = np.zeros((h, w), dtype=np.uint8)
    use_det = yolo_detections is not None and len(yolo_detections) > 0
    if use_det:
        for det in yolo_detections:
            if det.get("class") in ("power_line", "insulator", "target"):
                x1, y1, x2, y2 = det["bbox"]
                pad = 15
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                cv2.rectangle(corridor, (x1, y1), (x2, y2), 255, -1)
    else:
        edges = cv2.Canny(gray, 20, 80)
        min_len = max(50, min(h, w) // 4)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=min_len, maxLineGap=20)
        line_mask = np.zeros((h, w), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                pts = line.reshape(-1) if line.ndim > 1 else line
                if len(pts) >= 4:
                    x1, y1, x2, y2 = (int(pts[0]), int(pts[1]),
                                      int(pts[2]), int(pts[3]))
                    length = np.hypot(x2 - x1, y2 - y1)
                    if length >= min_len:
                        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 20)
        corridor = cv2.dilate(
            line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
        if np.sum(corridor > 0) / (h * w) > 0.4:
            corridor = np.zeros((h, w), dtype=np.uint8)

    # Step 3: 中等纹理 (排除水印锐边)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad_mag / (grad_mag.max() + 1e-6) * 255).astype(np.uint8)
    grad_blur = cv2.GaussianBlur(grad_norm, (5, 5), 0)
    _, texture_mask = cv2.threshold(grad_blur, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, high_grad = cv2.threshold(grad_blur, 180, 255, cv2.THRESH_BINARY)
    texture_mask = cv2.subtract(texture_mask, high_grad)

    # Step 4: 组合 AND
    if np.sum(corridor) > 0:
        combined = cv2.bitwise_and(cv2.bitwise_and(ice_color, texture_mask),
                                   corridor)
        if np.sum(combined) < 100:
            combined = cv2.bitwise_and(ice_color, corridor)
    else:
        combined = np.zeros((h, w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Step 5: 轮廓形状过滤
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros((h, w), dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect > 8.0 or aspect < 0.125:
            continue
        cv2.drawContours(final, [c], -1, 255, -1)

    return final


def pseudo_ice_mask_simple(img_bgr: np.ndarray) -> np.ndarray:
    """训练集伪标签生成器 (从 ice_detection/algorithms/ice_mask_generator.py 移植).

    HSV 低饱和 Otsu + 亮度过滤 + 边缘密度 + 形态学清理, 用于 ITL 训练预标注。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    _, low_sat = cv2.threshold(s_blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    _, bright = cv2.threshold(v_blur, int(np.mean(v)), 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.boxFilter((edges > 0).astype(np.float32),
                                 ddepth=-1, ksize=(15, 15))
    _, edge_mask = cv2.threshold((edge_density * 255).astype(np.uint8),
                                 10, 255, cv2.THRESH_BINARY)

    ice_mask = cv2.bitwise_and(low_sat, bright)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ice_mask, connectivity=8)
    cleaned = np.zeros_like(ice_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 200:
            cleaned[labels == i] = 255
    return cleaned
