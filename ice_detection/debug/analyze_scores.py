"""分析Mask R-CNN的检测分数分布"""
import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")
from maskrcnn_inference import load_detectron2_model, predict

model = load_detectron2_model()

test_dir = r"D:\dehaze_fusion\my_test\input"
import os
files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.JPG'))])

for fname in files:
    img = cv2.imread(os.path.join(test_dir, fname))
    result = predict(model, img, conf_thresh=0.0)  # 获取所有检测

    scores = result['scores']
    print(f"\n{fname}: {len(scores)} total detections (conf>=0.0)")
    if len(scores) > 0:
        for thresh in [0.5, 0.7, 0.8, 0.9, 0.95]:
            n = np.sum(scores >= thresh)
            print(f"  conf>={thresh}: {n} detections")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score percentiles: 25%={np.percentile(scores,25):.3f}, 50%={np.percentile(scores,50):.3f}, 75%={np.percentile(scores,75):.3f}, 90%={np.percentile(scores,90):.3f}")

        # 显示前10个检测的详细信息
        top_idx = np.argsort(-scores)[:10]
        print(f"  Top 10 detections:")
        for i in top_idx:
            x1, y1, x2, y2 = result['boxes'][i].astype(int)
            print(f"    score={scores[i]:.4f} box=[{x1},{y1},{x2},{y2}] size={x2-x1}x{y2-y1}")
