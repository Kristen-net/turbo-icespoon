"""
端到端去雾+覆冰检测流水线
==========================
去雾引擎 -> YOLOv8 覆冰检测 -> 结果可视化

使用方法:
    python end2end_pipeline.py --input hazy_power_line.png --output-dir output/
    python end2end_pipeline.py --input hazy_power_line.png --yolo-weights yolov8n.pt --fusion-level 4
"""

import os
import sys
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


# ============================================================
# 去雾引擎（简化版，从 fusion_inference 导入）
# ============================================================

# 将 fusion_inference.py 所在目录加入路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from fusion_inference import (
        load_image, save_image, tile_inference,
        HazeCLIPEngine, DiffDehazeEngine, WDMambaEngine,
        FusionEngine, LightweightCNN, LightUNet, WaveletMambaNet
    )
    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"[警告] 无法导入 fusion_inference: {e}")
    print("[警告] 将使用简化去雾模块")
    FUSION_AVAILABLE = False


# ============================================================
# 覆冰检测引擎 (YOLOv8)
# ============================================================

class IcingDetector:
    """输电线路覆冰检测引擎"""

    def __init__(self, weights_path=None, device='cuda', conf_threshold=0.3):
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.weights_path = weights_path
        self._load_model()

    def _load_model(self):
        """加载 YOLOv8 模型"""
        print("[覆冰检测] 加载 YOLOv8 模型...")

        # 尝试加载自定义训练的覆冰检测权重
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
                print(f"  [覆冰检测] 自定义模型加载成功: {self.weights_path}")
                return
            except Exception as e:
                print(f"  [覆冰检测] 自定义模型加载失败: {e}")

        # 尝试加载预训练 YOLOv8
        try:
            from ultralytics import YOLO
            # 下载预训练权重（国内镜像）
            yolo_path = self._download_yolo_weights()
            self.model = YOLO(yolo_path)
            print(f"  [覆冰检测] 预训练 YOLOv8 加载成功: {yolo_path}")
            print(f"  [覆冰检测] 注意: 使用预训练 COCO 权重，建议替换为覆冰检测专用权重")
        except ImportError:
            print(f"  [覆冰检测] ultralytics 未安装，使用简化检测器")
            self.model = SimpleEdgeDetector()
        except Exception as e:
            print(f"  [覆冰检测] YOLOv8 加载失败: {e}")
            self.model = SimpleEdgeDetector()

    def _download_yolo_weights(self):
        """下载 YOLOv8 权重（国内镜像）"""
        weights_dir = os.path.join(os.path.expanduser("~"), ".cache", "yolov8")
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "yolov8n.pt")

        if os.path.exists(weights_path):
            return weights_path

        print("  [覆冰检测] 下载 YOLOv8n 权重...")
        # 国内镜像源
        urls = [
            "https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        ]

        import urllib.request
        for url in urls:
            try:
                print(f"    尝试: {url[:60]}...")
                urllib.request.urlretrieve(url, weights_path)
                print(f"    下载成功: {weights_path}")
                return weights_path
            except Exception as e:
                print(f"    失败: {e}")
                continue

        print("  [覆冰检测] 所有权重源下载失败")
        return None

    def detect(self, img_tensor_or_path):
        """执行覆冰检测"""
        if self.model is None:
            return [], []

        start = time.time()

        # 如果输入是 tensor，转为 numpy
        if isinstance(img_tensor_or_path, torch.Tensor):
            img = img_tensor_or_path.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(img_tensor_or_path, cv2.IMREAD_COLOR)

        # YOLOv8 推理
        if hasattr(self.model, 'predict'):
            results = self.model.predict(img, conf=self.conf_threshold, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0]),
                        'class': int(box.cls[0]),
                        'class_name': r.names[int(box.cls[0])]
                    })
        else:
            # 简化检测器
            detections = self.model.detect(img)

        elapsed = time.time() - start
        return detections, elapsed

    def visualize(self, img_tensor_or_path, detections, output_path):
        """可视化检测结果"""
        if isinstance(img_tensor_or_path, torch.Tensor):
            img = img_tensor_or_path.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(img_tensor_or_path, cv2.IMREAD_COLOR)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            cls_name = det.get('class_name', f"class_{det['class']}")

            # 绘制边界框
            color = (0, 255, 0) if 'ice' in cls_name.lower() else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, img)
        return img


class SimpleEdgeDetector:
    """简化边缘检测器（无 YOLOv8 时的替代方案）"""

    def detect(self, img):
        """使用 Canny 边缘检测模拟覆冰检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    'bbox': [float(x), float(y), float(x + w), float(y + h)],
                    'confidence': min(area / 1000, 0.95),
                    'class': 0,
                    'class_name': 'potential_ice'
                })

        return detections


# ============================================================
# 端到端流水线
# ============================================================

class EndToEndPipeline:
    """去雾 + 覆冰检测端到端流水线"""

    def __init__(self, hazeclip_weights=None, diffdehaze_weights=None,
                 wdmamba_weights=None, yolo_weights=None,
                 fusion_level=4, device='cuda'):
        self.device = device
        self.fusion_level = fusion_level

        # 初始化去雾引擎
        self.dehaze_engine = None
        if FUSION_AVAILABLE:
            print("=" * 50)
            print("初始化端到端流水线")
            print("=" * 50)

            hazeclip = HazeCLIPEngine(hazeclip_weights or '', device)
            diffdehaze = DiffDehazeEngine(diffdehaze_weights, device=device, num_steps=15)
            wdmamba = WDMambaEngine(wdmamba_weights or '', device)
            self.dehaze_engine = FusionEngine(hazeclip, diffdehaze, wdmamba, device)
            print()

        # 初始化覆冰检测引擎
        self.icing_detector = IcingDetector(yolo_weights, device)

        print()
        print("[流水线] 初始化完成")
        print(f"  去雾引擎: {'融合 Level ' + str(fusion_level) if self.dehaze_engine else '未启用'}")
        print(f"  覆冰检测: {'YOLOv8' if hasattr(self.icing_detector.model, 'predict') else '简化检测器'}")
        print()

    def run(self, input_path, output_dir='output'):
        """执行端到端推理"""
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("端到端去雾 + 覆冰检测")
        print("=" * 60)
        print(f"输入: {input_path}")
        print(f"输出目录: {output_dir}")
        print()

        # Step 1: 读取图片
        print("[Step 1] 读取雾天图片...")
        img = load_image(input_path)
        print(f"  尺寸: {img.shape}")
        print(f"  显存占用: {img.element_size() * img.nelement() / 1024**2:.1f} MB")
        print()

        # 保存原始雾天图
        hazy_path = os.path.join(output_dir, "01_hazy_input.png")
        save_image(img, hazy_path)

        # Step 2: 去雾
        print("[Step 2] 执行去雾推理...")
        dehaze_start = time.time()

        if self.dehaze_engine:
            dehazed, dehaze_time, dehaze_vram = self.dehaze_engine.infer(
                img, fusion_level=self.fusion_level, fp16=True
            )
        else:
            # 简化去雾（直方图均衡化）
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            dehazed = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            dehaze_time = time.time() - dehaze_start
            dehaze_vram = 0.1

        print(f"  去雾耗时: {dehaze_time:.3f}s")
        print(f"  去雾显存: {dehaze_vram:.2f} GB")
        print()

        # 保存去雾结果
        dehazed_path = os.path.join(output_dir, "02_dehazed.png")
        save_image(dehazed, dehazed_path)

        # Step 3: 覆冰检测
        print("[Step 3] 执行覆冰检测...")
        detections, detect_time = self.icing_detector.detect(dehazed)
        print(f"  检测耗时: {detect_time:.3f}s")
        print(f"  检测到目标: {len(detections)} 个")
        for det in detections:
            print(f"    - {det['class_name']}: conf={det['confidence']:.3f}, bbox={det['bbox']}")
        print()

        # Step 4: 可视化
        print("[Step 4] 生成可视化结果...")
        vis_path = os.path.join(output_dir, "03_detection_result.png")
        self.icing_detector.visualize(dehazed, detections, vis_path)

        # 同时在雾天原图上检测（对比）
        hazy_detections, _ = self.icing_detector.detect(hazy_path)
        hazy_vis_path = os.path.join(output_dir, "04_hazy_detection.png")
        self.icing_detector.visualize(hazy_path, hazy_detections, hazy_vis_path)

        # Step 5: 生成报告
        print("[Step 5] 生成分析报告...")
        report = {
            'input': input_path,
            'image_size': list(img.shape),
            'dehazing': {
                'method': f'fusion_level_{self.fusion_level}',
                'time_seconds': round(dehaze_time, 3),
                'vram_gb': round(dehaze_vram, 2),
                'output': dehazed_path,
            },
            'icing_detection': {
                'method': 'YOLOv8' if hasattr(self.icing_detector.model, 'predict') else 'edge_detector',
                'time_seconds': round(detect_time, 3),
                'num_detections': len(detections),
                'detections': detections,
            },
            'comparison': {
                'dehazed_detections': len(detections),
                'hazy_detections': len(hazy_detections),
                'improvement': len(detections) - len(hazy_detections),
            },
            'total_time': round(dehaze_time + detect_time, 3),
        }

        report_path = os.path.join(output_dir, "05_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print("端到端流水线完成")
        print("=" * 60)
        print(f"总耗时: {dehaze_time + detect_time:.3f}s")
        print(f"  去雾: {dehaze_time:.3f}s (显存 {dehaze_vram:.2f} GB)")
        print(f"  检测: {detect_time:.3f}s")
        print(f"  检测目标: {len(detections)} 个 (去雾后) vs {len(hazy_detections)} 个 (雾天)")
        print()
        print("输出文件:")
        print(f"  01_hazy_input.png     - 雾天原图")
        print(f"  02_dehazed.png        - 去雾结果")
        print(f"  03_detection_result.png - 去雾后覆冰检测")
        print(f"  04_hazy_detection.png - 雾天覆冰检测（对比）")
        print(f"  05_report.json        - 分析报告")
        print("=" * 60)

        return report


def main():
    parser = argparse.ArgumentParser(description="端到端去雾+覆冰检测流水线")
    parser.add_argument('--input', type=str, required=True, help='输入雾天图片路径')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--fusion-level', type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--hazeclip-weights', type=str, default='')
    parser.add_argument('--diffdehaze-weights', type=str, default='')
    parser.add_argument('--wdmamba-weights', type=str, default='')
    parser.add_argument('--yolo-weights', type=str, default=None, help='YOLOv8 权重路径')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='检测置信度阈值')

    args = parser.parse_args()

    pipeline = EndToEndPipeline(
        hazeclip_weights=args.hazeclip_weights,
        diffdehaze_weights=args.diffdehaze_weights,
        wdmamba_weights=args.wdmamba_weights,
        yolo_weights=args.yolo_weights,
        fusion_level=args.fusion_level,
    )

    pipeline.run(args.input, args.output_dir)


if __name__ == '__main__':
    main()
