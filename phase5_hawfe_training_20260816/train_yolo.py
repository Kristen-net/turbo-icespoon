"""
YOLOv8 自定义模型训练脚本
使用自动标注的数据集训练输电线路检测模型
"""
import os
import sys
from ultralytics import YOLO

# ==================== 配置 ====================
# 数据集配置
DATA_YAML = r"D:\dehaze_fusion\yolo_dataset\data.yaml"

# 预训练权重 (从COCO迁移学习)
PRETRAINED = r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d\yolov8n.pt"

# 训练参数
EPOCHS = 50
IMGSZ = 640
BATCH = 8
DEVICE = "0"  # GPU

# 输出目录
OUTPUT_DIR = r"D:\dehaze_fusion\yolo_train_output"


def main():
    print("=" * 60)
    print("YOLOv8 自定义模型训练")
    print("=" * 60)
    
    # 检查数据集
    if not os.path.exists(DATA_YAML):
        print(f"错误: 找不到 {DATA_YAML}")
        print("请先运行 auto_label.py 生成数据集")
        return
    
    # 检查预训练权重
    if not os.path.exists(PRETRAINED):
        print(f"错误: 找不到预训练权重 {PRETRAINED}")
        return
    
    # 检查训练数据
    train_dir = r"D:\dehaze_fusion\yolo_dataset\images\train"
    val_dir = r"D:\dehaze_fusion\yolo_dataset\images\val"
    train_imgs = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(train_dir) else []
    val_imgs = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(val_dir) else []
    
    print(f"训练图片: {len(train_imgs)}")
    print(f"验证图片: {len(val_imgs)}")
    
    if len(train_imgs) < 10:
        print("警告: 训练图片太少, 建议至少100张以上")
    
    # 统计标注
    train_labels = r"D:\dehaze_fusion\yolo_dataset\labels\train"
    total_labels = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    if os.path.exists(train_labels):
        for f in os.listdir(train_labels):
            if f.endswith('.txt'):
                with open(os.path.join(train_labels, f)) as fp:
                    for line in fp:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            class_counts[cid] = class_counts.get(cid, 0) + 1
                            total_labels += 1
    
    print(f"总标注框: {total_labels}")
    class_names = ['insulator', 'power_line', 'ice', 'tower']
    for cid, count in class_counts.items():
        print(f"  {class_names[cid]}: {count}")
    
    if total_labels == 0:
        print("\n错误: 没有标注数据! 请检查 auto_label.py 输出")
        return
    
    # 加载模型
    print(f"\n加载预训练权重: {PRETRAINED}")
    model = YOLO(PRETRAINED)
    
    # 训练
    print(f"\n开始训练...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image size: {IMGSZ}")
    print(f"  Batch size: {BATCH}")
    print(f"  Device: {DEVICE}")
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=OUTPUT_DIR,
        name="power_line_yolo",
        exist_ok=True,
        patience=15,  # 早停
        save=True,
        plots=True,
        # RTX 5060 优化
        amp=False,  # 禁用AMP避免GitHub下载
        workers=4,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    # 训练完成
    best_model_path = os.path.join(OUTPUT_DIR, "power_line_yolo", "weights", "best.pt")
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  最佳权重: {best_model_path}")
    print(f"  结果目录: {os.path.join(OUTPUT_DIR, 'power_line_yolo')}")
    print(f"{'='*60}")
    
    # 快速验证
    if os.path.exists(best_model_path):
        print(f"\n用验证集快速测试...")
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH)
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
