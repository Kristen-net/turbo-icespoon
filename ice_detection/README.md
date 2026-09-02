# 覆冰检测模块 (Ice Detection)

本目录包含 IceWave-DehazeFormer 项目中所有与覆冰检测相关的代码、配置、调试工具和结果报告。
项目主体为图像去雾，覆冰检测作为独立附属模块，可单独使用或与去雾管线集成。

## 目录结构

```
ice_detection/
├── algorithms/               # 检测算法
│   ├── dcp_ice.py            # 暗通道先验 + HSV 覆冰检测
│   ├── end2end_ice_detection.py  # 端到端覆冰检测系统 (25KB, 核心文件)
│   ├── hazeclip_ice.py       # HazeCLIP 覆冰检测
│   ├── ice_mask_generator.py # 覆冰掩码生成器
│   ├── maskrcnn_inference.py # Mask R-CNN R50-FPN 推理 (detectron2→torchvision)
│   └── yolo_auto_label.py    # YOLO 自动标注 (HSV + Canny + Hough + 纹理)
├── training/                  # 检测训练
│   ├── train_yolo.py         # YOLOv8 训练脚本
│   └── yolo_training_results.csv  # YOLO 训练结果数据
├── configs/                   # 检测配置
│   ├── data.yaml             # YOLO 数据集配置
│   ├── yolo_dataset_manifest.txt  # 数据集文件清单
│   └── yolo_train_args.yaml  # YOLO 训练超参数
├── debug/                     # Mask R-CNN 调试与分析工具
│   ├── analyze_ckpt.py       # checkpoint 结构分析
│   ├── analyze_ckpt2.py      # checkpoint 结构分析 v2
│   ├── analyze_scores.py     # 检测分数分析
│   ├── compare_arch.py       # detectron2 vs torchvision 架构对比
│   ├── debug_forward.py      # 前向传播调试
│   ├── debug_forward2.py     # 前向传播调试 v2
│   ├── debug_maskrcnn.py     # Mask R-CNN 推理调试
│   ├── debug_weights.py     # 权重映射调试
│   ├── test_configs.py       # 推理参数优化测试
│   └── pipeline_monitor.py   # 处理管线监控
├── reports/                   # 检测报告
│   ├── generate_excel_report.py  # Excel 报告生成 (4 工作表)
│   └── processing_report.xlsx    # 处理结果报告
└── README.md                  # 本说明文件
```

## 检测算法演进

### 1. 暗通道先验 + HSV (dcp_ice.py)
- 原理: 雾和冰都会降低暗通道值，结合 HSV 颜色空间过滤白色/浅蓝色冰区域
- 适用: 简单场景的快速覆冰检测
- 来源: Phase 2 (2026-08-10)

### 2. 端到端覆冰检测 (end2end_ice_detection.py)
- 原理: 暗通道先验 + HSV + Canny 边缘 + Hough 直线 + 纹理分析的综合系统
- 特点: 25KB 核心文件，集成多种检测策略
- 来源: Phase 2 (2026-08-10)

### 3. HazeCLIP 覆冰检测 (hazeclip_ice.py)
- 原理: 利用 CLIP 语言-视觉对齐能力辅助覆冰区域识别
- 来源: Phase 2 (2026-08-10)

### 4. 覆冰掩码生成器 (ice_mask_generator.py)
- 原理: 基于去雾前后图像差异 + 颜色空间分析生成覆冰掩码
- 特点: `--ice-mask` 参数仅在检测到覆冰时生成掩码，非覆冰图像不产生黑图
- 来源: Phase 5 (2026-08-16)

### 5. YOLO 自动标注 + 训练 (yolo_auto_label.py, train_yolo.py)
- 原理: HSV 颜色过滤 + Canny 边缘检测 + Hough 直线检测 + 纹理分析自动标注
- 检测类别: insulator, power_line, ice, tower
- 来源: Phase 5 (2026-08-16)

### 6. Mask R-CNN (maskrcnn_inference.py)
- 原理: Mask R-CNN R50-FPN 实例分割，detectron2 训练权重迁移到 torchvision
- 特点: 同事提供的预训练模型，2 类 (背景+目标)，1 掩码通道
- 优化: NMS 去重 (IoU=0.3) + 面积过滤 (>200px) + 限制每图 20 个检测
- 来源: Phase 6 (2026-08-17)

## 与去雾主体的集成

覆冰检测模块通过以下方式与去雾主体 (`phase6_maskrcnn_20260817/dehaze_inference.py`) 集成:

1. `--ice-mask` 参数: 去雾后自动检测覆冰并生成掩码
2. `--compare` 参数: 生成去雾前后对比图，叠加检测结果
3. `--use-maskrcnn` 参数: 切换使用 Mask R-CNN 替代 YOLO 检测
4. ITL 覆冰感知损失 (`phase5_hawfe_training_20260816/itl_loss.py`): 训练阶段约束去雾网络保留覆冰特征

## 文件来源映射

| 原路径 | 新路径 |
|--------|--------|
| phase2/dcp_ice.py | ice_detection/algorithms/ |
| phase2/end2end_ice_detection.py | ice_detection/algorithms/ |
| phase2/hazeclip_ice.py | ice_detection/algorithms/ |
| phase5/ice_mask_generator.py | ice_detection/algorithms/ |
| phase5/auto_label.py | ice_detection/algorithms/yolo_auto_label.py |
| phase6/maskrcnn_inference.py | ice_detection/algorithms/ |
| phase5/train_yolo.py | ice_detection/training/ |
| results/yolo_training_results.csv | ice_detection/training/ |
| configs/data.yaml | ice_detection/configs/ |
| configs/yolo_dataset_manifest.txt | ice_detection/configs/ |
| configs/yolo_train_args.yaml | ice_detection/configs/ |
| phase6/analyze_*.py, debug_*.py, etc. | ice_detection/debug/ |
| phase6/generate_excel_report.py | ice_detection/reports/ |
| results/processing_report.xlsx | ice_detection/reports/ |
