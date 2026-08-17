# IceWave-DehazeFormer

## 雾天输电线路去雾与覆冰检测端到端耦合系统

本项目实现了一个基于深度学习的雾天输电线路图像去雾与覆冰检测系统，核心创新在于"去雾-覆冰感知端到端耦合"架构，融合了 CLIP 语言引导轻量 CNN、扩散生成式去雾和 Mamba/Transformer 新型骨干网络三条技术路线。

---

## 项目架构

```
前端轻量采集缓存 + 云端集中智能感知
```

系统遵循 **"前端轻量采集缓存 + 云端集中智能感知"** 的架构设计，核心算法创新为 **"去雾-覆冰感知端到端耦合"**，融合三条技术路线：
1. **CLIP 语言引导轻量 CNN** — 利用 CLIP 的语言-视觉对齐能力指导去雾
2. **扩散生成式去雾** — 基于 DDPM/DDPM 的生成式图像恢复
3. **Mamba/Transformer 新型骨干** — DehazeFormer (Transformer) 和 WDMamba (SSM)

---

## 核心创新

### HA-WFE (Haar Wavelet Feature Enhancement)
瓶颈层 Haar 小波特征增强模块，对四个子带进行差异化处理：
- **LL (低频)**: 保留全局结构信息
- **LH/HL (中频)**: 增强边缘和纹理细节
- **HH (高频)**: 抑制噪声和雾残留
- 零初始化残差设计，减少训练不稳定

### ITL (Ice-aware Triplet Loss)
覆冰感知损失函数，包含区域约束和边界约束：
- 区域约束: 覆冰区域与背景的分离损失
- 边界约束: 覆冰边界的平滑过渡损失

### CLIP 雾提示蒸馏
- 训练时使用 HazeCLIP 教师模型蒸馏
- 推理时无需 CLIP，轻量化部署
- Prompt dropout (50%) 避免 train-test gap

---

## 模型版本演进

| 版本 | 名称 | 描述 | 参数量 | 关键改进 |
|------|------|------|--------|---------|
| M1 | DehazeFormer-S 基线 | 原始 DehazeFormer-S | 基准 | 基线模型 |
| M2 | + HA-WFE v1 | 零初始化, Tanh, 共享alpha | +~10% | Haar小波特征增强(第一版) |
| M2p | + HA-WFE v2 | 正值初始化, Sigmoid, 独立alpha | +~12% | 改进初始化和激活函数 |
| M3 | + CLIP蒸馏 | HazeCLIP教师蒸馏 | +~13% | 语言引导去雾, 推理无需CLIP |
| M4 | + ITL覆冰感知 | 区域+边界约束 (推荐) | +~15% | 覆冰区域感知损失 |

**参数增量控制在基线模型的 10-15% 范围内。**

---

## 开发阶段时间线

### Phase 1: 初始框架搭建 (2026-08-04)
- `generate_haze.py` — 合成雾生成工具
- `benchmark.py` — 基准测试框架
- `end2end_pipeline.py` — 端到端流水线初版
- `quality_eval.py` — 图像质量评估工具 (PSNR, SSIM, 暗通道等)

### Phase 2: CLIP 与 Mamba 集成 (2026-08-10 ~ 2026-08-11)
**CLIP 路线:**
- `clone_clip.py` / `download_clip.py` — CLIP 模型下载
- `patch_clip.py` — CLIP 补丁适配
- `hazeclip_ice.py` — HazeCLIP 覆冰检测
- `fusion_inference.py` — 多模型融合推理

**Mamba 路线:**
- `wdmamba_inference.py` / `wdmamba_inference_v2.py` — WDMamba 推理
- `selective_scan_interface.py` (v1~v4) — 选择性扫描接口实现
- `mamba_ssm_init.py` — Mamba SSM 初始化
- **关键发现**: mamba_ssm CUDA 内核无法在 Windows 编译，纯 PyTorch 实现导致显著性能下降

**扩散模型路线:**
- `diffdehaze_inference.py` — DiffDehaze 推理
- `dehazesb_inference.py` — DehazeSB 推理

**端到端检测:**
- `end2end_ice_detection.py` — 端到端覆冰检测系统 (25KB, 核心文件)
- `dcp_ice.py` — 暗通道先验覆冰检测

### Phase 3: 多轨融合 (2026-08-11 ~ 2026-08-12)
- `fusion_3tracks.py` — 三轨融合 (DehazeFormer + HazeCLIP + WDMamba)
- `fusion_3tracks_auto.py` — 自动三轨融合
- `end2end_3tracks.py` / `end2end_3tracks_auto.py` — 端到端三轨
- `run_pipeline.py` — 流水线运行器
- `compare_old_new.py` — 新旧方法对比
- **关键发现**: 多模型融合可能稀释最佳单模型效果; HazeCLIP 单独使用常优于融合

### Phase 4: 数据集构建与基线训练 (2026-08-15)
**数据集构建:**
- `build_dataset.py` (v1~v3) — 数据集构建工具
- `check_data.py` / `extract_data.py` / `check_zips.py` — 数据检查与提取

**环境验证:**
- `check_torch.py` — PyTorch 环境检查
- `verify_dehazeformer.py` — DehazeFormer 验证
- `verify_env.py` — 环境完整验证

**权重管理:**
- `download_weights.py` (v1~v2) — 预训练权重下载
- `verify_weights.py` — 权重验证
- `verify_mct.py` — MCT 验证

**性能测试:**
- `test_speed.py` / `test_speed_mct.py` / `test_speed_standard.py` — 速度测试
- `test_batch_sizes.py` — 批量大小测试

**基线训练:**
- `train_m1.py` — M1 基线模型训练
- `ha_wfe.py` — HA-WFE 模块实现 (第一版)
- `train_m2.py` — M2 HA-WFE 模型训练

### Phase 5: HA-WFE v2 与多版本训练 (2026-08-16)
**HA-WFE v2:**
- `ha_wfe_v2.py` — HA-WFE 模块改进版 (正值初始化, Sigmoid, 独立alpha)
- `train_m2p.py` — M2p HA-WFE v2 训练

**监控与分析:**
- `read_tb.py` — TensorBoard 日志读取
- `check_progress.py` — 训练进度检查
- `analyze_saturation_impact.py` — 饱和度影响分析
- `check_cuda.py` — CUDA 兼容性检查
- `test_compat.py` — 兼容性测试

**CLIP 蒸馏 (M3):**
- `train_m3.py` — M3 CLIP 蒸馏训练
- `monitor_m3.py` — M3 训练监控
- `pipeline_m3_compare.py` — M3 模型流水线对比
- `clip_fog_prompt.py` — CLIP 雾提示集成模块
- **关键改进**: Prompt dropout (50%) + 低学习率 (5e-5) 稳定训练

**ITL 覆冰感知 (M4):**
- `itl_loss.py` — ITL 覆冰感知损失函数实现
- `train_m4.py` — M4 ITL 覆冰感知训练
- `monitor_m4.py` — M4 训练监控
- `pipeline_m4_compare.py` — M4 模型流水线对比

**覆冰检测与 YOLO:**
- `ice_mask_generator.py` — 覆冰掩码生成器
- `make_comparison.py` — 对比图生成
- `auto_label.py` — 自动标注工具 (HSV+边缘+纹理)
- `train_yolo.py` — YOLO 训练脚本

### Phase 6: Mask R-CNN 集成与 Excel 报告 (2026-08-17)
**核心推理脚本:**
- `dehaze_inference.py` — **主推理脚本** (38KB), 集成去雾+检测+覆冰掩码+对比图
- `generate_excel_report.py` — Excel 报告生成 (4个工作表)

**Mask R-CNN 集成:**
- `maskrcnn_inference.py` — Mask R-CNN 推理模块
- `test_configs.py` — Mask R-CNN 配置测试
- `analyze_ckpt.py` / `analyze_ckpt2.py` — Checkpoint 分析
- `compare_arch.py` — 架构对比
- `analyze_scores.py` — 分数分析
- `debug_maskrcnn.py` / `debug_weights.py` / `debug_forward.py` / `debug_forward2.py` — 调试工具

**流水线监控:**
- `pipeline_monitor.py` — 流水线监控工具

---

## 文件目录结构

```
icewave-dehazeformer/
├── README.md                           # 本文件
├── .gitignore
│
├── phase1_initial_20260804/            # Phase 1: 初始框架 (4个脚本)
├── phase2_clip_mamba_20260810_11/      # Phase 2: CLIP与Mamba集成 (21个脚本)
├── phase3_multitrack_fusion_20260811_12/ # Phase 3: 多轨融合 (7个脚本)
├── phase4_dataset_baseline_20260815/   # Phase 4: 数据集与基线 (19个脚本)
├── phase5_hawfe_training_20260816/     # Phase 5: HA-WFE与多版本训练 (19个脚本)
├── phase6_maskrcnn_20260817/           # Phase 6: Mask R-CNN集成 (13个脚本)
│
├── source_dehazeformer/                # DehazeFormer 源码 (关键文件)
├── source_hazeclip/                    # HazeCLIP 源码 (关键文件)
│
├── docs/                               # 文档
│   └── five_way_report.txt             # 五路去雾对比报告
│
├── configs/                            # 配置文件
│   ├── yolo_train_args.yaml            # YOLO 训练参数
│   ├── hazeclip_finetune.yaml          # HazeCLIP 微调配置
│   ├── hazeclip_inference.yaml         # HazeCLIP 推理配置
│   ├── hazeclip_pretrain.yaml          # HazeCLIP 预训练配置
│   ├── data.yaml                       # YOLO 数据集配置
│   └── yolo_dataset_manifest.txt       # 数据集清单
│
└── results/                            # 实验结果
    ├── processing_report.xlsx          # Excel 处理报告
    └── yolo_training_results.csv       # YOLO 训练结果
```

---

## 使用方法

### 环境要求

- Python 3.10+
- PyTorch 2.11.0+ (CUDA 12.8+ for RTX 5060/Blackwell)
- CUDA 12.8+
- GPU: NVIDIA RTX 5060 (8GB VRAM) 或更高

### 核心依赖

```
torch >= 2.11.0
torchvision
opencv-python
numpy
openpyxl
ultralytics (YOLOv8)
timm
```

### 推理命令

```bash
# 基本去雾
python dehaze_inference.py -i input/ -o output/

# 去雾 + 覆冰检测 + 对比图 (YOLO)
python dehaze_inference.py -i input/ -o output/ --ice-mask --compare

# 使用 Mask R-CNN 替代 YOLO
python dehaze_inference.py -i input/ -o output/ --ice-mask --compare --use-maskrcnn --no-retrain

# 指定模型 (m1/m2/m2p/m3/m4)
python dehaze_inference.py -i input/ -o output/ -m m4 --compare
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `-i / --input` | 输入图片路径或文件夹 |
| `-o / --output` | 输出图片路径或文件夹 |
| `-m / --model` | 模型选择: m1(基线), m2(HA-WFE v1), m2p(HA-WFE v2), m3(CLIP蒸馏), m4(ITL覆冰感知, 默认) |
| `--ice-mask` | 生成覆冰区域掩码 (仅在检测到覆冰时生成) |
| `--compare` | 生成原图 vs 去雾 vs 检测 vs 覆冰的并排对比图 |
| `--use-maskrcnn` | 使用同事提供的 Mask R-CNN 模型替代 YOLO |
| `--no-yolo` | 禁用 YOLO 检测 |
| `--no-retrain` | 跳过 YOLO 自动重训 |

### 输出文件结构

```
output/
├── image1.png                    # 去雾后图片
├── image1_yolo.png               # YOLO 检测标注图 (或 _maskrcnn.png)
├── image1_ice_mask.png            # 覆冰掩码 (仅有覆冰时生成)

compare/
├── compare_image1.png            # 并排对比图

processing_report.xlsx             # Excel 报告 (4个工作表)
```

### Excel 报告工作表

1. **处理总览** — 文件名、分辨率、亮度/对比度提升、检测器类型、覆冰判定
2. **去雾质量指标** — 亮度、对比度、饱和度、边缘密度、暗通道、信息熵 (原图 vs 去雾后)
3. **覆冰检测详情** — 检测器、覆冰面积比、覆冰区域数、最大覆冰区域
4. **统计汇总** — 总图片数、覆冰图片数、检测器分布、平均值统计

---

## 关键技术决策与经验

### 环境适配
- PyTorch 1.13.1+cpu 升级至 2.11.0+cu128 以支持 RTX 5060 (Blackwell, compute 12.0)
- mamba_ssm CUDA 内核无法在 Windows 编译; 纯 PyTorch 实现导致去雾性能显著下降 (输出近似恒等映射)
- PyTorch 2.6+ 修改了 torch.load 默认为 weights_only=True; 加载旧 checkpoint 需添加 weights_only=False

### 训练策略
- CLIP 雾提示注入需要 Prompt dropout (50%) 避免 train-test gap; 不加 dropout 会导致 PSNR 下降
- Prompt dropout + 低学习率 (5e-5) 稳定训练, 使模型在有/无 CLIP 提示时均表现良好
- HazeCLIP 教师模型训练时冻结, 推理时移除
- HA-WFE 高频子带在合成雾数据上激活极少; 真实雾数据可能需要调整初始化 (非零) 或移除 Tanh 约束

### 检测算法
- YOLO 自动标注使用 HSV 颜色 + Canny 边缘 + Hough 直线 + 纹理过滤的组合策略
- 覆冰检测采用走廊约束 (YOLO power_line/insulator 区域) + 白色特征 + 中等纹理的 AND 逻辑
- Mask R-CNN 权重从 detectron2 迁移到 torchvision, 需手动键名映射 (FPN层、ROI Head等)
- Mask R-CNN 分数饱和问题通过 NMS 去重 (IoU=0.3) + 面积过滤 (>200px) + 限制每图20个检测来缓解
- CLIPSurgery (CS-ViT-B/32) 返回空间特征 [B, 50, 512] (1 CLS + 49 spatial); 常规 CLIP 仅返回 [B, 512]

### 部署优化
- RTX 5060 8GB VRAM 需要 bf16、梯度检查点和分块推理
- dehaze_inference.py 通过文件哈希清单自动触发 YOLO 重训 (增量数据 + 迁移学习, 20 epochs)
- --ice-mask 参数仅在检测到覆冰时生成掩码 (非覆冰情况不生成黑色图片)

---

## 模型权重说明

以下模型权重因体积较大未包含在仓库中，需单独获取：

| 模型 | 文件 | 大小 | 说明 |
|------|------|------|------|
| HazeCLIP | model.pth | ~108MB | HazeCLIP 教师模型 |
| Mask R-CNN | model_0004999.pth | ~335MB | 同事提供的 detectron2 权重 |
| M1~M4 | m{1,2,2p,3,4}_best.pth | 5.6~15.6MB | IceWave 各版本检查点 |
| DehazeFormer | dehazeformer.pth | ~5.7MB | DehazeFormer 预训练权重 |
| YOLOv8 | best.pt / last.pt | ~5.9MB | YOLO 电力线检测权重 |
| YOLOv8n | yolov8n.pt | ~6.2MB | YOLOv8n 预训练权重 |

---

## 相关技术文档

- DehazeFormer: https://github.com/cecid3/DehazeFormer
- HazeCLIP: https://github.com/cuaecc/HazeCLIP
- CLIPSurgery: https://github.com/leonardopinto/clipsurgery

---

## 作者

Kristen-net (https://github.com/Kristen-net)

## 许可证

本项目仅供学术研究使用。
