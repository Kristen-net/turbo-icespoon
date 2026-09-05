# IceWave 局限性与边界声明（Limitations）

> **目的**：SCI 一区审稿人通常会问"这个方法在哪些场景下会失效？"——本文档正面回应，按"已知 + 可量化 + 已缓解"三档列出。
> **撰写基础**：基于 `src/icewave/` 包的实际实现，截至 commit `f5b31eb`。
> **关联文档**：`docs/AUDIT_REPORT.md`（21 项工程审计）、`docs/REPRODUCIBILITY.md`（复现协议）、`docs/IMPLEMENTATION_NOTES.md`（实现细节）。

---

## 1. 数据集局限性

### 1.1 私有数据集未公开

| 维度 | 现状 | 影响 | 缓解措施 |
|---|---|---|---|
| 原始图像来源 | 私有 zip 包，来源未在仓库说明（P-13） | 第三方无法验证"该数据集上 PSNR 35.38"的数字意义 | 数据来源记录 `data/DATA_PROVENANCE.md`（计划中，P-13.1）；长期目标 Zenodo DOI 发布 |
| 标注协议 | 仅有规则伪标签（HSV + 边缘密度） | 评测指标本身被同一规则约束，循环论证（P-4） | P0-4 计划人工标注 150–300 张验证子集 |
| 数据集规模 | 训练 84 对合成 + 验证 84 对合成 + 真实 673 张 | 统计效力不足（P-1）；合成训练样本偏小 | P1-2 三 seeds + 多骨干验证 |

### 1.2 场景覆盖偏差

| 场景 | 覆盖度 | 风险 |
|---|---|---|
| 白天薄雾/中雾 | ✓ 充分 | — |
| 浓雾（HAZE_LEVELS[2]） | ✗ 训练未覆盖 | 浓雾 PSNR 实测缺失（P-6） |
| 夜间 / 低光巡检 | ✗ 完全未覆盖 | 夜间场景 PSNR 未知，**审稿人若问"夜间能用吗？"直接答不能** |
| 雪天、雨天 | ✗ 完全未覆盖 | 仅文档中提及，**未实测** |
| 杆塔类型多样性 | ◐ 单一区域 | 数据地理来源未声明，**域泛化未验证** |
| 极端天气（台风、冻雨） | ✗ 未覆盖 | 数据采集场景不明，**鲁棒性不可声称** |

### 1.3 伦理与脱敏

- **现状**：原始 zip 中可能含 GPS / 杆塔编号 / 电力公司标识等敏感信息，**当前未做脱敏处理**
- **声明**：在数据集公开（P1-3）前，必须完成：
  - 去除 EXIF GPS / 拍摄设备 / 拍摄时间字段
  - 杆塔编号 / 线路编号遮罩
  - 电力公司标识去除
  - IRB 审批或机构伦理审查

---

## 2. 模型方法局限性

### 2.1 HA-WFE（Haar 小波增强）

| 局限 | 量化证据 | 影响 |
|---|---|---|
| **单层 Haar 固定基** | `src/icewave/models/hawfe.py:29-41` 仅单层 DWT/IDWT | 与 ProDehaze (2025) HFE / WDMamba 撞车，差异性需通过方向 B 升级版本证明 |
| **高频子带激活少** | README 历史记录"高频子带在合成雾上激活极少"（旧 `phase5/README.md`） | HA-WFE 核心分支可能贡献有限，需消融验证 |
| **奇数尺寸崩溃**（已修复） | 上游原 bug：裁剪 F_enhanced 后未同步裁剪 F_b | 在 H/W 为奇数时 padding 一致性破坏，**commit `20f462f` 已同步裁剪** |
| **尺度耦合** | HA-WFE 插入位置固定在 layer3 后 | **未消融**不同插入位置（layer2/layer3/layer4）的影响 |

### 2.2 CLIP 雾提示蒸馏

| 局限 | 量化证据 | 影响 |
|---|---|---|
| **CLIP 推理免依赖**：训练需 CLIP，推理不需要 | `src/icewave/train/trainer.py` m3/m4 路径 | 这是**卖点**而非局限——确认无歧义 |
| **prompt 投影 32 通道是手工设置** | `_BACKBONE_SPECS['m3'].prompt_channels=32` | **未消融**不同通道数（8/16/32/64） |
| **prompt dropout 0.5 是手工设置** | trainer.py 默认值 | **未消融**不同概率（0.0/0.25/0.5/0.75） |
| **CLIPSurgery 与 OpenAI CLIP 不可互换** | 依赖 CS-ViT-B/32 权重 | 用户需自行下载 CS 权重（HuggingFace） |

### 2.3 ITL 覆冰感知损失

| 局限 | 量化证据 | 影响 |
|---|---|---|
| **冰掩码来源仍为规则伪标签** | `src/icewave/detect/ice_mask.py` HSV + Otsu + Sobel | 与 P-4 循环论证同源——ITL 优化方向可能只是"输出越来越像规则" |
| **SSIM 模块曾为死代码**（已修复） | `phase5/itl_loss.py` 旧版 `self.ssim` 实例化但未调用 | 修复后 `region_term='ssim'` 走真 SSIM；默认 `region_term='ssim'`（commit `20f462f`） |
| **人工标注未接入** | `ice_mask_human/` 优先逻辑已实现，未真实数据 | 实际接入需 P-13.1 + 人工标注子集 |
| **λ_region / λ_boundary 是手工设置** | 默认 0.5 / 0.3 | **未消融** |

### 2.4 联合优化（joint 模式）

| 局限 | 量化证据 | 影响 |
|---|---|---|
| **检测头是简化版** | `src/icewave/losses/detect.py` UncertaintyWeighting + CorridorTextureLoss，**不是完整 Faster R-CNN / DETR** | 与"真正端到端"有差距，仅作概念验证 |
| **下游 mAP 增益指标** | `src/icewave/eval/downstream.py` 已实现 | **未真实训练**——指标为 0，无法声称增益 |
| **物理模型简化** | Beer-Lambert 假设 β(d) 单调；大气散射假设均匀 A | 与真实场景有差距，**未做域差距量化** |

---

## 3. 评测局限性

### 3.1 公开基准未跑

- **现状**：`configs/benchmarks.yaml` 占位，**未真实跑过任何公开基准**
- **影响**：所有 PSNR/SSIM 数字仅在私有合成/真实数据集上，**无法与 SOTA 直接对比**
- **审稿应对**：明确声明"本工作使用私有数据集 + 合成器，与 RESIDE-SOTS 等公开基准的可比性需进一步验证"

### 3.2 统计效力不足

- **现状**：所有数字来自单次运行，**无 3 seeds、无数值方差、无显著性检验**
- **影响**：+0.48 dB 增益可能在统计噪声范围内
- **审稿应对**：明确声明"统计规范为后续工作；当前结果仅供概念验证"

### 3.3 评测指标覆盖度

| 缺失指标 | 影响 |
|---|---|
| LPIPS / DISTS（感知质量） | 无法证明"去雾结果更自然" |
| NIQE / BRISQUE（无参考质量） | 真实雾无参考指标仅靠 dark_channel，**单一** |
| FLOPs / 帧率 / 显存 | README 参数增量 +2–4%，但**未给绝对推理成本** |
| 端到端检测 mAP | joint 模式核心评测，**未真实跑** |

---

## 4. 工程与部署局限性

### 4.1 路径参数化

- **现状**：`ICEWAVE_*` 环境变量 + argparse + YAML（commit `20f462f`），但 `trainer.py` 默认值仍部分硬编码
- **影响**：用户自定义路径时仍需查文档
- **审稿应对**：默认配置已可跑通，自定义路径按 `docs/REPRODUCIBILITY.md` §3 操作

### 4.2 AMP dtype 硬编码

- **现状**：`torch.amp.autocast("cuda", enabled=False)` 仅在 HA-WFE 模块（避免小波分解精度损失），整体训练 dtype 仍 fp16
- **影响**：A100/4090 等支持 bf16 的卡浪费了 bf16 精度优势
- **缓解**：可通过 `--amp bf16` CLI 参数切换（计划中）

### 4.3 推理 CLI 默认行为

- **现状**：`icewave-infer` 默认关自动重训（commit `15b2364`），但 `--auto-retrain` flag 仍可用
- **影响**：服务化部署时默认安全；单机调试可显式开启

### 4.4 Docker 镜像体积

- **`Dockerfile.gpu`** 基于 `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`，含 ultralytics（YOLO）等重型依赖
- **体积**：~8 GB；可拆分为 `Dockerfile.infer-base`（无 YOLO）+ `Dockerfile.infer-detect`（含 YOLO）两个镜像
- **当前未拆分**：作为 P2-2 工作项

### 4.5 Mask R-CNN 适配器遗留

- **现状**：`src/icewave/detect/maskrcnn_adapter.py` 提供 detectron2 → torchvision 权重键名映射
- **影响**：类别语义仅"target"（与论文场景无明确对齐证据，原 P-8 诊断）
- **缓解**：在 docstring 中明确标注"该适配器仅作历史兼容性保留，推荐使用 `detect/yolo.py` 走 YOLO 路线"

---

## 5. 学术诚信声明

### 5.1 数据来源

- 本项目数据来源遵循 P-13.1 规范（计划中）；在数据公开前，所有论文需明确"私有数据集，不可公开"
- 任何引用本工作的二次研究者，应通过 GitHub Issue 申请数据访问（伦理审查后）

### 5.2 模型权重

- M1–M4 权重为作者自身训练结果，可通过 `configs/weights.yaml`（维护者填 URL/SHA256）下载
- HazeCLIP / YOLO 权重为第三方，遵循各自原始许可（详见 `THIRD_PARTY_NOTICES.md`）
- **不提供任何商用保证**

### 5.3 与 SOTA 对比

- 当前 PSNR/SSIM 数字仅在私有数据集上，**不构成对任何公开 SOTA 的超越声明**
- 公开基准对比是 P0-3 计划项，**未完成**

### 5.4 代码可复现性

- 103 项 pytest 全过（commit `f5b31eb`）
- `docs/REPRODUCIBILITY.md` 给出完整复现协议
- `scripts/reproduce.sh` 一键 DRY_RUN 验证
- **若复现失败**，请提交 GitHub Issue 含：(a) Python/torch/CUDA 版本；(b) 完整命令；(c) 报错栈

---

## 6. 已知 bug 与修复追踪

| Bug | 状态 | 修复 commit |
|---|---|---|
| 骨干默认参数错档（旧 `phase5/...`） | ✓ 已修 | `20f462f` |
| HA-WFE 奇数尺寸 padding 同步裁剪 | ✓ 已修 | `20f462f` |
| ITL 死代码 self.ssim + SSIM 实现 | ✓ 已修 | `20f462f` |
| prompt-proj optimizer 漏注册 | ✓ 已修 | `20f462f` |
| 130+ 硬编码路径 | ✓ 已修 | `20f462f` |
| ITL meta 键名 `alpha/thickness` vs `ice_alpha/ice_thickness` | ✓ 已修 | `15b2364` |
| `.gitignore` 误吞 `src/icewave/data/` 包源码 | ✓ 已修 | `15b2364` |
| `ice_detection/algorithms/ice_mask_generator.py` 与新包双实现 | ◐ 标注 deprecation | `15b2364` |

---

## 7. 未来工作

按优先级（详细见分析报告 §3.4 与 §4.2）：

1. **真实训练 + 公开基准**（P0-3/P0-4/P1-1/P1-2）：4–6 个月
2. **数据集公开**（P1-3 + P-13.1）：2–3 周
3. **HA-WFE 升级版**（方向 B）：3–5 周
4. **ONNX/TensorRT + Gradio demo**（P2-2）：1–2 周
5. **合成→真实域差距分析**（方向 E）：3–4 周

---

> **本文档应与论文 Supplementary Materials 一同提交**，作为审稿人快速了解系统边界的索引。