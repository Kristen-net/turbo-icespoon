# IceWave 论文大纲与代码映射

> **目的**：为 SCI 一区投稿准备论文骨架。每节给出：**论文结构** → **代码实现位置** → **预期图表**。
> **目标期刊**：IEEE Trans. on Industrial Informatics / IEEE Trans. on Instrumentation and Measurement（一区备选 Applied Energy / Pattern Recognition）
> **撰写基础**：代码 commit `f5b31eb`

---

## 论文标题候选

1. **"Detection-Aware Ice-Aware Dehazing for Transmission-Line Inspection via Joint Optimization"**
2. **"IceWave: A Joint Dehazing-Detection Framework for Foggy Power-Line Images with Composite Degradation Modeling"**
3. **"From Cascade to Joint: Ice-Aware Dehazing with Task-Driven Restoration for Transmission Lines"**

推荐主投标题：**标题 1**（最贴合 IEEE TII 主题 + 强调 joint 创新）。

---

## 论文结构（建议 8 页正文 + 4 页附录 = 12 页）

### Section 1: Introduction (1.5 页)

**论文段落**：
1.1 雾天输电线路巡检的应用背景（电网安全）
1.2 现有去雾方法在覆冰场景的局限（小波撞车问题，引用 ProDehaze/WDMamba）
1.3 现有级联系统（先去雾再检测）的循环论证与脱节
1.4 本文贡献（4 条 bullet）

**预期图表**：图 1 系统整体框架图（去雾主干 + 检测感知分支 + 复合退化合成器）

**对应代码**：
- 系统总览：`README.md` + `src/icewave/__init__.py`
- 复合退化：`src/icewave/data/degradation.py`（`synthesize_hazy_iced`）

**贡献清单（4 条）**：
1. **复合退化物理模型**：首次形式化"雾 + 覆冰"两步成像（Beer-Lambert × Koschmieder），公式与代码对应 §3.1
2. **检测感知联合优化**：把级联升级为端到端，检测损失反向传播指导去雾主干，§3.2
3. **HA-WFE 自适应频域选择**：在单层 Haar 基础上引入透射率引导的频域门控，区分于 ProDehaze 固定 HFE，§3.3
4. **下游 mAP 增益评测协议**：定义 ΔmAP = mAP_dehazed - mAP_hazy 作为恢复对任务的价值指标，§4.3

---

### Section 2: Related Work (1.5 页)

2.1 单图去雾：FFA-Net / Restormer / DehazeFormer / MB-TaylorFormer
2.2 CLIP 引导去雾：CLIPHaze / HazeCLIP / DA-CLIP
2.3 扩散去雾：ProDehaze / Diff-Plugin
2.4 覆冰检测：传统方法 + 深度方法（YOLOv8 / Mask R-CNN / DeepLabV3+）
2.5 联合优化：检测感知损失 / 多任务学习 / 不确定性加权

**关键引用**：
- He et al. (CVPR'09) DCP（暗通道先验）
- Song et al. (TIP'23) DehazeFormer
- Wu et al. (ECCV'24) CLIPHaze
- ProDehaze (2025) — **必引，与 HA-WFE 撞车**
- Kendall & Gal (ICML'17) 不确定性加权

---

### Section 3: Method (3.5 页)

#### 3.1 复合退化物理模型 (0.8 页)

**论文段落**：
- 形式化雾 + 覆冰两步成像：J_ice → I = J_ice·t + A·(1-t)
- α = 1 - exp(-β·d) Beer-Lambert
- 三档预设（thin/medium/dense）

**预期图表**：图 2 复合退化流程图（清晰图 → 冰层叠加 → 大气散射 → 雾图）

**对应代码**：
- `src/icewave/data/degradation.py:123-141` `synthesize_haze`
- `src/icewave/data/degradation.py` `ice_composite` (Beer-Lambert)
- `src/icewave/data/degradation.py:55-62` `HAZE_PRESETS`
- `src/icewave/data/degradation.py:84-102` `make_ice_thickness`

**关键公式**：

```
J_ice(x) = (1 - α(x)) · J(x) + α(x) · texture(x)       # Beer-Lambert
I(x)     = J_ice(x) · t(x) + A · (1 - t(x))             # Koschmieder
α(x)     = 1 - exp(-β · d(x))                            # 透射-厚度耦合
```

#### 3.2 检测感知联合优化 (1 页)

**论文段落**：
- 损失函数：L_total = L_rec + λ_det · L_det + σ-aware 多任务加权
- 不确定性加权：Kendall & Gal 公式
- 检测感知损失：检测框内特征保持 + CorridorTextureLoss（走廊白色纹理）

**预期图表**：图 3 joint 模式架构图（DehazeFormer 骨干 + 检测头分支 + σ 加权）

**对应代码**：
- `src/icewave/losses/detect.py` `UncertaintyWeighting`
- `src/icewave/losses/detect.py` `CorridorTextureLoss`
- `src/icewave/train/trainer.py` `build_model('joint')` 路径
- `src/icewave/eval/downstream.py` `compute_delta_map`

#### 3.3 HA-WFE 自适应频域选择 (1 页)

**论文段落**：
- 单层 Haar DWT/IDWT 数学
- LL 分支：SCA + 可选 CLIP 雾提示调制
- 三个高频子带：深度可分离门控 + 子带独立 α
- 关键差异化：**透射率引导的频域门控**（vs ProDehaze 固定 HFE）

**预期图表**：图 4 HA-WFE 模块结构（LL 分支 + 三个高频子带 + 透射率调制）

**对应代码**：
- `src/icewave/models/hawfe.py:29-41` `HaarDWT2d`
- `src/icewave/models/hawfe.py:55-189` `HAWFE` (v1) + `HAWFEv2`
- `src/icewave/models/hawfe.py:300+` `IceWaveDehazeFormer`
- `src/icewave/models/hawfe.py:400+` `build_model`

#### 3.4 ITL 覆冰感知损失 (0.7 页)

**论文段落**：
- 公式：L_itl = λ_region · L_region + λ_boundary · L_boundary
- L_region：冰区内 masked SSIM（默认）+ 加权 L1
- L_boundary：边界带 Sobel 梯度 L1
- 冰掩码来源：人工标注（ice_mask_human/）优先 → 规则伪标签兜底

**对应代码**：
- `src/icewave/losses/itl.py:56-78` `ITLLoss`
- `src/icewave/losses/itl.py:14-17` 公式注释
- `src/icewave/detect/ice_mask.py` 掩码生成器

---

### Section 4: Experiments (3 页)

#### 4.1 实验设置 (0.7 页)

**论文段落**：
- 数据集：私有数据集 + RESIDE-SOTS（户外/室内）
- 评测指标：PSNR / SSIM / LPIPS / FLOPs / 帧率 / 下游 ΔmAP
- 对比方法 ≥ 10 个
- 训练设置：3 seeds + 显著性检验

**对应代码**：
- `configs/benchmarks.yaml`（占位）
- `configs/train/*.yaml`（m1/m2/m2p/m3/m4/joint）
- `scripts/reproduce.sh` 一键流水线

#### 4.2 合成数据集结果 (0.8 页)

**预期表格**：Table 1 PSNR/SSIM 对比（M1-M4 vs 8 个 SOTA）

**对应代码**：
- `src/icewave/eval/benchmark.py`
- `src/icewave/eval/metrics.py`

#### 4.3 真实数据集 + 下游任务 (1 页)

**预期表格**：Table 2 真实雾无参考指标（dark_channel / NIQE）+ Table 3 下游 ΔmAP

**对应代码**：
- `src/icewave/eval/downstream.py`
- `src/icewave/detect/yolo.py` `YOLODetector`

#### 4.4 消融实验 (0.7 页)

**预期表格**：Table 5 λ_region/λ_boundary 消融 + HA-WFE 子带消融 + CLIP prompt 通道消融

**对应代码**：
- `configs/train/*.yaml` 配置文件支持超参扫描

---

### Section 5: Discussion (0.5 页)

5.1 与 ProDehaze 等小波类工作的差异化（**审稿人必问**）
5.2 合成→真实域差距的来源分析（**对应分析报告 A-5**）
5.3 局限性与未来工作（详见 `docs/LIMITATIONS.md`）

---

### Section 6: Conclusion (0.3 页)

总结 4 条贡献。

---

## 附录结构（4 页）

- **A1**：训练超参与配置（`configs/` 全部 yaml + 表格）
- **A2**：合成器参数表（HAZE_PRESETS 三档）
- **A3**：检测感知损失推导细节
- **A4**：可复现性协议（`docs/REPRODUCIBILITY.md` 完整版）

---

## 图表清单（汇总）

| 编号 | 内容 | 类型 | 对应代码 |
|---|---|---|---|
| 图 1 | 系统整体框架 | 系统图 | — |
| 图 2 | 复合退化流程 | 物理流程图 | `data/degradation.py` |
| 图 3 | joint 架构 | 网络结构图 | `models/hawfe.py` |
| 图 4 | HA-WFE 模块细节 | 模块结构图 | `models/hawfe.py:55-189` |
| 图 5 | 检测感知分支 | 损失流程图 | `losses/detect.py` |
| 图 6 | 定性对比（合成 + 真实） | 图像 | `eval/benchmark.py` |
| 表 1 | 合成 PSNR/SSIM | 数据表 | — |
| 表 2 | 真实雾无参考 | 数据表 | — |
| 表 3 | 下游 ΔmAP | 数据表 | `eval/downstream.py` |
| 表 4 | 推理效率 | 数据表 | — |
| 表 5 | 消融实验 | 数据表 | — |
| 表 6 | 超参数附录 | 数据表 | `configs/train/*.yaml` |

---

## 写作顺序建议

1. **先写 Methods §3.1 + §3.3 + §3.4**（纯公式 + 代码对应，最稳）
2. **再写 Experiments §4.4 消融**（消融可基于现有 M1/M2/M2p 数字 + 假想 M3/M4）
3. **再写 Introduction §1.4 贡献**（基于 Methods 与 Experiments 倒推）
4. **最后写 Discussion §5**（基于 Limitations + 审稿人预问）

**关键技巧**：每个公式都必须能在代码里找到 `path:line` 指针，避免"论文方法描述与实现不符"（P-8 历史教训）。

---

## 投稿配套材料清单

| 材料 | 来源 | 必需 |
|---|---|---|
| 论文 PDF (12 页) | 上述结构 | ✓ |
| 源代码 zip | `turbo-icespoon` HEAD `f5b31eb` | ✓ |
| 训练权重 | `populate_manifest.py` 生成 + HuggingFace 上传 | ✓ |
| 测试集（私有） | `data/DATA_PROVENANCE.md` 文档化 | ✓（投稿声明私有） |
| REPRODUCING.md | `docs/REPRODUCIBILITY.md` | ✓ |
| THIRD_PARTY_NOTICES.md | `THIRD_PARTY_NOTICES.md` | ✓ |
| LIMITATIONS.md | `docs/LIMITATIONS.md` | 推荐 |
| AUDIT_REPORT.md | `docs/AUDIT_REPORT.md` | 可选（审稿人可选读） |

---

> **本文档应与论文初稿同步维护**。每完成一节，回填"对应代码"列的具体行号。