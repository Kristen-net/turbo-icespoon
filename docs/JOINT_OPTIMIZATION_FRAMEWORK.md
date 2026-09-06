# Detection-Aware Joint Optimization Framework for Ice-Aware Dehazing of Foggy Power-Line Images

> **文件定位**：将现有级联式"先去雾再检测"重构为**检测感知联合优化**的顶层技术方案与实施计划。
> **目标受众**：SCI 一区投稿作者 + 工程实现者。
> **关联代码**：基于 commit `8bc038a` 的代码骨架（`src/icewave/` 全包，`build_model('joint')` 已通）。
> **关联文档**：`docs/AUDIT_REPORT.md`、`docs/PAPER_OUTLINE.md`、`docs/IMPLEMENTATION_NOTES.md`、父目录 `turbo-icespoon分析报告.md`。

---

## 0. 一句话总览（BLUF）

把"去雾 ↔ 检测"从**级联**改为**端到端联合训练**：以**复合退化物理模型**作为可微合成器；以**下游检测任务梯度 + 框内特征保持**替换"基于规则伪标签的冰区约束"；以**学习任务不确定性**自动平衡重建与检测；以**ΔmAP 评测协议**度量去雾对任务真实价值。

---

## 1. 贡献清单（论文 4 个 Bullet 项）

| ID | 贡献 | 与既有工作的区分度 | 落地代码锚点 |
|----|------|---------|---------|
| **C1** | 雾 + 覆冰复合退化成像模型（Beer-Lambert × Koschmieder × 镜面反射 × 边缘模糊） | 现有工作多独立处理雾或覆冰；合成器未能量化建模 | `src/icewave/data/degradation.py`（既有，扩字段） |
| **C2** | 检测感知端到端联合训练（box-feature preservation + detectability loss） | 真正替换 ITL 规则伪标签约束（**打破循环论证**）；与"走廊纹理保持"（既有 P1-1 概念验证）相比升级为 box-aware | `src/icewave/losses/detect.py`（新增 `BoxFeaturePreservationLoss` 与 `DetectabilityLoss`） |
| **C3** | 可学习任务不确定性加权（Kendall & Gal log σ²） | 替代手工调 λ；与既有 `UncertaintyWeighting` 兼容 | `src/icewave/losses/detect.py:UncertaintyWeighting`（既有，扩 K） |
| **C4** | 下游任务增益评测协议（ΔmAP / gap / recovery ratio） | 与公开基准 PSNR 互补，回答"去雾是否真对下游有用" | `src/icewave/eval/downstream.py`（既有，扩统计量） |

**与既有 P1-1 的关系**：当前 P1-1 是**概念验证**（走廊纹理保持 + 不确定性加权 + 下游指标脚手架）。本文方案将其升级为**完整端到端联合训练**（加入 box-feature / detectability 真检测梯度通道），并补齐合成器物理字段，**改写为 SCI 一区核心创新**。

---

## 2. 论文 §3.1 正式问题定义

### 2.1 符号表（先定义后使用）

| 符号 | 含义 | 张量形状 |
|------|------|--------|
| $J$ | 清晰无雾无冰图像（GT） | $H\times W\times 3$ |
| $I$ | 观测的雾 + 覆冰图像 | $H\times W\times 3$ |
| $D(x)$ | 冰层厚度图（每像素） | $H\times W$ |
| $\beta$ | 冰层 Beer-Lambert 消光系数 | 标量 |
| $\alpha_{\text{ice}}(D)$ | 冰层不透明度 = $1-\exp(-\beta D)$ | $H\times W$ |
| $S(x)$ | 冰面镜面反射（前向散射 + 环境光） | $H\times W\times 3$ |
| $t(x)$ | 大气透射率（0~1） | $H\times W$ |
| $A$ | 大气光（RGB，向量） | $3$ |
| $\mathcal{K}$ | 边缘模糊核（冰诱导 + 离焦） | $k_h\times k_w$ |
| $R(\cdot;\theta)$ | 待训练去雾主干（含 HA-WFE + CLIP fog prompt） | 神经网络 |
| $D(\cdot;\phi)$ | 检测头（YOLOv8 / DETR；可冻结骨干 + 微调 neck+head） | 神经网络 |
| $\mathcal{B}^{*}$ | 人工标注的 GT 框集 | $N\times 4$ |

### 2.2 复合退化模型

把"雾天覆冰输电线"成像分解为 **4 个物理级联**：

$$\boxed{\;
I \;=\; \mathcal{K} \star \Big[\,\underbrace{(J \odot T_0(D) + \alpha_{\text{ice}}(D) \odot S)}_{\text{Step A: 冰层复合 (Beer-Lambert)}}\;\Big] \;\odot\; t(x)\;+\; A\odot(1 - t(x))\;\Big]\;}
\tag{1}$$

其中：

- $T_0(D) = \exp(-\beta D)$：冰层透射（Beer-Lambert）
- $\alpha_{\text{ice}}(D) = 1 - \exp(-\beta D) \in [0,1)$：冰层不透明度
- $S$ 的物理分解：$S = R_{\text{spec}}\cdot I_{\text{env}} + R_{\text{trans}}\cdot J_{\text{trans}}$，表征镜面反射（$R_{\text{spec}}$，强方向性）+ 半透明次表面散射（$R_{\text{trans}}$，各向同性漫射），二者权重按冰晶形态分布采样得到。
- $\mathcal{K} \star [\cdot]$：高斯 + 各向异性（沿导线方向拉长）的二维卷积，模拟冰层在轮廓处的扩散性模糊。**默认** $\mathcal{K}$ 为 $5\times 5$、$\sigma=[1.0, 2.5]$、沿水平方向扩散（电线主导方向）。

**逆向分解（去雾网络要学的就是这个算子的逆）**：给定 $I$，恢复 $J$，并显式地最小化以 $J$ 为输入时检测器在 GT 框上的任务损失。

### 2.3 与既有 `synthesize_hazy_iced` 的差异

| 维度 | 既有（P1-1 状态） | 本文方案（升级点） |
|------|----------|---------|
| 冰层透射 | 仅 $\alpha$（无 $S$） | 加入镜面反射 $S$ 与半透明散射项 |
| 大气散射 | Koschmieder，$A$ 标量 | 保留，允许空间不均（远处 $A$ 较高） |
| 边缘模糊 | **缺失** | 引入各向异性 $\mathcal{K}$ |
| 物理可微 | 仅前向合成；不能反传 | 在 PyTorch 中以 `Conv2d(weight=)` 形式存在，**全链路可微** |
| 元数据返回 | `{t_map, A, ice_alpha, ice_thickness}` | 增加 `S, blur_kernel_id, ice_morphology` |

### 2.4 问题的形式化（作为损失期望）

$$\min_{\theta,\phi}\;\mathbb{E}_{(I,J,\mathcal{B}^{*}) \sim \mathcal{D}}\Big[\;\mathcal{L}_{\text{recon}}\big(R_\theta(I), J\big) \;+\; \mathcal{L}_{\text{det-task}}\big(D_\phi(R_\theta(I)), \mathcal{B}^{*}\big) \;\Big]
\tag{2}$$

$R_\theta(\cdot)$ 是去雾主干（含 HA-WFE + CLIP 雾提示）；$D_\phi(\cdot)$ 是检测器（**骨干冻结、head 微调**，避免检测器本身漂移）。$R_\theta$ 的全部参数（HA-WFE $\alpha_{*}/\beta$、CLIP prompt 投影、DehazeFormer 全部 block）在训练中被 §3 的检测梯度反向传播更新——这就是"joint"语义。

---

## 3. 论文 §3.2 检测感知损失设计

### 3.1 损失函数族总览

把"检测感知"拆成 **4 个子项**，每项有明确物理含义：

$$\boxed{\;\mathcal{L}_{\text{det-aware}} \;=\; \underbrace{\mathcal{L}_{\text{box-feat}}}_{\text{框内特征保持}} \;+\; \underbrace{\mathcal{L}_{\text{detectability}}}_{\text{可检测性代理}} \;+\; \underbrace{\mathcal{L}_{\text{box-align}}}_{\text{框对齐辅助}} \;+\; \underbrace{\mathcal{L}_{\text{corridor}}}_{\text{走廊纹理（保留作辅助）}} \;}\;
\tag{3}$$

> **关键设计决定**：把 `ITLLoss`（基于规则伪标签的冰区约束）从主损失中**降级或剔除**，改由 $\mathcal{L}_{\text{box-feat}}$ 直接在 GT 框区域上做特征方向的语义保持——**彻底切断"规则造标签→规则评效果"的循环论证**。同时保留 `ITLLoss`（仅当有人工冰掩码时启用）作为可选正交约束。

### 3.2 子项 1：框内特征保持损失 $\mathcal{L}_{\text{box-feat}}$

设 $\Phi_l(\cdot; x)$ 为检测器 $D_\phi$ 在第 $l$ 层（取 neck 的 P3/P4/P5 三个尺度的输出）、空间位置 $x$ 处的特征向量。对每张图、每个 GT 框 $b_i=(x_1,y_1,x_2,y_2)$：

$$\mathcal{L}_{\text{box-feat}} \;=\; \sum_{l \in \{P3,P4,P5\}}\; \sum_{i=1}^{N}\; \frac{1}{|b_i|}\,\sum_{x \in b_i}\; \Big[\,1 - \cos\big(\Phi_l(R_\theta(I); x),\;\Phi_l(J; x)\big)\,\Big]
\tag{4}$$

- 余弦相似度保留**方向**而非幅度，避免重建噪声破坏特征语义。
- 跨尺度加和：P3 负责小目标（冰瘤、绝缘子伞裙），P5 负责大目标（杆塔），二者梯度互补。
- $J$（清晰图）的检测器特征作为锚点，提供"目标域分布"——这是把"GT 真实分布"显式注入去雾训练的桥梁。

### 3.3 子项 2：可检测性代理损失 $\mathcal{L}_{\text{detectability}}$

去雾网络可能过平滑到检测器什么也不看见；为防止这种现象：

$$\mathcal{L}_{\text{detectability}} \;=\; -\,\frac{1}{N}\;\sum_{n=1}^{N}\; \hat{p}_{\text{obj}}^{(n)} \cdot \mathbb{1}\big[\,\hat{p}_{\text{cls}^{(n)}}^{*} > \tau_{\text{cls}}\,\big]
\tag{5}$$

- $\hat{p}_{\text{obj}}^{(n)}$：检测器在 anchor $n$ 的 objectness 分数。
- $\mathbb{1}[\cdot]$：仅在 GT 类别的类别概率 $\hat{p}_{\text{cls}^{(n)}}^{*} > \tau_{\text{cls}}$（默认 0.3）时计入；忽略背景 anchor。
- **负号**：最大化"被检测的置信度"，**等价于告诉去雾网络：去雾结果要让检测器高自信**。

实现要点：检测器需提供 (a) raw logits & objectness；(b) decode 前向可微路径（YOLOv8 的 `model.model[-1]` 之前的 head 可被替换为 **soft decode**）。

### 3.4 子项 3：框对齐辅助损失 $\mathcal{L}_{\text{box-align}}$

让"去雾前后"的检测结果在**框几何层面**与 GT 接近：

$$\mathcal{L}_{\text{box-align}} \;=\; \frac{1}{N}\,\sum_{n=1}^{N}\; \Big[\,\lambda_{\text{CIoU}}\,(1 - \text{CIoU}(\hat{b}_n, b_n^{*})) \;+\; \lambda_{\text{DFL}}\, DFL(\hat{c}_n, c_n^{*})\,\Big]
\tag{6}$$

- CIoU：同时考虑中心点距离、长宽比、IoU；YOLOv8 默认就是它。
- DFL（Distribution Focal Loss）：在 YOLOv8 中是关键正则项，保留可避免回归损失主导。
- 该损失通过 $D_\phi$ 反传回 $R_\theta$，**只有 $D_\phi$ head 微调 + $R_\theta$ 全参数更新**这一拓扑才可用。

### 3.5 子项 4：走廊纹理保持损失 $\mathcal{L}_{\text{corridor}}$（降级为辅助）

保留既有 `CorridorTextureLoss`，仅在"走廊掩码"（来自预训练检测器对清晰图的框膨胀得到，**不来自规则伪标签**）区域作为辅助正则。建议权重 $0.1 \sim 0.3$，主损失被 $\mathcal{L}_{\text{box-feat}}$ 取代。

### 3.6 子项 5（可选）：冰面物理一致性损失 $\mathcal{L}_{\text{ice-phys}}$

用合成器元数据（含 $T_0(D)$ 与 $\alpha_{\text{ice}}(D)$）做半监督——这是"复合退化建模范儿"的工程价值兑现：

$$\mathcal{L}_{\text{ice-phys}} \;=\; \lambda_{\text{trans}}\, \big\| \hat{T}(R_\theta(I)) - T^*(D)\big\|_1 \;+\; \lambda_{\text{ice-}\alpha}\, \big\| 1 - \hat{\alpha}(R_\theta(I)) - \alpha_{\text{ice}}(D)\big\|_1
\tag{7}$$

$\hat{T}(\cdot), \hat{\alpha}(\cdot)$ 是去雾主干的两个轻量辅助 head（各 1×1 conv → sigmoid），输出估计的物理量。该项**只在合成数据上有监督**（真实雾无 $D$ 真值），但可作为合成→真实迁移的正则。

---

## 4. 论文 §3.3 多任务不确定性加权

### 4.1 公式（沿用 Kendall & Gal, CVPR'18）

$$\boxed{\;\mathcal{L}_{\text{total}} \;=\; \sum_{i=1}^{K}\; \exp(-s_i)\cdot \mathcal{L}_i\;+\;\frac{1}{2}\,\sum_{i=1}^{K} s_i\;}\;
\tag{8}$$

$s_i = \log \sigma_i^2$ 初始化为 0（即初始权重 1.0）；$K$ 在 joint 模式下取 5：

| $i$ | 损失项 | 物理含义 |
|------|------|--------|
| 1 | $\mathcal{L}_{\text{recon}}$ | 像素级重建（L1 + SSIM） |
| 2 | $\mathcal{L}_{\text{box-feat}}$ | 检测框内特征保持（C3） |
| 3 | $\mathcal{L}_{\text{detectability}}$ | 可检测性置信度（C3） |
| 4 | $\mathcal{L}_{\text{box-align}}$ | CIoU + DFL（C3） |
| 5 | $\mathcal{L}_{\text{ice-phys}}$ | 物理量一致性（C1, 合成数据） |

### 4.2 与 ITL 的协同

若 P0-4 人工标注完成且冰掩码可靠，可把 `ITLLoss` 作为第 6 项加入，作为正交支撑；否则不启用。所有 $s_i$ 一并被 `optimizer` 更新，进入 checkpoint 中的 `state["uncertainty"]` 字段（已有 `Trainer` 支持）。

### 4.3 不确定性加权 vs 手工 $\lambda$

| 维度 | 手工 $\lambda$ | 学习 $s_i$ |
|------|--------------|-----------|
| 调参工作量 | 高（K 维网格） | 几乎零（一次训练即可） |
| 任务尺度自适应 | 否 | 是（梯度量级吸收） |
| 训练稳定性 | 对 LR 敏感 | 对 LR 较不敏感 |
| 与既有实现差异 | — | API 兼容（`UncertaintyWeighting(num_losses=K)`） |

---

## 5. 论文 §3.4 训练流程

### 5.1 双阶段训练（避免检测器早期噪声干扰去雾学习）

| 阶段 | 训练内容 | 冻结 | 学习 | epochs | 必要性 |
|------|---------|------|------|--------|--------|
| **Stage-A (warm-up)** | 仅 $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{ice-phys}}$ | $D_\phi$（全冻结） | HA-WFE、DehazeFormer、CLIP prompt 投影、$\hat T$ / $\hat\alpha$ head | 5~10 | 让重建稳定，避免检测头随机梯度污染去雾主干 |
| **Stage-B (joint)** | 全 5 项损失 + 不确定性加权 | $D_\phi$ 的 **backbone**；neck 末层微调 | HA-WFE、DehazeFormer、CLIP prompt 投影、辅助 head、$D_\phi$ neck+head、不确定性 $s_i$ | 30~50 | 真正端到端联合优化 |

> 关键：Stage-B 不是"全部可训练"——**检测器 backbone 必须冻结**，否则会因去雾结果"太好检测"导致检测器放弃学真实分布而只学去雾分布，违反**外泛化假设**。只微调 neck+head 是折中。

### 5.2 训练伪代码（落到 PyTorch）

```python
# joint 模式主循环（伪代码，paper supplementary）
for stage, cfg in [("A", warm_cfg), ("B", joint_cfg)]:
    if stage == "B":
        detector.backbone.requires_grad_(False)       # 冻结 backbone
        detector.head.requires_grad_(True)             # 微调 head
        uncertainty = UncertaintyWeighting(K=5).to(device)

    for batch in loader:                               # batch: (I, J, mask, bbox, cls)
        I, J, mask_ice, bboxes, classes = [t.to(device) for t in batch]

        # Stage-A: 不读 bboxes/classes, 走老路
        R = dehaze_net(I)
        if stage == "A":
            loss = recon(R, J) + ice_phys(R, J)
            loss.backward(); optimizer.step(); continue

        # Stage-B: 全损失
        # (1) 检测前向（detector 在 R 上）
        det_out = detector(R)                          # dict: {'cls': [B,N,C], 'box': [B,N,4], 'obj': [B,N]}

        # (2) 各项损失
        L1 = l1_loss(R, J) + 0.1 * ssim_loss(R, J)
        L2 = box_feat_loss(R, J, bboxes, detector)     # 余弦，多尺度
        L3 = detectability_loss(det_out, classes)
        L4 = box_align_loss(det_out, bboxes, classes)  # CIoU + DFL
        L5 = ice_phys_loss(R_hat_aux, J, mask_ice)

        # (3) 不确定性加权
        loss = uncertainty([L1, L2, L3, L4, L5])
        loss.backward()
        clip_grad_norm_(params, max_norm=10.0)
        optimizer.step(); scheduler.step()
```

### 5.3 完整训练超参（建议起点）

```yaml
# configs/train/joint_v2.yaml
seed: 42
model:
  version: joint_v2          # 新 version
  backbone: s
  init_checkpoint: ${ICEWAVE_WEIGHTS_DIR}/checkpoints/m4_best.pth

data:
  root: ${ICEWAVE_DATA_ROOT}/dataset
  patch_size: 192
  with_bboxes: true          # 新增：dataset 须读 bbox txt

train:
  stages:
    warm_up:
      epochs: 8
      lr: 1.0e-4
      freeze_detector: true
    joint:
      epochs: 40
      lr: 5.0e-5
      freeze_detector_backbone: true     # 只冻结 backbone
      grad_clip: 10.0

losses:
  K: 5
  lambda_recon_l1: 1.0
  lambda_recon_ssim: 0.1
  lambda_box_feat: 0.5
  lambda_detect: 0.2
  lambda_box_align: 0.5
  lambda_ice_phys: 0.3
  s_init: 0.0                   # log sigma^2 起点
```

---

## 6. 论文 §3.5 下游任务增益评测协议

### 6.1 协议公式

**单次去雾增益**：

$$\Delta_{\text{gain}} \;=\; \text{mAP}_{\text{dehazed}} - \text{mAP}_{\text{hazy}}
\tag{9}$$

**残差收益空间（Headroom）**：

$$\Delta_{\text{gap}} \;=\; \text{mAP}_{\text{clear}} - \text{mAP}_{\text{dehazed}}
\tag{10}$$

**归一化恢复率**：

$$R \;=\; \frac{\Delta_{\text{gain}}}{\text{mAP}_{\text{clear}} - \text{mAP}_{\text{hazy}}},\;\;R \in (-\infty, 1]
\tag{11}$$

**物理含义**：$R=1$ 表示"去雾完全恢复了清晰图带给检测器的全部能力"；$R<1$ 表示"去雾仍有提升空间"；$R<0$ 表示"去雾反而让检测更差"（**审稿人最关心的负面结果**）。

### 6.2 与单 PSNR 评测的关键差异

| 维度 | 单 PSNR | $\Delta_{\text{gap}}$ + $R$ |
|------|---------|--------------|
| 评测对象 | 图像保真度 | 去雾对**任务**的价值 |
| 与 SOTA 比较能力 | 是 | 间接（通过 gap 是否小于 SOTA） |
| 对下游真实工业价值 | **无直接证据** | **强证据**（电力巡检最终落在检出数） |
| 算力成本 | 低 | 中（跑两遍 YOLO） |

### 6.3 既有 `eval/downstream.py` 的升级点

| 既有输出 | 本文升级 | 改动代码 |
|----------|---------|---------|
| $\text{mAP}_{\text{hazy/dehazed/clear}}$ | 同左 | 已就绪 |
| 无 | $\Delta_{\text{gain}}$、$\Delta_{\text{gap}}$、$R$（含 95% CI） | `src/icewave/eval/downstream.py` 新增 `compute_gain_stats(...)` |
| 无 | 按雾档（thin/medium/dense）分组的 $R$ 表 | `src/icewave/eval/downstream.py` 读 haze level metadata |
| 无 | 与基线（Cascade/M1/M4）的成对 $t$ 检验 | 同上 |

### 6.4 报告样例（论文表格预期形态）

| Method | mAP_hazy | mAP_dehazed | mAP_clear | $\Delta_{\text{gain}}$ | $R$ (%) |
|--------|----------|-------------|-----------|--------------|---------|
| Cascade (M4→YOLO, uncoupled) | 0.412 | 0.483 | 0.602 | +0.071 | 37.4 |
| **Joint (本文)** | 0.412 | 0.531 | 0.602 | **+0.119** | **62.8** |
| Ablation − $\mathcal{L}_{\text{box-feat}}$ | 0.412 | 0.498 | 0.602 | +0.086 | 45.3 |
| Ablation − 不确定性加权 | 0.412 | 0.514 | 0.602 | +0.102 | 53.7 |

> 三种子实验汇总，附 95% CI 与显著性标记。这是投稿审稿重点表格。

---

## 7. 数据集扩展（论文 §4.1）

### 7.1 P0-4 子任务（A1）—— 人工标注验证子集

**输入**：既有合成验证集 84 对 + 真实 673 张。
**工作量**：150-300 张，类别分布：
- insulator（含冰 / 无冰）
- power_line（含冰 / 无冰）
- ice（独立标注）
- tower（含冰 / 无冰）

**格式**：YOLO txt（与检测器训练同格式），坐标归一化。
**协议**：
- 双人独立标注 → Cohen's κ ≥ 0.75 才能进入验证集
- 冰区边界由第二标注员专门核对
- 提供 `ice_mask_human/` 目录优先于 `ice_mask_rule/`（已有逻辑，见 `data/dataset.py`）

### 7.2 P0-5 增补——合成器扩展字段

`src/icewave/data/degradation.py` 当前缺：**specular $S$**、**半透明散射层**、**边缘模糊核** $\mathcal{K}$、**morphology label**。本次升级要点：

```python
# 新增 dataclass 字段（data/degradation.py）
@dataclass
class IceParams:
    enabled: bool = False
    extinction: float = 2.5
    coverage: float = 0.5
    max_thickness: float = 1.0
    texture_strength: float = 0.15
    # —— 新字段 ——
    specular_strength: float = 0.4   # 镜面反射比例
    translucent_strength: float = 0.3 # 半透明散射比例
    blur_sigma_x: float = 1.0        # 各向异性模糊 σ_x
    blur_sigma_y: float = 2.5        # σ_y（沿导线方向拉长）
    morphology: str = "shell"        # shell / sleeve / lobe / cluster
```

合成入口 `synthesize_hazy_iced` 增加 `apply_blur=True, with_speckle=True` 标志；元数据 dict 新增 `S, blur_id, morphology_label`。

---

## 8. 实施计划（按用户拆分）

### 8.1 A1 —— 归入 P0-4 子任务（**2 周**）

| 工作日 | 里程碑 | 产物 | 验证 |
|--------|--------|------|------|
| **D1-D3** | 制定人工标注协议（双标注、Cohen's κ） | `data/ANNOTATION_PROTOCOL.md` | 评审通过 |
| **D4-D10** | LabelImg 标注 150-300 张（双人并行） | `data/dataset/val/labels_human/*.txt` | 双人 κ 计算脚本（`tests/test_annotation_kappa.py`） |
| **D11** | 数据集接入：`IceAwareDataset` 优先读 `labels_human/` | `data/dataset.py` 改造 | `tests/test_dataset.py::test_human_label_priority` |
| **D11** | 引入 `BoxFeaturePreservationLoss` 骨架（无 YOLO 检测器的前向可微替代） | `losses/detect.py` 新增 `BoxFeatStub`（基于 Sobel + 像素特征） | `tests/test_box_feat_loss.py` |
| **D12-D14** | 在 m4 训练流中启用 $\mathcal{L}_{\text{box-feat}}$（warm-up 阶段），跑通 M4→BoxFeat 1 次训练 | `configs/train/m4_boxfeat.yaml` + 1 次训练日志 | 验指标对比（`m4 vs m4+boxfeat`） |

**退出标准**：
- 150 张人工标注完成，$\kappa \ge 0.75$
- `BoxFeatStub` 的单元测试通过
- M4 + BoxFeat vs M4 在 PSNR/SSIM 上**不退化**（允许 ±0.2 dB 之内），证明该损失未破坏重建

### 8.2 A2 —— 升级为 P1-1 顶级（**4–6 周**）

| 周次 | 里程碑 | 产物 | 验证 |
|------|--------|------|------|
| **W1** | 扩展 `synthesize_hazy_iced`（镜面 $S$、模糊 $\mathcal{K}$、morphology） | `degradation.py` + `tests/test_composite_degradation.py`（新） | `python -m pytest tests/test_composite_degradation.py -v` |
| **W1** | `IceAwareDataset` 支持 bbox 标签读取 | `data/dataset.py` | `tests/test_bbox_loading.py` |
| **W2** | 引入可微检测头：`DetectorWrapper`（冻结 YOLOv8 backbone，可微 head） | `models/detector_wrapper.py` + `tests/test_detector_wrapper.py` | yolo 权重加载 + 前向 + 反传梯度到 R_θ（数值检验） |
| **W2** | 实现 `BoxFeaturePreservationLoss`（多尺度，余弦） | `losses/detect.py` 新增 | `tests/test_box_feat_full.py`（与 stub 对齐） |
| **W3** | 实现 `DetectabilityLoss` 与 `BoxAlignLoss` | `losses/detect.py` | `tests/test_detect_losses.py` 扩展 |
| **W3** | `UncertaintyWeighting(K=5)` 接入，含 warm-up 阶段 | `train/trainer.py` 新增 `version='joint_v2'` 路径 | `tests/test_uncertainty_K5.py` |
| **W4** | 训练主循环贯通（Stage-A 5 ep + Stage-B 30 ep） | `configs/train/joint_v2.yaml` + 端到端 1 次训练 | 训练 Loss 曲线各分量均为有限值、无 NaN |
| **W4** | `eval/downstream.py` 升级：compute_gain_stats、按雾档分组、CI 估计 | `tests/test_downstream_gain.py` | 在 M4 已有 checkpoint 上跑 baseline 数字 |
| **W5** | 真训练（合成 + 真实混合，6 epochs 试训 + 30 epochs 主训） | `runs/joint_v2/checkpoints/` + `metrics.json` | `Δ_gain` 与 $R$ 与 Cascade 基线对比，**目标 $R > 0.45$** |
| **W5** | 消融：去掉 $\mathcal{L}_{\text{box-feat}}$ / 去掉不确定性加权 / 冻结主干 | 4 组训练（每组 ~12 GPU-h） | 各组 $R$ 列表 + 显著性检验 |
| **W6** | 写论文 §3.2 / §3.3 / §4.3 完整正文 | `docs/paper/section_3_2_joint.md` + `section_3_3_uncertainty.md` + `section_4_3_eval.md` | 章节自审 + 3 轮内部评审 |

**退出标准**：
- 端到端 1 次完整训练无 NaN、各 Loss 项收敛
- $R > 0.45$（vs Cascade 基线 $R \approx 0.37$）
- 4 组消融全部完成，箱线图 + $t$ 检验可用
- 论文 §3.2-§3.4、§4.3 完整文字版 4 节

### 8.3 里程碑甘特图（文字版）

```
          W1   W2   W3   W4   W5   W6
A1:       |||---||--|
 标注协议
 标注
 接 dataset
 引入 BoxFeat stub
 A1 退出

A2:           |||---|||---|||---|||
 扩展合成器
 接 bbox 可微检测头
 实现三损失
 Uncertainty K=5
 端到端训练
 Δ_gain 评测
 消融 (4 组)
 论文 §3.2-§4.3
 A2 退出
```

### 8.4 GPU / 算力预算

| 阶段 | GPU 数 | GPU-h | 备注 |
|------|--------|-------|------|
| A1 (M4+BoxFeat 1 次训练) | 1 | 4 | 8 ep warm-up |
| A2 W4 端到端 1 次完整 | 1×A100 | 28 | Stage-A 8ep + Stage-B 40ep |
| A2 W5 消融 4 组 × 12h | 1 | 48 | 共享同一 GPU 串行 |
| **合计** | — | **80 GPU-h** | 折合 A100 ≈ ¥640 |
| 缓冲 + 失败回退 | 1 | +30 | +¥240 |
| **总计** | — | **~110 GPU-h ≈ ¥880** | AutoDL 现价 |

### 8.5 风险与回退

| 风险 | 影响 | 概率 | 回退方案 |
|------|------|------|---------|
| YOLOv8 检测梯度数值不稳定 | Stage-B NaN | 中 | 改用 DETR head（在 `models/detector_wrapper.py` 抽象层切换） |
| 不确定性加权退化（$\sigma$ 漂移） | 训练震荡 | 低 | 限制 $\log \sigma^2 \in [-6, 6]$（论文 §3.3 中讨论） |
| $R$ 不显著高于 Cascade | 论文创新降级 | 中 | 仅提交 $\mathcal{L}_{\text{box-feat}}$ 单项（A1），仍属有意义贡献 |
| 浓雾档增益为负 | 重要负面结果 | 中 | 论文中正面承认，给出"浓雾需要专用恢复分支"的未来工作 |
| 人工标注 κ 不达标 | 训练带噪声 | 低 | 引入第三标注员仲裁 |

---

## 9. 关键代码改动清单（落到文件与函数）

| 文件 | 新增/修改 | 函数 / 类 | 行数估计 |
|------|----------|---------|---------|
| `src/icewave/data/degradation.py` | 修改 | `IceParams` 增 4 字段；`synthesize_hazy_iced` 增 `apply_blur, with_speckle` | +90 行 |
| `src/icewave/data/dataset.py` | 修改 | `IceAwareDataset` 增 `return_bboxes` 模式，输出 `(I, J, ice_mask, corridor, bbox, cls)` | +50 行 |
| `src/icewave/models/detector_wrapper.py` | 新增 | `DetectorWrapper(load_yolo)`, `soft_decode_heads()`, `extract_neck_features()` | +200 行 |
| `src/icewave/losses/detect.py` | 修改 | 新增 `BoxFeaturePreservationLoss`, `DetectabilityLoss`, `BoxAlignLoss`, `IcePhysicalLoss` | +180 行 |
| `src/icewave/train/trainer.py` | 修改 | 新增 `version='joint_v2'` 路径与 Stage-A/B 切换；`UncertaintyWeighting(K=5)` 接入 | +120 行 |
| `src/icewave/eval/downstream.py` | 修改 | 新增 `compute_gain_stats()`, `per_haze_level_report()`, `paired_t_test()` | +150 行 |
| `configs/train/joint_v2.yaml` | 新增 | 完整 Stage-A/B + 5 项损失配置 | +50 行 |
| `tests/test_composite_degradation.py` | 新增 | 验证镜面/模糊/morphology 字段已加入 | +100 行 |
| `tests/test_detector_wrapper.py` | 新增 | 验证检测头可微、特征提取回传正常 | +80 行 |
| `tests/test_box_feat_full.py` | 新增 | 全功能 BoxFeat（含多尺度） | +90 行 |
| `tests/test_detect_losses.py` | 修改 | 增加 3 个新损失的数值与梯度测试 | +120 行 |
| `tests/test_downstream_gain.py` | 新增 | 验证 $\Delta_{\text{gain}}$、$R$、分组、t 检验 | +90 行 |

**总计增量**：~1320 行（含测试）

---

## 10. 论文写作配套（直接可拷）

### 10.1 §3.2 章节摘要（约 400 字，SCI 顶级风格起手）

> "We propose a **detection-aware joint optimization** framework that replaces the conventional cascade of separate dehazing and detection networks with an end-to-end trainable system. Unlike prior approaches that couple dehazing and detection only through post-hoc image quality metrics (PSNR/SSIM), our framework propagates the detection task gradient back to the dehazing backbone through three differentiable surrogates: (i) a **box-internal feature preservation** loss that anchors the dehazed feature distribution, across multiple detection-neck scales, to the feature distribution of the haze-free oracle image; (ii) a **detectability loss** that penalizes over-smoothing by maximizing the detector's objectness confidence on the dehazed output; and (iii) a **box-alignment loss** that combines CIoU regression with distribution focal loss. These three surrogates, together with the reconstruction loss and an optional ice-layer physical consistency loss, are balanced by **learnable task uncertainties** (Kendall & Gal, 2018). The joint stage keeps the detector backbone frozen and fine-tunes only the neck and head, ensuring that the detector does not drift toward the dehazed distribution alone."

### 10.2 §4.3 评测指标章节起手

> "Traditional dehazing benchmarks report only PSNR/SSIM, which measure image fidelity but not downstream utility. We introduce the **downstream task gain** protocol: a single YOLOv8 detector with frozen weights is run on (a) hazy input, (b) dehazed output, and (c) haze-free ground truth; the three resulting mAP@0.5 values yield two derived quantities — the **dehaze gain** $\Delta_{\text{gain}}=\text{mAP}_{\text{dehazed}}-\text{mAP}_{\text{hazy}}$ and the **normalized recovery ratio** $R=\Delta_{\text{gain}}/(\text{mAP}_{\text{clear}}-\text{mAP}_{\text{hazy}})$. A perfect dehazer achieves $R=1$; negative $R$ indicates that dehazing actively hurts detection."

### 10.3 审稿应对（Q&A 预判）

| 审稿问题 | 应答要点 |
|---------|---------|
| "为什么不直接用 YOLOv7 / DETR？" | 实验对比在消融章节展示 head 替换便利性；主干冻结使替换仅改 ~30 行 |
| "为什么要冻结 backbone？" | 否则检测器被去雾输出分布污染；论文 §3.3 实验有 ablation |
| "$R=0.628$ 与已有工作可比吗？" | 这是新指标，首次定义，**直接对比对象是 Cascade 基线** |
| "盒内余弦对重建没贡献是因为清晰图也未必真" | 是的；论文承认$\mathcal{L}_{\text{recon}}$ 才是重建主项，box-feat 是检测感知约束 |
| "为何不直接端到端训练检测器全部？" | 见 Stage-B 拓扑：head 微调更稳且与 frozen-backbone 评估协议一致 |
| "不确定性加权是否会让某项 loss 变成 0 失效？" | 加 0.5·$\sum s_i$ 正则防止 $s \to \infty$，且每 5 epoch 输出 $s_i$ 监控 |

---

## 11. 与既有 P1-1 的关系（兼容性保证）

| 既有 | 本文 | 兼容 |
|------|------|------|
| `build_model('joint')` | `build_model('joint_v2')` | 新增 version；旧 `joint` 行为不变 |
| `UncertaintyWeighting(num_losses=4)` | `UncertaintyWeighting(num_losses=5)` | K 是参数，新版本调成 5 即可 |
| `eval/downstream.py` mAP 输出 | 同上 + $\Delta_{\text{gain}}/R$ | 旧输出保持，向下兼容 |
| `ITLLoss` | 仍可用，作为可选 $\mathcal{L}_{\text{ice-phys}}$ 正交项 | 完全保留 |
| `CorridorTextureLoss` | 降为辅助项（可选） | 仍可用 |

**向旧用户保证**：现有 M4/joint 配置和检查点不需迁移；只是新增一条 `joint_v2` 路径，达到 SCI 一区核心创新点。

---

## 12. 总结（执行摘要）

- **方案核心**：用"复合退化模型 + 检测感知损失（box-feat + detectability + box-align）+ 不确定性加权 + ΔmAP 评测"四件套替代级联。
- **创新点(C2-C4)** 均直接对接审稿人会问的："去雾对下游真的有用吗？"
- **A1 (P0-4 子任务, 2 周)**：人工标注 + BoxFeat stub 起步，**即可发表最低限度贡献**。
- **A2 (P1-1 顶级, 4-6 周)**：端到端真训练，**核心创新兑现**。
- **总 GPU 预算**：~110 GPU-h ≈ ¥880（AutoDL A100）。
- **最坏回退**：A1 单项完成也可作为一项有意义的增量贡献投稿。

**下一步建议**（给作者）：
1. 本周内确认 A1 启动（标注协议 + 招募标注员）
2. W1 同步扩展 `synthesize_hazy_iced` 物理字段（无 GPU 依赖，纯算法工作）
3. W2 启动 `DetectorWrapper` 与 3 个新损失的开发

---

> **本文档撰写基础**：commit `8bc038a` (HEAD)。**所有列出的代码改动尚未提交**，将在 A1/A2 实施过程中分批落库。
