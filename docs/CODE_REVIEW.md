# turbo-icespoon (IceWave) 代码审查报告

> **审查对象**: https://github.com/Kristen-net/turbo-icespoon @ commit `11e52df`（含本报告提交 = `ca9b27a`）
> **审查基线**: 相对原项目（vendored DehazeFormer + HazeCLIP）的全部自有改动
> **审查方式**: 全量人工阅读 `src/icewave/**` + `scripts/` + 关键路径本地复现验证（Python 3.11 venv, torch 2.14 CPU）
> **审查日期**: 2026-09-07
> **结论速览**: 工程化质量**总体优秀**（分层清晰/文档密度高/146 个测试/CI 四 job），但发现 **2 个 P0 级训练正确性 bug（均已本地实证）**、5 个 P1 级缺陷。⚠️ **P0-1 与 P0-2 共同意味着：W5 跑出的 joint_v2 训练结果（Best PSNR=18.90）与消融结论（"检测感知损失损害泛化"）均不可作为论文证据，须修复后重训。**

---

## 目录

1. [改动整体概述与主要目的](#1-改动整体概述与主要目的)
2. [逐模块改动内容与影响范围](#2-逐模块改动内容与影响范围)
3. [代码质量评估](#3-代码质量评估)
4. [潜在 bug / 安全隐患 / 边界情况（按严重度分级）](#4-潜在-bug--安全隐患--边界情况)
5. [改进建议与最佳实践](#5-改进建议与最佳实践)
6. [附录：已复现 bug 的最小复现脚本](#6-附录已复现-bug-的最小复现脚本)

---

## 1. 改动整体概述与主要目的

原项目 = DehazeFormer（去雾骨干，BSD-3）+ HazeCLIP（CLIP 蒸馏教师）。自有改动把它重构成一个面向"雾天输电线路覆冰检测"的科研系统，共 5 类：

| # | 改动类别 | 位置 | 目的 |
|---|---------|------|------|
| ① | **模型创新** | `src/icewave/models/hawfe.py` | HA-WFE（Haar 小波增强）以标准 nn.Module 子类插入 DehazeFormer 瓶颈层，替换旧 monkey-patch；CLIP 雾提示分支（M3/M4） |
| ② | **损失函数** | `src/icewave/losses/` | ITL 覆冰感知损失（修复 3 处旧 bug）、走廊纹理保持损失、Kendall&Gal 不确定性加权（joint 模式） |
| ③ | **数据管线** | `src/icewave/data/` | 雾+覆冰复合退化物理合成器（Beer-Lambert × Koschmieder）、场景级切分数据集构建器、人工标注优先的 IceAwareDataset |
| ④ | **训练/评测/推理** | `src/icewave/train/ eval/ infer/` | YAML 配置驱动训练器（5 种模式）、公开基准 harness、下游任务增益（ΔmAP）评测、去默认自动重训的推理 CLI |
| ⑤ | **工程化** | CI/Docker/文档/测试 | 4-job CI、Apache-2.0 + 第三方 NOTICE、96 测试函数、可复现性材料包 |

**主要目的**（从提交历史与文档推断，且达成度较高）：把一次性的实验脚本仓库升级为 SCI 一区可投稿的规范化研究代码库——路径参数化、检查点兼容、循环论证修复（人工标注优先）、可复现协议。

**遗留的"计划态"**：`phase1~6/`、`ice_detection/` 旧脚本目录仍保留（约 100+ 文件，双实现已标注 deprecation），是刻意保留的溯源副本，但构成维护负担（见 §5.1）。

---

## 2. 逐模块改动内容与影响范围

### 2.1 `src/icewave/models/`（模型层）★改动质量高

| 文件 | 改动 | 影响范围 |
|------|------|---------|
| `dehazeformer.py` | vendored，仅 timm API 兼容性修改（NOTICE 声明） | 全部模型的地基 |
| `hawfe.py` | **核心创新**。HAWFE v1/v2（零初始化 Tanh / Sigmoid 门控）、HaarDWT/IWT、IceWaveDehazeFormer 子类、`build_model()` 工厂（m1/m2/m2p/m3/m4/joint）、`load_checkpoint()` | 所有训练与推理；旧检查点 `strict=True` 直接加载（有测试保证） |
| `prompt.py` | CLIPFogPrompt（CS-ViT-B/32 空间特征→32 通道提示）、HazeCLIPTeacher（MSBDN 教师，冻结）；路径三级解析（参数>env>默认） | M3/M4/joint 训练；推理不依赖 CLIP（卖点） |

**亮点**：`_BACKBONE_SPECS` 显式列骨干超参并注明"不可依赖类默认值"（曾捕获检查点键错位）；HA-WFE 内部强制 fp32 + autocast 关闭，保证小波数值精度与旧检查点行为一致；奇数尺寸 padding 同步裁剪（修复上游 bug）。

### 2.2 `src/icewave/losses/`（损失层）★改动质量高

| 文件 | 改动 | 影响范围 |
|------|------|---------|
| `itl.py` | 修复旧 ITL 三 bug（文档不符/batch 尺寸依赖/空掩码断梯度）；`ssim_map()` 导出供 eval 复用 | M4/joint 训练；eval/metrics |
| `detect.py` | CorridorTextureLoss（对数对比度保持 + 梯度 L1）、UncertaintyWeighting（可学习 log σ²） | joint 训练 |

**亮点**：空掩码返回 `pred.sum()*0.0`（graph-connected 零）避免 "does not require grad" 崩溃——这是很多同类代码的隐性坑，此处处理正确且有注释。

### 2.3 `src/icewave/data/`（数据层）⚠️ 含 P0 bug

| 文件 | 改动 | 影响范围 |
|------|------|---------|
| `degradation.py` | HazeParams/IceParams/HAZE_PRESETS、`synthesize_hazy_iced`（冰层复合→大气散射两步，无冰时与经典 ASM 逐字节相等） | 全部合成数据 |
| `dataset.py` | IceAwareDataset：hazy/clear 按 `_hazeN` 后缀配对、人工标注优先、验证中心裁剪、走廊膨胀 | 全部训练 |
| `build_dataset.py` | 暗通道过滤 clear/border、场景级切分（by_zip）、浓雾进轮转、路径参数化 | 数据集构建 |

### 2.4 `src/icewave/train/`（训练层）

`trainer.py`：配置驱动、prompt 投影层进 optimizer（修复旧 bug）、config_snapshot.yaml 落盘（含 git commit）、metrics.json 每 epoch 更新；`config.py`：`${ENV:default}` 展开 + deep_update；`cli.py`：`--override key.subkey=value`。

### 2.5 `src/icewave/eval/`（评测层）

`metrics.py`（PSNR/SSIM/冰区指标/伪标签一致性）、`benchmark.py`（公开基准 harness，padding 推理）、`downstream.py`（**自研 mAP@0.5 all-point 实现，不依赖 pycocotools**；ΔmAP 协议）。

### 2.6 `src/icewave/detect/`（检测层）

`yolo.py`（ultralytics 可选依赖 + 清晰报错）、`ice_mask.py`（规则伪标签，docstring 明示"仅兜底"）、`maskrcnn_adapter.py`（detectron2→torchvision 键映射，标注"历史兼容保留"）。

### 2.7 `src/icewave/utils/` + 工程件

`paths.py`（5 个 `ICEWAVE_*` 环境变量三级解析）、`seed.py`（worker_init_fn 齐全）；CI 4-job 分层（base→dev/detect/clis，needs 依赖正确）；Docker 双镜像；CITATION.cff；NOTICE.md 许可声明完整（BSD-3/AGPL/OpenAI 权重均如实标注）。

### 2.8 A2/W2–W6 增量提交（`0223bb7`…`11e52df`，+5800 行）

按 JOINT_OPTIMIZATION_FRAMEWORK 落地的第一批实现：

| 文件 | 内容 | 评价 |
|------|------|------|
| `models/detector_wrapper.py` | Stub/YOLOv8 双模式检测器封装（冻结 backbone） | ⚠️ 结构清晰但存在 P0-2 梯度死路与 P2-5 解析错误 |
| `losses/detect.py`（+391 行） | BoxFeaturePreservation / Detectability / BoxAlign(CIoU+DFL) / IcePhysical + σ clamp | 数值实现好，但整条链被 P0-2 废掉梯度 |
| `train/trainer.py`（+300 行） | `train_joint_v2` 双阶段（Stage-A warm-up + Stage-B joint, K=5） | 流程完整；`batch[-2]/batch[-1]` 取 bbox 的索引约定略脆弱 |
| `data/dataset.py`（+105 行） | `return_bboxes` + YOLO 标签读取 | ⚠️ 引入第四路裁剪错位（见 P0-1） |
| `data/degradation.py`（+155 行） | 镜面反射/各向异性模糊/morphology 字段 | 元数据 A bug（P1-1）未修 |
| `eval/downstream.py`（+143 行） | gain_stats/雾档分组/成对 t 检验 | ⚠️ scipy 未声明（P1-5） |
| `scripts/run_*.py` ×8 + `run_ablation_chain.ps1` | W5 训练/消融/评估一次性脚本 | ⚠️ 硬编码绝对路径回归（P2-6） |
| `reports/w5-experiment-report.html` | W5 消融分析（PSNR 曲线/AP/雷达图） | 呈现质量高，但底层实验受 P0-1/P0-2 混杂 |
| `docs/paper/section_3_2/3_3/4_3.md` | 论文正文三节 | 与代码同步意识好；数字须待重跑 |
| `configs/train/*.yaml` ×8 | joint_v2/消融/smoke 配置 | 配置齐全 |

**总评**：A2 提交的**框架完整性与测试数量**（+50 测试）值得肯定，但两个 P0 级正确性 bug 都藏在这批新代码中，且恰好破坏了它要证明的核心主张——**实现速度快于验证深度**的典型症状。

---

## 3. 代码质量评估

### 3.1 可读性 — **A（优秀）**

- 每个模块头部 docstring 讲清"旧版问题 → 本版改法"，如 `itl.py` 列出修复的 3 处旧 bug 及验证测试名，评审可追溯性极好。
- 关键决策有"为什么"注释（如 hawfe.py:341 为何不可依赖类默认值）。
- 中文注释一致、术语统一（雾档 thin/medium/dense、走廊 corridor）。
- 小瑕疵：`dataset.py:152-153` 存在死变量 `seed_y/seed_x`（恰是 P0 bug 的线索）；`infer/cli.py:136` `h, w` 未使用。

### 3.2 可维护性 — **A-**

**优点**：
- 单一事实来源：路径全走 `utils.paths`；模型构造全走 `build_model`；配置快照含 git commit，实验↔代码可对账。
- 测试设计聪明：detect 冒烟测试无需安装 600MB ultralytics；模型兼容性测试锁定旧检查点键名。
- 向下兼容策略明确：新增不破坏旧行为（`joint` 新 version 而非改 `m4`）。

**扣分项**：
- `phase1~6/` + `ice_detection/` 双实现虽标 deprecation，但仍是 ~100 文件的认知负担与 grep 噪音（`ice_mask_generator.py` 双实现已在审计中标注）。
- `benchmark.py` 的 `_dehaze` 被 `infer/cli.py` 和 `eval/downstream.py` 跨层 import（eval→infer 方向的私有函数 `_dehaze` 成为事实公共 API），应提升为 `icewave.infer.core`。
- `source_hazeclip/` 是不完整副本，导入失败靠运行期报错指引（`prompt.py:55-63` 处理得体，但属长期债）。

### 3.3 性能 — **B**

| 问题 | 位置 | 影响 |
|------|------|------|
| `lpips_score` 每图新建 LPIPS 模型 | `eval/metrics.py:42` | 基准评测慢一个量级（每图加载 AlexNet），大基准不可用；应模块级缓存 |
| DataLoader 默认 `num_workers=0` | `configs/train/*.yaml` | Windows 友好但 Linux 训练吃不满 IO；建议按平台给默认值 |
| `make_ice_texture` 5×5 手写双循环卷积 | `data/degradation.py:105-117` | 小图无碍；可用 `cv2.filter2D` 一行替（清晰性 trade-off，可不动） |
| `evaluate_set` GT 匹配为 O(D×G) 纯 Python 双循环 | `eval/downstream.py:139-150` | 千级框内可接受；万级需向量化 |

训练数值性能设计是加分项：bf16 优先 + HA-WFE 内强制 fp32 + GradScaler 仅 fp16 启用，dtype 策略显式且正确。

---

## 4. 潜在 bug / 安全隐患 / 边界情况

> 严重度：**P0 = 影响训练正确性必须先修**；P1 = 功能性缺陷；P2 = 边界/健壮性；P3 = 卫生问题。
> 标注 ✅ 已复现 的条目附验证脚本（§6）。

### P0-2（✅ 已实证）检测感知三损失对去雾主干的梯度全部被切断 —— joint_v2 "端到端联合优化"实际未发生

**位置**: `src/icewave/models/detector_wrapper.py:239-270`（新增于 A2-W2 提交 `0223bb7`）

两处叠加导致 L2/L3/L4 对去雾网络 **梯度恒为零**：

```python
def extract_neck_features(self, img):        # L2 (BoxFeat) 调用路径
    with torch.no_grad():                    # ← 整条特征提取包进 no_grad
        feats = self.backbone(img)
    ...
    return feats                             # YOLOv8 路径: 已 detach

def forward(self, img):                      # L3/L4 (Detectability/BoxAlign) 调用路径
    feats = self.backbone(img)
    if not any(p.requires_grad for p in self.backbone.parameters()):
        feats = [f.detach() for f in feats]  # ← backbone 冻结 ⇒ 显式 detach
    return self.neck(feats)
```

"冻结检测器 backbone 参数"（`requires_grad_(False)`）本不影响 **梯度对输入图像** 的回传——autograd 对输入仍建图。但代码误以为冻结即断链，又主动 `detach()`，把唯一通往去雾网络 `pred` 的梯度路径杀死了。

**本地实证输出**（Stub 模式，`freeze_backbone=True`，与本仓库 `joint_v2_train.yaml`/`smoke_test.yaml` 配置一致）：

```
L2 (BoxFeat)        pred.grad 存在但范数 = 0.0   ← graph-connected 零主导, 检测路径零贡献
L3 (Detectability)  pred.grad is None            ← 无计算图
L4 (BoxAlign)       pred.grad is None            ← 无计算图
neck 参数            有非零梯度                    ← 损失只在训练检测器 neck
```

**影响链**（严重）：
1. W5 joint_v2 训练中，Stage-B 去雾主干实际只被 L1（重建）更新——**论文 §3.2 的核心主张（检测梯度指导去雾）在代码层面未发生**；
2. 消融组间差异并非来自检测感知损失（它们梯度为零），而来自不确定性加权对 L1 有效权重的间接扰动 + 随机性——`reports/w5-experiment-report.html` 的"检测感知损失损害泛化"结论**不可信**；
3. 若照此写论文，审稿人复现代码即穿帮。

**修复**（~15 行）：
- `extract_neck_features` 去掉 `no_grad` 包裹；锚点侧（clear 图特征）可用 `torch.no_grad()` 单独处理；
- `forward` 删除 `detach()` 分支（冻结参数本身已足够阻止 backbone 权重更新）;
- 补测试：`loss.backward() 后 pred.grad.norm() > 0`（当前 17 个 detect loss 测试全部只断言数值，无一个断言梯度可达）。

### P0-1（✅ 已复现，且被新提交放大）训练模式 hazy/clear/ice/bbox 四路裁剪互不对齐 —— 监督信号被破坏

**位置**: `src/icewave/data/dataset.py:151-158, 172-173`

```python
if self.is_train:
    seed_y = np.random.randint(0, max(1, hazy.shape[0] - ps + 1))   # ← 计算后从未使用
    seed_x = np.random.randint(0, max(1, hazy.shape[1] - ps + 1))   # ← 死变量
    hazy = _crop(hazy, ps, center=False)    # 内部独立再随机
    clear = _crop(clear, ps, center=False)  # 又一次独立随机 → 与 hazy 不同位置!
    ...
    ice_arr = _crop(ice_arr, ps, center=not self.is_train)  # 第三次独立随机
```

`_crop(center=False)` 每次调用自行调 `np.random.randint` 取偏移，因此 **hazy、clear、ice_mask 三者各自裁在不同位置**。只要图像大于 patch_size（192），L1/SSIM/ITL 全部在比较"不同图像区域的像素"。

- 复现输出：`consecutive crops aligned: False`（§6 脚本）
- **⚠️ A2 提交（`0223bb7`）未修复此 bug，反而引入第四路不对齐**：`dataset.py:243-247` 新增 bbox 加载用**第一次**随机采样 `crop_y/crop_x` 平移坐标，而 hazy 用第二次（`_crop` 内部）、clear 第三次、ice 第四次——四个张量各自对齐到不同的裁剪窗口。BoxFeat/Detectability/BoxAlign 收到的框与检测器实际看到的图像**系统性错位**（偏移差可达 图像-128 像素）。
- **⚠️ W5 训练全程处于该状态**：`joint_v2_train.yaml` 用 512×512 合成图（`make_synthetic_augmented.py`）配 `patch_size: 128`，随机偏移域宽达 384px。W5 报告的低 PSNR（18.90）与"域迁移问题"结论中，**裁剪错位是未受控的最大混杂变量**。
- **为什么 146 个测试没抓到**：`test_patch_crop_val_centered` 只测验证集中心裁剪；`test_bbox_loading.py` 测试用了固定 crop_y/crop_x 参数直调 `_load_yolo_labels`，绕过了 `__getitem__` 的随机路径；训练随机裁剪对齐仍**零覆盖**。
- **修复**（10 行内）：`__getitem__` 先取一次 `(y0, x0)`，对 hazy/clear/ice 用显式窗口切片、bbox 用同一 `(y0, x0)` 平移；同步补测试 `test_train_crop_alignment`。

### P1-5（✅ 已实证）scipy 为未声明依赖，且 import 位于守卫之前 —— 测试套件在干净环境必红

**位置**: `src/icewave/eval/downstream.py:334`（新增于 `0223bb7`）

`paired_t_test()` 第一行就 `from scipy import stats`，之后才做 `n < 2` 早退。`scipy` 未出现在 `pyproject.toml`/`requirements.txt`/`environment.yml` 任何依赖声明中。本地干净 venv 实测：

```
FAILED tests/test_downstream_gain.py::TestPairedTTest::test_too_few_samples
ModuleNotFoundError: No module named 'scipy'
```

CI `test-base` job 只装 `pip install -e .` + pytest（无 scipy，timm/torchvision/opencv 均不传递依赖 scipy）→ 最新提交的 CI 应为红。修复：scipy 进 `[project.optional-dependencies]` 或用纯 numpy 实现 t 检验（成对 t 统计量一行可写）；import 移到守卫之后。

### P2-5 `_parse_detect_output` 对真实 YOLOv8 的输出布局解析错误（潜伏 bug）

**位置**: `src/icewave/models/detector_wrapper.py:156-183`

YOLOv8 Detect 头每 anchor 输出 `4×reg_max(=64) DFL 通道 + nc 类别`，**没有独立 objectness 通道**。当前代码假设 `4 + 4×4 = 20` 处有 obj、`box_dim=20` 布局——一旦 `detector_weights` 传入真实 YOLOv8 权重（W5 用 Stub 模式所以没炸），`obj`/`cls` 会从错误的通道切片，三个损失全部失真。另 `grid_h = grid_w = int(sqrt(n_anchors))` 仅在方形输入下碰巧正确。修复：按 ultralytics `Detect.forward` 的真实输出契约重写，或直接调用 `model.model.model[-1]` 的官方解析。

### P2-6 W5 评估脚本重新引入硬编码绝对路径 —— 对 P0-1 重构成果的回归

**位置**: `scripts/run_real_eval.py:8-24`（及 `run_eval_part2.py`/`run_full_eval.py`/`run_downstream_eval.py` 同模式）

```python
REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)                                             # import 期副作用
yolo_weights = Path(r"D:\dehaze_fusion\yolo_train_output\power_line_yolo\weights\best.pt")
```

整个 P0-1 重构消灭的 130+ 处硬编码路径（`.trae-cn`、`D:\dehaze_fusion`），在 8 个新 `scripts/run_*.py` 中原样回归，且含 `os.chdir` import 期副作用。这些脚本在作者本机之外**不可运行**。修复：改用 `icewave.utils.paths` + argparse 参数；一次性实验脚本建议移入 `scripts/experiments/` 并在 README 标注"作者本机专用"或直接参数化。附带卫生项：`reports/_shared/fonts/*.ttf`（二进制字体）与 `echarts.min.js` 入库 ~230KB，可接受但应在 .gitattributes 标 binary；`make_synthetic_data.py` 与 `make_synthetic_augmented.py` 双实现。

### P1-1（✅ 已复现）`synthesize_hazy_iced` 元数据 A 与实际合成 A 不一致

**位置**: `src/icewave/data/degradation.py:207-212`

`synthesize_haze` 内部已用 rng 采了 3 次 uniform 生成 A；`with_metadata=True` 时外层**再采 3 次**生成 `meta["A"]`——两次采样值不同，元数据记录的是"从未参与合成"的 A。

- 复现输出：`metadata A consistent: False`（§6 脚本）
- **影响**：当前仅影响"用元数据做监督"的路径（暂未启用）；但 A2 计划中的 $\mathcal{L}_{\text{ice-phys}}$ / 透射率监督会**学到错误目标**。修复方式：让 `synthesize_haze` 返回 A，或抽出采样。
- 注：`t_map` 一致（同一 rng 序列内先于第二次 A 采样），仅 `A` 错位。

### P1-2 `--detector maskrcnn` 路径必然 TypeError

**位置**: `src/icewave/eval/downstream.py:249` vs `detect/maskrcnn_adapter.py:121-123`

```python
detector = MaskRCNNDetector(args.detector_weights, conf=args.conf)  # 下游 CLI
def __init__(self, weights, device="cpu", class_name="target", min_area=200, iou_thresh=0.3)  # 实际签名无 conf
```

选择 maskrcnn 时直接 `TypeError: unexpected keyword 'conf'`。冒烟测试只跑 yolo 路径，未覆盖。修复：给 `MaskRCNNDetector.__init__` 加 `conf` 参数或用 `score_thresh` 映射。

### P1-3 数据集构建器的 zip 输入完全不可用

**位置**: `src/icewave/data/build_dataset.py:65-67`

`--src` 为 zip 文件时 `_list_images` 返回 `{zip路径: [zip路径]}`，随后 `cv2.imread(zip文件)` 返回 None → 全部被拒 → `RuntimeError("过滤后无清晰图")`。`import zipfile, shutil` 是死导入——**承诺了从未实现的解压**。docstring 与 CLI help 都写着"zip 包或已展开目录"。修复：实现 zip 解压到临时目录，或删掉 zip 宣称。

### P1-4 m3/m4 推理协议不一致（有提示 vs 无提示）

**位置**: `src/icewave/eval/benchmark.py:36`、`infer/cli.py` 全程 vs `train/trainer.py:288-291`

训练验证 `validate()` **始终**注入 CLIP 提示；而 benchmark `_dehaze` 与推理 CLI 调 `model(x)` 不传 `fog_prompt`（`clip_prompt=None` → 走无提示分支）。同一 M4 权重两套推理协议：

- 训练期 val PSNR（带提示）与公开基准/部署 PSNR（无提示）**不可比**；
- prompt_drop_prob=0.5 训练使无提示路径可用，但性能缺口未量化。

建议：CLI/benchmark 增加 `--fog-prompt on|off|auto` 并在结果 JSON 记录协议字段。

### P2-1 checkpoint 路径一律按 m4 构建模型

**位置**: `eval/benchmark.py:102-104`。`--models xxx.pth` 时 `build_model("m4")` 硬编码——传入 m2p/m3 checkpoint 会因键不匹配在 `strict=True` 处崩溃（还好是崩溃不是静默错）。应从 checkpoint 的 `version` 字段构建。

### P2-2 `--train-if-missing` 训练后仍找不到权重

**位置**: `infer/cli.py:86-92`。`train_yolo("ice_detection/configs/data.yaml")`：①硬编码相对路径；②训练产物落在 `outputs/yolo/` 而 YOLODetector 随后仍从原 `--detector-weights` 路径加载 → FileNotFoundError。建议该 flag 直接移除（默认关闭已是正确方向，残留路径是陷阱）。

### P2-3 `torch.load(weights_only=False)` × 4 处 —— 反序列化安全隐患

**位置**: `models/hawfe.py:384`、`models/prompt.py:194`、`train/trainer.py:104`、`detect/maskrcnn_adapter.py:139`。恶意 checkpoint 可执行任意代码（pickle 反序列化）。本仓库权重目前自产自用风险低，但 `configs/weights.yaml` 计划对外发布下载 URL 后即为**供应链入口**。建议：state_dict 场景改 `weights_only=True`（PyTorch≥2.0 对纯张量 dict 可用），并在 download_weights.py 加 SHA256 校验（CITATION/REPRODUCIBILITY 已有此意识）。

### P2-4 边界情况集合

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| a | `eval/metrics.py:24` | 完全相同图像 PSNR=inf → mean=inf → `json.dumps` 输出非标 `Infinity` | clamp 到 99.0 或 None |
| b | `train/cli.py:24-25` | `--override train.epochs` 时若 `train` 键是标量，`setdefault` 抛 AttributeError | 先校验 isinstance(dict) |
| c | `data/dataset.py:53-58` | 掩码 imread 失败静默返回 (0,0) → 当作"无冰"继续训练，数据丢失无告警 | raise 或至少 warning 计数 |
| d | `data/dataset.py:120` | clear 只认 `{base}.png`，jpg GT 被静默跳过 | glob 建索引时记录真实扩展名 |
| e | `train/trainer.py:267-271` | checkpoint 保存 `state["uncertainty"]` 但 `init_checkpoint` 恢复时不加载（断点续训丢 σ） | 恢复时一并加载 |
| f | `eval/downstream.py:140-150` | GT 匹配取全体最优 IoU，若该 GT 已被占则计 FP（不回退次优未匹配 GT）——与 VOC 标准略有出入，极端重复框场景低估 AP | 遍历 IoU≥thr 的未匹配 GT |
| g | `models/prompt.py:48-49, 164` | `sys.path.insert` 全局副作用（vendored 导入），多进程 worker 下重复插入 | 可接受，建议幂等判断已有 |

### P3 卫生项

- `dataset.py` 死变量（P0-1 线索）、`infer/cli.py` 未用 `h,w`、`build_dataset.py` 死导入。
- `phase1~6/`、`ice_detection/` 旧目录建议在投稿前移入 `legacy/` 或单独 tag 归档。
- 历史安全：此前泄露的两枚 PAT 已撤销、已迁 SSH 推送 ✅；建议投稿公开前跑一次 `git log -p | grep -iE "ghp_|token|secret"` 终检。

---

## 5. 改进建议与最佳实践

### 5.0 ⚠️ 最紧急：W5 实验结论必须作废重跑

P0-1（四路裁剪错位）+ P0-2（检测损失梯度死路）+ P1-5（scipy）三个 bug 都在 W5 训练/消融路径上。当前 `reports/w5-experiment-report.html` 的全部数字（含"检测感知损失损害泛化"核心发现）**不具备论文证据效力**。正确顺序：

1. 修 P0-1 → 修 P0-2 → 补两类梯度/对齐回归测试 → 修 P1-5 让 CI 回绿；
2. 重跑 smoke test（1+1 epoch，确认 `pred.grad` 非零、loss 曲线合理）；
3. 重跑 W5 四组消融（预算不变，~110 GPU-h 内）；
4. 重写 W5 报告；论文 §3.2/§4.3 数字以重跑为准。

### 5.1 优先级排序（投稿路线图视角）

| 优先级 | 动作 | 工作量 | 阻塞的下游 |
|--------|------|--------|-----------|
| **立即** | 修 P0-1 裁剪对齐（含 bbox 同窗）+ 补对齐测试 | 0.5 天 | 一切真实训练（W5 重跑、P0-3 基线） |
| **立即** | 修 P0-2 梯度死路（去 no_grad/detach）+ 补 `pred.grad>0` 测试 | 0.5 天 | joint_v2 框架的立论基础 |
| **立即** | 修 P1-5 scipy 依赖声明 + import 移到守卫后 | 0.5 小时 | CI 回绿、测试套件可信 |
| **本周** | 修 P1-1 元数据 A；P1-2 maskrcnn TypeError；P1-3 zip 输入（或删宣称） | 各 0.5 天 | 物理监督 / 评测 CLI 可用性 |
| **本周** | P1-4 统一推理协议 + 结果 JSON 记录协议字段 | 1 天 | 基准数字可信性 |
| **重训前** | P2-5 重写 YOLOv8 输出解析（真实权重路径） | 1 天 | 真实检测器接入 joint_v2 |
| **启动训练前** | P2-3 weights_only + SHA256 校验 | 1 天 | 权重发布安全 |
| **顺手** | P2-6 脚本参数化 / P2-1 / P2-2 / P2-4 / P3 | 合计 1.5 天 | 仓库卫生 |

### 5.2 测试缺口（按风险补）

1. **训练随机裁剪 hazy/clear/ice/bbox 四路同窗**（P0-1 的直接回归测试）——最高优先。
2. **检测损失对去雾输入的梯度非零断言**（P0-2 的直接回归测试）——`loss.backward(); assert pred.grad is not None and pred.grad.norm() > 0`。当前 17 个 detect loss 测试全部只断言损失数值，无一个断言梯度可达，这正是 P0-2 溜过去的原因。
3. **scipy 缺失环境的 t 检验早退路径**（P1-5）。
4. maskrcnn 分支 CLI 冒烟（P1-2 这类"签名漂移"靠 `--help` 冒烟测不出）。
5. `synthesize_hazy_iced(with_metadata=True)` 物理一致性：用 `t_map + A + clear` 重建 == hazy（正是 §6 脚本的断言，可直接转正为测试）。
6. benchmark `--models *.pth` 非 m4 checkpoint 的报错路径。

### 5.2.1 新增代码的正面评价（A2/W2–W6 提交）

为公平起见，A2 提交也有多处高质量实现：

- `UncertaintyWeighting` 升级支持 `s_init/s_clamp`（σ 漂移约束）——正是 JOINT_OPTIMIZATION_FRAMEWORK §8.5 预案的正确落地；
- `test_bbox_loading.py` 对 padding/越界框/空标签覆盖到位（只是没接进 `__getitem__` 随机路径）;
- `compute_gain_stats`（Δ/R + 95% CI）、`per_haze_level_report`、`paired_t_test` 三件套补齐了评测协议；
- W5 报告如实记录负面结果（"检测感知损失损害泛化"）而非隐藏——科研诚信态度正确，只是结论恰好被 bug 混杂；
- smoke test 脚本（1+1 epoch）先验证训练流程再上正式训练的流程意识很好。

### 5.3 结构性建议

- 把 `benchmark._dehaze` 提升为 `icewave/infer/core.py::dehaze_bgr()` 公共函数，消除 eval→eval 的私有跨模块引用。
- `evaluate_set` 的 mAP 实现已有自研价值（无 pycocotools 依赖），但建议加一个与 `torchvision.ops` 或 ultralytics 内置 mAP 的数值一致性测试（一次性 golden 文件即可），防自研指标漂移。
- `lpips_score` 模块级缓存 LPIPS 实例（`functools.lru_cache`）。
- 配置面：`train.num_workers` 按 `sys.platform` 给默认（win32→0, linux→4），在 `config.py` 展开时注入。

### 5.4 值得保持的实践（不要在重构中丢掉）

- 模块头"旧版问题→本版改法"docstring 模式；
- graph-connected 零张量处理空掩码；
- config_snapshot.yaml 含 git commit 的实验对账机制；
- vendored 代码 NOTICE.md 许可台账；
- CI 分层安装矩阵（可选依赖不拖慢主链路）。

---

## 6. 附录：已复现 bug 的最小复现脚本

```python
# 环境: 仓库根目录, src 已可导入
import numpy as np
from icewave.data.dataset import _crop

# --- P0-1: 连续两次随机裁剪不对齐 ---
img = np.zeros((256, 256, 3), dtype=np.uint8); img[64:192, 64:192] = 255
np.random.seed(1)
hazy = _crop(img, 192, center=False)
clear = _crop(img, 192, center=False)
assert not np.array_equal(hazy, clear)   # ← __getitem__ 内 hazy/clear 即此关系

# --- P0-2: 检测感知三损失对 pred 的梯度为零/无图 ---
import torch
from icewave.models.detector_wrapper import DetectorWrapper
from icewave.losses.detect import (BoxFeaturePreservationLoss,
                                   DetectabilityLoss, BoxAlignLoss)
det = DetectorWrapper(weights=None, freeze_backbone=True)   # 同 W5 Stub 配置
pred = torch.rand(2, 3, 128, 128, requires_grad=True)
clear_t = torch.rand(2, 3, 128, 128)
boxes = torch.tensor([[0.2, 0.2, 0.8, 0.8]])

L2 = BoxFeaturePreservationLoss(feature_type="detector")(pred, clear_t, boxes, detector=det)
L2.backward()
assert pred.grad.norm().item() == 0.0    # ← 梯度恰为零

det_out = det(pred2 := torch.rand(2, 3, 128, 128, requires_grad=True))
DetectabilityLoss()(det_out, torch.tensor([[0, 1], [2, 3]])).backward()
assert pred2.grad is None                # ← 无计算图

# --- P1-1: 元数据 A 与实际 A 不一致 ---
from icewave.data.degradation import HazeParams, synthesize_hazy_iced
clear = np.full((64, 64, 3), 128, dtype=np.uint8)
h2, meta = synthesize_hazy_iced(clear, HazeParams(0.4, 0.7, 220),
                                rng=np.random.default_rng(7), with_metadata=True)
t = meta["t_map"][:, :, None]
recon = np.clip(clear.astype(np.float32) * t
                + np.array(meta["A"])[None, None, :] * (1 - t), 0, 255).astype(np.uint8)
assert not np.array_equal(recon, h2)     # 若 A 一致应逐字节相等

# --- P1-5: 干净环境 (无 scipy) ---
# python -m pytest tests/test_downstream_gain.py::TestPairedTTest -q
# → ModuleNotFoundError: No module named 'scipy' (n<2 早退路径也应失败)
```

> 本报告全部文件行号基于 commit `11e52df`。修复后请以本报告 §5.0–§5.1 清单逐项回勾。
