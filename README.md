# IceWave-DehazeFormer

> 雾天输电线路去雾与覆冰检测研究框架 · 面向 SCI 一区投稿目标的工程化重构与创新改造
> Apache-2.0 | PyTorch ≥ 2.0 | Python ≥ 3.10

针对雾天输电线路巡检场景, 本项目以 **DehazeFormer** 为骨干, 集成 **HA-WFE 小波
特征增强**、**CLIP 雾提示蒸馏**、**ITL 覆冰感知损失** 三大创新模块, 并提出面向
SCI 一区的"**检测感知联合优化框架**" (joint 训练模式) 与"**下游任务增益**"评测协议,
将"去雾"从独立重建任务升级为下游检测任务的协同恢复。

> 一句话: 一次去雾, 同时给出更清晰的图和更可靠的覆冰检测结果。

---

## 1. 与旧版的关键差异

| 维度 | 旧版 (`phase6/dehaze_inference.py` 等) | 新版 (`src/icewave/`) |
|------|-----------------------------------------|-----------------------|
| 代码形态 | 130+ 处硬编码 `D:\` `.trae-cn` 路径 | 路径全部参数化 (YAML + `ICEWAVE_*` 环境变量) |
| 模型集成 | 运行时 monkey-patch `forward_features` | `IceWaveDehazeFormer` 标准 `nn.Module` 子类 (可导出 ONNX) |
| 训练入口 | 6 个 `train_m*.py` 各写一份 Config | `icewave-train --config configs/train/*.yaml` |
| 推理入口 | `python dehaze_inference.py ...` 内嵌 YOLO 自动重训 | `icewave-infer --input ... --model m4` |
| 测试 | 0 项 | 93 项 (CPU 验证, 含检查点键名/数值兼容性) |
| 环境矩阵 | README 硬绑 PyTorch 2.11+/CUDA 12.8+/RTX 5060 | `pyproject.toml` 声明 `torch≥2.0`, 给 CPU/多 CUDA 版本容器镜像 |
| 许可 | README"仅供学术研究", LICENSE=Apache-2.0, 二者冲突 | README 末尾明确 Apache-2.0 条款 + 第三方依赖条款 (NOTICE.md) |

---

## 2. 仓库结构

```
turbo-icespoon/
├── src/icewave/                      # ★ 唯一主代码包 (pip install -e .)
│   ├── models/      (hawfe.py, dehazeformer.py, prompt.py)
│   ├── losses/      (itl.py, detect.py)
│   ├── data/        (dataset.py, degradation.py, build_dataset.py)
│   ├── detect/      (ice_mask.py, yolo.py, maskrcnn_adapter.py)
│   ├── train/       (trainer.py, config.py, cli.py)
│   ├── infer/       (cli.py)
│   ├── eval/        (benchmark.py, downstream.py, metrics.py)
│   └── utils/       (paths.py, seed.py)
│
├── configs/
│   ├── train/{m1,m2,m2p,m3,m4,joint}.yaml   # 各训练模式配置
│   ├── benchmarks.yaml                       # 公开基准 harness 配置
│   └── hazeclip_*.yaml                       # ⚠️ 历史遗留, 见下方说明
│
├── tests/                              # 93 项 pytest, CPU 可跑
├── scripts/download_weights.py         # 权重下载 + SHA256 校验
├── docs/                               # 审计报告 + 实施记录
│   ├── AUDIT_REPORT.md                 # 21 项代码 vs 文档审计 (本次新增)
│   ├── IMPLEMENTATION_NOTES.md         # P0/P1 改动逐项说明
│   └── five_way_report.txt             # 历史五路对比报告 (未改动)
│
├── pyproject.toml  requirements.txt  environment.yml
├── Dockerfile.cpu  Dockerfile.gpu    .github/workflows/ci.yml
├── LICENSE (Apache-2.0)  NOTICE.md  CITATION.cff
│
├── source_dehazeformer/                # 历史 vendored 骨干 (P0-1a 已迁入 src)
├── source_hazeclip/                    # 历史 HazeCLIP 副本 (教师蒸馏, 可选)
│
├── phase1~phase6/                      # ⚠️ 历史开发过程目录, 仅供考古
└── ice_detection/                      # ⚠️ 旧版检测附属模块, 见下方说明
```

### ⚠️ 历史目录与新包的关系

| 旧路径 | 替代位置 | 状态 |
|--------|----------|------|
| `ice_detection/algorithms/ice_mask_generator.py` | `src/icewave/detect/ice_mask.py` | 旧文件头部已加 deprecation 指向新包 |
| `ice_detection/algorithms/yolo_auto_label.py` 等 | `src/icewave/detect/yolo.py` | 同上 |
| `ice_detection/algorithms/maskrcnn_inference.py` | `src/icewave/detect/maskrcnn_adapter.py` | 同上 |
| `phase5/itl_loss.py` | `src/icewave/losses/itl.py` | 修复三处 bug |
| `phase5/ha_wfe_v2.py` / `phase4/ha_wfe.py` | `src/icewave/models/hawfe.py` | 子类化 + 修复奇数尺寸 bug |
| `phase6/dehaze_inference.py` | `src/icewave/infer/cli.py` (`icewave-infer`) | 去除 YOLO 自动重训 |
| `phase4~5/train_m*.py` | `src/icewave/train/cli.py` (`icewave-train`) | 配置驱动的统一入口 |
| `configs/hazeclip_*.yaml` | `configs/train/{m3,m4,joint}.yaml` | hazeclip_*.yaml 顶部加注释 |

旧目录保留以便追溯开发过程与恢复历史检查点, 但**所有新工作请在 `src/icewave/` 下进行**。

---

## 3. 模型版本演进

| 版本 | 名称 | 骨干 | HA-WFE | CLIP 提示 | ITL | 联合优化 | 参数量 | 相对 m1 增量 |
|------|------|------|--------|-----------|-----|----------|--------|---------------|
| m1 | DehazeFormer-S 基线 | S | — | — | — | — | 1.285 M | — |
| m2 | + HA-WFE v1 | S | v1 (零初值/Tanh) | — | — | — | 1.315 M | +2.32 % |
| m2p | + HA-WFE v2 | S | v2 (正初值/Sigmoid) | — | — | — | 1.315 M | +2.32 % |
| m3 | + CLIP 雾提示 | S | v2 | ✓ | — | — | 1.336 M | +4.01 % |
| m4 | + ITL 覆冰感知 | S | v2 | ✓ | ✓ | — | 1.336 M | +4.01 % |
| **joint** | + 检测感知联合 | S | v2 | ✓ | ✓ | ✓ | 1.336 M | +4.01 % |

> 参数量基于 `build_model(v)` 构造后 `sum(p.numel())` 实测; 与 README 旧版声称
> "+10%~+15%" 不同 (旧数字未实测, 现已修正)。
> CLIP 提示分支的投影层仅在 m3/m4/joint 中加载, m3→m4 严格模型参数量不变
> (差异在训练流程而非参数)。

`joint` 模式是本项目面向 SCI 一区的**核心创新训练模式**: M4 + 走廊纹理保持
损失 + Kendall & Gal 不确定性加权, 让去雾主干保留"检测关键特征", 打破"先去雾
再检测"级联范式, 实现任务协同优化。

---

## 4. 三大创新模块

### 4.1 HA-WFE (Haar Wavelet Feature Enhancement)

瓶颈层 Haar 小波特征增强, 对四个子带差异化处理:

- **LL (低频)**: SCA 通道注意力 + 可选 CLIP 雾提示调制
- **LH/HL (中频)**: 深度可分离门控 × 残差校正
- **HH (高频)**: 同 LH/HL 结构, 独立 α

v1 (m2) 与 v2 (m2p) 的差异: v1 用零初始化 + Tanh + 共享 α, v2 用正初始化 (0.1) +
Sigmoid 门控 + 子带独立 α。两种风格保留作消融维度。

**工程修复**: 旧 `phase4/ha_wfe.py` 在奇数空间尺寸输入时会因 `F_enhanced` 已裁回
原尺寸、`F_b` 仍为 padding 后尺寸而崩溃; 本包同步裁剪 `F_b` (偶数路径行为不变)。

### 4.2 CLIP 雾提示蒸馏 (m3/m4/joint)

CLIPSurgery (CS-ViT-B/32) 冻结编码雾图 → 49 个空间 token → 1×1 卷积投影为
32 通道提示 `M_h` → 注入 HA-WFE 的 LL 分支。教师 HazeCLIP (MSBDN) 输出 L1 蒸馏
损失, `λ_kd=0.05`。训练时 50 % prompt dropout, 推理时不需要 CLIP。

**工程修复**: 旧 `train_m3.py` 只优化 `model.parameters()`, `CLIPFogPrompt.proj`
随机初始化且永不更新; 新版 `train_prompt_proj=true` 强制投影层进入 optimizer。

### 4.3 ITL 覆冰感知损失 (m4/joint)

$$
\mathcal{L}_{\text{ITL}} = \lambda_{\text{region}} \mathcal{L}_{\text{region}} + \lambda_{\text{boundary}} \mathcal{L}_{\text{boundary}}
$$

- $\mathcal{L}_{\text{region}} = \mathbb{E}_{x \in \text{ice}} \| \hat J - J \|_1 + w_{\text{ssim}} \cdot (1 - \text{SSIM}_{\text{ice}}(\hat J, J))$
  - 冰区加权 L1 + 冰区加权 SSIM (默认 `region_term='ssim'`)
- $\mathcal{L}_{\text{boundary}} = \mathbb{E}_{x \in \partial \text{ice}} \| \nabla \hat J - \nabla J \|_1$
  - 边界带 (膨胀掩码减去原掩码) 上的 Sobel 梯度幅值 L1
- 默认 `λ_region=0.5`, `λ_boundary=0.3`, `w_ssim=0.1`, 边界膨胀核 7

**工程修复三处** (旧 `phase5/itl_loss.py` → 新 `src/icewave/losses/itl.py`):

1. 文档声称 SSIM 区域项、实现是加权 L1 → 默认改为真 SSIM 区域项
2. 区域损失随 batch 大小变化 (旧在 (B,3,H,W) 取均值) → 改为按"冰像素 × 通道" masked 均值
3. 空掩码返回 `0.0` Python float, 反传时报"does not require grad" → 改为
   `pred.sum() * 0.0` graph-connected 零张量 (任意 pred 的零倍数仍是 pred 计算图
   的一部分, 反向传播时梯度正常累积)

### 4.4 检测感知联合优化 (joint 独有)

P1-1 新增, 是面向 SCI 一区的核心创新:

- **复合退化物理模型** (`data/degradation.py`):
  冰层复合 (Beer-Lambert: $\alpha = 1 - e^{-\beta d}$) → 大气散射 (Koschmieder)
  无冰时严格退化为经典 ASM (逐字节相等, 保证与旧管线兼容)
- **走廊纹理保持损失** (`losses/detect.py`):
  仅在冰区膨胀走廊内最小化预测与清晰的梯度差异, 抑制走廊外过度锐化
- **不确定性加权** (`losses/detect.py`):
  Kendall & Gal 多任务损失可学习权重 σ, 自动平衡 recon / kd / itl / corridor 四项

### 4.5 下游任务增益指标 (P1-1, eval/downstream.py)

```
ΔmAP = mAP(hazy) − mAP(dehazed)   ←  去雾对检测的提升
gap   = mAP(clear) − mAP(dehazed)  ←  残留性能差距
```

回应审稿人对"PSNR 提升不等于真实价值"的质疑, 提供**任务级量化证据**。

---

## 5. 环境

`pyproject.toml` 声明 `torch≥2.0` (含 CPU-only), 运行时根据 GPU 自动选 bf16/fp16。
实际验证矩阵 (CI 后续补全):

| 场景 | 命令 |
|------|------|
| CPU 开发/测试 | `pip install -e ".[dev]"` (PyTorch 官方 CPU 轮子) |
| 单 GPU 训练 (A100/A10/V100) | `pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install -e ".[train,all]"` |
| RTX 5090/5060 (Blackwell, sm_120) | `pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install -e ".[all]"` |
| Conda 复现 | `conda env create -f environment.yml && conda activate icewave` |
| Docker CPU 推理 | `docker build -f Dockerfile.cpu -t icewave:cpu .` |
| Docker GPU 训练 | `docker build -f Dockerfile.gpu -t icewave:gpu .` |

⚠️ **关于 ultralytics (YOLO)**: AGPL-3.0 许可, 学术研究使用无限制; 闭源商业分发
需评估传染性 (见 NOTICE.md)。`pip install -e ".[detect]"` 才会安装。

---

## 6. 路径参数化

四个环境变量, 全部可选 (缺省指向仓库内):

```bash
export ICEWAVE_DATA_ROOT=/path/to/data           # 数据集根
export ICEWAVE_WEIGHTS_DIR=/path/to/weights      # 检查点/教师/检测权重
export ICEWAVE_OUTPUT_DIR=/path/to/outputs       # 训练输出/推理输出
export ICEWAVE_CLIP_DIR=/path/to/clip            # CLIP 缓存 (可选)
```

YAML 配置支持 `${ICEWAVE_DATA_ROOT}/dataset` 占位符 + 缺省值 (`load_config` 自动展开)。

---

## 7. 快速上手

### 7.1 安装

```bash
git clone https://github.com/Kristen-net/turbo-icespoon
cd turbo-icespoon
pip install -e ".[all]"          # CPU/教学用; GPU 见上文矩阵
```

### 7.2 准备数据与权重

```bash
# 1. 构造数据集 (场景级切分, 边界图默认排除)
icewave-build-dataset --src /path/to/raw_images \
    --out $ICEWAVE_DATA_ROOT/dataset --val-ratio 0.1

# 2. 下载权重 (维护者需先在 scripts/download_weights.py 的 MANIFEST 填入 URL/SHA256)
python scripts/download_weights.py --models m4
# 或全量
python scripts/download_weights.py --all
```

### 7.3 训练

```bash
icewave-train --config configs/train/m4.yaml
# 覆盖超参: icewave-train --config configs/train/m4.yaml --override train.epochs=60
# joint 模式: icewave-train --config configs/train/joint.yaml
```

### 7.4 评测

```bash
# 公开基准 (RESIDE-SOTS 等)
icewave-eval-benchmark --config configs/benchmarks.yaml --models m4 \
    --benchmarks reside_sots_indoor

# 下游增益指标 (同一检测器在 hazy / dehazed / clear 三套图上的 mAP 差)
icewave-eval-downstream --detector yolo --detector-weights weights/yolo/power_line_best.pt \
    --hazy-dir $ICEWAVE_DATA_ROOT/val/hazy \
    --clear-dir $ICEWAVE_DATA_ROOT/val/clear \
    --dehaze-model m4
```

### 7.5 推理

```bash
# 仅去雾
icewave-infer --input data/test/hazy --model m4

# 去雾 + 覆冰检测 (YOLO) + 规则冰掩码 + 对比图
icewave-infer --input data/test/hazy --model m4 \
    --detector yolo --detector-weights weights/yolo/power_line_best.pt \
    --conf 0.25

# 使用 Mask R-CNN 替代 YOLO (语义与覆冰对齐未经人工校验, 见 docs/IMPLEMENTATION_NOTES.md §4)
icewave-infer --input data/test/hazy --model m4 \
    --detector maskrcnn --detector-weights weights/maskrcnn/model_0004999.pth
```

### 7.6 输出结构

```
<ICEWAVE_OUTPUT_DIR>/infer/
├── dehazed/         # 去雾图
├── ice_mask/        # 规则式冰掩码 (去雾图上; 伪标签, 论文指标应以人工标注为准)
├── annotated/       # YOLO/MaskRCNN 标注图
├── compare/         # 原图 | 去雾 并排对比
└── report.csv       # 每图检测数 + 冰覆盖率
```

### 7.7 测试

```bash
pytest tests/ -v   # 93 项, CPU 即可
```

---

## 8. 模型权重

仓库不分发权重, 通过 `scripts/download_weights.py` 下载并做 SHA256 校验。
维护者需将权重托管到下列任一位置, 并填入脚本的 `MANIFEST`:

- GitHub Releases (与本仓库绑定)
- HuggingFace Hub (推荐, 可挂项目页)
- Zenodo (数据集 DOI)

| 权重 | 大小 | 用途 |
|------|------|------|
| `checkpoints/{m1..m4,joint}_best.pth` | ~5–15 MB | 各版本去雾检查点 |
| `hazeclip/model.pth` | ~108 MB | HazeCLIP 教师 (m3/m4/joint 蒸馏用) |
| `yolo/power_line_best.pt` | ~5.9 MB | YOLOv8 覆冰/线路检测 (AGPL-3.0) |
| `maskrcnn/model_0004999.pth` | ~335 MB | Mask R-CNN 迁移权重 (语义未对齐) |

URL/SHA256 占位符需维护者填写 (审计编号 M1/M6)。

---

## 9. 许可证与第三方归属

- **本项目**: Apache-2.0 (见 `LICENSE`)
- **DehazeFormer 骨干** (`src/icewave/models/dehazeformer.py`): BSD-3-Clause, vendored 自 [cecid3/DehazeFormer](https://github.com/cecid3/DehazeFormer), 仅做 timm API 兼容性修改
- **HazeCLIP 教师** (`source_hazeclip/`): 副本不完整, 使用前需补全 `build_model.py` / `simple_tokenizer.py`; 上游许可需查阅原仓库
- **ultralytics (YOLO)**: AGPL-3.0; 学术研究无限制, 闭源商业分发需评估
- **公开数据集** (RESIDE / O-HAZE / I-HAZE): 各自数据使用条款, 本项目不分发

详见 `NOTICE.md`。

---

## 10. 引用

```bibtex
@software{icewave_dehazeformer_2026,
  author = {Kristen-net},
  title  = {IceWave-DehazeFormer: Ice-Aware Dehazing for Transmission-Line Inspection},
  year   = {2026},
  url    = {https://github.com/Kristen-net/turbo-icespoon}
}
```

---

## 11. 引用本工作

作者: [Kristen-net](https://github.com/Kristen-net) (GitHub: Kristen-net)
若本工作对你的研究有帮助, 欢迎引用 (见 `CITATION.cff`)。

---

## 12. 已知风险与下一步

1. **权重托管 URL/SHA256 待维护者填写** (审计 M1/M6)
2. **HazeCLIP 副本不完整** (审计 M2/NOTICE.md §3), m3/m4/joint 训练前需补全
3. **Mask R-CNN 类别语义** (target 与覆冰对应未经人工校验)
4. **ultralytics AGPL** (商业闭源分发需评估)
5. **审计报告 21 项中 19 项随本 README 修复**, M1/M2/M5 三项需维护者介入 (见 `docs/AUDIT_REPORT.md`)

完整审计详见 `docs/AUDIT_REPORT.md`, 实施记录详见 `docs/IMPLEMENTATION_NOTES.md`。