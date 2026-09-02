# IceWave-DehazeFormer 创新改造实施记录

> 本文档记录本次针对 SCI 一区投稿的工程化重构与创新改造的**全部改动**，
> 与 `turbo-icespoon分析报告.md` 中的 P0/P1/P2 改进方案一一对应。
> 分支: `feature/sci-refactor-p0-p1`

---

## 1. 改动总览

| 优先级 | 编号 | 内容 | 状态 | 关键文件 |
|--------|------|------|------|----------|
| P0 | P0-1 | 消除硬编码路径 + 种子管理 | ✅ | `src/icewave/utils/paths.py`, `seed.py` |
| P0 | P0-1a | 模型层重构 (monkey-patch → 子类) | ✅ | `src/icewave/models/hawfe.py` |
| P0 | P0-1b | ITL 损失修复 | ✅ | `src/icewave/losses/itl.py` |
| P0 | P0-2a | 工程骨架 (包结构/环境) | ✅ | `src/icewave/**`, `pyproject.toml` |
| P0 | P0-2b | 推理 CLI 重写 (去自动重训) | ✅ | `src/icewave/infer/cli.py` |
| P0 | P0-2c | 打包/CI/Docker/许可 | ✅ | `Dockerfile.*`, `.github/`, `LICENSE` 等 |
| P0 | P0-3 | 公开基准 harness | ✅ | `src/icewave/eval/benchmark.py` |
| P0 | P0-4 | 循环论证修复 (人工标注优先) | ✅ | `src/icewave/eval/metrics.py`, `data/dataset.py` |
| P0 | P0-5 | 数据集构建器修复 | ✅ | `src/icewave/data/build_dataset.py` |
| P1 | P1-1 | 联合优化框架 (复合退化+检测感知损失) | ✅ | `data/degradation.py`, `losses/detect.py`, `train/trainer.py` |
| P1 | P1-1 | 下游任务增益指标 | ✅ | `src/icewave/eval/downstream.py` |

---

## 2. 关键改动说明 (原因 + 预期效果)

### 2.1 模型层重构 (P0-1a)

**改动**: 旧代码用运行时 monkey-patch (`model.forward_features = closure`) 把
HA-WFE 注入 DehazeFormer; 现改为 `IceWaveDehazeFormer(DehazeFormer)` 子类,
`forward(x, fog_prompt=None)` 显式传提示。

**原因**:
1. monkey-patch 是运行时闭包, 阻断 ONNX/TorchScript 导出 (云部署受阻);
2. 提示通过 `model.clip_prompt` 全局副作用传递, 隐式且易错。

**预期效果**: 检查点键名与旧版完全一致 (`load_state_dict(strict=True)` 可用),
前向数值逐位一致; 同时获得可导出性与显式数据流。

**验证**: `tests/test_model_compat.py` 23 项全过, 含"旧 monkey-patch 模型导出的
state_dict 能 strict 加载进新类"与"新旧前向逐位相等"。

### 2.2 ITL 损失修复 (P0-1b)

**改动**: 三处修复 (见 `losses/itl.py` docstring):
1. 文档声称 SSIM 区域项、实现却是加权 L1 → 默认 `region_term='ssim'`;
2. 区域损失随 batch 大小变化 (旧在 (B,3,H,W) 取均值) → 改为 masked 均值;
3. 空掩码返回 `0.0` 常量致反传报错 → 改为 graph-connected 零张量。

**预期效果**: 指标可复现 (batch 无关), 空掩码不崩溃, 论文表述与代码一致。

### 2.3 复合退化物理模型 (P1-1)

**改动**: 新增两级退化: 冰层复合 (Beer-Lambert) → 大气散射 (ASM),
见 `data/degradation.py`。

**原因**: 旧数据合成只做 ASM 雾, 覆冰与雾的相互作用从未建模; 且无退化真值。

**预期效果**: 支撑"检测感知去雾"新问题切入, 提供 t_map/A/alpha 等真值供
透射率监督等扩展实验; 无冰时严格退化为旧 ASM (有测试保证兼容)。

### 2.4 下游任务增益指标 (P1-1)

**改动**: `eval/downstream.py` 计算同一检测器在雾/去雾/清晰三套图上的 mAP 差。

**原因**: 回应"创新点不突出"质疑 —— 证明去雾真正提升下游检测性能 (ΔmAP),
而非仅提升 PSNR。

**预期效果**: 提供 "ΔmAP = mAP_dehazed - mAP_hazy" 与 "gap = mAP_clear - mAP_dehazed"
两个可入论文的量化证据。

### 2.5 数据集构建器修复 (P0-5)

**改动** (`data/build_dataset.py`): 边界图默认排除、场景级切分 (by_zip)、
浓雾档进入测试轮转。

**原因**: 旧版边界图污染 GT、逐图随机切分致场景泄漏、浓雾档从不评测。

### 2.6 循环论证修复 (P0-4)

**改动**: `ice_mask_human/` 目录优先于伪标签; 冰区指标一律以人工标注为参照。

**原因**: 旧版"规则造标签→训练→同一规则评指标"是循环论证, 审稿人必问。

---

## 3. 实施中发现并修复的上游 bug

### 3.1 DehazeFormer 骨干参数表缺失 (真实 bug)

`build_model` 原依赖 `DehazeFormer` 类默认值, 但类默认 `depths=[16,16,16,8,8]`
是 **B 档**, 而 `dehazeformer_s()` 工厂是 `[8,8,8,4,4]`。导致 m2p/m3/m4 构造出
错误深度的模型, 与旧检查点键名错位。

**修复**: 显式 `_BACKBONE_SPECS` 表 (s/t/b 三档完整工厂参数)。

### 3.2 HA-WFE 奇数尺寸 padding 崩溃 (上游原始 bug)

旧实现 (phase4/ha_wfe.py, phase5/ha_wfe_v2.py) 在奇数空间尺寸输入时:
裁剪了 `F_enhanced` 回原尺寸, 却保留 padding 后的 `F_b`, 二者相乘形状不匹配
必崩。旧代码从未触发 (实际特征图恒为偶数)。

**修复**: padding 分支同步裁剪 `F_b`。偶数路径行为完全不变 (严格兼容),
奇数路径从"崩溃"变为"正确工作"。

---

## 4. 已知风险与待办 (需用户确认/补充)

1. **权重托管**: `scripts/download_weights.py` 的 `MANIFEST` 中 URL/SHA256 为
   占位符, 需用户提供可托管位置 (GitHub Releases / HuggingFace) 并填写。
2. **HazeCLIP 副本不完整**: `source_hazeclip/` 缺 `build_model.py` /
   `simple_tokenizer.py`, CLIP 提示分支需补全后才能真正训练 m3/m4/joint。
   已用 `import_vendored_clip()` 给出清晰报错与修复指引。
3. **检测类别语义**: Mask R-CNN 迁移权重为 2 类 (背景+target), "target" 与
   "覆冰"的对应关系未经人工校验, 论文使用前需明确。
4. **ultralytics AGPL**: 检测功能用 YOLO, 商业闭源分发需评估 (见 NOTICE.md)。
5. **本地无 GPU/数据**: 测试套件在 CPU 上验证了数值正确性与检查点兼容性,
   但真实训练效果需在 GPU + 数据上跑通后确认。

---

## 5. 复现指引 (增强可复现性)

```bash
# 1. 环境
conda env create -f environment.yml && conda activate icewave
# 或 CPU 快速体验
pip install -e ".[dev]"

# 2. 数据与权重 (路径参数化)
export ICEWAVE_DATA_ROOT=/path/to/data
export ICEWAVE_WEIGHTS_DIR=/path/to/weights
python scripts/download_weights.py --models m4

# 3. 训练
icewave-train --config configs/train/m4.yaml

# 4. 评测 (公开基准 + 下游增益)
icewave-eval-benchmark --config configs/benchmarks.yaml --models m4 --benchmarks reside_sots_indoor
icewave-eval-downstream --detector yolo --detector-weights weights/yolo/power_line_best.pt \
    --hazy-dir data/val/hazy --labels-dir data/val/labels --dehaze-model m4 --clear-dir data/val/clear

# 5. 推理
icewave-infer --input data/test/hazy --model m4 --detector yolo --detector-weights weights/yolo/power_line_best.pt
```

---

## 6. 测试

`pytest tests/` → 93 项全过 (CPU 环境), 覆盖:
- 模型检查点键名/数值兼容性 (`test_model_compat.py`)
- ITL 损失三处修复 (`test_itl_loss.py`)
- 复合退化物理性质 (`test_degradation.py`)
- 检测感知损失与不确定性加权 (`test_detect_losses.py`)
- 数据集配对/人工标注优先 (`test_dataset.py`)
- 路径参数化/配置加载 (`test_paths_config.py`)
- 下游增益 mAP 实现 (`test_downstream.py`)
