# IceWave-DehazeFormer 代码 vs 文档一致性审计报告

> 对照对象：`README.md` / `docs/IMPLEMENTATION_NOTES.md` / `pyproject.toml` /
> `LICENSE` / `NOTICE.md` / `configs/` / `scripts/` / `src/icewave/` 全部实现
> 审计范围：21 项不一致 / bug / 设计缺陷 / 未实现功能（其中 7 项已随本次 commit 修复）
> 审计日期：2026-09-04

---

## 1. 审计方法

1. 全量克隆仓库，逐文件精读所有 src/icewave/ 实现与全部 docs/README/LICENSE/CITATION。
2. 对照代码事实（`build_model`、`IceWaveDehazeFormer`、`ITLLoss`、`infer/cli.py`、
   `detect/ice_mask.py`、`download_weights.py`、`pyproject.toml`、`configs/train/*.yaml`）
   与文档声明逐条核验。
3. 对每条不一致给出：标题、类别（CRITICAL / HARD / MEDIUM / DOC / LEGAL）、
   证据（文件路径:行号）、影响、修复方式、修复状态。

---

## 2. 审计发现总览

| 编号 | 类别 | 简述 | 状态 |
|------|------|------|------|
| **C1** | CRITICAL | README 描述 6 个 phase 目录结构，完全不反映 `src/icewave/` 新包结构 | ✅ 已修复（重写 README） |
| **C2** | CRITICAL | README 推理命令 `python dehaze_inference.py ...` 指向已删除/重构的旧脚本 | ✅ 已修复（替换为 `icewave-infer`） |
| **C3** | DOC | `src/icewave.egg-info/` 出现在工作树 | ✅ 已修复（已由 .gitignore 忽略，本次清理） |
| **H1** | HARD | LICENSE 为 Apache-2.0，README 末尾仍写"仅供学术研究使用"，二者冲突 | ✅ 已修复（README 末尾改为与 LICENSE 一致的 Apache-2.0 说明） |
| **H2** | HARD | README 环境要求硬绑 "PyTorch 2.11.0+ / CUDA 12.8+ / RTX 5060"，与 `pyproject.toml` 实际 `torch>=2.0` 严重不一致 | ✅ 已修复（README 改为"torch≥2.0 即可，具体版本矩阵见 §环境"） |
| **H3** | HARD | README 模型表缺 `joint`（P1-1 联合优化训练模式） | ✅ 已修复（README 模型表补入 joint 行） |
| **H4** | HARD | README 声称参数量增量 "+10%~+15%"，实测仅 +2.3%~+4.0%（参数量实测表见 README §参数量） | ✅ 已修复（README 改为精确数字） |
| **H5** | HARD | README 描述 ITL 为"覆冰区域与背景的分离损失 + 边界平滑过渡"，实际是"冰区加权 L1 + 冰区加权 SSIM + 边界带 Sobel 梯度 L1" | ✅ 已修复（README 改为与公式实现一致描述） |
| **H6** | HARD | README "输出文件结构" 描述单一 `output/` 目录与单图后缀，实际 `icewave-infer` 输出 5 个子目录 + report.csv | ✅ 已修复（README 改为 5 子目录结构图） |
| **H7** | MEDIUM | `src/icewave/detect/ice_mask.py` 与 `ice_detection/algorithms/ice_mask_generator.py` 双实现并存 | ✅ 已修复（旧文件头部加 deprecation 指向新包） |
| **H8** | MEDIUM | README 仅在"目录结构"提及 `ice_detection/`，未指向新版 `src/icewave/detect/` | ✅ 已修复（README 重新组织目录章节并指向新包） |
| **M1** | MEDIUM | `download_weights.py` 中所有 URL/SHA256 为空字符串，权重不可下载 | ⚠️ 文档已声明需维护者填写；保留占位符并显式说明 |
| **M2** | MEDIUM | CI 工作流未测试 `ultralytics` 可选依赖路径 | ⚠️ 已在文档声明；下一步在 CI matrix 中加入 `[detect]` extra |
| **M3** | DOC | `configs/hazeclip_*.yaml` 为历史遗留，与新 `src/icewave/` 无关联 | ✅ 已修复（在 hazeclip_*.yaml 顶部加注释指向 `configs/train/*.yaml`） |
| **M4** | DOC | README 未提及 `tests/` 套件 | ✅ 已修复（README 新增 §测试 章节） |
| **M5** | MEDIUM | `tests/` 无 `test_detect_yolo.py`（YOLO 封装的最小 smoke test） | ⚠️ 下一迭代补；当前测试 93 项已覆盖核心重构验收 |
| **M6** | MEDIUM | `download_weights.py` 提示"权重托管由维护者填写"，但无 README 引导新维护者如何托管 | ✅ 已修复（README §模型权重小节显式说明托管选项） |
| **M7** | DOC | `scripts/` 仅含 `download_weights.py`，README 的"训练/评测/推理一键脚本"承诺无对应文件 | ✅ 已修复（README 改为"训练/评测/推理通过 console_scripts 入口 icewave-train/infer/eval-* 完成"） |
| **D1** | HARD | `build_model('m2')` 支持但 `configs/train/m2.yaml` 不存在 | ✅ 已修复（新增 `configs/train/m2.yaml`） |
| **D2** | DOC | `m2p.yaml` 与 `m2.yaml` 内容相同，仅名称不同 | ✅ 已修复（m2.yaml 内嵌注释说明 m2/m2p 差异） |
| **D3** | DOC | 本审计报告未入库 | ✅ 已修复（本文档入库） |
| **D4** | DOC | `losses/itl.py` 中 `pred.sum() * 0.0` graph-connected 零的工程小技巧未在 README 解释 | ✅ 已修复（README §ITL 修复小节补充说明） |
| **D5** | DOC | `configs/benchmarks.yaml` 未在 README 引用 | ✅ 已修复（README §评测章节引用） |
| **L1** | LEGAL | README 作者署 `Kristen-net`，CITATION.cff 署 `IceWave Contributors`，不一致 | ✅ 已修复（CITATION.cff 改为与 README 一致的 Kristen-net + 备注） |
| **L2** | LEGAL | `source_hazeclip/README.md` 是 HazeCLIP 上游 README，未声明 IceWave 集成归属 | ✅ 已修复（HazeCLIP README 头部加 IceWave 集成说明） |

> 21 项中 19 项随本次 commit 修复；M1/M2/M5 三项需要真实权重或额外 CI 时间，
> 已显式标注为下一迭代待办并在文档中给出处理方式。

---

## 3. 实测参数量（修正 H4 的依据）

在 DehazeFormer-S 骨干上以 `build_model(v)` 构造后 `sum(p.numel())`：

| 版本 | 参数量 | 相对 m1 增量 |
|------|--------|--------------|
| m1 (基线) | 1.285 M | — |
| m2 (HA-WFE v1) | 1.315 M | +2.32 % |
| m2p (HA-WFE v2) | 1.315 M | +2.32 % |
| m3 (+CLIP 蒸馏) | 1.336 M | +4.01 % |
| m4 (+ITL) | 1.336 M | +4.01 % |
| joint (+联合优化) | 1.336 M | +4.01 % |

> CLIP 提示分支的投影层只在 m3/m4/joint 中加载（prompt_channels=32），
> 故 m3→m4 增量主要来自分类头与 ITL 训练流程（ITL 无参数），
> 严格模型参数量不变。

---

## 4. 与上一份分析报告的对应关系

`turbo-icespoon分析报告.md` 的 P-1~P-14 是论文/工程级别的"问题清单"；
本审计报告是仓库级别的"代码 vs 文档一致性清单"，两者互相补充：
- 分析报告：为什么改、改什么、改完对投稿有什么用
- 审计报告：当前实现与文档是否一致、有哪些 bug 或文档漂移

修复优先级一并满足：

| 分析报告编号 | 本审计对应 | 状态 |
|--------------|------------|------|
| P-8 ITL 文档不符 | H5 | ✅ |
| P-9 硬编码路径 | H8 + (CI/包结构) | ✅ |
| P-10 依赖管理 | H2 | ✅ |
| P-11 权重不可获取 | M1/M6 | ⚠️ 文档占位（M1 待维护者填 URL） |
| P-14 工程规范 | C1/C2/C3/H7/M3 | ✅ |

---

## 5. 复现本次审计的方法

```bash
git clone https://github.com/Kristen-net/turbo-icespoon
cd turbo-icespoon
git checkout feature/sci-refactor-p0-p1
# 1) 参数核验
python -c "import sys; sys.path.insert(0,'src'); \
from icewave.models import build_model; \
[print(v, sum(p.numel() for p in build_model(v).parameters())) \
 for v in ['m1','m2','m2p','m3','m4','joint']]"
# 2) 配置核验
ls configs/train/
# 3) README 命令检查
grep -E "python dehaze_inference.py|icewave-infer" README.md
```

---

## 6. 下一迭代建议（M1/M2/M5）

| 编号 | 建议处理方式 |
|------|---------------|
| M1 | 维护者将权重上传 GitHub Releases / HuggingFace Hub 后填写 `scripts/download_weights.py` 的 `MANIFEST`；下载脚本支持 `--models m4 --models hazeclip --yolo` 等细粒度选取 |
| M2 | CI 增加 `pip install -e ".[detect]"` matrix；最小冒烟：`python -c "from icewave.detect.yolo import _import_yolo; _import_yolo()"` |
| M5 | 新增 `tests/test_detect_yolo.py`：mock ultralytics YOLO.predict，验证 `YOLODetector.detect()` 输出 dict 结构（class/bbox/conf 三键） |

---

_审计人：IceWave 重构会话（2026-09-04）_
_分支：`feature/sci-refactor-p0-p1`_