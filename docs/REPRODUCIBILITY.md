# Reproducibility Statement (SCI 投稿级可复现性声明)

> 本文件按 IEEE TPAMI / Elsevier Neurocomputing 等 SCI 一区期刊 *Reproducibility /
> Reproducibility and Code Availability* 章节惯例整理。可直接复制到论文末尾或
> supplementary materials。

## R.1 系统需求 (Hardware)

| 组件 | 最低 | 推荐 |
|---|---|---|
| CPU | 4 核 x86_64 | 8 核 |
| RAM | 8 GB | 16 GB |
| GPU (训练) | NVIDIA 6GB VRAM (m1/m2) | 24GB (m3/m4/joint with HazeCLIP) |
| GPU (推理) | NVIDIA 4GB | 任意 CUDA 卡 |
| 磁盘 | 10 GB (代码) + 5 GB (权重) + 50 GB (数据集) | 100 GB SSD |

CPU-only 可跑全部推理与 m1/m2 训练 (慢 ~30x); m3/m4/joint 强烈建议 GPU。

## R.2 软件版本 (Software Stack)

| 组件 | 版本 | 锁定方式 |
|---|---|---|
| Python | ≥3.10, <3.13 | `pyproject.toml::requires-python` |
| PyTorch | ≥2.0, <3.0 | `pyproject.toml::dependencies` |
| torchvision | ≥0.15 | 同上 |
| OpenCV (headless) | ≥4.8 | 同上 |
| timm | ≥0.9 | 同上 |
| ultralytics (detect) | ≥8.2 | `pyproject.toml::optional-dependencies.detect` |
| CUDA (若用 GPU) | 11.8 / 12.x (按 PyTorch 匹配) | `Dockerfile.gpu` 锁定 |

环境变量:

```bash
export ICEWAVE_DATA_ROOT=/path/to/dataset       # 含 train/ val/ test/ 子目录
export ICEWAVE_WEIGHTS_DIR=/path/to/weights     # 与 configs/weights.yaml rel 字段对齐
export ICEWAVE_OUTPUT_DIR=/path/to/outputs      # 训练/推理/评测产物根
export ICEWAVE_CLIP_DIR=/path/to/clip            # 仅 m3/m4/joint 用, CLIP/CLIP-ViT 权重
export ICEWAVE_HAZECLIP_WEIGHTS=...             # 仅 m3/m4/joint 用
```

## R.3 一键复现 (One-shot Reproduction)

仓库根目录执行:

```bash
bash scripts/reproduce.sh
```

该脚本依序完成:
1. 创建 conda 环境 (Python 3.11 + 依赖)
2. `pip install -e ".[all]"` (含 detect/train/quality)
3. `python scripts/download_weights.py --all` (从 `configs/weights.yaml` 拉取权重)
4. `icewave-build-dataset --root $ICEWAVE_DATA_ROOT --config configs/benchmarks.yaml` (合成退化集)
5. `icewave-train --config configs/train/m4.yaml` (跑单模型训练, 替换为 m1/m2/m2p/m3/joint 即跑对应模型)
7. `icewave-eval-benchmark --weights weights/checkpoints/m4_best.pth --output outputs/benchmarks`
8. `icewave-eval-downstream --weights weights/checkpoints/m4_best.pth --dataset voc` (ΔmAP 计算)

完整复现预计耗时 (RTX 3090):
- 步骤 1-3: ~10 分钟
- 步骤 4 (数据集合成): ~30 分钟 (取决于规模)
- 步骤 5 (m4 训练): ~6 小时
- 步骤 6 (评测): ~20 分钟
- 步骤 7 (下游 mAP): ~40 分钟
- **总计 ~7 小时**

## R.4 随机种子与数值一致性

```bash
export PYTHONHASHSEED=42
python -c "import torch; torch.manual_seed(42); torch.cuda.manual_seed_all(42)"
```

`src/icewave/utils/seed.py::set_seed` 提供统一入口, 训练器默认 seed=42。

数值容差: 同一硬件与 CUDA 版本下, 相同 seed 的指标波动应 < 0.5% (PSNR) / 0.3% (SSIM) / 0.5 mAP (下游检测)。CPU 与 GPU 间有 ≤ 1% 数值差异 (浮点累积)。

## R.5 数据集

| 集 | 来源 | 大小 | 标注 |
|---|---|---|---|
| 输电线路清晰原图 | 用户私有 / 可申请 | ~5000 张 | 无 |
| 雾霾合成 (大气散射 + 冰 Beer-Lambert) | `src/icewave/data/degradation.py::synthesize_hazy_iced` | 程序生成 | 自动 α/β/厚度 |
| 人工覆冰掩码 | `ice_mask_human/` 子目录优先 | ~1200 张 | 二值 (255=冰) |
| VOC 2007 (下游检测) | <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/> | ~10k 张 | 20 类, 含 person/car 可作代理 |

数据集获取:
- 公开子集: 通过 <https://huggingface.co/datasets/Kristen-net/icewave-sample> 获取示例 (TBD)
- 完整研究数据集: 见论文 supplementary 或邮件联系作者

## R.6 训练配置与超参

每个模型有独立 YAML 配置:

| 配置 | 文件 | 关键超参 |
|---|---|---|
| m1 | `configs/train/m1.yaml` | DehazeFormer-S 骨干 + HA-WFE v1 |
| m2 | `configs/train/m2.yaml` | + Tanh+共享 α 门控 |
| m2p | `configs/train/m2p.yaml` | + Sigmoid+独立 α 门控 |
| m3 | `configs/train/m3.yaml` | + HazeCLIP 提示蒸馏 |
| m4 | `configs/train/m4.yaml` | + ITL 覆冰感知损失 |
| joint | `configs/train/joint.yaml` | + 覆冰检测 + 下游 mAP 联合 |

每个 YAML 自带完整数据增强、损失权重、学习率调度、batch size、epoch 数。

## R.7 已测试清单 (Verified Components)

| 类别 | 测试 | 验证范围 |
|---|---|---|
| 模型兼容性 | `tests/test_model_compat.py` (10 项) | 6 种 build_model 模式, 检查点键名一致性, 数值边界 |
| ITL 损失 | `tests/test_itl_loss.py` (12 项) | L1/SSIM/Sobel 梯度项, graph-connected 零, 空掩码 |
| 检测损失 | `tests/test_detect_losses.py` (11 项) | CorridorTextureLoss 不变性, UncertaintyWeighting |
| 数据退化 | `tests/test_degradation.py` (6 项) | 物理一致性 (Beer-Lambert + 大气散射) |
| 数据集 | `tests/test_dataset.py` (12 项) | 场景级切分, 人工标注优先, 数据增强 |
| 路径参数化 | `tests/test_paths_config.py` (18 项) | ICEWAVE_* 变量优先级, 回退, 跨平台 |
| 下游评估 | `tests/test_downstream.py` (24 项) | ΔmAP, all-point 插值 AP 边界 |
| YOLO 检测 | `tests/test_detect_yolo.py` (10 项) | 常量, filter_ice, draw, 错误处理 |

**总计 103 项全过** (CI `test-base` job 验证; CPU 可跑, 无 GPU 依赖)。

## R.8 引用格式 (Citation)

```bibtex
@software{icewave_dehazeformer_2026,
  title  = {IceWave-DehazeFormer: Ice-Aware Dehazing for Transmission-Line Inspection},
  author = {Kristen-net and IceWave contributors},
  year   = {2026},
  url    = {https://github.com/Kristen-net/turbo-icespoon},
  note   = {Apache-2.0; weights via scripts/download_weights.py}
}
```

如使用 HazeCLIP 蒸馏模块, 请同时引用 Cheng et al., AAAI 2024。

## R.9 已知差异与限制

详见 `docs/LIMITATIONS.md` (待补) 与 `docs/AUDIT_REPORT.md` (21 项审计)。

---

如发现 reproducibility 漏洞, 请发 issue 或直接联系维护者; 我们承诺 7 天内响应。