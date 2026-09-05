# Third-Party Notices (完整三方依赖声明表)

> 本文件按 [REUSE Software](https://reuse.software/) 与多数 SCI 一区期刊 (IEEE TPAMI,
> Springer LNCS, Elsevier Neurocomputing) 投稿要求逐项列出本项目所用三方依赖, 含
> 许可、来源、修改点、商业使用约束。本表与 `NOTICE.md` 互补, NOTICE 仅列项目自身
> 贡献声明, 本表覆盖所有引用。

## 1.1 深度学习框架与基础库

| 组件 | 版本 | 许可 | 用途 | 修改点 |
|---|---|---|---|---|
| [PyTorch](https://pytorch.org/) | ≥2.0 | BSD-3-Clause | 骨干框架 | 无源码修改 |
| [torchvision](https://github.com/pytorch/vision) | ≥0.15 | BSD-3-Clause | 视觉工具集 | 无源码修改 |
| [NumPy](https://numpy.org/) | ≥1.24 | BSD-3-Clause | 数值计算 | 无源码修改 |
| [timm](https://github.com/huggingface/pymmar) | ≥0.9 | Apache-2.0 | 兼容层导入 | 见 §1.2 |

## 1.2 代码 vendoring (随仓库再分发)

| 组件 | 原始仓库 | 许可 | vendoring 位置 | 修改点 |
|---|---|---|---|---|
| DehazeFormer 骨干 | <https://github.com/IDEA-Research/DehazeFormer> | Apache-2.0 | `src/icewave/models/dehazeformer.py` | timm 兼容适配 (`try: from timm.layers import ... except: from timm.models.layers import ...`) |
| HazeCLIP 模型 | <https://github.com/cooper12121/HazeCLIP> (论文: Cheng et al., AAAI 2024) | 研究使用许可 (未明确开源) | `source_hazeclip/` | 仅供研究/蒸馏; 见 `source_hazeclip/README.md` 上游来源说明 |

## 1.3 可选 extras 依赖 (按需)

| Extra | 组件 | 许可 | 商业使用约束 |
|---|---|---|---|
| `train` | [torchmetrics](https://github.com/Lightning-AI/torchmetrics) ≥1.0 | Apache-2.0 | 无 |
| `quality` | [lpips](https://github.com/richzhang/PerceptualSimilarity) ≥0.1 | BSD-2-Clause (AlexNet 权重: 非商业) | AlexNet 子权重仅供研究; 若需商业部署请自行替换骨干 |
| `detect` | [ultralytics](https://github.com/ultralytics/ultralytics) ≥8.2 (YOLOv8) | **AGPL-3.0** | **⚠ 商业使用需评估**: 集成到闭源商业产品须开源整体应用, 或购买商业许可 |
| `dev` | [pytest](https://pytest.org/), pytest-cov | MIT | 无 |

## 1.4 第三方权重 (随脚本下载, 不随仓库分发)

| 组件 | 来源 | 许可 | 备注 |
|---|---|---|---|
| IceWave 预训练权重 (.pth) | 见 `configs/weights.yaml` | Apache-2.0 (项目自身) | 通过 `scripts/download_weights.py` 下载并 SHA256 校验 |
| HazeCLIP 教师权重 | 见 `source_hazeclip/` | 见上游 README | 可选, 仅 m3/m4/joint 训练需要 |
| YOLOv8 覆冰检测权重 (.pt) | 见 `configs/weights.yaml` | **AGPL-3.0** (随 ultralytics) | 与 ultralytics 许可一致; 商业约束同 §1.3 |

## 1.5 SCI 期刊投稿合规说明

多数 SCI 一区期刊 (IEEE / Springer / Elsevier) 要求作者在论文末尾或 supplementary 中
声明三方依赖。本表已按 *Component · License · Source · Modification · Commercial use*
五维结构整理, 可直接复制到论文的 *Acknowledgements* 或 *Reproducibility Statement*。

**特别提示**: 若审稿人质疑 YOLOv8 AGPL 许可兼容性, 可提供两种应对:
1. **学术发表**: AGPL-3.0 不限制论文发表与学术演示, 无问题
2. **商业化/产品化**: 改用 [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (Apache-2.0)
   或 [YOLOv5](https://github.com/ultralytics/yolov5) 的老版本 (GPL-3.0 已受 AGPL-3.0
   约束, 同问题); 推荐使用 [RF-DETR](https://github.com/roboflow/rfdetr) (Apache-2.0)
   作下游兼容版

---

如发现本表遗漏, 请发 issue 或 PR; 维护者将逐项核实并补充。