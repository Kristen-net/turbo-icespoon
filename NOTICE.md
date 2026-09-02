# NOTICE — 第三方代码与依赖许可声明

本项目 (IceWave-DehazeFormer) 主体以 Apache-2.0 发布, 但集成/引用了以下
第三方代码与模型, 其许可可能与本项目不同, 使用时请遵守各自条款。

## 1. DehazeFormer (vendored 骨干)

- 来源: https://github.com/cecid3/DehazeFormer
- 路径: `src/icewave/models/dehazeformer.py` (仅做 timm API 兼容性修改)
- 许可: BSD-3-Clause
- 说明: 保留其原始版权声明于文件头部 docstring 未覆盖的源码注释中,
  未改动任何计算逻辑。

## 2. HA-WFE / HAWFEv2 模块

- 来源: 项目作者原创 (phase4/ha_wfe.py, phase5/ha_wfe_v2.py)
- 许可: 随本项目 Apache-2.0

## 3. HazeCLIP (教师蒸馏, 可选)

- 来源: 第三方 CLIP 蒸馏实现 (路径 `source_hazeclip/`)
- 许可: 需查阅上游仓库; 当前仓库内的副本不完整, 使用前需补全
  `build_model.py` / `simple_tokenizer.py` 并按上游许可处理。

## 4. CLIP 预训练权重

- OpenAI CLIP 模型权重, 许可需遵循 OpenAI 模型许可协议。
- 本项目未分发权重, 由 `scripts/download_weights.py` 或用户自行获取。

## 5. ultralytics (YOLO 检测, 可选依赖)

- 许可: AGPL-3.0 (Ultralytics 对部分代码使用 AGPL)
- 重要: AGPL 传染性较强, 若将本项目用于**闭源商业**分发, 需评估
  ultralytics 依赖的许可影响; 学术研究使用不受限。

## 6. 公开数据集

- RESIDE (SOTS-Indoor/Outdoor): 遵循其数据使用条款。
- O-HAZE / I-HAZE (NTIRE): 遵循 NTIRE 数据使用条款。
- 本项目不分发数据集, 需用户按原许可自行下载。

---

如对某一第三方组件的许可存在疑问, 请以该组件上游仓库的许可文件为准。
