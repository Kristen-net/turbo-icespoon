# HazeCLIP: Towards Language Guided Real-World Image Dehazing

> ⚠️ **IceWave-DehazeFormer 集成说明 (2026-09)**
>
> 本目录是从 [HazeCLIP 官方仓库](https://github.com/cuaecc/HazeCLIP) 的
> **vendored 副本**, 仅供 IceWave 的 m3/m4/joint 训练模式作为 CLIP 教师蒸馏
> 模块引用。IceWave 项目本身由 [Kristen-net](https://github.com/Kristen-net)
> 维护 (Apache-2.0), 本目录代码版权与许可以**上游 HazeCLIP 仓库**为准。
>
> 使用前请注意:
> 1. 本副本**不完整**: 缺 `build_model.py` 与 `simple_tokenizer.py`, 训练
>    IceWave m3/m4/joint 前需按上游仓库补全;
> 2. 上游许可与引用信息见下文, IceWave 不对 HazeCLIP 论文方法主张权利;
> 3. 新工作请直接使用 [pip 依赖](https://github.com/cuaecc/HazeCLIP) 或 git
>    submodule, 而非依赖本副本 (审计编号 L2)。
>
> ---
>
> 以下为 HazeCLIP 上游 README 内容 (版权: Ruiyi Wang et al., MIT/Apache)。

<a href="https://arxiv.org/abs/2407.13719"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b"></a>[![License](https://img.shields.io/badge/License-MIT-929292)](https://www.apache.org/licenses/LICENSE-2.0)

This repository contains the implementation of the paper "HazeCLIP: Towards Language Guided Real-World Image Dehazing".

We present HazeCLIP, a language-guided adaptation framework designed to enhance the real-world performance of pre-trained dehazing networks.

![teaser](assets/method.png)
![teaser](assets/comparisons.png)

## 🛠️ Setup

Set up conda environment via

```bash
conda create -n HazeCLIP python=3.9
conda activate HazeCLIP
pip install -r requirements.txt
```
## 🚀 Usage
Please modify the corresponding yaml configuration file before running the command.

### 🏋️ Inference

Download checkpoint from [Baidu Yun ](https://pan.baidu.com/s/1TxVUKOrNRGI19BaSBDbwIg)(code: haze) and put it in ./weights/ folder.

```py
python inference.py --config configs/inference.yaml
```



### 🚀 Training

#### Pre-training

Download synthetic data from [RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing) and put it under ./data/ folder.

```python
python pretrain.py --config configs/pretrain.yaml
```

#### Fine-tuning

Download fine-tuning dataset from [Baidu Yun](https://pan.baidu.com/s/1TxVUKOrNRGI19BaSBDbwIg) (code: haze) and put it under ./data/ folder.

```pyth
python finetune.py --config configs/finetune.yaml
```




### 🎓 Citation

If you find our work helpful, please consider cite our work as

```bibtex
@inproceedings{wang2024hazeclip,
  title     = {HazeCLIP: Towards Language Guided Real-World Image Dehazing},
  author    = {Wang, Ruiyi and Li, Wenhao and Liu, Xiaohong and Li, Chunyi and
               Zhang, Zicheng and Min, Xiaohong and Zhai, Guangtao},
  booktitle = {ECCV},
  year      = {2024},
  eprint    = {2407.13719},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
}
```

### 🎫 Acknowledgement (上游)

Parts of the codes are adopted from [RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing),
[CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery) and
[CLIP-LIT](https://github.com/ZhexinLiang/CLIP-LIT). Thanks for their work!

---

### IceWave 集成补充 (2026-09)

在 IceWave 中, HazeCLIP 仅作为**可选教师**在 m3/m4/joint 训练模式使用:

```python
# src/icewave/models/prompt.py
from source_hazeclip import clip  # 教师编码器
# 提示分支: CLIPSurgery 49 token → 1x1 conv → 32 通道提示 M_h → HA-WFE 的 LL 分支
```

详见 `src/icewave/train/trainer.py` 中 `HazeCLIPTeacher` 的调用方式。

**审计依据**: IceWave 仓库代码 vs 文档一致性审计 L2 (见 `docs/AUDIT_REPORT.md`)。