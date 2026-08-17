import sys
print("=" * 60)
print("环境验证报告")
print("=" * 60)

# PyTorch + CUDA
import torch
cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
print(f"[PyTorch] {torch.__version__} | CUDA: {cuda_ok} | GPU: {gpu_name}")

# torchvision
import torchvision
print(f"[torchvision] {torchvision.__version__}")

# HazeCLIP deps
import omegaconf; print(f"[omegaconf] {omegaconf.__version__}")
import einops; print(f"[einops] {einops.__version__}")
import ftfy; print(f"[ftfy] OK")
import regex; print(f"[regex] OK")
import tqdm; print(f"[tqdm] {tqdm.__version__}")

# DehazeSB deps
import dominate; print(f"[dominate] OK")
import pytorch_msssim; print(f"[pytorch_msssim] OK")
import lpips; print(f"[lpips] OK")
import transformers; print(f"[transformers] {transformers.__version__}")
import pyiqa; print(f"[pyiqa] OK")

# DiffDehaze-GAN deps
import diffusers; print(f"[diffusers] {diffusers.__version__}")
import torchmetrics; print(f"[torchmetrics] {torchmetrics.__version__}")
import torch_fidelity; print(f"[torch_fidelity] OK")

# WDMamba deps
import scipy; print(f"[scipy] {scipy.__version__}")
import skimage; print(f"[scikit-image] {skimage.__version__}")
import lmdb; print(f"[lmdb] OK")
import addict; print(f"[addict] OK")
import yapf; print(f"[yapf] OK")

# mamba_ssm check
try:
    import mamba_ssm
    print(f"[mamba_ssm] OK")
except ImportError:
    print(f"[mamba_ssm] NOT INSTALLED (needs CUDA Toolkit + nvcc)")

try:
    import causal_conv1d
    print(f"[causal_conv1d] OK")
except ImportError:
    print(f"[causal_conv1d] NOT INSTALLED (needs CUDA Toolkit + nvcc)")

print("=" * 60)
if cuda_ok:
    print("结论: PyTorch CUDA 环境正常, RTX 5060 可用")
else:
    print("警告: CUDA 不可用!")
print("=" * 60)
