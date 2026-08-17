import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    x = torch.randn(4, 3, 192, 192, device="cuda")
    y = x * 2
    print(f"GPU tensor test: {y.shape}, device={y.device}")
else:
    print("CUDA not available")
