"""从HuggingFace镜像下载DehazeFormer权重"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

save_path = r"D:\dehaze_fusion\DehazeFormer\pretrained"
os.makedirs(save_path, exist_ok=True)

print("Downloading DehazeFormer weights from HF mirror...")
file_path = hf_hub_download(
    repo_id="IDKiro/DehazeFormer_Demo",
    filename="saved_models/dehazeformer.pth",
    repo_type="space",
    local_dir=save_path,
)
print(f"Downloaded to: {file_path}")
print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")

# 验证权重能否加载
import torch
ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict):
    if "model" in ckpt:
        model_keys = list(ckpt["model"].keys())
        print(f"\nCheckpoint contains model state dict with {len(model_keys)} keys")
        print(f"First 5 keys: {model_keys[:5]}")
    elif "state_dict" in ckpt:
        model_keys = list(ckpt["state_dict"].keys())
        print(f"\nCheckpoint contains state_dict with {len(model_keys)} keys")
        print(f"First 5 keys: {model_keys[:5]}")
    else:
        model_keys = list(ckpt.keys())
        print(f"\nCheckpoint top-level keys ({len(model_keys)}): {model_keys[:10]}")
else:
    print(f"\nCheckpoint type: {type(ckpt)}")
