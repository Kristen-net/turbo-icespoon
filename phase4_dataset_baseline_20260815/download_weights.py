"""尝试从HuggingFace镜像下载DehazeFormer权重"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import list_repo_files, hf_hub_download

# 尝试不同的repo名
repos = [
    "IDKiro/DehazeFormer_Demo",
    "IDKiro/DehazeFormer",
    "IDKiro/dehazeformer",
]

for repo in repos:
    try:
        files = list_repo_files(repo, repo_type="space")
        print(f"[Space] {repo}: {files}")
    except Exception as e:
        print(f"[Space] {repo}: {e}")
    
    try:
        files = list_repo_files(repo, repo_type="model")
        print(f"[Model] {repo}: {files}")
    except Exception as e:
        print(f"[Model] {repo}: {e}")
