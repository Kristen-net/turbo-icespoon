"""下载并解压 CLIP 源码"""
import urllib.request
import zipfile
import io
import os
import sys

print("下载 CLIP 源码...")
urls = [
    "https://ghproxy.com/https://github.com/openai/CLIP/archive/refs/heads/main.zip",
    "https://mirror.ghproxy.com/https://github.com/openai/CLIP/archive/refs/heads/main.zip",
    "https://github.com/openai/CLIP/archive/refs/heads/main.zip",
]

data = None
for url in urls:
    try:
        print(f"  尝试: {url[:60]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        data = resp.read()
        print(f"  下载成功! 大小: {len(data)/1024:.0f} KB")
        break
    except Exception as e:
        print(f"  失败: {e}")

if data is None:
    print("所有下载源均失败!")
    sys.exit(1)

# 解压
target = "D:/dehaze_fusion/CLIP_src"
os.makedirs(target, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as zf:
    zf.extractall(target)

# 检查解压结果
for root, dirs, files in os.walk(target):
    for f in files:
        if f.endswith(".py"):
            print(f"  找到: {os.path.join(root, f)}")
            break
    if files:
        break

# 找到实际的 CLIP 目录（通常是 CLIP-main/）
clip_dirs = [d for d in os.listdir(target)]
print(f"解压目录: {clip_dirs}")
print("CLIP 源码下载完成!")
