"""尝试多个镜像源克隆 CLIP"""
import subprocess
import os
import sys

# git 可执行文件路径
git_bin = r"C:\Users\2457025871\miniconda3\envs\dehaze_fusion\Library\bin\git.exe"
target = r"D:\dehaze_fusion\CLIP_src"

# 多个镜像源
mirrors = [
    ("gitclone.com", "https://gitclone.com/github.com/openai/CLIP.git"),
    ("kgithub.com", "https://kgithub.com/openai/CLIP.git"),
    ("ghproxy.com", "https://ghproxy.com/https://github.com/openai/CLIP.git"),
    ("直接GitHub", "https://github.com/openai/CLIP.git"),
]

for name, url in mirrors:
    print(f"\n尝试 {name}: {url}")
    if os.path.exists(target):
        import shutil
        shutil.rmtree(target, ignore_errors=True)

    try:
        result = subprocess.run(
            [git_bin, "clone", "--depth", "1", url, target],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        )
        if result.returncode == 0:
            print(f"  克隆成功!")
            # 验证
            clip_file = os.path.join(target, "clip", "__init__.py")
            if os.path.exists(clip_file):
                print(f"  验证: {clip_file} 存在")
                print("CLIP 源码克隆完成!")
                sys.exit(0)
            else:
                print(f"  警告: clip/__init__.py 不存在")
                # 列出目录
                for f in os.listdir(target):
                    print(f"    {f}")
        else:
            print(f"  失败: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  超时（120秒）")
    except Exception as e:
        print(f"  错误: {e}")

print("\n所有 git 镜像均失败，尝试 PyPI 替代包...")

# 尝试从 PyPI 安装替代包
import urllib.request
# clip-anytorch 是 OpenAI CLIP 的 PyPI 镜像
pypi_packages = [
    "clip-anytorch",
    "openai-clip",
]

pip_bin = r"C:\Users\2457025871\miniconda3\envs\dehaze_fusion\Scripts\pip.exe"
for pkg in pypi_packages:
    print(f"\n尝试 pip install {pkg}...")
    try:
        result = subprocess.run(
            [pip_bin, "install", pkg, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
             "--trusted-host", "pypi.tuna.tsinghua.edu.cn"],
            capture_output=True, text=True, timeout=120
        )
        print(f"  stdout: {result.stdout[:300]}")
        if result.returncode == 0:
            print(f"  安装成功!")
            # 验证
            result2 = subprocess.run(
                [r"C:\Users\2457025871\miniconda3\envs\dehaze_fusion\python.exe", "-c", "import clip; print('clip OK:', clip.__file__)"],
                capture_output=True, text=True, timeout=30
            )
            print(f"  验证: {result2.stdout.strip()}")
            if result2.returncode == 0:
                sys.exit(0)
        else:
            print(f"  失败: {result.stderr[:200]}")
    except Exception as e:
        print(f"  错误: {e}")

print("\n所有方法均失败。")
