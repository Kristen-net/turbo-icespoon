"""修补 CLIP 的 pkg_resources 兼容问题"""
import os

clip_file = r"C:\Users\2457025871\miniconda3\envs\dehaze_fusion\lib\site-packages\clip\clip.py"

with open(clip_file, 'r', encoding='utf-8') as f:
    content = f.read()

old = "from pkg_resources import packaging"
new = """try:
    from pkg_resources import packaging
except (ModuleNotFoundError, ImportError):
    import packaging"""

if old in content:
    content = content.replace(old, new, 1)
    with open(clip_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("CLIP clip.py 修补成功!")
else:
    # 检查是否已经被修补过
    if "import packaging" in content:
        print("CLIP 已经被修补过，无需重复操作")
    else:
        print("未找到目标代码，请检查文件内容")
        print("前10行:")
        for i, line in enumerate(content.split('\n')[:10], 1):
            print(f"  {i}: {line}")
