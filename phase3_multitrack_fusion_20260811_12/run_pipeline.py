"""
一键去雾-覆冰检测管线
使用方法:
  1. 把你的图片放到 D:\dehaze_fusion\test_images\ 目录
  2. 运行: python run_pipeline.py
  3. 结果在 D:\dehaze_fusion\pipeline_output\

自动执行: HazeCLIP → DehazeSB → DCP → 三赛道融合 → YOLOv8 检测 + 覆冰估计
"""
import os
import sys
import shutil
import glob
import subprocess
import time

# ==================== 配置 ====================
# Python 路径 (yolov8 环境)
PYTHON = r"C:\Users\2457025871\miniconda3\envs\yolov8\python.exe"

# 输入目录 (你的新数据集放这里)
INPUT_DIR = r"D:\dehaze_fusion\test_images"

# 各赛道目录
HAZECLIP_DIR = r"D:\dehaze_fusion\HazeCLIP"
DEHAZESB_DIR = r"D:\dehaze_fusion\DehazeSB"
DIFFDEHAZE_DIR = r"D:\dehaze_fusion\DiffDehaze-GAN"

# 各赛道输入目录
HAZECLIP_INPUT = os.path.join(HAZECLIP_DIR, "images")
DEHAZESB_INPUT = os.path.join(DEHAZESB_DIR, "test_data")

# 各赛道输出目录
HAZECLIP_OUTPUT = os.path.join(HAZECLIP_DIR, "outputs")
DEHAZESB_OUTPUT = os.path.join(DEHAZESB_DIR, "output", "dehazesb_results")
DCP_OUTPUT = os.path.join(DIFFDEHAZE_DIR, "output", "diffdehaze_results")

# 融合和检测输出
FUSION_OUTPUT = r"D:\dehaze_fusion\fusion_3tracks_output"
END2END_OUTPUT = r"D:\dehaze_fusion\end2end_3tracks_output"

# 支持的图像格式
IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')

# 脚本路径
SCRIPTS_DIR = r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d"
SCRIPT_DEHAZESB = os.path.join(SCRIPTS_DIR, "dehazesb_inference.py")
SCRIPT_DCP = os.path.join(SCRIPTS_DIR, "diffdehaze_inference.py")
SCRIPT_FUSION = os.path.join(SCRIPTS_DIR, "fusion_3tracks_auto.py")
SCRIPT_END2END = os.path.join(SCRIPTS_DIR, "end2end_3tracks_auto.py")


def find_images(directory):
    """查找目录中的所有图像文件"""
    images = []
    for ext in IMAGE_EXTS:
        images.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(images)


def clear_directory(directory):
    """清空目录内容但保留目录"""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(directory, exist_ok=True)


def run_step(step_name, script_path, cwd=None):
    """运行一个步骤"""
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=cwd or os.path.dirname(script_path),
        capture_output=False
    )
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"  [ERROR] {step_name} 失败 (返回码 {result.returncode})")
        return False
    
    print(f"  [OK] {step_name} 完成 ({elapsed:.1f}s)")
    return True


def main():
    print("=" * 60)
    print("  一键去雾-覆冰检测管线")
    print("  HazeCLIP → DehazeSB → DCP → 融合 → YOLOv8 检测")
    print("=" * 60)
    
    # 1. 检查输入
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"\n已创建输入目录: {INPUT_DIR}")
        print(f"请把你的图片放进去，然后重新运行此脚本。")
        return
    
    images = find_images(INPUT_DIR)
    if not images:
        print(f"\n输入目录 {INPUT_DIR} 中没有图片！")
        print(f"请把你的图片放进去 (支持 png/jpg/bmp/tif)")
        return
    
    print(f"\n找到 {len(images)} 张图片:")
    for img in images:
        print(f"  - {os.path.basename(img)} ({os.path.getsize(img)/1024:.1f} KB)")
    
    # 2. 清空旧输出
    print(f"\n[0/6] 清空旧输出...")
    for d in [HAZECLIP_OUTPUT, DEHAZESB_OUTPUT, DCP_OUTPUT, FUSION_OUTPUT, END2END_OUTPUT]:
        clear_directory(d)
        os.makedirs(d, exist_ok=True)
    
    # 3. 分发图片到各赛道输入目录
    print(f"\n[0/6] 分发图片到各赛道...")
    # HazeCLIP
    clear_directory(HAZECLIP_INPUT)
    os.makedirs(HAZECLIP_INPUT, exist_ok=True)
    for img in images:
        shutil.copy2(img, HAZECLIP_INPUT)
    
    # DehazeSB
    clear_directory(DEHAZESB_INPUT)
    os.makedirs(DEHAZESB_INPUT, exist_ok=True)
    for img in images:
        shutil.copy2(img, DEHAZESB_INPUT)
    
    print(f"  已复制 {len(images)} 张图片到 HazeCLIP/images/ 和 DehazeSB/test_data/")
    
    # 4. 赛道 1: HazeCLIP
    print(f"\n[1/6] 运行 HazeCLIP 去雾...")
    hazeclip_inference = os.path.join(HAZECLIP_DIR, "inference.py")
    hazeclip_config = os.path.join(HAZECLIP_DIR, "configs", "inference.yaml")
    
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, hazeclip_inference, "--config", hazeclip_config],
        cwd=HAZECLIP_DIR
    )
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"  [ERROR] HazeCLIP 失败")
        return
    hazeclip_results = find_images(HAZECLIP_OUTPUT)
    print(f"  [OK] HazeCLIP 完成 ({elapsed:.1f}s), 生成 {len(hazeclip_results)} 张图")
    
    # 5. 赛道 2: DehazeSB
    print(f"\n[2/6] 运行 DehazeSB 去雾...")
    if not run_step("DehazeSB", SCRIPT_DEHAZESB):
        print("  跳过 DehazeSB, 继续其他赛道")
    dehazesb_results = find_images(DEHAZESB_OUTPUT)
    print(f"  生成 {len(dehazesb_results)} 张图")
    
    # 6. 赛道 3: DCP
    print(f"\n[3/6] 运行 DCP 去雾...")
    if not run_step("DCP", SCRIPT_DCP):
        print("  跳过 DCP, 继续融合")
    dcp_results = find_images(DCP_OUTPUT)
    print(f"  生成 {len(dcp_results)} 张图")
    
    # 7. 融合
    print(f"\n[4/6] 运行三赛道融合...")
    if not run_step("融合", SCRIPT_FUSION):
        print("  [ERROR] 融合失败")
        return
    fusion_results = [f for f in find_images(FUSION_OUTPUT) if 'fusion_final' in f]
    print(f"  生成 {len(fusion_results)} 张最终融合图")
    
    # 8. 端到端检测
    print(f"\n[5/6] 运行 YOLOv8 检测 + 覆冰估计...")
    if not run_step("端到端检测", SCRIPT_END2END):
        print("  [ERROR] 端到端检测失败")
        return
    
    # 9. 完成
    print(f"\n[6/6] 全部完成!")
    print(f"\n{'='*60}")
    print(f"  结果输出:")
    print(f"  融合去雾图: {FUSION_OUTPUT}")
    print(f"  检测结果:   {END2END_OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
