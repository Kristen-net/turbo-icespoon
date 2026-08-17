"""
数据预处理流水线 v1.0
输入: D:\DATA_ALL\extracted 下所有未分类图像
输出: 
  D:\DATA_ALL\dataset\clear\        - 清晰图 (作为GT)
  D:\DATA_ALL\dataset\hazy_real\    - 真实雾图 (测试用)
  D:\DATA_ALL\dataset\train\hazy\   - 合成雾图 (训练输入)
  D:\DATA_ALL\dataset\train\clear\  - 对应清晰图 (训练GT)
  D:\DATA_ALL\dataset\val\hazy\
  D:\DATA_ALL\dataset\val\clear\
  D:\DATA_ALL\dataset\test\hazy\    - 真实雾图 (无GT, 仅视觉评测)
  D:\DATA_ALL\dataset\test\clear\   - 从清晰图中抽一部分作为有GT的测试
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ==================== 配置 ====================
SRC_DIR = r'D:\DATA_ALL\extracted'
DST_DIR = r'D:\DATA_ALL\dataset'

# 分类阈值 (暗通道均值, 归一化到[0,1])
DARK_CLEAR_THRESH = 0.18   # 低于此值 = 清晰图
DARK_HAZY_THRESH = 0.35    # 高于此值 = 真实雾图
# 中间的 = 边界模糊, 归入清晰图(因为可以合成雾)

# 合成雾参数
HAZE_LEVELS = [
    (0.6, 0.75, 180),   # 薄雾: t_min, t_max, A_val
    (0.4, 0.6, 200),    # 中雾
    (0.2, 0.4, 220),    # 浓雾
]
SYNTH_PER_CLEAR = 2  # 每张清晰图合成2张雾图 (薄+中)

# 数据集划分
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# ==================== 工具函数 ====================

def compute_dark_channel(img, patch_size=15):
    """计算暗通道均值 (归一化到[0,1])"""
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)


def list_all_images(root_dir):
    """递归收集所有图像路径"""
    images = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过 dataset 目录自己
        if 'dataset' in root:
            continue
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(root, f))
    return sorted(images)


def estimate_atmospheric_light(img, dark_ch, top_percent=0.001):
    """从最亮的暗通道像素估计大气光A"""
    h, w = dark_ch.shape
    num_pixels = h * w
    top_n = max(int(num_pixels * top_percent), 1)
    
    # 找暗通道最亮的前0.1%像素位置
    flat_dark = dark_ch.flatten()
    indices = np.argsort(flat_dark)[-top_n:]
    
    # 取这些位置在原图中的RGB均值
    flat_img = img.reshape(-1, 3).astype(np.float32)
    A = np.mean(flat_img[indices], axis=0)
    # 限制范围
    A = np.clip(A, 180, 240)
    return A


def synthesize_haze(img, t_min, t_max, A_val, use_depth_guided=True):
    """
    合成雾图: I = J * t + A * (1 - t)
    t图由随机深度图生成, 保证空间不均匀性
    """
    h, w = img.shape[:2]
    
    # 生成透射率图 (模拟不同深度)
    if use_depth_guided:
        # 用随机高斯模糊生成平滑的深度图
        noise = np.random.randn(h, w).astype(np.float32)
        # 多次下采样+上采样制造大尺度变化
        for scale in [8, 4, 2]:
            nh, nw = h // scale, w // scale
            small = cv2.resize(noise, (nw, nh))
            noise = cv2.resize(small, (w, h))
        noise = cv2.GaussianBlur(noise, (51, 51), 20)
        
        # 归一化到 [t_min, t_max]
        n_min, n_max = noise.min(), noise.max()
        if n_max - n_min < 1e-6:
            t_map = np.ones((h, w), dtype=np.float32) * (t_min + t_max) / 2
        else:
            t_map = t_min + (noise - n_min) / (n_max - n_min) * (t_max - t_min)
    else:
        # 均匀雾
        t_val = random.uniform(t_min, t_max)
        t_map = np.ones((h, w), dtype=np.float32) * t_val
    
    t_map = np.clip(t_map, 0.05, 0.95)
    
    # 大气光 (三通道略有差异, 更真实)
    A = np.array([A_val + random.randint(-5, 5),
                  A_val + random.randint(-3, 3),
                  A_val + random.randint(-8, 2)], dtype=np.float32)
    A = np.clip(A, 170, 245)
    
    # 合成: I = J * t + A * (1 - t)
    img_f = img.astype(np.float32)
    t_3ch = t_map[:, :, np.newaxis]
    hazy = img_f * t_3ch + A * (1 - t_3ch)
    hazy = np.clip(hazy, 0, 255).astype(np.uint8)
    
    return hazy


def copy_and_rename(src_path, dst_dir, new_name):
    """复制文件并重命名"""
    os.makedirs(dst_dir, exist_ok=True)
    ext = os.path.splitext(src_path)[1].lower()
    dst_path = os.path.join(dst_dir, new_name + ext)
    shutil.copy2(src_path, dst_path)
    return dst_path


def save_image(img, dst_dir, name):
    """保存图像"""
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, name + '.png')
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return path


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("IceWave 数据集构建流水线")
    print("=" * 60)
    
    # Step 1: 收集所有图像
    print("\n[Step 1] 收集所有图像...")
    all_images = list_all_images(SRC_DIR)
    print(f"  找到 {len(all_images)} 张图像")
    
    # Step 2: 计算暗通道值分类
    print("\n[Step 2] 按暗通道值分类 (清晰/边界/有雾)...")
    clear_imgs = []
    hazy_imgs = []
    border_imgs = []
    
    for i, img_path in enumerate(all_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        dark_val = compute_dark_channel(img)
        
        if dark_val < DARK_CLEAR_THRESH:
            clear_imgs.append((img_path, dark_val))
        elif dark_val > DARK_HAZY_THRESH:
            hazy_imgs.append((img_path, dark_val))
        else:
            border_imgs.append((img_path, dark_val))
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(all_images)}")
    
    print(f"  清晰图: {len(clear_imgs)} (暗通道<{DARK_CLEAR_THRESH})")
    print(f"  边界图: {len(border_imgs)} (中间区域, 归入清晰)")
    print(f"  真实雾图: {len(hazy_imgs)} (暗通道>{DARK_HAZY_THRESH})")
    
    # 边界图归入清晰图 (可以合成雾)
    clear_imgs.extend(border_imgs)
    print(f"  -> 最终清晰图: {len(clear_imgs)}")
    
    # Step 3: 划分数据集
    print("\n[Step 3] 划分训练/验证/测试集...")
    
    random.shuffle(clear_imgs)
    random.shuffle(hazy_imgs)
    
    n_clear = len(clear_imgs)
    n_hazy = len(hazy_imgs)
    
    # 清晰图划分: 75%训练(合成雾) + 10%验证 + 15%测试
    n_train_clear = int(n_clear * TRAIN_RATIO)
    n_val_clear = int(n_clear * VAL_RATIO)
    n_test_clear = n_clear - n_train_clear - n_val_clear
    
    train_clear = clear_imgs[:n_train_clear]
    val_clear = clear_imgs[n_train_clear:n_train_clear + n_val_clear]
    test_clear = clear_imgs[n_train_clear + n_val_clear:]
    
    # 真实雾图全部作为测试集 (无GT, 视觉评测)
    test_hazy_real = hazy_imgs
    
    print(f"  训练清晰图: {len(train_clear)}")
    print(f"  验证清晰图: {len(val_clear)}")
    print(f"  测试清晰图: {len(test_clear)}")
    print(f"  真实雾图(测试): {len(test_hazy_real)}")
    
    # Step 4: 合成训练雾图
    print("\n[Step 4] 合成训练集雾图...")
    
    train_hazy_dir = os.path.join(DST_DIR, 'train', 'hazy')
    train_clear_dir = os.path.join(DST_DIR, 'train', 'clear')
    os.makedirs(train_hazy_dir, exist_ok=True)
    os.makedirs(train_clear_dir, exist_ok=True)
    
    train_count = 0
    for i, (img_path, dark_val) in enumerate(train_clear):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        base_name = f'train_{i:04d}'
        
        # 保存清晰图GT
        save_image(img, train_clear_dir, base_name)
        
        # 合成不同浓度雾图
        for level_idx, (t_min, t_max, A_val) in enumerate(HAZE_LEVELS[:SYNTH_PER_CLEAR]):
            hazy_img = synthesize_haze(img, t_min, t_max, A_val)
            haze_name = f'{base_name}_haze{level_idx}'
            save_image(hazy_img, train_hazy_dir, haze_name)
            train_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"  已合成 {i+1}/{len(train_clear)} 组, 共{train_count}张雾图")
    
    print(f"  训练集: {len(train_clear)}张清晰图 → {train_count}张合成雾图")
    
    # Step 5: 合成验证雾图 (每张清晰图合成1张中雾)
    print("\n[Step 5] 合成验证集雾图...")
    
    val_hazy_dir = os.path.join(DST_DIR, 'val', 'hazy')
    val_clear_dir = os.path.join(DST_DIR, 'val', 'clear')
    os.makedirs(val_hazy_dir, exist_ok=True)
    os.makedirs(val_clear_dir, exist_ok=True)
    
    val_count = 0
    for i, (img_path, dark_val) in enumerate(val_clear):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        base_name = f'val_{i:04d}'
        save_image(img, val_clear_dir, base_name)
        
        # 合成中雾 (固定参数保证可复现)
        t_min, t_max, A_val = HAZE_LEVELS[1]  # 中雾
        np.random.seed(i + 1000)
        random.seed(i + 1000)
        hazy_img = synthesize_haze(img, t_min, t_max, A_val)
        save_image(hazy_img, val_hazy_dir, f'{base_name}_haze')
        val_count += 1
    
    print(f"  验证集: {val_count}对 (hazy + clear)")
    
    # Step 6: 测试集
    print("\n[Step 6] 构建测试集...")
    
    # 6a: 有GT的测试 (从清晰图合成)
    test_hazy_dir = os.path.join(DST_DIR, 'test', 'hazy_synthetic')
    test_clear_dir = os.path.join(DST_DIR, 'test', 'clear')
    os.makedirs(test_hazy_dir, exist_ok=True)
    os.makedirs(test_clear_dir, exist_ok=True)
    
    test_count = 0
    for i, (img_path, dark_val) in enumerate(test_clear):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        base_name = f'test_{i:04d}'
        save_image(img, test_clear_dir, base_name)
        
        # 合成中雾
        t_min, t_max, A_val = HAZE_LEVELS[1]
        np.random.seed(i + 5000)
        random.seed(i + 5000)
        hazy_img = synthesize_haze(img, t_min, t_max, A_val)
        save_image(hazy_img, test_hazy_dir, f'{base_name}_haze')
        test_count += 1
    
    print(f"  有GT测试: {test_count}对")
    
    # 6b: 真实雾图 (无GT, 视觉评测)
    test_real_dir = os.path.join(DST_DIR, 'test', 'hazy_real')
    os.makedirs(test_real_dir, exist_ok=True)
    
    for i, (img_path, dark_val) in enumerate(test_hazy_real):
        img = cv2.imread(img_path)
        if img is None:
            continue
        base_name = f'real_{i:04d}'
        save_image(img, test_real_dir, base_name)
    
    print(f"  真实雾图测试: {len(test_hazy_real)}张 (无GT)")
    
    # Step 7: 输出统计
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)
    print(f"\n目录结构: {DST_DIR}")
    print(f"  train/")
    print(f"    hazy/     : {train_count} 张 (合成雾图, 训练输入)")
    print(f"    clear/    : {len(train_clear)} 张 (清晰GT)")
    print(f"  val/")
    print(f"    hazy/     : {val_count} 张 (合成雾图)")
    print(f"    clear/    : {val_count} 张 (清晰GT)")
    print(f"  test/")
    print(f"    hazy_synthetic/ : {test_count} 张 (有GT, 定量评测)")
    print(f"    clear/          : {test_count} 张 (清晰GT)")
    print(f"    hazy_real/      : {len(test_hazy_real)} 张 (真实雾, 定性评测)")
    
    total_synth = train_count + val_count + test_count
    print(f"\n总计合成雾图: {total_synth} 对")
    print(f"真实雾图: {len(hazy_imgs)} 张")


if __name__ == '__main__':
    main()
