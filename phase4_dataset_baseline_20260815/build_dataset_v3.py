"""
数据预处理流水线 v3.0 - 加速版
改进: 
1. 缩略图计算暗通道 (速度提升10倍+)
2. 分阶段输出进度
3. 大图像直接缩放到训练尺寸再保存
"""

import os
import cv2
import numpy as np
import random
import zipfile
import time

# ==================== 配置 ====================
ZIP_DIR = r'D:\DATA_ALL'
DST_DIR = r'D:\DATA_ALL\dataset'

# 分类阈值
DARK_CLEAR_THRESH = 0.18
DARK_HAZY_THRESH = 0.35

# 缩略图大小 (用于快速暗通道计算)
THUMB_SIZE = 256

# 训练/验证图像尺寸 (统一到这个尺寸，省显存)
TRAIN_SIZE = 512

# 合成雾参数
HAZE_LEVELS = [
    (0.6, 0.75, 180),   # 薄雾
    (0.4, 0.6, 200),    # 中雾
    (0.2, 0.4, 220),    # 浓雾
]
SYNTH_PER_CLEAR = 2

# 数据集划分
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def read_image_from_zip(zf, name, max_size=None):
    """从zip读取图像，可选缩放到max_size"""
    data = zf.read(name)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if max_size and (img.shape[0] > max_size or img.shape[1] > max_size):
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def list_images_in_zip(zf):
    return [n for n in zf.namelist() if n.lower().endswith(IMAGE_EXTS)]


def compute_dark_channel_fast(img, patch_size=15):
    """快速暗通道计算 - 先缩小"""
    h, w = img.shape[:2]
    if max(h, w) > THUMB_SIZE:
        scale = THUMB_SIZE / max(h, w)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        thumb = img
    
    min_val = np.min(thumb, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)


def synthesize_haze(img, t_min, t_max, A_val):
    """合成雾图"""
    h, w = img.shape[:2]
    
    # 生成空间不均匀透射率
    noise = np.random.randn(h // 8, w // 8).astype(np.float32)
    noise = cv2.resize(noise, (w, h))
    noise = cv2.GaussianBlur(noise, (31, 31), 10)
    
    n_min, n_max = noise.min(), noise.max()
    if n_max - n_min < 1e-6:
        t_map = np.ones((h, w), dtype=np.float32) * (t_min + t_max) / 2
    else:
        t_map = t_min + (noise - n_min) / (n_max - n_min) * (t_max - t_min)
    
    t_map = np.clip(t_map, 0.05, 0.95)
    
    A = np.array([A_val + random.randint(-5, 5),
                  A_val + random.randint(-3, 3),
                  A_val + random.randint(-8, 2)], dtype=np.float32)
    A = np.clip(A, 170, 245)
    
    img_f = img.astype(np.float32)
    t_3ch = t_map[:, :, np.newaxis]
    hazy = img_f * t_3ch + A * (1 - t_3ch)
    return np.clip(hazy, 0, 255).astype(np.uint8)


def save_image(img, dst_dir, name):
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, name + '.png')
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return path


def main():
    t0 = time.time()
    print("=" * 60)
    print("IceWave 数据集构建流水线 v3.0 (加速版)")
    print("=" * 60)
    
    # Step 1: 收集图像
    print("\n[Step 1] 收集图像...")
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.lower().endswith('.zip')]
    
    all_images = []
    for zname in zip_files:
        zpath = os.path.join(ZIP_DIR, zname)
        with zipfile.ZipFile(zpath, 'r') as zf:
            imgs = list_images_in_zip(zf)
            for img_name in imgs:
                all_images.append((zpath, img_name))
    
    print(f"  共 {len(all_images)} 张图像 ({len(zip_files)}个zip)")
    
    # Step 2: 快速分类
    print(f"\n[Step 2] 快速分类 (缩略图暗通道)...")
    clear_imgs = []
    hazy_imgs = []
    border_imgs = []
    
    current_zip = None
    zf = None
    
    for i, (zpath, img_name) in enumerate(all_images):
        if zpath != current_zip:
            if zf: zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        # 只读缩略图用于分类
        img_thumb = read_image_from_zip(zf, img_name, max_size=THUMB_SIZE)
        if img_thumb is None:
            continue
        
        dark_val = compute_dark_channel_fast(img_thumb)
        item = (zpath, img_name, dark_val)
        
        if dark_val < DARK_CLEAR_THRESH:
            clear_imgs.append(item)
        elif dark_val > DARK_HAZY_THRESH:
            hazy_imgs.append(item)
        else:
            border_imgs.append(item)
        
        if (i + 1) % 200 == 0:
            print(f"  已分类 {i+1}/{len(all_images)} (清晰:{len(clear_imgs)} 雾:{len(hazy_imgs)} 边界:{len(border_imgs)})")
    
    if zf: zf.close()
    
    clear_imgs.extend(border_imgs)
    print(f"\n  最终: 清晰图={len(clear_imgs)}, 真实雾图={len(hazy_imgs)}")
    
    # Step 3: 划分
    print("\n[Step 3] 划分数据集...")
    random.shuffle(clear_imgs)
    random.shuffle(hazy_imgs)
    
    n = len(clear_imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train_clear = clear_imgs[:n_train]
    val_clear = clear_imgs[n_train:n_train + n_val]
    test_clear = clear_imgs[n_train + n_val:]
    test_hazy_real = hazy_imgs
    
    print(f"  训练: {len(train_clear)}, 验证: {len(val_clear)}, 测试: {len(test_clear)}")
    print(f"  真实雾图: {len(test_hazy_real)}")
    
    # Step 4: 训练集合成
    print(f"\n[Step 4] 合成训练雾图 ({len(train_clear)}张清晰 × {SYNTH_PER_CLEAR} = {len(train_clear)*SYNTH_PER_CLEAR}张)...")
    
    train_hazy_dir = os.path.join(DST_DIR, 'train', 'hazy')
    train_clear_dir = os.path.join(DST_DIR, 'train', 'clear')
    
    train_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val) in enumerate(train_clear):
        if zpath != current_zip:
            if zf: zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name, max_size=TRAIN_SIZE)
        if img is None:
            continue
        
        base_name = f'train_{i:04d}'
        save_image(img, train_clear_dir, base_name)
        
        for level_idx in range(SYNTH_PER_CLEAR):
            t_min, t_max, A_val = HAZE_LEVELS[level_idx]
            hazy_img = synthesize_haze(img, t_min, t_max, A_val)
            save_image(hazy_img, train_hazy_dir, f'{base_name}_haze{level_idx}')
            train_count += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  训练集: {i+1}/{len(train_clear)} ({train_count}张雾图) [{elapsed:.0f}s]")
    
    if zf: zf.close()
    print(f"  完成: {train_count}张训练雾图")
    
    # Step 5: 验证集
    print("\n[Step 5] 合成验证集雾图...")
    
    val_hazy_dir = os.path.join(DST_DIR, 'val', 'hazy')
    val_clear_dir = os.path.join(DST_DIR, 'val', 'clear')
    
    val_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val) in enumerate(val_clear):
        if zpath != current_zip:
            if zf: zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name, max_size=TRAIN_SIZE)
        if img is None: continue
        
        base_name = f'val_{i:04d}'
        save_image(img, val_clear_dir, base_name)
        
        np.random.seed(i + 1000)
        random.seed(i + 1000)
        t_min, t_max, A_val = HAZE_LEVELS[1]
        hazy_img = synthesize_haze(img, t_min, t_max, A_val)
        save_image(hazy_img, val_hazy_dir, f'{base_name}_haze')
        val_count += 1
    
    if zf: zf.close()
    print(f"  完成: {val_count}对")
    
    # Step 6: 测试集
    print("\n[Step 6] 构建测试集...")
    
    test_hazy_dir = os.path.join(DST_DIR, 'test', 'hazy_synthetic')
    test_clear_dir = os.path.join(DST_DIR, 'test', 'clear')
    test_real_dir = os.path.join(DST_DIR, 'test', 'hazy_real')
    
    test_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val) in enumerate(test_clear):
        if zpath != current_zip:
            if zf: zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name, max_size=TRAIN_SIZE)
        if img is None: continue
        
        base_name = f'test_{i:04d}'
        save_image(img, test_clear_dir, base_name)
        
        np.random.seed(i + 5000)
        random.seed(i + 5000)
        t_min, t_max, A_val = HAZE_LEVELS[1]
        hazy_img = synthesize_haze(img, t_min, t_max, A_val)
        save_image(hazy_img, test_hazy_dir, f'{base_name}_haze')
        test_count += 1
    
    if zf: zf.close()
    
    # 真实雾图
    real_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val) in enumerate(test_hazy_real):
        if zpath != current_zip:
            if zf: zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name, max_size=TRAIN_SIZE)
        if img is None: continue
        save_image(img, test_real_dir, f'real_{i:04d}')
        real_count += 1
    
    if zf: zf.close()
    
    # 汇总
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)
    print(f"\n输出目录: {DST_DIR}")
    print(f"  train/hazy/     : {train_count} 张 (合成雾图)")
    print(f"  train/clear/    : {len(train_clear)} 张 (清晰GT)")
    print(f"  val/hazy/       : {val_count} 张")
    print(f"  val/clear/      : {val_count} 张")
    print(f"  test/hazy_synth : {test_count} 张 (有GT)")
    print(f"  test/clear/     : {test_count} 张")
    print(f"  test/hazy_real/ : {real_count} 张 (真实雾)")
    print(f"\n总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
