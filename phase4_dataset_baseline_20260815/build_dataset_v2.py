"""
数据预处理流水线 v2.0
改进: 直接从zip读取图像，避免中文路径导致的cv2.imread失败
"""

import os
import cv2
import numpy as np
import random
import zipfile
import io

# ==================== 配置 ====================
ZIP_DIR = r'D:\DATA_ALL'
DST_DIR = r'D:\DATA_ALL\dataset'

# 分类阈值 (暗通道均值, 归一化到[0,1])
DARK_CLEAR_THRESH = 0.18
DARK_HAZY_THRESH = 0.35

# 合成雾参数 (薄/中/浓)
HAZE_LEVELS = [
    (0.6, 0.75, 180),
    (0.4, 0.6, 200),
    (0.2, 0.4, 220),
]
SYNTH_PER_CLEAR = 2  # 每张清晰图合成2张雾图

# 数据集划分
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# ==================== 工具函数 ====================

def read_image_from_zip(zf, name):
    """从zip文件中读取图像，避免中文路径问题"""
    data = zf.read(name)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img


def list_images_in_zip(zf):
    """列出zip中所有图像"""
    return [n for n in zf.namelist() if n.lower().endswith(IMAGE_EXTS)]


def compute_dark_channel(img, patch_size=15):
    """计算暗通道均值 (归一化到[0,1])"""
    min_val = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_val.astype(np.float32), kernel)
    return float(np.mean(dark) / 255.0)


def synthesize_haze(img, t_min, t_max, A_val, use_depth_guided=True):
    """合成雾图: I = J * t + A * (1 - t)"""
    h, w = img.shape[:2]
    
    if use_depth_guided:
        # 用随机高斯模糊生成平滑的深度图 (模拟空间不均匀雾)
        noise = np.random.randn(h, w).astype(np.float32)
        for scale in [8, 4, 2]:
            nh, nw = max(h // scale, 1), max(w // scale, 1)
            small = cv2.resize(noise, (nw, nh))
            noise = cv2.resize(small, (w, h))
        noise = cv2.GaussianBlur(noise, (51, 51), 20)
        
        n_min, n_max = noise.min(), noise.max()
        if n_max - n_min < 1e-6:
            t_map = np.ones((h, w), dtype=np.float32) * (t_min + t_max) / 2
        else:
            t_map = t_min + (noise - n_min) / (n_max - n_min) * (t_max - t_min)
    else:
        t_val = random.uniform(t_min, t_max)
        t_map = np.ones((h, w), dtype=np.float32) * t_val
    
    t_map = np.clip(t_map, 0.05, 0.95)
    
    # 大气光 (三通道略有差异)
    A = np.array([A_val + random.randint(-5, 5),
                  A_val + random.randint(-3, 3),
                  A_val + random.randint(-8, 2)], dtype=np.float32)
    A = np.clip(A, 170, 245)
    
    # 合成
    img_f = img.astype(np.float32)
    t_3ch = t_map[:, :, np.newaxis]
    hazy = img_f * t_3ch + A * (1 - t_3ch)
    hazy = np.clip(hazy, 0, 255).astype(np.uint8)
    
    return hazy


def save_image(img, dst_dir, name):
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, name + '.png')
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return path


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("IceWave 数据集构建流水线 v2.0 (从zip直接读取)")
    print("=" * 60)
    
    # Step 1: 收集所有zip中的图像
    print("\n[Step 1] 收集所有zip中的图像...")
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.lower().endswith('.zip')]
    
    all_images = []  # (zip_path, inner_path)
    
    for zname in zip_files:
        zpath = os.path.join(ZIP_DIR, zname)
        with zipfile.ZipFile(zpath, 'r') as zf:
            imgs = list_images_in_zip(zf)
            for img_name in imgs:
                all_images.append((zpath, img_name))
    
    print(f"  找到 {len(all_images)} 张图像 (来自{len(zip_files)}个zip)")
    
    # Step 2: 计算暗通道值分类
    print("\n[Step 2] 按暗通道值分类 (清晰/边界/有雾)...")
    clear_imgs = []
    hazy_imgs = []
    border_imgs = []
    
    # 按zip分组处理，减少zip打开次数
    current_zip = None
    zf = None
    
    for i, (zpath, img_name) in enumerate(all_images):
        if zpath != current_zip:
            if zf:
                zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name)
        if img is None:
            continue
        
        dark_val = compute_dark_channel(img)
        
        item = (zpath, img_name, dark_val, img.shape[:2])
        if dark_val < DARK_CLEAR_THRESH:
            clear_imgs.append(item)
        elif dark_val > DARK_HAZY_THRESH:
            hazy_imgs.append(item)
        else:
            border_imgs.append(item)
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(all_images)}")
    
    if zf:
        zf.close()
    
    print(f"  清晰图: {len(clear_imgs)} (暗通道<{DARK_CLEAR_THRESH})")
    print(f"  边界图: {len(border_imgs)} (中间区域, 归入清晰)")
    print(f"  真实雾图: {len(hazy_imgs)} (暗通道>{DARK_HAZY_THRESH})")
    
    # 边界图归入清晰图
    clear_imgs.extend(border_imgs)
    print(f"  -> 最终清晰图: {len(clear_imgs)}")
    
    # Step 3: 划分数据集
    print("\n[Step 3] 划分训练/验证/测试集...")
    
    random.shuffle(clear_imgs)
    random.shuffle(hazy_imgs)
    
    n_clear = len(clear_imgs)
    n_hazy = len(hazy_imgs)
    
    n_train_clear = int(n_clear * TRAIN_RATIO)
    n_val_clear = int(n_clear * VAL_RATIO)
    n_test_clear = n_clear - n_train_clear - n_val_clear
    
    train_clear = clear_imgs[:n_train_clear]
    val_clear = clear_imgs[n_train_clear:n_train_clear + n_val_clear]
    test_clear = clear_imgs[n_train_clear + n_val_clear:]
    test_hazy_real = hazy_imgs
    
    print(f"  训练清晰图: {len(train_clear)}")
    print(f"  验证清晰图: {len(val_clear)}")
    print(f"  测试清晰图: {len(test_clear)}")
    print(f"  真实雾图(测试): {len(test_hazy_real)}")
    
    # Step 4: 合成训练雾图
    print("\n[Step 4] 合成训练集雾图...")
    
    train_hazy_dir = os.path.join(DST_DIR, 'train', 'hazy')
    train_clear_dir = os.path.join(DST_DIR, 'train', 'clear')
    
    train_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val, shape) in enumerate(train_clear):
        if zpath != current_zip:
            if zf:
                zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name)
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
    
    if zf:
        zf.close()
    
    print(f"  训练集: {len(train_clear)}张清晰图 → {train_count}张合成雾图")
    
    # Step 5: 合成验证雾图
    print("\n[Step 5] 合成验证集雾图...")
    
    val_hazy_dir = os.path.join(DST_DIR, 'val', 'hazy')
    val_clear_dir = os.path.join(DST_DIR, 'val', 'clear')
    
    val_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val, shape) in enumerate(val_clear):
        if zpath != current_zip:
            if zf:
                zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name)
        if img is None:
            continue
        
        base_name = f'val_{i:04d}'
        save_image(img, val_clear_dir, base_name)
        
        t_min, t_max, A_val = HAZE_LEVELS[1]
        np.random.seed(i + 1000)
        random.seed(i + 1000)
        hazy_img = synthesize_haze(img, t_min, t_max, A_val)
        save_image(hazy_img, val_hazy_dir, f'{base_name}_haze')
        val_count += 1
    
    if zf:
        zf.close()
    
    print(f"  验证集: {val_count}对 (hazy + clear)")
    
    # Step 6: 测试集
    print("\n[Step 6] 构建测试集...")
    
    test_hazy_dir = os.path.join(DST_DIR, 'test', 'hazy_synthetic')
    test_clear_dir = os.path.join(DST_DIR, 'test', 'clear')
    
    test_count = 0
    current_zip = None
    zf = None
    
    for i, (zpath, img_name, dark_val, shape) in enumerate(test_clear):
        if zpath != current_zip:
            if zf:
                zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name)
        if img is None:
            continue
        
        base_name = f'test_{i:04d}'
        save_image(img, test_clear_dir, base_name)
        
        t_min, t_max, A_val = HAZE_LEVELS[1]
        np.random.seed(i + 5000)
        random.seed(i + 5000)
        hazy_img = synthesize_haze(img, t_min, t_max, A_val)
        save_image(hazy_img, test_hazy_dir, f'{base_name}_haze')
        test_count += 1
    
    if zf:
        zf.close()
    
    print(f"  有GT测试: {test_count}对")
    
    # 真实雾图
    test_real_dir = os.path.join(DST_DIR, 'test', 'hazy_real')
    
    current_zip = None
    zf = None
    real_count = 0
    
    for i, (zpath, img_name, dark_val, shape) in enumerate(test_hazy_real):
        if zpath != current_zip:
            if zf:
                zf.close()
            zf = zipfile.ZipFile(zpath, 'r')
            current_zip = zpath
        
        img = read_image_from_zip(zf, img_name)
        if img is None:
            continue
        base_name = f'real_{i:04d}'
        save_image(img, test_real_dir, base_name)
        real_count += 1
    
    if zf:
        zf.close()
    
    print(f"  真实雾图测试: {real_count}张 (无GT)")
    
    # Step 7: 输出统计
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)
    print(f"\n目录结构: {DST_DIR}")
    print(f"  train/")
    print(f"    hazy/     : {train_count} 张 (合成雾图)")
    print(f"    clear/    : {len(train_clear)} 张 (清晰GT)")
    print(f"  val/")
    print(f"    hazy/     : {val_count} 张")
    print(f"    clear/    : {val_count} 张")
    print(f"  test/")
    print(f"    hazy_synthetic/ : {test_count} 张 (有GT)")
    print(f"    clear/          : {test_count} 张")
    print(f"    hazy_real/      : {real_count} 张 (真实雾, 无GT)")
    
    total_synth = train_count + val_count + test_count
    print(f"\n总计合成雾图: {total_synth} 对")
    print(f"真实雾图: {real_count} 张")


if __name__ == '__main__':
    main()
