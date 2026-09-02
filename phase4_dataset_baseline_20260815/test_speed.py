"""测试 - 只做分类，看看速度"""
import os
import cv2
import numpy as np
import zipfile
import time

ZIP_DIR = r'D:\DATA_ALL'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

t0 = time.time()
zip_files = [f for f in os.listdir(ZIP_DIR) if f.lower().endswith('.zip')]

count = 0
for zname in zip_files:
    zpath = os.path.join(ZIP_DIR, zname)
    print(f'Processing {zname} ...', flush=True)
    with zipfile.ZipFile(zpath, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(IMAGE_EXTS)]
        print(f'  {len(names)} images', flush=True)
        
        for i, name in enumerate(names[:50]):  # 只测前50张
            data = zf.read(name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f'  FAILED: {name}', flush=True)
                continue
            count += 1
            if i == 0:
                print(f'  first image shape: {img.shape}', flush=True)

elapsed = time.time() - t0
print(f'\nTotal: {count} images in {elapsed:.1f}s', flush=True)
print(f'Average: {elapsed/count*1000:.0f}ms per image', flush=True)
